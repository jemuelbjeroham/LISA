import asyncio
import json
import logging

from jsonschema import validate
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from lisa.agents.base import BaseAgent
from lisa.agents.technical_specialist.communication import TaskChannel, secure_line
from lisa.agents.technical_specialist.models import Evidence, SpecialistResult
from lisa.guardrails.hitl.dispatcher import check_tool
from lisa.guardrails.hitl.policy import GuardError
from lisa.message_context import with_system_time
from lisa.prompts.loader import load_prompt

logger = logging.getLogger(__name__)

AUTHORIZED_MCP_TOOLS = frozenset({"search_knowledge", "web_search", "web_search_with_content"})


class Specialist(BaseAgent):
    @staticmethod
    def bind_tools(model):
        return model

    def __init__(self, model, mcp_client, *, system_prompt=None, max_iterations=6, timeout_seconds=60):
        super().__init__(model, system_prompt or load_prompt("technical_specialist/specialist_v2.txt"))
        self.raw_model = model
        self.mcp_client = mcp_client
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds
        self.tool_schemas = {}
        self.initialized = False

    async def initialize_tools(self):
        discovered = await self.mcp_client.list_tools()
        schemas = []
        for entry in discovered.tools:
            if entry.name not in AUTHORIZED_MCP_TOOLS:
                continue
            self.tool_schemas[entry.name] = entry.input_schema
            schemas.append({"type": "function", "function": {"name": entry.name,
                "description": entry.description or "", "parameters": entry.input_schema}})
        self.model = self.raw_model.bind_tools([secure_line, *schemas], tool_choice="auto")
        self.initialized = True

    async def run(self, state, channel: TaskChannel | None = None):
        evidence, limitations = [], []
        if channel is None or channel.closed:
            return self._failed("Missing active task context.")
        if not self.initialized:
            return self._failed("Specialist tools were not initialized.")
        messages = [SystemMessage(content=self.system_prompt), *state.get("messages", []),
                    HumanMessage(content=f"Active task_id: {channel.task_id}")]
        await self._progress(channel, "I'm assessing the request and deciding which sources to check.")
        for iteration in range(self.max_iterations):
            try:
                response = await asyncio.wait_for(self.model.ainvoke(with_system_time(messages)), self.timeout_seconds)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Specialist model failed: category=%s", type(exc).__name__, exc_info=False)
                return self._failed("Investigation model unavailable.", evidence)
            messages.append(response)
            if not response.tool_calls:
                break
            for call in response.tool_calls[:8]:
                name, arguments = call.get("name"), call.get("args")
                call_id = call.get("id", "")
                try:
                    if not isinstance(arguments, dict):
                        raise TypeError("invalid_arguments")
                    if name == "secure_line":
                        if arguments.get("message_type") == "progress":
                            raise PermissionError("progress_is_execution_controlled")
                        if arguments.get("sender") != "specialist" or arguments.get("recipient") != "butler":
                            raise PermissionError("invalid_sender_or_recipient")
                        result = await secure_line.ainvoke({**arguments, "channel": channel})
                        if result["accepted"] and arguments.get("message_type") == "clarification":
                            return SpecialistResult(status="needs_input", summary=arguments["payload"],
                                evidence=evidence, limitations=limitations, next_action="request_clarification")
                        observation = json.dumps(result)
                    else:
                        if name not in self.tool_schemas or name not in AUTHORIZED_MCP_TOOLS:
                            raise PermissionError("unauthorized_tool")
                        validate(arguments, self.tool_schemas[name])
                        await check_tool(name)
                        description = {"search_knowledge": "the internal knowledge base",
                            "web_search": "public search results", "web_search_with_content": "public source content"}[name]
                        await self._progress(channel, f"I'm checking {description}.")
                        output = await asyncio.wait_for(self.mcp_client.call_tool(name, arguments), self.timeout_seconds)
                        if output.is_error:
                            raise RuntimeError("tool_reported_failure")
                        observation = "\n".join(item.text for item in output.content if hasattr(item, "text"))
                        evidence.append(Evidence(tool=name, call_id=call_id, content=observation))
                        await self._progress(channel, f"The check of {description} returned. I'm reviewing the response; it may not contain sufficient evidence.")
                    messages.append(ToolMessage(content=observation, tool_call_id=call_id))
                except asyncio.CancelledError:
                    raise
                except GuardError as exc:
                    messages.append(ToolMessage(content=json.dumps(exc.result()), tool_call_id=call_id))
                except Exception as exc:
                    logger.exception("Specialist tool failed: tool=%s category=%s", name, type(exc).__name__, exc_info=False)
                    limitation = f"Tool {name} failed ({type(exc).__name__}); no result verified."
                    limitations.append(limitation)
                    if name in self.tool_schemas:
                        await self._progress(channel, "A source check failed. I'll account for that limitation in the findings.")
                    messages.append(ToolMessage(content=json.dumps({"accepted": False, "error": limitation}), tool_call_id=call_id))
            if len(response.tool_calls) > 8:
                return self._failed("Too many tool calls in a single step.", evidence)
        else:
            limitations.append("Investigation iteration limit reached.")
        await self._progress(channel, "I'm consolidating the findings and checking their limitations.")
        return await self._finish(messages, evidence, limitations)

    async def _progress(self, channel, text):
        receipt = await secure_line.ainvoke({"task_id": channel.task_id, "sender": "specialist",
            "recipient": "butler", "message_type": "progress", "payload": text, "channel": channel})
        if receipt["accepted"] and channel.on_progress is not None:
            channel.on_progress(text)

    async def _finish(self, messages, evidence, limitations):
        for attempt in range(2):
            try:
                structured = self.raw_model.with_structured_output(SpecialistResult)
                result = await asyncio.wait_for(structured.ainvoke(with_system_time(messages)), self.timeout_seconds)
                result = SpecialistResult.model_validate(result)
                result = result.model_copy(update={"evidence": evidence, "limitations": [*result.limitations, *limitations]})
                if result.status == "completed" and (limitations or not evidence):
                    result = result.model_copy(update={"status": "partial", "limitations": [*result.limitations,
                        "Completion could not be verified against sufficient successful tool execution."]})
                return SpecialistResult.model_validate(result.model_dump())
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Specialist result validation failed: category=%s", type(exc).__name__, exc_info=False)
                messages = [*messages, HumanMessage(content="Return a valid structured result; report limitations honestly.")]
        return self._failed("Structured investigation result could not be validated.", evidence)

    @staticmethod
    def _failed(limitation, evidence=None):
        return SpecialistResult(status="failed", summary="Investigation could not be completed.",
            evidence=evidence or [], limitations=[limitation], next_action="present_result")