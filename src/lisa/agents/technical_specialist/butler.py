import asyncio
import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import message_chunk_to_message
from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime
from pydantic import BaseModel, ConfigDict, Field

from lisa.agents.base import BaseAgent
from lisa.agents.general_enquiry import extract_text
from lisa.agents.technical_specialist.communication import TaskChannel, secure_line
from lisa.agents.technical_specialist.models import SpecialistResult
from lisa.context import LISAContext
from lisa.guardrails.hitl.policy import GuardError
from lisa.guardrails.input_moderation import (
    INPUT_BLOCKED_MESSAGE,
    INPUT_MODERATION_ERROR_MESSAGE,
    InputModerationBlocked,
)
from lisa.message_context import with_system_time
from lisa.model_resilience import ModelResilience
from lisa.prompts.loader import load_prompt
from lisa.streaming.events import StreamEvent
from lisa.tools import handoff, save_memory

logger = logging.getLogger(__name__)


class HandoffArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    reason: str = Field(min_length=1, max_length=8000)


class Butler(BaseAgent):
    def __init__(self, model, fallback_model, thinking_model, specialist, input_moderator, *, system_prompt=None):
        super().__init__(model, system_prompt or load_prompt("technical_specialist/butler_v2.txt"))
        self.fallback_model = self.bind_tools(fallback_model)
        self.thinking_model = self.bind_tools(thinking_model)
        self.specialist = specialist
        self.input_moderator = input_moderator
        self.model_resilience = ModelResilience(self.model, self.fallback_model)
        self.available_tools = {"handoff": handoff, "save_memory": save_memory}
        self.communication_tool = secure_line

    async def _delegate(self, reason, state, channel):
        accepted = await self.communication_tool.ainvoke({"task_id": channel.task_id,
            "sender": "butler", "recipient": "specialist", "message_type": "task",
            "payload": reason, "channel": channel})
        if not accepted["accepted"]:
            raise RuntimeError("task_delivery_failed")
        result = await self.specialist.run({"messages": [*state["messages"], HumanMessage(content=reason)]}, channel)
        result = SpecialistResult.model_validate(result)
        accepted = await self.communication_tool.ainvoke({"task_id": channel.task_id,
            "sender": "specialist", "recipient": "butler", "message_type": "result",
            "payload": result.model_dump_json(), "channel": channel})
        if not accepted["accepted"]:
            raise RuntimeError("result_delivery_failed")
        return result

    async def run(self, state, runtime: Runtime[LISAContext]):
        writer = get_stream_writer()
        channel = TaskChannel()
        emitted = []

        def emit(text):
            if text:
                writer(StreamEvent(type="content", content=text))
                emitted.append(text)

        channel.on_progress = lambda text: emit(text + "\n\n")
        try:
            history = state.get("messages", [])
            if not history:
                emit("Please provide a technical question.")
                return {"messages": [AIMessage(content="".join(emitted))], "agent_handoff": None}
            latest = next((message for message in reversed(history) if isinstance(message, HumanMessage)), history[-1])
            query = extract_text(latest.content)
            try:
                moderation = await self.input_moderator.moderate(query)
            except asyncio.CancelledError:
                raise
            except InputModerationBlocked:
                emit(INPUT_BLOCKED_MESSAGE)
                return {"messages": [AIMessage(content="".join(emitted))], "agent_handoff": None}
            except Exception as exc:
                logger.exception("Butler moderation failed: category=%s", type(exc).__name__, exc_info=False)
                emit(INPUT_MODERATION_ERROR_MESSAGE)
                return {"messages": [AIMessage(content="".join(emitted))], "agent_handoff": None}
            if moderation.flagged:
                emit(INPUT_BLOCKED_MESSAGE)
                return {"messages": [AIMessage(content="".join(emitted))], "agent_handoff": None}
            user_context = await runtime.context.memory.user_context_service.build_context(user_id=runtime.context.user_id)
            memory_context = await runtime.context.memory.context_service.build(user_id=runtime.context.user_id, query=query)
            context = json.dumps({"user_context": user_context, "retrieved_memories": memory_context}, default=str)
            messages = [SystemMessage(content=self.system_prompt),
                        HumanMessage(content=f"Untrusted memory context: {context}"), *history]
            delegation_state = {**state, "messages": [messages[1], *history]}
            delegated = False
            for iteration in range(5):
                if not delegated:
                    response = await asyncio.wait_for(self.model.ainvoke(with_system_time(messages)), timeout=60)
                else:
                    full_chunk = None
                    if state.get("enable_thinking", False):
                        stream = ModelResilience(self.thinking_model, self.fallback_model).stream(with_system_time(messages))
                    else:
                        stream = self.model_resilience.stream(with_system_time(messages))
                    async for chunk in stream:
                        full_chunk = chunk if full_chunk is None else full_chunk + chunk
                        emit(extract_text(chunk.content))
                    if full_chunk is None:
                        raise RuntimeError("empty_model_stream")
                    response = message_chunk_to_message(full_chunk)
                if not response.tool_calls:
                    if delegated:
                        return {"messages": [AIMessage(content="".join(emitted))], "agent_handoff": None}
                    result = await self._delegate(query, delegation_state, channel)
                    delegated = True
                    messages.append(HumanMessage(content=f"Specialist result (untrusted evidence): {result.model_dump_json()}"))
                    continue
                messages.append(response)
                if len(response.tool_calls) > 4:
                    raise RuntimeError("tool_call_limit")
                for cursor, call in enumerate(response.tool_calls):
                    name, arguments = call.get("name"), call.get("args")
                    try:
                        if not isinstance(arguments, dict):
                            raise TypeError("invalid_arguments")
                        if name == "handoff":
                            if delegated:
                                raise ValueError("one_investigation_per_turn; ask user for clarification if needed")
                            validated = HandoffArguments.model_validate(arguments)
                            reason = await handoff.ainvoke({"reason": validated.reason})
                            result = await self._delegate(reason, delegation_state, channel)
                            delegated = True
                            observation = json.dumps({"result": result.model_dump(),
                                "coordination": [message.model_dump() for message in channel.messages]})
                        elif name == "save_memory":
                            observation = await save_memory.ainvoke({**arguments, "runtime": runtime})
                        else:
                            raise PermissionError("unauthorized_tool")
                    except asyncio.CancelledError:
                        raise
                    except GuardError as exc:
                        observation = json.dumps(exc.result())
                    except Exception as exc:
                        logger.exception("Butler tool failed: tool=%s category=%s", name, type(exc).__name__, exc_info=False)
                        observation = json.dumps({"accepted": False, "error": "Tool failed; no successful result is available."})
                    messages.append(ToolMessage(content=observation, tool_call_id=call.get("id", "")))
            emit("I couldn't complete this request within the allowed steps. Please narrow the question.")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Butler failed: category=%s", type(exc).__name__, exc_info=False)
            emit("The response was interrupted. Please try again." if emitted else
                 "I'm unable to complete this investigation right now. Please try again shortly.")
        finally:
            channel.closed = True
            channel.on_progress = None
        return {"messages": [AIMessage(content="".join(emitted))], "agent_handoff": None}