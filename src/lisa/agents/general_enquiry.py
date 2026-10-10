
import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.utils import message_chunk_to_message
from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from lisa.agents.base import BaseAgent
from lisa.context import LISAContext
from lisa.model_resilience import ModelResilience
from lisa.state import LISAState
from lisa.streaming.events import StreamEvent
from lisa.tools import handoff, save_memory

logger = logging.getLogger(__name__)


class GeneralEnquiry(BaseAgent):

    def __init__(
        self,
        model: BaseChatModel,
        fallback_model: BaseChatModel,
        thinking_model: BaseChatModel,
        system_prompt: str,
    ):
        super().__init__(
            model=model,
            system_prompt=system_prompt,
        )

        self.fallback_model = self.bind_tools(fallback_model)
        self.thinking_model = self.bind_tools(thinking_model)

        self.model_resilience = ModelResilience(
            primary=self.model,
            fallback=self.fallback_model,
            inactivity_timeout_seconds=10.0,
        )

        self.available_tools = {
            "handoff": handoff,
            "save_memory": save_memory,
        }

    async def run(
        self,
        state: LISAState,
        runtime: Runtime[LISAContext],
    ):
        last_message = state["messages"][-1]

        user_query = (
            last_message.content
            if isinstance(last_message.content, str)
            else str(last_message.content)
        )

        memory_context = (
            await runtime.context.memory.context_service.build(
                user_id=runtime.context.user_id,
                query=user_query,
            )
        )

        system_content = self.system_prompt

        if memory_context:
            system_content = f"{system_content}\n\n{memory_context}"

        messages = [
            SystemMessage(content=system_content),
            *state["messages"],
        ]

        writer = get_stream_writer()

        # Continue the model/tool cycle until the model produces
        # a response without requesting another tool.
        while True:
            response_chunks = []
            full_chunk = None

            if state["enable_thinking"]:
                model = self.thinking_model

                logger.info(
                    "GeneralEnquiry using thinking model: %s",
                    type(model).__name__,
                )

                stream = model.astream(messages)

            else:
                logger.info(
                    "GeneralEnquiry using resilient model: "
                    "primary=%s fallback=%s",
                    type(self.model).__name__,
                    type(self.fallback_model).__name__,
                )

                stream = self.model_resilience.stream(messages)

            # Stream the model response immediately while also
            # assembling the complete message for tool inspection.
            async for chunk in stream:
                full_chunk = (
                    chunk
                    if full_chunk is None
                    else full_chunk + chunk
                )

                reasoning = chunk.additional_kwargs.get(
                    "reasoning_content"
                )

                if reasoning:
                    writer(
                        StreamEvent(
                            type="reasoning",
                            content=reasoning,
                        )
                    )

                if chunk.content:
                    writer(
                        StreamEvent(
                            type="content",
                            content=chunk.content,
                        )
                    )

                    response_chunks.append(chunk.content)

            if full_chunk is None:
                logger.warning(
                    "GeneralEnquiry received an empty model stream"
                )

                return {
                    "messages": [
                        AIMessage(content="")
                    ]
                }

            assistant_message = message_chunk_to_message(
                full_chunk
            )

            # Handle handoff before executing any other tools.
            for tool_call in assistant_message.tool_calls:
                if tool_call["name"] == "handoff":
                    logger.info(
                        "Agent requested handoff: %s",
                        tool_call["args"],
                    )

                    return {
                        "agent_handoff": tool_call["args"],
                    }

            # Preserve the assistant message, including its tool calls.
            messages.append(assistant_message)

            # No tool calls means this is the final model response.
            if not assistant_message.tool_calls:
                final_response = "".join(response_chunks)

                return {
                    "messages": [
                        AIMessage(content=final_response)
                    ]
                }

            # Execute requested tools and feed their results back
            # into the conversation before invoking the model again.
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call["name"]
                tool_args = dict(tool_call["args"])

                tool = self.available_tools.get(tool_name)

                if tool is None:
                    logger.warning(
                        "Unknown tool requested: %s",
                        tool_name,
                    )

                    messages.append(
                        ToolMessage(
                            content=f"Unknown tool: {tool_name}",
                            tool_call_id=tool_call["id"],
                        )
                    )
                    continue

                if tool_name == "save_memory":
                    tool_args["runtime"] = runtime

                logger.info(
                    "GeneralEnquiry executing tool: %s",
                    tool_name,
                )

                result = await tool.ainvoke(tool_args)

                messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call["id"],
                    )
                )

            # The next iteration calls the model again with
            # the assistant tool calls and their results.
