import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.config import get_stream_writer

from lisa.agents.base import BaseAgent
from lisa.model_resilience import ModelResilience
from lisa.state import LISAState
from lisa.streaming.events import StreamEvent

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

        self.thinking_model = thinking_model

        self.model_resilience = ModelResilience(
            primary=self.model,
            fallback=self.fallback_model,
            inactivity_timeout_seconds=10.0,
        )

    async def run(self, state: LISAState):

        messages = [
            SystemMessage(content=self.system_prompt),
            *state["messages"],
        ]

        writer = get_stream_writer()

        response_chunks = []

        if state["enable_thinking"]:
            model = self.thinking_model
            logger.info(
                "GeneralEnquiry using thinking model: %s",
                type(model).__name__,
            )

            stream = model.astream(messages)

        else:
            model = self.model_resilience
            logger.info(
                "GeneralEnquiry using resilient model: primary=%s fallback=%s",
                type(self.model).__name__,
                type(self.fallback_model).__name__,
            )

            stream = model.stream(messages)

        async for chunk in stream:

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

            if chunk.tool_calls:
                for tool_call in chunk.tool_calls:
                    if tool_call["name"] == "handoff":
                        logger.info(
                            "Agent requested handoff: %s",
                            tool_call["args"],
                        )

                        return {
                            "agent_handoff": tool_call["args"],
                        }

            if chunk.content:
                writer(
                    StreamEvent(
                        type="content",
                        content=chunk.content,
                    )
                )

                response_chunks.append(chunk.content)

        final_response = "".join(response_chunks)

        return {
            "messages": [
                AIMessage(content=final_response)
            ]
        }