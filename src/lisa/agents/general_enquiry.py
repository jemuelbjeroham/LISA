
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.config import get_stream_writer

from lisa.agents.base import BaseAgent
from lisa.state import LISAState
from lisa.streaming.events import StreamEvent


class GeneralEnquiry(BaseAgent):

    def __init__(self, model: BaseChatModel, thinking_model: BaseChatModel, system_prompt: str):
        super().__init__(
            model=model,
            system_prompt=system_prompt,
        )
        self.thinking_model = thinking_model

    async def run(self, state: LISAState):

        model = (
            self.thinking_model
            if state["enable_thinking"]
            else self.model
        )

        messages = [
            SystemMessage(content=self.system_prompt),
            *state["messages"],
        ]

        writer = get_stream_writer()

        response_chunks = []
        async for chunk in model.astream(messages):

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

            final_response = "".join(response_chunks)

        return {
            "messages": [
                AIMessage(content=final_response)
            ]
        }


    async def stream(self, state: LISAState):

        model = (
            self.thinking_model
            if state["enable_thinking"]
            else self.model
        )

        messages = [
            SystemMessage(content=self.system_prompt),
            *state["messages"],
        ]

        async for chunk in model.astream(messages):
            yield chunk