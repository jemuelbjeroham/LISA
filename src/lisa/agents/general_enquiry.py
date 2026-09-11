from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage

from lisa.agents.base import BaseAgent
from lisa.state import LISAState


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

        response_chunks = []
        async for chunk in model.astream(messages):
            response_chunks.append(chunk)

        final_response = "".join(
            chunk.content
            for chunk in response_chunks
            if chunk.content
        )
        return {
            "messages": [AIMessage(content=final_response)]
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