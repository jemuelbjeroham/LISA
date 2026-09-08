from langchain_core.messages import AIMessage, SystemMessage

from lisa.agents.base import BaseAgent
from lisa.state import LISAState


class GeneralEnquiry(BaseAgent):

    async def run(self, state: LISAState):

        messages = [
            SystemMessage(content=self.system_prompt),
            *state["messages"],
        ]

        response_chunks = []
        async for chunk in self.model.astream(messages):
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

        messages = [
            SystemMessage(content=self.system_prompt),
            *state["messages"],
        ]

        async for chunk in self.model.astream(messages):
            yield chunk