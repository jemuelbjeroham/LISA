from langchain_core.messages import SystemMessage

from lisa.agents.base import BaseAgent
from lisa.state import LISAState


class GeneralEnquiry(BaseAgent):

    async def run(self, state: LISAState):

        messages = [
            SystemMessage(content=self.system_prompt),
            *state["messages"],
        ]

        response = self.model.invoke(messages)

        return {
            "messages": [response]
        }

    async def stream(self, state: LISAState):

        messages = [
            SystemMessage(content=self.system_prompt),
            *state["messages"],
        ]

        async for chunk in self.model.astream(messages):
            yield chunk