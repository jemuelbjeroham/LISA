from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage

from lisa.memory.decision import MemoryAgentResult


class MemoryAgent:
    def __init__(self, model: BaseChatModel, prompt: str):
        self.model = model.with_structured_output(MemoryAgentResult)
        self.prompt = prompt

    async def analyze(
            self,
            messages: list[BaseMessage],
    ) -> MemoryAgentResult:
        conversation = [
            SystemMessage(content=self.prompt),
            *messages,
        ]

        return await self.model.ainvoke(conversation)