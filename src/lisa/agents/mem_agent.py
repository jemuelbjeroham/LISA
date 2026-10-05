from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage

from lisa.memory.decision import MemoryAgentResult
from lisa.memory.models import Memory


class MemoryAgent:
    def __init__(self, model: BaseChatModel, prompt: str):
        self.model = model.with_structured_output(MemoryAgentResult)
        self.prompt = prompt

    async def analyze(
            self,
            messages: list[BaseMessage],
            existing_memories: list[Memory] | None = None,
    ) -> MemoryAgentResult:
        memory_context = ""

        if existing_memories:
            memory_context = "\n\nExisting memories:\n" + "\n".join(
                f"- ID: {memory.id} | "
                f"Type: {memory.type.value} | "
                f"Content: {memory.content}"
                for memory in existing_memories
            )
        conversation = [
            SystemMessage(content=self.prompt + memory_context),
            *messages,
        ]

        return await self.model.ainvoke(conversation)