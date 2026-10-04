from uuid import UUID

from langchain_core.messages import BaseMessage

from lisa.agents.mem_agent import MemoryAgent
from lisa.memory.decision import MemoryAction
from lisa.memory.models import Memory
from lisa.memory.service import MemoryService


class MemoryWorkflow:
    def __init__(
            self,
            agent: MemoryAgent,
            service: MemoryService,
    ) -> None:
        self.agent = agent
        self.service = service

    async def process(
            self,
            user_id: UUID,
            messages: list[BaseMessage],
    ) -> list[Memory]:
        result = await self.agent.analyze(messages)
        saved_memories: list[Memory] = []

        for decision in result.decisions:
            if decision.action != MemoryAction.SAVE:
                continue

            if (
                decision.content is None
                or decision.scope is None
                or decision.scope is None
                or decision.type is None
                or decision.source is None
            ):
                continue

            memory = Memory(
                user_id=user_id,
                scope=decision.scope,
                type=decision.type,
                content=decision.content,
                source=decision.source,
                confidence=decision.confidence,
                importance=decision.importance,
            )

            saved = await self.service.create(memory)
            saved_memories.append(saved)

        return saved_memories