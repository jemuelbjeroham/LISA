import logging
from uuid import UUID

from langchain_core.messages import BaseMessage

from lisa.agents.mem_agent import MemoryAgent
from lisa.memory.decision import MemoryAction
from lisa.memory.models import Memory
from lisa.memory.service import MemoryService

logger = logging.getLogger(__name__)

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
        existing_memories = await self.service.list_by_user(user_id)
        result = await self.agent.analyze(
            messages,
            existing_memories=existing_memories,
        )
        logger.info(
            "MemoryAgent returned %d decisions: %s",
            len(result.decisions),
            [
                {
                    "action": decision.action,
                    "content": decision.content,
                    "reason": decision.reason,
                }
                for decision in result.decisions
            ],
        )
        saved_memories: list[Memory] = []

        for decision in result.decisions:
            logger.info(
                "Checking memory decision: action=%r, expected=%r, match=%s",
                decision.action,
                MemoryAction.SAVE,
                decision.action == MemoryAction.SAVE,
            )
            if decision.action != MemoryAction.SAVE:
                continue

            logger.info(
                "Decision fields: content=%r, scope=%r, type=%r, source=%r, "
                "confidence=%r, importance=%r",
                decision.content,
                decision.scope,
                decision.type,
                decision.source,
                decision.confidence,
                decision.importance,
            )

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

            duplicate = await self.service.find_duplicate(memory)

            if duplicate is not None:
                logger.info(
                    "Duplicate memory detected: memory_id=%s, content=%r",
                    duplicate.id,
                    duplicate.content,
                )
                continue

            saved = await self.service.create(memory)
            saved_memories.append(saved)

        return saved_memories