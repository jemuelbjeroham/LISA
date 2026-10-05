import logging
from datetime import UTC, datetime
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
                "Checking memory decision: action=%r",
                decision.action,
            )

            if decision.action == MemoryAction.UPDATE:
                if decision.target_memory_id is None:
                    logger.warning(
                        "UPDATE decision missing target_memory_id; skipping"
                    )
                    continue

                if (
                    decision.content is None
                    or decision.scope is None
                    or decision.type is None
                    or decision.source is None
                ):
                    logger.warning(
                        "UPDATE decision missing required fields; skipping"
                    )
                    continue

                target = await self.service.get(
                    decision.target_memory_id
                )

                if target is None:
                    logger.warning(
                        "UPDATE target memory not found: memory_id=%s",
                        decision.target_memory_id,
                    )
                    continue

                if target.user_id != user_id:
                    logger.warning(
                        "UPDATE target belongs to a different user: memory_id=%s",
                        target.id,
                    )
                    continue

                now = datetime.now(UTC)

                # Close the historical memory.
                target.valid_to = now
                target.updated_at = now

                await self.service.update(target)

                # Create the new current version.
                updated_memory = Memory(
                    user_id=user_id,
                    scope=decision.scope,
                    type=decision.type,
                    content=decision.content,
                    source=decision.source,
                    confidence=decision.confidence,
                    importance=decision.importance,
                    valid_from=now,
                    valid_to=None,
                )

                saved = await self.service.create(updated_memory)
                saved_memories.append(saved)

                logger.info(
                    "Memory superseded: old_id=%s, new_id=%s",
                    target.id,
                    saved.id,
                )

                continue



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
                valid_from=now,
                valid_to=None,
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