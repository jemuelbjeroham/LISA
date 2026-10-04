from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from lisa.infrastructure.postgres.models import MemoryModel
from lisa.memory.models import (
    Memory,
    MemoryScope,
    MemorySource,
    MemoryType,
)
from lisa.memory.repository import MemoryRepository


class PostgresMemoryRepository(MemoryRepository):
    def __init__(self, session: AsyncSession):
        self.session = session

    @staticmethod
    def _to_model(memory: Memory) -> MemoryModel:
        return MemoryModel(
            id=memory.id,
            user_id=memory.user_id,
            scope=memory.scope.value,
            type=memory.type.value,
            content=memory.content,
            source=memory.source.value,
            confidence=memory.confidence,
            importance=memory.importance,
            created_at=memory.created_at,
            updated_at=memory.updated_at,
            last_accessed_at=memory.last_accessed_at,
        )

    @staticmethod
    def _to_domain(model: MemoryModel) -> Memory:
        return Memory(
            id=model.id,
            user_id=model.user_id,
            scope=MemoryScope(model.scope),
            type=MemoryType(model.type),
            content=model.content,
            source=MemorySource(model.source),
            confidence=model.confidence,
            importance=model.importance,
            created_at=model.created_at,
            updated_at=model.updated_at,
            last_accessed_at=model.last_accessed_at,
        )

    async def create(self, memory: Memory) -> Memory:
        model = self._to_model(memory)

        self.session.add(model)
        await self.session.commit()
        await self.session.refresh(model)

        return self._to_domain(model)

    async def get(
            self,
            memory_id: UUID,
    ) -> Memory | None:
        result = await self.session.execute(
            select(MemoryModel).where(
                MemoryModel.id == memory_id
            )
        )

        model = result.scalar_one_or_none()

        if model is None:
            return None

        return self._to_domain(model)

    async def update(self, memory: Memory) -> Memory:
        model = await self.session.get(
            MemoryModel,
            memory.id,
        )

        if model is None:
            raise ValueError(
                f"Memory not found: {memory.id}"
            )

        model.user_id = memory.user_id
        model.scope = memory.scope.value
        model.type = memory.type.value
        model.content = memory.content
        model.source = memory.source.value
        model.confidence = memory.confidence
        model.importance = memory.importance
        model.created_at = memory.created_at
        model.updated_at = memory.updated_at
        model.last_accessed_at = memory.last_accessed_at

        await self.session.commit()
        await self.session.refresh(model)

        return self._to_domain(model)

    async def delete(self, memory_id: UUID) -> None:
        model = await self.session.get(
            MemoryModel,
            memory_id,
        )

        if model is None:
            return

        await self.session.delete(model)
        await self.session.commit()

    async def find_duplicate(
            self,
            memory: Memory,
    ) -> Memory | None:

        result = await self.session.execute(
            select(MemoryModel).where(
                MemoryModel.user_id == memory.user_id,
                MemoryModel.scope == memory.scope.value,
                MemoryModel.type == memory.type.value,
            )
        )

        normalized_content = " ".join(
            memory.content.split()
        ).casefold()

        for model in result.scalars():
            existing_content = " ".join(
                model.content.split()
            ).casefold()

            if existing_content == normalized_content:
                return self._to_domain(model)

        return None
    