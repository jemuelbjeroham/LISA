from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from lisa.infrastructure.postgres.memory_repository import (
    PostgresMemoryRepository,
)
from lisa.infrastructure.postgres.models import MemoryModel
from lisa.memory.models import Memory
from lisa.memory.retriever import MemoryRetriever


class PostgresMemoryRetriever(MemoryRetriever):
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.repository = PostgresMemoryRepository(session)

    async def retrieve(
            self,
            user_id: UUID,
            query: str,
            top_k: int = 5,
    ) -> list[Memory]:
        if top_k <= 0:
            return []

        statement = (
            select(MemoryModel)
            .where(
                MemoryModel.user_id == user_id,
                MemoryModel.confidence >= 0.5,
            )
            .order_by(
                MemoryModel.importance.desc(),
                MemoryModel.updated_at.desc(),
            )
            .limit(top_k)
        )

        result = await self.session.execute(statement)

        models = result.scalars().all()

        return [
            self.repository._to_domain(model)
            for model in models
        ]