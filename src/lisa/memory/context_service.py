from uuid import UUID

from lisa.infrastructure.postgres.memory_retriever import (
    PostgresMemoryRetriever,
)
from lisa.memory.context import MemoryContextBuilder


class MemoryContextService:
    def __init__(
        self,
        retriever: PostgresMemoryRetriever,
        builder: MemoryContextBuilder,
    ):
        self.retriever = retriever
        self.builder = builder

    async def build(
        self,
        user_id: UUID,
        query: str,
        *,
        top_k: int = 5,
    ) -> str:
        memories = await self.retriever.retrieve(
            user_id,
            query,
            top_k=top_k,
        )

        return self.builder.build(memories)