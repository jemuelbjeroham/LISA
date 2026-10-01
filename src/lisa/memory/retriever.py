from typing import Protocol
from uuid import UUID

from lisa.memory.retriever import Memory


class MemoryRetriever(Protocol):
    async def retrieve(
            self,
            user_id: UUID,
            query: str,
            top_k: int = 5,
    ) -> list[Memory]:
        ...
        