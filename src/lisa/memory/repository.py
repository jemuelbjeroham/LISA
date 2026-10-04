from typing import Protocol
from uuid import UUID

from lisa.memory.models import Memory


class MemoryRepository(Protocol):

    async def create(
            self,
            memory: Memory,
    ) -> Memory:
        ...

    async def get(
            self,
            memory_id: UUID,
    ) -> Memory | None:
        ...

    async def update(
            self,
            memory: Memory,
    ) -> Memory:
        ...

    async def delete(
            self,
            memory_id: UUID,
    ) -> None:
        ...

    async def find_duplicate(
            self,
            memory: Memory,
    ) -> Memory | None:
        ...

    