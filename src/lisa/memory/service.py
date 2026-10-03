from uuid import UUID

from lisa.memory.models import Memory
from lisa.memory.repository import MemoryRepository


class MemoryService:
    def __init__(self, repository: MemoryRepository) -> None:
        self.repository = repository

    async def create(self, memory: Memory) -> Memory:
        return await self.repository.create(memory)

    async def get(self, memory_id: UUID) -> Memory | None:
        return await self.repository.get(memory_id)

    async def update(self, memory: Memory) -> Memory:
        return await self.repository.update(memory)

    async def delete(self, memory_id: UUID) -> None:
        await self.repository.delete(memory_id)
