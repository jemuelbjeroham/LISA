from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from lisa.infrastructure.postgres.memory_repository import (
    PostgresMemoryRepository,
)
from lisa.memory.retriever import MemoryRetriever
from lisa.memory.service import MemoryService


@dataclass
class MemoryDependencies:
    service: MemoryService
    retriever: MemoryRetriever

def memory_dependencies(session: AsyncSession) -> MemoryDependencies:
    repository = PostgresMemoryRepository(session)
    service = MemoryService(repository)
    retriever = MemoryRetriever(service)

    return MemoryDependencies(
        service=MemoryService(repository),
        retriever=retriever,
    )