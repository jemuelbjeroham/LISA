from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from lisa.infrastructure.postgres.memory_repository import (
    PostgresMemoryRepository,
)
from lisa.infrastructure.postgres.memory_retriever import (
    PostgresMemoryRetriever,
)
from lisa.memory.service import MemoryService


@dataclass
class MemoryDependencies:
    service: MemoryService
    retriever: PostgresMemoryRetriever


def memory_dependencies(session: AsyncSession) -> MemoryDependencies:
    repository = PostgresMemoryRepository(session)
    retriever = PostgresMemoryRetriever(session)

    return MemoryDependencies(
        service=MemoryService(repository),
        retriever=retriever,
    )