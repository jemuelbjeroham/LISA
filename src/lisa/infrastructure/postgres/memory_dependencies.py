from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from lisa.infrastructure.postgres.memory_repository import (
    PostgresMemoryRepository,
)
from lisa.memory.context import MemoryContextBuilder
from lisa.memory.context_service import MemoryContextService
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

    context_service = MemoryContextService(
        retriever=retriever,
        builder=MemoryContextBuilder()
    )
    return MemoryDependencies(
        service=MemoryService(repository),
        retriever=retriever,
        context_service=context_service,
    )