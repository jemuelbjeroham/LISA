from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from lisa.infrastructure.postgres.memory_repository import (
    PostgresMemoryRepository,
)
from lisa.memory.context import MemoryContextBuilder
from lisa.memory.context_service import MemoryContextService
from lisa.memory.retriever import MemoryRetriever
from lisa.memory.service import MemoryService
from lisa.memory.user_context.repository import UserContextRepository
from lisa.memory.user_context.service import UserContextService


@dataclass
class MemoryDependencies:
    service: MemoryService
    retriever: MemoryRetriever
    context_service: MemoryContextService
    user_context_service: UserContextService

def memory_dependencies(session: AsyncSession) -> MemoryDependencies:
    repository = PostgresMemoryRepository(session)
    service = MemoryService(repository)
    retriever = MemoryRetriever(service)

    context_service = MemoryContextService(
        retriever=retriever,
        builder=MemoryContextBuilder()
    )

    user_context_repository = UserContextRepository(session)
    user_context_service = UserContextService(user_context_repository)

    return MemoryDependencies(
        service=MemoryService(repository),
        retriever=retriever,
        context_service=context_service,
        user_context_service=user_context_service,
    )