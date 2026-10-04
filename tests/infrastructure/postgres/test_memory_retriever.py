from datetime import UTC, datetime, timedelta
from uuid import uuid4

from lisa.config import Settings
from lisa.infrastructure.postgres.database import Database
from lisa.infrastructure.postgres.memory_repository import (
    PostgresMemoryRepository,
)
from lisa.infrastructure.postgres.memory_retriever import (
    PostgresMemoryRetriever,
)
from lisa.memory.models import (
    Memory,
    MemoryScope,
    MemorySource,
    MemoryType,
)


async def test_memory_retriever_filters_and_ranks_memories():
    database = Database(Settings())
    user_id = uuid4()
    another_user_id = uuid4()
    now = datetime.now(UTC)

    memories = [
        Memory(
            user_id=user_id,
            scope=MemoryScope.USER,
            type=MemoryType.PREFERENCE,
            content="High importance, older memory",
            source=MemorySource.EXPLICIT,
            confidence=1.0,
            importance=0.9,
            created_at=now - timedelta(days=10),
            updated_at=now - timedelta(days=10),
        ),
        Memory(
            user_id=user_id,
            scope=MemoryScope.USER,
            type=MemoryType.FACT,
            content="Lower importance, newer memory",
            source=MemorySource.EXPLICIT,
            confidence=1.0,
            importance=0.7,
            created_at=now,
            updated_at=now,
        ),
        Memory(
            user_id=user_id,
            scope=MemoryScope.USER,
            type=MemoryType.FACT,
            content="Low confidence memory",
            source=MemorySource.INFERRED,
            confidence=0.2,
            importance=1.0,
            created_at=now,
            updated_at=now,
        ),
        Memory(
            user_id=another_user_id,
            scope=MemoryScope.USER,
            type=MemoryType.FACT,
            content="Another user's memory",
            source=MemorySource.EXPLICIT,
            confidence=1.0,
            importance=1.0,
            created_at=now,
            updated_at=now,
        ),
    ]

    try:
        async with database.session() as session:
            repository = PostgresMemoryRepository(session)
            retriever = PostgresMemoryRetriever(session)

            for memory in memories:
                await repository.create(memory)

            results = await retriever.retrieve(
                user_id=user_id,
                query="user preferences",
                top_k=2,
            )

            assert len(results) == 2
            assert results[0].content == "High importance, older memory"
            assert results[1].content == "Lower importance, newer memory"

            assert all(memory.user_id == user_id for memory in results)
            assert all(memory.confidence >= 0.5 for memory in results)

    finally:
        async with database.session() as session:
            repository = PostgresMemoryRepository(session)
            for memory in memories:
                await repository.delete(memory.id)

        await database.close()