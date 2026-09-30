from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from lisa.infrastructure.postgres.database import Database
from lisa.infrastructure.postgres.memory_repository import (
    PostgresMemoryRepository,
)
from lisa.memory.models import Memory, MemoryScope, MemorySource, MemoryType


async def test_memory_repository_crud():
    from lisa.config import Settings

    database = Database(Settings())

    try:
        async with database.session() as session:
            assert isinstance(session, AsyncSession)

            repository = PostgresMemoryRepository(session)

            user_id = uuid4()
            memory = Memory(
                user_id=user_id,
                scope=MemoryScope.USER,
                type=MemoryType.PREFERENCE,
                content="User prefers concise technical explanations.",
                source=MemorySource.EXPLICIT,
                confidence=1.0,
                importance=0.8,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            # Create
            created = await repository.create(memory)

            assert created.id == memory.id
            assert created.user_id == user_id
            assert created.content == memory.content

            # Read
            retrieved = await repository.get(memory.id)

            assert retrieved is not None
            assert retrieved.id == memory.id
            assert retrieved.scope == MemoryScope.USER
            assert retrieved.type == MemoryType.PREFERENCE
            assert retrieved.source == MemorySource.EXPLICIT
            assert retrieved.content == (
                "User prefers concise technical explanations."
            )

            # Update
            retrieved.content = "User prefers concise technical explanations with examples."
            retrieved.importance = 0.9
            retrieved.updated_at = datetime.now(timezone.utc)

            updated = await repository.update(retrieved)

            assert updated.content == (
                "User prefers concise technical explanations with examples."
            )
            assert updated.importance == 0.9

            # Delete
            await repository.delete(memory.id)

            deleted = await repository.get(memory.id)

            assert deleted is None

    finally:
        await database.close()