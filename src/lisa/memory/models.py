from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class MemoryType(StrEnum):
    FACT = "fact"
    PREFERENCE = "preference"
    INSTRUCTION = "instruction"
    EPISODIC = "episodic"
    PROJECT = "project"

class MemoryScope(StrEnum):
    USER = "user"
    PROJECT = "project"

class MemorySource(StrEnum):
    EXPLICIT = "explicit"
    INFERRED = "inferred"

class Memory(BaseModel):
    id: UUID = Field(default_factory=uuid4)

    user_id: UUID
    scope: MemoryScope
    type: MemoryType

    content: str

    valid_from: datetime | None = None
    valid_to: datetime | None = None

    recorded_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )

    source: MemorySource
    confidence: float = Field(ge=0.0, le=1.0)
    importance: float = Field(ge=0.0, le=1.0)

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )
    last_accessed_at: datetime | None = None


