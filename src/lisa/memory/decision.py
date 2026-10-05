from uuid import UUID
from pydantic import BaseModel, Field

from lisa.memory.models import MemoryScope, MemorySource, MemoryType


class MemoryAction:
    IGNORE = "ignore"
    SAVE = "save"
    UPDATE = "update"

class MemoryDecision(BaseModel):
    """A single decision made by the memory agent."""

    action: str = Field(
        description="Whether to ignore, save, or update this memory.",
    )
    content: str | None = Field(
        default=None,
        description="A concise, normalized memory statement.",
    )
    scope: MemoryScope | None = None
    type: MemoryType | None = None
    source: MemorySource | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    importance: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = Field(
        default="",
        description="Brief explanation for the decision",
    )
    target_memory_id: UUID | None = Field(
        default=None,
        description=(
            "ID of the existing memory being updated. "
            "Only set when the decision updates a specific existing memory."
        ),
    )

class MemoryAgentResult(BaseModel):
    """The memory agent's decision for a conversation analysis"""

    decisions: list[MemoryDecision] = Field(default_factory=list)