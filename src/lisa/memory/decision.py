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
    scope: MemoryScope
    type: MemoryType
    source: MemorySource
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
    preference_category: str | None = Field(
        default=None,
        description=(
            "Category for a saved preference, such as communication "
            "or response_style. Required when saving a preference."
        ),
    )
    preference_key: str | None = Field(
        default=None,
        description=(
            "Stable key identifying the preference, such as "
            "preferred_name or explanation_format."
        ),
    )
    preference_value: str | int | float | bool | list[str] | dict[str, str] | None = Field(
        default=None,
        description=(
            "Structured value of the preference. Use a suitable "
            "JSON-compatible value rather than a natural-language sentence."
        ),
    )


class MemoryAgentResult(BaseModel):
    """The memory agent's decision for a conversation analysis"""

    decisions: list[MemoryDecision] = Field(default_factory=list)