
from datetime import UTC, datetime
from typing import Annotated

from langchain_core.tools import InjectedToolArg, tool
from langgraph.runtime import Runtime

from lisa.context import LISAContext
from lisa.memory.models import (
    Memory,
    MemoryScope,
    MemorySource,
    MemoryType,
)


@tool
def handoff(reason: str) -> str:
    """Handoff the current request back to the orchestrator"""
    return reason

@tool
async def save_memory(
    content: str,
    memory_type: MemoryType,
    scope: MemoryScope,
    runtime: Annotated[Runtime[LISAContext], InjectedToolArg],
) -> str:
    """Save information the user explicitly asks LISA to remember.

    Use this tool when the user explicitly asks to remember, save,
    or store a fact, preference, instruction, or project detail.
    Do not claim the information was saved unless this tool succeeds.
    """
    content = content.strip()

    if not content:
        return "Memory was not saved: content cannot be empty"

    user_id = runtime.context.user_id
    memory_service = runtime.context.memory.service

    memory = Memory(
        user_id=user_id,
        scope=scope,
        type=memory_type,
        content=content,
        source=MemorySource.EXPLICIT,
        confidence=1.0,
        importance=0.8,
        valid_from=datetime.now(UTC),
        valid_to=None,
    )

    duplicate = await memory_service.find_duplicate(memory)

    if duplicate is not None:
        return (
            "An equivalent memory already exists. "
            "No duplicate was created."
        )

    await memory_service.create(memory)

    return f"Memory saved successfully: {content}"