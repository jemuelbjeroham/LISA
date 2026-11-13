
from datetime import UTC, datetime
from typing import Annotated

from langchain_core.tools import InjectedToolArg, tool
from langgraph.runtime import Runtime

from lisa.context import LISAContext
from lisa.guardrails.hitl.context import (
    current_execution,
    native_write,
    require_native_write,
)
from lisa.guardrails.hitl.dispatcher import check_tool
from lisa.guardrails.hitl.policy import GuardError, MemoryArguments
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
    context = current_execution.get()
    if context is None:
        raise GuardError("identity_unavailable", 401)
    if runtime.context.user_id is None or runtime.context.user_id != context.principal.user_id:
        raise GuardError("not_authorized")
    await check_tool("save_memory")
    arguments = MemoryArguments(content=content, memory_type=memory_type, scope=scope)
    token = native_write.set(runtime.context.user_id)
    try:
        return await _save_memory(arguments.content, arguments.memory_type, arguments.scope, runtime)
    finally:
        native_write.reset(token)


async def _save_memory(content, memory_type, scope, runtime):
    require_native_write(runtime.context.user_id)
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