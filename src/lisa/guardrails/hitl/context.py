from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from uuid import UUID

from lisa.guardrails.hitl.policy import GuardError


@dataclass(frozen=True)
class Principal:
    user_id: UUID
    session_id: UUID
    role: str


@dataclass(frozen=True)
class ExecutionContext:
    principal: Principal
    conversation_id: UUID
    service: object
    resuming_execution_id: UUID | None = None


current_execution: ContextVar[ExecutionContext | None] = ContextVar("hitl_execution", default=None)
native_write: ContextVar[UUID | None] = ContextVar("hitl_native_write", default=None)


@contextmanager
def execution_scope(context):
    token = current_execution.set(context)
    try:
        yield
    finally:
        current_execution.reset(token)


def require_native_write(user_id):
    if user_id is None or native_write.get() is None or native_write.get() != user_id:
        raise GuardError("approval_required")


def automatic_memory_enabled():
    return False


def native_runtime(user_id, memory):
    from langgraph.runtime import Runtime

    from lisa.context import LISAContext
    return Runtime(context=LISAContext(user_id=user_id, memory=memory))