from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Literal
from uuid import uuid4

from langchain_core.tools import InjectedToolArg, tool
from pydantic import BaseModel, ConfigDict, Field


class SecureMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    task_id: str = Field(min_length=1, max_length=100)
    sender: Literal["butler", "specialist"]
    recipient: Literal["butler", "specialist"]
    message_type: Literal["task", "clarification", "finding", "result", "progress"]
    payload: str = Field(min_length=1, max_length=100000)


@dataclass
class TaskChannel:
    task_id: str = field(default_factory=lambda: str(uuid4()))
    messages: list[SecureMessage] = field(default_factory=list)
    closed: bool = False
    on_progress: Callable[[str], None] | None = None

    def deliver(self, message: SecureMessage) -> dict:
        if self.closed or message.task_id != self.task_id:
            return {"accepted": False, "error": "inactive_or_mismatched_task"}
        if message.sender == message.recipient:
            return {"accepted": False, "error": "invalid_recipient"}
        if len(self.messages) >= 128:
            return {"accepted": False, "error": "message_limit"}
        self.messages.append(message)
        return {"accepted": True, "task_id": self.task_id, "recipient": message.recipient}


@tool
def secure_line(
    task_id: str,
    sender: Literal["butler", "specialist"],
    recipient: Literal["butler", "specialist"],
    message_type: Literal["task", "clarification", "finding", "result", "progress"],
    payload: str,
    channel: Annotated[TaskChannel | None, InjectedToolArg] = None,
) -> dict:
    """Deliver a task-scoped message between Butler and Specialist, not externally."""
    if channel is None:
        return {"accepted": False, "error": "missing_task_context"}
    try:
        message = SecureMessage(task_id=task_id, sender=sender, recipient=recipient,
                                message_type=message_type, payload=payload)
    except ValueError:
        return {"accepted": False, "error": "invalid_message"}
    return channel.deliver(message)