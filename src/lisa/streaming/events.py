from typing import Literal

from pydantic import BaseModel


class StreamEvent(BaseModel):
    type: Literal["reasoning", "content", "approval_required", "approval_updated", "tool_execution_status", "execution_suspended", "execution_completed"]
    content: str = ""
    event_id: str | None = None
    execution_id: str | None = None
    approval_id: str | None = None
    revision: int | None = None
    status: str | None = None
