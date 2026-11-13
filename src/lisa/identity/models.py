from typing import Literal
from uuid import UUID

from pydantic import BaseModel


class User(BaseModel):
    id: UUID
    display_name: str | None = None
    role: Literal["admin", "member"] = "member"
    session_id: UUID | None = None

