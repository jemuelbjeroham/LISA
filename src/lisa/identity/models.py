from uuid import UUID

from pydantic import BaseModel


class User(BaseModel):
    id: UUID
    display_name: str | None = None

