from typing import Literal

from pydantic import BaseModel


class StreamEvent(BaseModel):
    type: Literal["reasoning", "content"]
    content: str
