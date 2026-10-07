
from typing import Annotated, TypedDict
from uuid import UUID

from langgraph.graph.message import add_messages

from lisa.routing import Route


class LISAState(TypedDict):
    user_id: UUID
    messages: Annotated[list, add_messages]
    route: Route | None
    active_route: Route | None
    agent_handoff: dict | None
    knowledge_context: list[str]
    enable_thinking: bool