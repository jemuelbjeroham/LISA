from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages

from lisa.routing import Route


class LISAState(TypedDict):
    messages: Annotated[list, add_messages]
    route: Route | None

