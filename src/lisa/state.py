from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class LISAState(TypedDict):
    messages: Annotated[list, add_messages]

