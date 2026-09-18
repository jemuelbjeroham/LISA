from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages

from lisa.routing import AgentDecision, Route


class LISAState(TypedDict):
    messages: Annotated[list, add_messages]
    route: Route | None
    active_route: Route | None
    agent_decision: AgentDecision | None
    knowledge_context: list[str]
    enable_thinking: bool