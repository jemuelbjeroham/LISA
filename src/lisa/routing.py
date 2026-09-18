from enum import StrEnum

from pydantic import BaseModel
from typing import Literal

class Route(StrEnum):
    GENERAL_ENQUIRY = "general_enquiry"
    ISSUE_REPORTING = "issue_reporting"
    TECHNICAL_CLARIFICATION = "technical_clarification"

class RoutingAction(StrEnum):
    CONTINUE = "continue"
    ROUTE = "route"

class RoutingDecision(BaseModel):
    route: Route

class RoutingPolicy:

    def decide(
            self,
            active_route: Route | None,
    ) -> RoutingAction:

        if active_route is not None:
            return RoutingAction.CONTINUE

        return RoutingAction.ROUTE

class AgentDecision(BaseModel):
    action: Literal["respond", "handoff"]