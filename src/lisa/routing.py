from enum import StrEnum

from pydantic import BaseModel


class Route(StrEnum):
    GENERAL_ENQUIRY = "general_enquiry"
    ISSUE_REPORTING = "issue_reporting"
    TECHNICAL_CLARIFICATION = "technical_clarification"

class RoutingDecision(BaseModel):
    route: Route
    