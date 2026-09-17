from enum import StrEnum

from lisa.state import LISAState


class RoutingAction(StrEnum):
    CONTINUE = "continue"
    ROUTE = "route"

class RoutingPolicy:

    def decide(self, state: LISAState) -> RoutingAction:
        if state["active_route"] is not None:
            return RoutingAction.CONTINUE

        return RoutingAction.ROUTE