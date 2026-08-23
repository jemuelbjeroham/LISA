from langchain_core.language_models import BaseChatModel

from lisa.routing import Route, RoutingDecision
from lisa.state import LISAState


class Orchestrator:
    def __init__(self, model: BaseChatModel):
        self.model = model.with_structured_output(RoutingDecision)

    def route(self, state: LISAState):
        pass