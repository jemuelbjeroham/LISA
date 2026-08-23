from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage

from lisa.routing import RoutingDecision
from lisa.state import LISAState


class Orchestrator:
    def __init__(self, model: BaseChatModel, routing_prompt: str):
        self.model = model.with_structured_output(RoutingDecision)
        self.routing_prompt = routing_prompt

    def route(self, state: LISAState):
        messages = [
            SystemMessage(content=self.routing_prompt),
            *state["messages"],
        ]
        decision = self.model.invoke(messages)

        return {
            "route": decision.route,
        }