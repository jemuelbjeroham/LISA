import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage

from lisa.routing import RoutingDecision
from lisa.state import LISAState

logger = logging.getLogger(__name__)

class Orchestrator:
    def __init__(self, model: BaseChatModel, routing_prompt: str):
        self.model = model.with_structured_output(RoutingDecision)
        self.routing_prompt = routing_prompt

    def route(self, state: LISAState):
        messages = [
            SystemMessage(content=self.routing_prompt),
            *state["messages"],
        ]
        try:
            logger.info("sending message to LLM to classify the intent")
            decision = self.model.invoke(messages)
            logger.info("LLM classified the intent and chose the route: %s", decision.route)

            return {
                "route": decision.route,
                "active_route": decision.route,
            }

        except Exception:
            logger.exception("orchestrator failure occured")
            raise