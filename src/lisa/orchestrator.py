import logging
import time

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage

from lisa.routing import Route, RoutingDecision
from lisa.state import LISAState

logger = logging.getLogger(__name__)

MAX_ROUTING_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 0.5


class Orchestrator:
    def __init__(self, model: BaseChatModel, routing_prompt: str):
        self.model = model.with_structured_output(RoutingDecision)
        self.routing_prompt = routing_prompt

    def route(self, state: LISAState):
        messages = [
            SystemMessage(content=self.routing_prompt),
            *state["messages"],
        ]

        for attempt in range(1, MAX_ROUTING_ATTEMPTS + 1):
            try:
                logger.info(
                    "Classifying user intent: attempt=%d/%d",
                    attempt,
                    MAX_ROUTING_ATTEMPTS,
                )

                decision = self.model.invoke(messages)

                # Validate the result even when structured output is enabled.
                decision = RoutingDecision.model_validate(decision)

                logger.info(
                    "Intent classified successfully: route=%s",
                    decision.route.value,
                )

                return {
                    "route": decision.route,
                    "active_route": decision.route,
                    "agent_handoff": None,
                }

            except Exception:
                logger.exception(
                    "Routing attempt failed: attempt=%d/%d",
                    attempt,
                    MAX_ROUTING_ATTEMPTS,
                )

                if attempt < MAX_ROUTING_ATTEMPTS:
                    time.sleep(RETRY_DELAY_SECONDS * attempt)

        # Controlled fallback: keep the conversation moving.
        fallback_route = Route.GENERAL_ENQUIRY

        logger.error(
            "Routing retries exhausted; using fallback route=%s",
            fallback_route.value,
        )

        return {
            "route": fallback_route,
            "active_route": fallback_route,
            "agent_handoff": None,
        }