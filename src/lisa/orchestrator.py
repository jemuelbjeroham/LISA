import logging
import time
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage

from lisa.jev_router import JevRouter
from lisa.routing import Route, RoutingDecision
from lisa.state import LISAState

logger = logging.getLogger(__name__)

MAX_ROUTING_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 0.5


class Orchestrator:
    """Routes each new request: Jev first, the LLM router as fallback.

    The LLM router (`model`) handles the request when Jev is not configured,
    when the Jev call fails, or when Jev's answers do not settle the route.
    """

    def __init__(
        self,
        model: BaseChatModel,
        routing_prompt: str,
        jev_router: JevRouter | None = None,
    ) -> None:
        self.model = model.with_structured_output(RoutingDecision)
        self.routing_prompt = routing_prompt
        self.jev_router = jev_router

    def route(self, state: LISAState) -> dict[str, Any]:
        decision = self._route_with_jev(state)
        if decision is not None:
            return self._result(decision.route)
        return self._route_with_llm(state)

    def _route_with_jev(self, state: LISAState) -> RoutingDecision | None:
        if self.jev_router is None:
            return None
        try:
            decision = self.jev_router.classify(state["messages"])
        except Exception:
            logger.exception("Jev routing failed; falling back to LLM router")
            return None
        if decision is None:
            logger.info("Jev routing unsettled; falling back to LLM router")
        return decision

    @staticmethod
    def _result(route: Route) -> dict[str, Any]:
        return {
            "route": route,
            "active_route": route,
            "agent_handoff": None,
        }

    def _route_with_llm(self, state: LISAState) -> dict[str, Any]:
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

                raw_decision = self.model.invoke(messages)
                decision = RoutingDecision.model_validate(raw_decision)

                logger.info(
                    "Intent classified successfully: route=%s",
                    decision.route.value,
                )

                return self._result(decision.route)

            except Exception:
                logger.exception(
                    "Routing attempt failed: attempt=%d/%d",
                    attempt,
                    MAX_ROUTING_ATTEMPTS,
                )

                if attempt < MAX_ROUTING_ATTEMPTS:
                    time.sleep(RETRY_DELAY_SECONDS * attempt)

        fallback_route = Route.GENERAL_ENQUIRY

        logger.error(
            "Routing retries exhausted; using fallback route=%s",
            fallback_route.value,
        )

        return self._result(fallback_route)