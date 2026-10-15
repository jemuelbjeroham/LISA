
import logging
from functools import partial

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from lisa.agents.general_enquiry import GeneralEnquiry
from lisa.agents.technical_clarification import TechnicalClarificationAgent
from lisa.context import LISAContext
from lisa.orchestrator import Orchestrator
from lisa.routing import Route, RoutingAction, RoutingPolicy
from lisa.state import LISAState

logger = logging.getLogger(__name__)

MAX_HANDOFFS = 3


def route_entry(state: LISAState, routing_policy: RoutingPolicy) -> str:
    action = routing_policy.decide(state["active_route"])

    if action == RoutingAction.CONTINUE:
        return state["active_route"].value

    return "orchestrator"


def route_from_state(state: LISAState) -> Route:
    return state["route"]


def record_handoff(state: LISAState) -> dict:
    handoff = state.get("agent_handoff")

    if handoff is None:
        return {"handoff_count": state.get("handoff_count", 0)}

    count = state.get("handoff_count", 0) + 1

    logger.info(
        "Agent handoff recorded: count=%d limit=%d handoff=%s",
        count,
        MAX_HANDOFFS,
        handoff,
    )

    if count > MAX_HANDOFFS:
        logger.warning(
            "Maximum handoff count exceeded; stopping routing."
        )

        return {
            "handoff_count": count,
            "agent_handoff": None,
            "messages": [
                AIMessage(
                    content=(
                        "I'm having trouble routing this request to the "
                        "right assistant. Please rephrase your question "
                        "or start a new request."
                    )
                )
            ],
        }

    return {"handoff_count": count}


def route_after_handoff(state: LISAState) -> str:
    if state.get("agent_handoff") is not None and state.get(
        "handoff_count", 0
    ) <= MAX_HANDOFFS:
        return "orchestrator"

    return "end"


def build_graph(
    orchestrator: Orchestrator,
    technical_clarification_agent: TechnicalClarificationAgent,
    general_enquiry_agent: GeneralEnquiry,
    routing_policy: RoutingPolicy,
):
    builder = StateGraph(LISAState, context_schema=LISAContext)

    builder.add_node("orchestrator", orchestrator.route)
    builder.add_node(
        "technical_clarification",
        technical_clarification_agent.run,
    )
    builder.add_node("general_enquiry", general_enquiry_agent.run)
    builder.add_node("record_handoff", record_handoff)

    builder.add_conditional_edges(
        START,
        partial(route_entry, routing_policy=routing_policy),
        {
            "orchestrator": "orchestrator",
            Route.TECHNICAL_CLARIFICATION.value: "technical_clarification",
            Route.GENERAL_ENQUIRY.value: "general_enquiry",
        },
    )

    builder.add_conditional_edges(
        "orchestrator",
        route_from_state,
        {
            Route.TECHNICAL_CLARIFICATION: "technical_clarification",
            Route.GENERAL_ENQUIRY: "general_enquiry",
        },
    )

    builder.add_edge("technical_clarification", "record_handoff")
    builder.add_edge("general_enquiry", "record_handoff")

    builder.add_conditional_edges(
        "record_handoff",
        route_after_handoff,
        {
            "orchestrator": "orchestrator",
            "end": END,
        },
    )

    return builder.compile()
