from functools import partial

from langgraph.graph import END, START, StateGraph

from lisa.agents.general_enquiry import GeneralEnquiry
from lisa.agents.technical_clarification import TechnicalClarificationAgent
from lisa.orchestrator import Orchestrator
from lisa.routing import Route, RoutingAction, RoutingPolicy
from lisa.state import LISAState


def route_entry(state: LISAState, routing_policy: RoutingPolicy) -> str:

    action = routing_policy.decide(
        state["active_route"]
    )

    if action == RoutingAction.CONTINUE:
        return state["active_route"].value

    return "orchestrator"
    
def route_from_state(state: LISAState) -> Route:
    return state["route"]

def build_graph(
        orchestrator: Orchestrator,
        technical_clarification_agent: TechnicalClarificationAgent,
        general_enquiry_agent: GeneralEnquiry,
        routing_policy: RoutingPolicy,
):
    builder = StateGraph(LISAState)

    builder.add_node("orchestrator", orchestrator.route)
    builder.add_node("technical_clarification", technical_clarification_agent.run)
    builder.add_node("general_enquiry", general_enquiry_agent.run)

    builder.add_conditional_edges(
        START,
        partial(route_entry, routing_policy=routing_policy),
        {
            "orchestrator": "orchestrator",
            Route.TECHNICAL_CLARIFICATION.value: "technical_clarification",
            Route.GENERAL_ENQUIRY.value: "general_enquiry",
        }
    )
    # builder.add_edge(START, "orchestrator")
    builder.add_conditional_edges("orchestrator", route_from_state,
                                  {
                                      Route.TECHNICAL_CLARIFICATION: "technical_clarification",
                                      Route.GENERAL_ENQUIRY: "general_enquiry"
                                  },
                                  )
    builder.add_edge("technical_clarification", END)
    builder.add_edge("general_enquiry", END)

    return builder.compile()
