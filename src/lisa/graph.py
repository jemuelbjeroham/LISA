from langgraph.graph import END, START, StateGraph

from lisa.agents.technical_clarification import TechnicalClarificationAgent
from lisa.orchestrator import Orchestrator
from lisa.routing import Route
from lisa.state import LISAState


def route_from_state(state: LISAState) -> Route:
    return state["route"]

def build_graph(orchestrator: Orchestrator, technical_clarification_agent: TechnicalClarificationAgent):
    builder = StateGraph(LISAState)

    builder.add_node("orchestrator", orchestrator.route)
    builder.add_node("technical_clarification", technical_clarification_agent.run)

    builder.add_edge(START, "orchestrator")
    builder.add_conditional_edges("orchestrator", route_from_state,
                                  {
                                      Route.TECHNICAL_CLARIFICATION: "technical_clarification",
                                  },
                                  )
    builder.add_edge("technical_clarification", END)

    return builder.compile()
