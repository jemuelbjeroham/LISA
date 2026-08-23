from langgraph.graph import END, START, StateGraph

from lisa.orchestrator import Orchestrator
from lisa.state import LISAState


def build_graph(orchestrator: Orchestrator):
    builder = StateGraph(LISAState)

    builder.add_node("orchestrator", orchestrator.route)
    builder.add_edge(START, "orchestrator")
    builder.add_edge("orchestrator", END)

    return builder.compile()
