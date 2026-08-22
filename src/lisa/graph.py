from langgraph.graph import END, START, StateGraph

from lisa.state import LISAState


def orchestrator(state: LISAState) -> dict:
    print("Orchestrator received: ", state["messages"])
    return {}

def build_graph() -> StateGraph:
    builder = StateGraph(LISAState)

    builder.add_node("orchestrator", orchestrator)
    builder.add_edge(START, "orchestrator")
    builder.add_edge("orchestrator", END)

    return builder.compile()
