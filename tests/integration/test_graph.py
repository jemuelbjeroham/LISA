from langchain_core.messages import HumanMessage

from lisa.graph import graph


def test_lisa_graph_executes():
    initial_state ={
        "messages": [HumanMessage(content="Hello LISA")]
    }

    result = graph.invoke(initial_state)

    assert "messages" in result