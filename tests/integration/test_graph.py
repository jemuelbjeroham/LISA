from unittest.mock import Mock

from langchain_core.messages import HumanMessage

from lisa.application import LISA
from lisa.routing import Route, RoutingDecision


def test_lisa_graph_executes():
    model = Mock()

    model.with_structured_output.return_value = model
    model.invoke.return_value = RoutingDecision(
        route=Route.GENERAL_ENQUIRY
    )

    lisa = LISA(model)

    initial_state = {
        "messages": [HumanMessage(content="Hello LISA")],
        "route": None,
    }

    result = lisa.graph.invoke(initial_state)

    assert "messages" in result
    assert result["route"] == Route.GENERAL_ENQUIRY