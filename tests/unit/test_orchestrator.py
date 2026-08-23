from unittest.mock import Mock

from langchain_core.messages import HumanMessage

from lisa.orchestrator import Orchestrator
from lisa.routing import Route, RoutingDecision


def test_orchestrator_sends_routing_prompt_and_messages():
    model = Mock()

    model.with_structured_output.return_value = model
    model.invoke.return_value = RoutingDecision(
        route=Route.TECHNICAL_CLARIFICATION
    )

    routing_prompt = "Classify the user's primary intent."

    orchestrator = Orchestrator(
        model=model,
        routing_prompt=routing_prompt,
    )

    user_message = HumanMessage(
        content="Why does a Firewall rule get stuck in deleting state?"
    )

    state = {
        "messages": [user_message],
        "route": None,
    }

    orchestrator.route(state)

    model.invoke.assert_called_once()

    sent_messages = model.invoke.call_args.args[0]

    assert sent_messages[0].content == routing_prompt
    assert sent_messages[1] == user_message