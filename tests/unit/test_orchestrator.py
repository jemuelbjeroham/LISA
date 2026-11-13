from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage

from lisa import orchestrator as orchestrator_module
from lisa.orchestrator import MAX_ROUTING_ATTEMPTS, Orchestrator
from lisa.routing import Route, RoutingDecision


def router(*decisions):
    model = MagicMock()
    model.with_structured_output.return_value.invoke.side_effect = list(decisions)
    return Orchestrator(model, "Routing policy"), model


@pytest.mark.parametrize(
    "message,route",
    [
        ("Hi, how are you?", Route.GENERAL_ENQUIRY),
        ("What is the SkyPath failover procedure?", Route.KNOWLEDGE_ASSISTANT),
        ("The Amagi feed for channel 4 is down", Route.TROUBLESHOOTING_AGENT),
    ],
)
def test_orchestrator_returns_model_route_for_each_agent(message, route):
    orchestrator, model = router(RoutingDecision(route=route))
    state = {"messages": [HumanMessage(content=message)], "route": None}

    result = orchestrator.route(state)

    assert result == {"route": route, "active_route": route, "agent_handoff": None}
    model.with_structured_output.assert_called_once_with(RoutingDecision)
    sent = model.with_structured_output.return_value.invoke.call_args.args[0]
    assert sent[0].content == "Routing policy"
    assert sent[1].content == message


def test_orchestrator_accepts_dict_decision():
    orchestrator, _ = router({"route": "troubleshooting_agent"})
    result = orchestrator.route({"messages": [HumanMessage(content="Fix it")]})
    assert result["route"] is Route.TROUBLESHOOTING_AGENT


def test_orchestrator_retries_then_falls_back_to_general_enquiry(monkeypatch):
    monkeypatch.setattr(orchestrator_module.time, "sleep", lambda _: None)
    orchestrator, model = router(*([{"route": "technical_clarification"}] * MAX_ROUTING_ATTEMPTS))

    result = orchestrator.route({"messages": [HumanMessage(content="?")]})

    assert result["route"] is Route.GENERAL_ENQUIRY
    assert model.with_structured_output.return_value.invoke.call_count == MAX_ROUTING_ATTEMPTS


def test_orchestrator_recovers_after_transient_failure(monkeypatch):
    monkeypatch.setattr(orchestrator_module.time, "sleep", lambda _: None)
    orchestrator, _ = router(RuntimeError("timeout"), RoutingDecision(route=Route.KNOWLEDGE_ASSISTANT))
    assert orchestrator.route({"messages": [HumanMessage(content="Docs?")]})["route"] is Route.KNOWLEDGE_ASSISTANT


def test_route_enum_exposes_exactly_the_three_main_agents():
    assert {route.value for route in Route} == {"general_enquiry", "knowledge_assistant", "troubleshooting_agent"}
