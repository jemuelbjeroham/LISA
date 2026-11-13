import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from lisa import orchestrator as orchestrator_module
from lisa.jev_router import (
    MAX_EARLIER_MESSAGES,
    NEEDS_INTERNAL_DOCS,
    WANTS_PROBLEM_SOLVED,
    WANTS_TICKET,
    JevRouter,
    build_routing_state,
    decide_route,
    parse_jev_routing_prompt,
)
from lisa.orchestrator import Orchestrator
from lisa.prompts.loader import load_prompt
from lisa.routing import Route, RoutingDecision

PROMPT_PATH = "orchestrator/jev_routing_v1.json"
PROMPT = parse_jev_routing_prompt(load_prompt(PROMPT_PATH))
BACKGROUND = {"assistant": "LISA"}


def answers(ticket: float, problem: float, docs: float) -> dict[str, float]:
    return {WANTS_TICKET: ticket, WANTS_PROBLEM_SOLVED: problem, NEEDS_INTERNAL_DOCS: docs}


def jev_client(ticket: float, problem: float, docs: float) -> MagicMock:
    client = MagicMock()
    client.system_one.return_value = SimpleNamespace(
        model="jev-1.14.0",
        nouls={name: SimpleNamespace(noul=p) for name, p in answers(ticket, problem, docs).items()},
    )
    return client


@pytest.mark.parametrize(
    "probabilities,route",
    [
        # Ticket requests win even when they describe a failure.
        (answers(0.95, 0.9, 0.1), Route.GENERAL_ENQUIRY),
        (answers(0.05, 0.9, 0.9), Route.TROUBLESHOOTING_AGENT),
        (answers(0.05, 0.1, 0.9), Route.KNOWLEDGE_ASSISTANT),
        (answers(0.05, 0.1, 0.1), Route.GENERAL_ENQUIRY),
        # Any judgment in the unsettled band, before a clear yes, escalates.
        (answers(0.5, 0.9, 0.1), None),
        (answers(0.05, 0.5, 0.9), None),
        (answers(0.05, 0.1, 0.5), None),
        # A clear earlier yes is not affected by later unsettled answers.
        (answers(0.05, 0.9, 0.5), Route.TROUBLESHOOTING_AGENT),
    ],
)
def test_decide_route_applies_decision_table(probabilities, route):
    assert decide_route(probabilities) is route


def test_build_routing_state_uses_latest_user_message_and_skips_non_conversation():
    messages = [
        SystemMessage(content="system"),
        HumanMessage(content="The Amagi feed is down"),
        AIMessage(content="", tool_calls=[{"name": "search", "args": {}, "id": "1"}]),
        ToolMessage(content="tool output", tool_call_id="1"),
        AIMessage(content="Which channel?"),
        HumanMessage(content="CH398"),
    ]

    state = build_routing_state(messages, BACKGROUND)

    assert state["latest_user_message"] == "CH398"
    assert state["assistant"] == "LISA"
    assert state["earlier_conversation"] == [
        {"from": "user", "text": "The Amagi feed is down"},
        {"from": "LISA", "text": "Which channel?"},
    ]


def test_build_routing_state_omits_empty_history_and_caps_its_length():
    assert "earlier_conversation" not in build_routing_state([HumanMessage(content="Hi")], BACKGROUND)

    history = [HumanMessage(content=f"m{i}") for i in range(MAX_EARLIER_MESSAGES + 5)]
    state = build_routing_state(history, BACKGROUND)
    assert len(state["earlier_conversation"]) == MAX_EARLIER_MESSAGES


def test_build_routing_state_without_user_message_returns_none():
    assert build_routing_state([AIMessage(content="Hello")], BACKGROUND) is None


def test_jev_router_sends_all_questions_in_one_request():
    client = jev_client(0.05, 0.92, 0.2)

    decision = JevRouter(client, PROMPT).classify([HumanMessage(content="Why is CH398 black?")])

    assert decision == RoutingDecision(route=Route.TROUBLESHOOTING_AGENT)
    client.system_one.assert_called_once()
    state, questions = client.system_one.call_args.args
    assert state["latest_user_message"] == "Why is CH398 black?"
    assert state["internal_systems"] == PROMPT.state["internal_systems"]
    assert set(questions) == {WANTS_TICKET, WANTS_PROBLEM_SOLVED, NEEDS_INTERNAL_DOCS}
    assert questions[WANTS_TICKET].instructions == PROMPT.questions[WANTS_TICKET].instructions


def test_shipped_prompt_defines_every_decision_table_question():
    assert set(PROMPT.questions) == {WANTS_TICKET, WANTS_PROBLEM_SOLVED, NEEDS_INTERNAL_DOCS}
    assert PROMPT.state


def test_prompt_missing_a_question_is_rejected():
    raw = json.loads(load_prompt(PROMPT_PATH))
    del raw["questions"][NEEDS_INTERNAL_DOCS]
    with pytest.raises(ValueError, match="exactly the questions"):
        parse_jev_routing_prompt(json.dumps(raw))


def test_prompt_setting_conversation_state_is_rejected():
    raw = json.loads(load_prompt(PROMPT_PATH))
    raw["state"]["latest_user_message"] = "spoofed"
    with pytest.raises(ValueError, match="may not set"):
        parse_jev_routing_prompt(json.dumps(raw))


def llm_router(*decisions):
    model = MagicMock()
    model.with_structured_output.return_value.invoke.side_effect = list(decisions)
    return model


def test_orchestrator_uses_jev_route_without_calling_llm():
    model = llm_router()
    orchestrator = Orchestrator(model, "prompt", jev_router=JevRouter(prompt=PROMPT, client=jev_client(0.02, 0.03, 0.95)))

    result = orchestrator.route({"messages": [HumanMessage(content="How do I submit the form?")]})

    assert result["route"] is Route.KNOWLEDGE_ASSISTANT
    model.with_structured_output.return_value.invoke.assert_not_called()


def test_orchestrator_falls_back_to_llm_when_jev_is_unsettled():
    model = llm_router(RoutingDecision(route=Route.KNOWLEDGE_ASSISTANT))
    orchestrator = Orchestrator(model, "prompt", jev_router=JevRouter(prompt=PROMPT, client=jev_client(0.02, 0.03, 0.5)))

    result = orchestrator.route({"messages": [HumanMessage(content="What is SkyPath?")]})

    assert result["route"] is Route.KNOWLEDGE_ASSISTANT
    model.with_structured_output.return_value.invoke.assert_called_once()


def test_orchestrator_falls_back_to_llm_when_jev_errors(monkeypatch):
    monkeypatch.setattr(orchestrator_module.time, "sleep", lambda _: None)
    client = MagicMock()
    client.system_one.side_effect = RuntimeError("TypeSafe unavailable")
    model = llm_router(RoutingDecision(route=Route.TROUBLESHOOTING_AGENT))
    orchestrator = Orchestrator(model, "prompt", jev_router=JevRouter(client, PROMPT))

    result = orchestrator.route({"messages": [HumanMessage(content="Feed is down")]})

    assert result == {
        "route": Route.TROUBLESHOOTING_AGENT,
        "active_route": Route.TROUBLESHOOTING_AGENT,
        "agent_handoff": None,
    }
