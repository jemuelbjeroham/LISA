import pytest
from langchain_core.messages import AIMessage, HumanMessage

from lisa.conversation.serialization import ConversationStateSerializer
from lisa.routing import Route
from lisa.state import LISAState


def test_state_round_trip():

    state: LISAState = {
        "messages": [
            HumanMessage(content="How do I troubleshoot a firewall?"),
            AIMessage(content="Start by checking the firewall logs."),
        ],
        "route": Route.TROUBLESHOOTING_AGENT,
        "active_route": Route.TROUBLESHOOTING_AGENT,
        "agent_handoff": None,
        "knowledge_context": [
            "Firewall troubleshooting procedure."
        ],
        "enable_thinking": False,
    }

    serialized = ConversationStateSerializer.serialize(state)

    restored = ConversationStateSerializer.deserialize(
        serialized
    )

    assert restored["route"] == Route.TROUBLESHOOTING_AGENT
    assert restored["active_route"] == Route.TROUBLESHOOTING_AGENT
    assert restored["agent_handoff"] is None
    assert restored["knowledge_context"] == [
        "Firewall troubleshooting procedure."
    ]
    assert restored["enable_thinking"] is False

    assert len(restored["messages"]) == 2

    assert isinstance(restored["messages"][0], HumanMessage)
    assert restored["messages"][0].content == (
        "How do I troubleshoot a firewall?"
    )

    assert isinstance(restored["messages"][1], AIMessage)
    assert restored["messages"][1].content == (
        "Start by checking the firewall logs."
    )


def test_retired_route_returns_to_orchestrator():
    state = {
        "messages": [], "route": "technical_clarification",
        "active_route": "technical_clarification", "agent_handoff": None,
        "knowledge_context": [], "enable_thinking": False,
    }
    restored = ConversationStateSerializer.deserialize(state)
    assert restored["route"] is None
    assert restored["active_route"] is None


def test_unknown_route_is_not_silently_accepted():
    state = {
        "messages": [], "route": "unknown", "active_route": None,
        "agent_handoff": None, "knowledge_context": [], "enable_thinking": False,
    }
    with pytest.raises(ValueError):
        ConversationStateSerializer.deserialize(state)