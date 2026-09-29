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
        "route": Route.TECHNICAL_CLARIFICATION,
        "active_route": Route.TECHNICAL_CLARIFICATION,
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

    assert restored["route"] == Route.TECHNICAL_CLARIFICATION
    assert restored["active_route"] == Route.TECHNICAL_CLARIFICATION
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