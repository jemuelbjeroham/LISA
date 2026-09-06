from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from lisa.graph import build_graph
from lisa.routing import Route


@pytest.mark.anyio
async def test_graph_routes_general_enquiry_to_general_enquiry_agent():
    orchestrator = MagicMock()

    orchestrator.route.return_value = {
        "route": Route.GENERAL_ENQUIRY,
    }

    technical_clarification_agent = MagicMock()
    general_enquiry_agent = MagicMock()

    general_enquiry_agent.run = MagicMock(
        return_value={
            "messages": [
                AIMessage(content="Hello! How can I help you?")
            ]
        }
    )

    graph = build_graph(
        orchestrator=orchestrator,
        technical_clarification_agent=technical_clarification_agent,
        general_enquiry_agent=general_enquiry_agent,
    )

    state = {
        "messages": [
            HumanMessage(content="Hello")
        ],
        "route": None,
    }

    result = await graph.ainvoke(state)

    assert result["route"] == Route.GENERAL_ENQUIRY
    assert result["messages"][-1].content == (
        "Hello! How can I help you?"
    )

    general_enquiry_agent.run.assert_called_once()

    agent_state = general_enquiry_agent.run.call_args.args[0]

    assert agent_state["route"] == Route.GENERAL_ENQUIRY
    assert len(agent_state["messages"]) == 1
    assert agent_state["messages"][0].content == "Hello"

    technical_clarification_agent.run.assert_not_called()