from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from lisa.agents.technical_clarification import TechnicalClarificationAgent
from lisa.graph import build_graph
from lisa.orchestrator import Orchestrator
from lisa.routing import Route


@pytest.mark.anyio
async def test_graph_routes_to_technical_clarification_agent():
    model = Mock()

    structured_model = Mock()
    structured_model.invoke.return_value = Mock(
        route=Route.TECHNICAL_CLARIFICATION
    )

    model.with_structured_output.return_value = structured_model

    technical_model = Mock()
    technical_model.invoke.return_value = AIMessage(
        content="This is the technical explanation."
    )

    retriever = Mock()
    retriever.retrieve = AsyncMock(
        return_value=["Relevant technical documentation."]
    )

    orchestrator = Orchestrator(
        model=model,
        routing_prompt="Route the user request.",
    )

    technical_clarification_agent = TechnicalClarificationAgent(
        model=technical_model,
        retriever=retriever,
        system_prompt="You are a technical assistant.",
    )

    general_enquiry_agent = Mock()

    graph = build_graph(
        orchestrator=orchestrator,
        technical_clarification_agent=technical_clarification_agent,
        general_enquiry_agent=general_enquiry_agent,
    )

    state = {
        "messages": [
            HumanMessage(content="How do I troubleshoot BGP?")
        ],
        "route": None,
    }

    result = await graph.ainvoke(state)

    assert result["route"] == Route.TECHNICAL_CLARIFICATION
    assert result["messages"][-1].content == (
        "This is the technical explanation."
    )

    retriever.retrieve.assert_awaited_once_with(
        "How do I troubleshoot BGP?"
    )

    general_enquiry_agent.run.assert_not_called()