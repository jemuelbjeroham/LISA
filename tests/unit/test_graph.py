from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from lisa.graph import build_graph
from lisa.routing import Route, RoutingPolicy


@pytest.mark.parametrize("route", list(Route))
@pytest.mark.parametrize("active", [False, True])
async def test_graph_routes_to_selected_agent(route, active):
    orchestrator = MagicMock()
    orchestrator.route.return_value = {"route": route, "active_route": route}
    agents = {name.value: MagicMock() for name in Route}
    for name, agent in agents.items():
        agent.run.return_value = {"messages": [AIMessage(content=name)]}

    graph = build_graph(
        orchestrator=orchestrator,
        knowledge_assistant_agent=agents[Route.KNOWLEDGE_ASSISTANT.value],
        troubleshooting_agent=agents[Route.TROUBLESHOOTING_AGENT.value],
        general_enquiry_agent=agents[Route.GENERAL_ENQUIRY.value],
        routing_policy=RoutingPolicy(),
    )

    state = {
        "messages": [HumanMessage(content="Hello")],
        "route": None,
        "active_route": route if active else None,
        "agent_handoff": None,
    }

    result = await graph.ainvoke(state)

    assert result["messages"][-1].content == route.value
    assert orchestrator.route.call_count == (0 if active else 1)
    for name, agent in agents.items():
        assert agent.run.call_count == (1 if name == route.value else 0)


async def test_graph_routes_handoff_between_new_agents():
    orchestrator = MagicMock()
    orchestrator.route.side_effect = [
        {"route": Route.KNOWLEDGE_ASSISTANT, "active_route": Route.KNOWLEDGE_ASSISTANT, "agent_handoff": None},
        {"route": Route.TROUBLESHOOTING_AGENT, "active_route": Route.TROUBLESHOOTING_AGENT, "agent_handoff": None},
    ]
    knowledge = MagicMock()
    knowledge.run.return_value = {"agent_handoff": {"reason": "A specific fault needs investigation."}}
    troubleshooting = MagicMock()
    troubleshooting.run.return_value = {"messages": [AIMessage(content="Investigating.")]}
    general = MagicMock()
    graph = build_graph(orchestrator, knowledge, troubleshooting, general, RoutingPolicy())

    result = await graph.ainvoke({
        "messages": [HumanMessage(content="Help with the feed.")],
        "route": None,
        "active_route": None,
        "agent_handoff": None,
    })

    assert result["messages"][-1].content == "Investigating."
    assert result["handoff_count"] == 1
    assert orchestrator.route.call_count == 2
    knowledge.run.assert_called_once()
    troubleshooting.run.assert_called_once()
    general.run.assert_not_called()