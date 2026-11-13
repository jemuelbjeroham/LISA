"""LISA.chat / LISA.stream_chat running through the real compiled graph.

The orchestrator and the three agents are stubs; everything between them
(graph routing, state handling, persistence, streaming) is the real code.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.config import get_stream_writer

from lisa.application import LISA
from lisa.graph import build_graph
from lisa.routing import Route, RoutingPolicy
from lisa.streaming.events import StreamEvent


class UserScopedStore:
    """Minimal ConversationStore keyed by (user_id, conversation_id)."""

    def __init__(self):
        self.data = {}

    async def get(self, user_id, conversation_id):
        return self.data.get((user_id, conversation_id))

    async def save(self, user_id, conversation_id, state):
        self.data[(user_id, conversation_id)] = state


class StubAgent:
    def __init__(self, name):
        self.name = name
        self.seen = []
        self.result = None

    async def run(self, state, runtime):
        self.seen.append((list(state["messages"]), runtime.context.user_id))
        if self.result is not None:
            return self.result
        get_stream_writer()(StreamEvent(type="content", content=f"from {self.name}"))
        return {"messages": [AIMessage(content=f"from {self.name}")]}


def make_app(monkeypatch, *routes):
    store = UserScopedStore()
    orchestrator = SimpleNamespace(route=MagicMock(side_effect=[
        {"route": r, "active_route": r, "agent_handoff": None} for r in routes]))
    agents = {route: StubAgent(route.value) for route in Route}
    app = LISA(conversation_store=store)
    app.graph = build_graph(
        orchestrator=orchestrator,
        knowledge_assistant_agent=agents[Route.KNOWLEDGE_ASSISTANT],
        troubleshooting_agent=agents[Route.TROUBLESHOOTING_AGENT],
        general_enquiry_agent=agents[Route.GENERAL_ENQUIRY],
        routing_policy=RoutingPolicy(),
    )
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=object())
    session.__aexit__ = AsyncMock(return_value=False)
    app.database = SimpleNamespace(session=lambda: session)
    monkeypatch.setattr("lisa.application.memory_dependencies", lambda session: SimpleNamespace())
    return app, store, orchestrator, agents


@pytest.mark.parametrize("route", list(Route))
async def test_chat_routes_to_each_main_agent_and_persists(monkeypatch, route):
    app, store, _, agents = make_app(monkeypatch, route)
    user_id, conversation_id = uuid4(), uuid4()

    reply = await app.chat(conversation_id, "Hello", user_id)

    assert reply == f"from {route.value}"
    messages, seen_user = agents[route].seen[0]
    assert seen_user == user_id
    assert messages[-1].content == "Hello"
    for other, agent in agents.items():
        assert len(agent.seen) == (1 if other == route else 0)
    saved = store.data[(user_id, conversation_id)]
    assert [m.content for m in saved["messages"]] == ["Hello", f"from {route.value}"]
    assert saved["active_route"] == route


async def test_follow_up_continues_with_active_agent_without_rerouting(monkeypatch):
    app, _, orchestrator, agents = make_app(monkeypatch, Route.TROUBLESHOOTING_AGENT)
    user_id, conversation_id = uuid4(), uuid4()

    await app.chat(conversation_id, "Feed is down", user_id)
    await app.chat(conversation_id, "Still down", user_id)

    assert orchestrator.route.call_count == 1
    assert len(agents[Route.TROUBLESHOOTING_AGENT].seen) == 2
    history = agents[Route.TROUBLESHOOTING_AGENT].seen[1][0]
    assert [m.content for m in history] == ["Feed is down", "from troubleshooting_agent", "Still down"]


async def test_conversations_are_isolated_per_user(monkeypatch):
    app, store, _, _ = make_app(monkeypatch, Route.GENERAL_ENQUIRY, Route.GENERAL_ENQUIRY)
    conversation_id = uuid4()
    alice, bob = uuid4(), uuid4()

    await app.chat(conversation_id, "Alice here", alice)
    await app.chat(conversation_id, "Bob here", bob)

    assert store.data[(alice, conversation_id)]["messages"][0].content == "Alice here"
    assert store.data[(bob, conversation_id)]["messages"][0].content == "Bob here"


async def test_stream_chat_yields_agent_events_and_persists(monkeypatch):
    app, store, _, _ = make_app(monkeypatch, Route.KNOWLEDGE_ASSISTANT)
    user_id, conversation_id = uuid4(), uuid4()

    events = [event async for event in app.stream_chat(conversation_id, "Runbook?", user_id)]

    assert [(e.type, e.content) for e in events] == [("content", "from knowledge_assistant")]
    saved = store.data[(user_id, conversation_id)]
    assert saved["messages"][-1].content == "from knowledge_assistant"


async def test_chat_reports_pending_approval_when_guardrail_suspends(monkeypatch):
    app, _, _, agents = make_app(monkeypatch, Route.TROUBLESHOOTING_AGENT)

    agents[Route.TROUBLESHOOTING_AGENT].result = {
        "messages": [AIMessage(content="")], "hitl_suspended": "execution-1"}
    reply = await app.chat(uuid4(), "Create a ticket", uuid4())

    assert reply.startswith("Approval required")


async def test_chat_requires_database(monkeypatch):
    app, _, _, _ = make_app(monkeypatch, Route.GENERAL_ENQUIRY)
    app.database = None
    with pytest.raises(RuntimeError):
        await app.chat(uuid4(), "Hello", uuid4())


def test_application_requires_conversation_store():
    with pytest.raises(ValueError):
        LISA()


async def test_existing_state_is_loaded(monkeypatch):
    app, store, _, _ = make_app(monkeypatch, Route.GENERAL_ENQUIRY)
    user_id, conversation_id = uuid4(), uuid4()
    store.data[(user_id, conversation_id)] = {
        "messages": [HumanMessage(content="Earlier"), AIMessage(content="Reply")],
        "route": None, "active_route": None, "agent_handoff": None, "knowledge_context": [],
    }

    await app.chat(conversation_id, "Now", user_id)

    assert [m.content for m in store.data[(user_id, conversation_id)]["messages"]] == [
        "Earlier", "Reply", "Now", "from general_enquiry"]
