from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

from lisa.agents.base import BaseAgent
from lisa.agents.technical_specialist.butler import Butler
from lisa.agents.technical_specialist.communication import TaskChannel, secure_line
from lisa.agents.technical_specialist.models import SpecialistResult
from lisa.agents.technical_specialist.specialist import Specialist


@pytest.fixture(autouse=True)
def allow_guarded_tools(request, monkeypatch):
    """Specialist tool calls pass through the HITL guard; allow them unless a test opts out."""
    if "real_guard" in request.keywords:
        return
    monkeypatch.setattr(
        "lisa.agents.technical_specialist.specialist.check_tool",
        AsyncMock(return_value=SimpleNamespace(write=False)),
    )


def test_secure_line_scopes_delivery_and_rejects_missing_context():
    channel = TaskChannel()
    arguments = {"task_id": channel.task_id, "sender": "butler", "recipient": "specialist",
                 "message_type": "task", "payload": "Investigate"}
    assert secure_line.invoke(arguments)["accepted"] is False
    assert secure_line.invoke({**arguments, "channel": channel})["accepted"] is True
    assert channel.messages[0].payload == "Investigate"
    other = TaskChannel()
    assert secure_line.invoke({**arguments, "channel": other})["accepted"] is False
    assert other.messages == []
    channel.closed = True
    assert secure_line.invoke({**arguments, "channel": channel})["accepted"] is False


def test_secure_line_invalid_payload_and_recipient_fail():
    channel = TaskChannel()
    arguments = {"task_id": channel.task_id, "sender": "specialist", "recipient": "specialist",
                 "message_type": "finding", "payload": "evidence", "channel": channel}
    assert secure_line.invoke(arguments)["accepted"] is False
    assert secure_line.invoke({**arguments, "recipient": "butler", "payload": " "})["accepted"] is False
    assert channel.messages == []


def test_specialist_result_validates_incomplete_status():
    with pytest.raises(ValidationError):
        SpecialistResult(status="needs_input", summary="Which device?", next_action="present_result")
    with pytest.raises(ValidationError):
        SpecialistResult(status="failed", summary="Unavailable", next_action="present_result")


def specialist_fixture(responses, *, tool_error=False):
    model = MagicMock()
    model.bind_tools.return_value = model
    model.ainvoke = AsyncMock(side_effect=responses)
    model.with_structured_output.return_value.ainvoke = AsyncMock(return_value={
        "status": "completed", "summary": "Found procedure", "findings": ["Check link"],
        "next_action": "present_result"})
    client = SimpleNamespace(list_tools=AsyncMock(return_value=SimpleNamespace(tools=[
        SimpleNamespace(name="search_knowledge", description="Search", input_schema={"type": "object",
            "properties": {"query": {"type": "string"}}, "required": ["query"]}),
        SimpleNamespace(name="unauthorized", description="No", input_schema={})])),
        call_tool=AsyncMock(return_value=SimpleNamespace(is_error=tool_error,
            content=[SimpleNamespace(text='{"results":[{"content":"Check link","source":"https://example.com"}]}')])))
    return Specialist(model, client), model, client


@pytest.mark.anyio
async def test_specialist_react_observes_multiple_calls_and_returns_actual_evidence():
    call = {"name": "search_knowledge", "args": {"query": "link"}, "id": "one"}
    agent, model, client = specialist_fixture([AIMessage(content="", tool_calls=[call]),
        AIMessage(content="", tool_calls=[{**call, "id": "two"}]), AIMessage(content="done")])
    await agent.initialize_tools()
    result = await agent.run({"messages": [HumanMessage(content="Check link")]}, TaskChannel())
    assert isinstance(agent, BaseAgent)
    assert "LISA" in agent.system_prompt
    assert set(agent.tool_schemas) == {"search_knowledge"}
    assert client.call_tool.await_count == 2
    assert [item.call_id for item in result.evidence] == ["one", "two"]
    assert "https://example.com" in result.evidence[0].content
    assert result.status == "completed"
    assert any(getattr(item, "type", "") == "tool" for item in model.ainvoke.call_args.args[0])


@pytest.mark.anyio
async def test_specialist_tool_error_is_not_success():
    agent, _, _ = specialist_fixture([AIMessage(content="", tool_calls=[{
        "name": "search_knowledge", "args": {"query": "x"}, "id": "one"}]), AIMessage(content="done")], tool_error=True)
    await agent.initialize_tools()
    result = await agent.run({"messages": []}, TaskChannel())
    assert result.status == "partial"
    assert result.evidence == []
    assert result.limitations


@pytest.mark.anyio
async def test_specialist_clarification_reaches_butler():
    channel = TaskChannel()
    agent, _, _ = specialist_fixture([AIMessage(content="", tool_calls=[{"name": "secure_line", "id": "clarify",
        "args": {"task_id": channel.task_id, "sender": "specialist", "recipient": "butler",
                 "message_type": "clarification", "payload": "Which firewall?"}}])])
    await agent.initialize_tools()
    result = await agent.run({"messages": []}, channel)
    assert result.status == "needs_input"
    assert channel.messages[-1].recipient == "butler"
    assert result.summary == "Which firewall?"


@pytest.mark.anyio
async def test_butler_streams_before_generation_completes(monkeypatch):
    import asyncio

    from langchain_core.messages import AIMessageChunk

    events = []
    reached_first = asyncio.Event()
    release = asyncio.Event()
    model = MagicMock()
    model.bind_tools.return_value = model
    model.ainvoke = AsyncMock(return_value=AIMessage(content="unverified draft must not stream", tool_calls=[
        {"name": "handoff", "args": {"reason": "Check link"}, "id": "delegate"}]))

    async def stream(messages):
        yield AIMessageChunk(content="Verified ")
        reached_first.set()
        await release.wait()
        yield AIMessageChunk(content="finding.")

    model.astream = stream
    specialist = SimpleNamespace(run=AsyncMock(return_value=SpecialistResult(status="completed",
        summary="Link checked", next_action="present_result")))
    moderator = SimpleNamespace(moderate=AsyncMock(return_value=SimpleNamespace(flagged=False)))
    agent = Butler(model, model, model, specialist, moderator)
    monkeypatch.setattr("lisa.agents.technical_specialist.butler.get_stream_writer", lambda: events.append)
    memory = SimpleNamespace(user_context_service=SimpleNamespace(build_context=AsyncMock(return_value="")),
        context_service=SimpleNamespace(build=AsyncMock(return_value="")))
    runtime = SimpleNamespace(context=SimpleNamespace(user_id="user", memory=memory))
    pending = asyncio.create_task(agent.run({"messages": [HumanMessage(content="Check link")]}, runtime))
    await asyncio.wait_for(reached_first.wait(), 2)
    assert not pending.done()
    assert [event.content for event in events] == ["Verified "]
    release.set()
    result = await pending
    assert result["messages"][0].content == "Verified finding."
    assert set(agent.available_tools) == {"handoff", "save_memory"}
    assert isinstance(agent, BaseAgent)
    assert "LISA" in agent.system_prompt
    channel = specialist.run.call_args.args[1]
    assert channel.closed
    assert [message.message_type for message in channel.messages] == ["task", "result"]


@pytest.mark.anyio
@pytest.mark.parametrize("route", ["knowledge_assistant", "troubleshooting_agent", "general_enquiry"])
async def test_existing_graph_routes_to_supplied_agents(route):
    from lisa.graph import build_graph
    from lisa.routing import Route, RoutingPolicy

    agents = {name: SimpleNamespace(run=AsyncMock(return_value={"messages": [AIMessage(content=name)],
              "agent_handoff": None})) for name in ("knowledge_assistant", "troubleshooting_agent", "general_enquiry")}
    orchestrator = SimpleNamespace(route=AsyncMock(return_value={"route": Route(route)}))
    graph = build_graph(orchestrator, agents["knowledge_assistant"], agents["troubleshooting_agent"],
                        agents["general_enquiry"], RoutingPolicy())
    result = await graph.ainvoke({"messages": [HumanMessage(content="question")], "active_route": None})
    assert result["messages"][-1].content == route
    for name, agent in agents.items():
        assert agent.run.await_count == (1 if name == route else 0)


@pytest.mark.anyio
async def test_specialist_malformed_result_retries_are_bounded():
    agent, model, _ = specialist_fixture([AIMessage(content="done")])
    model.with_structured_output.return_value.ainvoke = AsyncMock(return_value={"status": "imaginary"})
    await agent.initialize_tools()
    result = await agent.run({"messages": []}, TaskChannel())
    assert result.status == "failed"
    assert model.with_structured_output.return_value.ainvoke.await_count == 2


@pytest.mark.anyio
async def test_specialist_stops_at_limit_and_rejects_unavailable_tool():
    agent, model, client = specialist_fixture([AIMessage(content="", tool_calls=[
        {"name": "unauthorized", "args": {}, "id": "denied"}])])
    agent.max_iterations = 1
    await agent.initialize_tools()
    result = await agent.run({"messages": []}, TaskChannel())
    client.call_tool.assert_not_called()
    assert result.status == "partial"
    assert "iteration limit" in " ".join(result.limitations)
    assert model.ainvoke.await_count == 1


def butler_fixture(result=None):
    model = MagicMock()
    model.bind_tools.return_value = model
    model.ainvoke = AsyncMock(return_value=AIMessage(content="", tool_calls=[{
        "name": "handoff", "args": {"reason": "Investigate"}, "id": "delegate"}]))
    specialist = SimpleNamespace(run=AsyncMock(return_value=result or SpecialistResult(
        status="completed", summary="Verified", next_action="present_result")))
    agent = Butler(model, model, model, specialist,
        SimpleNamespace(moderate=AsyncMock(return_value=SimpleNamespace(flagged=False))))
    runtime = SimpleNamespace(context=SimpleNamespace(user_id="user", memory=SimpleNamespace(
        user_context_service=SimpleNamespace(build_context=AsyncMock(return_value="")),
        context_service=SimpleNamespace(build=AsyncMock(return_value="")))))
    return agent, model, specialist, runtime


@pytest.mark.anyio
async def test_butler_cancellation_closes_channel(monkeypatch):
    import asyncio

    from langchain_core.messages import AIMessageChunk

    agent, model, specialist, runtime = butler_fixture()
    reached = asyncio.Event()

    async def stream(messages):
        yield AIMessageChunk(content="First")
        reached.set()
        await asyncio.Event().wait()

    model.astream = stream
    events = []
    monkeypatch.setattr("lisa.agents.technical_specialist.butler.get_stream_writer", lambda: events.append)
    pending = asyncio.create_task(agent.run({"messages": [HumanMessage(content="task")]}, runtime))
    await asyncio.wait_for(reached.wait(), 2)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert specialist.run.call_args.args[1].closed
    assert [event.content for event in events] == ["First"]


@pytest.mark.anyio
async def test_butler_interrupted_stream_does_not_replay(monkeypatch):
    from langchain_core.messages import AIMessageChunk

    agent, model, _, runtime = butler_fixture()
    calls = []

    async def stream(messages):
        calls.append(1)
        yield AIMessageChunk(content="First")
        raise RuntimeError("private detail")

    model.astream = stream
    events = []
    monkeypatch.setattr("lisa.agents.technical_specialist.butler.get_stream_writer", lambda: events.append)
    await agent.run({"messages": [HumanMessage(content="task")]}, runtime)
    assert len(calls) == 1
    assert events[0].content == "First"
    assert "interrupted" in events[1].content


@pytest.mark.anyio
async def test_butler_clarification_continues_using_conversation_history(monkeypatch):
    from langchain_core.messages import AIMessageChunk

    result = SpecialistResult(status="needs_input", summary="Which firewall?", next_action="request_clarification")
    agent, model, specialist, runtime = butler_fixture(result)

    async def stream(messages):
        assert "Which firewall?" in messages[-1].content
        yield AIMessageChunk(content="Which firewall?")

    model.astream = stream
    monkeypatch.setattr("lisa.agents.technical_specialist.butler.get_stream_writer", lambda: lambda event: None)
    first = await agent.run({"messages": [HumanMessage(content="Check the firewall")]}, runtime)
    history = [HumanMessage(content="Check the firewall"), *first["messages"], HumanMessage(content="FW-1")]
    await agent.run({"messages": history}, runtime)
    assignment = specialist.run.call_args.args[0]["messages"]
    assert any(message.content == "FW-1" for message in assignment)
    assert any(message.content == "Which firewall?" for message in assignment)
    assert specialist.run.call_args_list[0].args[1].task_id != specialist.run.call_args_list[1].args[1].task_id


@pytest.mark.anyio
async def test_communication_failure_is_not_delivered():
    agent, _, specialist, _ = butler_fixture()
    agent.communication_tool = SimpleNamespace(ainvoke=AsyncMock(return_value={"accepted": False}))
    with pytest.raises(RuntimeError, match="task_delivery_failed"):
        await agent._delegate("task", {"messages": []}, TaskChannel())
    specialist.run.assert_not_called()


@pytest.mark.anyio
async def test_specialist_cancellation_propagates():
    import asyncio

    agent, model, _ = specialist_fixture([])
    model.ainvoke = AsyncMock(side_effect=asyncio.CancelledError)
    await agent.initialize_tools()
    with pytest.raises(asyncio.CancelledError):
        await agent.run({"messages": []}, TaskChannel())


@pytest.mark.anyio
async def test_application_receives_live_graph_chunks_and_persists_completion(monkeypatch):
    import asyncio
    from uuid import uuid4

    from langchain_core.messages import AIMessageChunk

    from lisa.application import LISA
    from lisa.graph import build_graph
    from lisa.routing import Route, RoutingPolicy
    from lisa.streaming.events import StreamEvent

    agent, model, _, runtime = butler_fixture()
    release = asyncio.Event()
    finished = asyncio.Event()

    async def stream(messages):
        yield AIMessageChunk(content="First ")
        await release.wait()
        yield AIMessageChunk(content="second")
        finished.set()

    model.astream = stream
    store = SimpleNamespace(get=AsyncMock(return_value=None), save=AsyncMock())
    app = LISA(conversation_store=store)
    app.graph = build_graph(SimpleNamespace(route=AsyncMock(return_value={"route": Route.TROUBLESHOOTING_AGENT})),
        SimpleNamespace(run=AsyncMock()), agent, SimpleNamespace(run=AsyncMock()), RoutingPolicy())
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=object())
    session.__aexit__ = AsyncMock(return_value=False)
    app.database = SimpleNamespace(session=lambda: session)
    monkeypatch.setattr("lisa.application.memory_dependencies", lambda session: runtime.context.memory)
    output = app.stream_chat(uuid4(), "Investigate", uuid4())
    first = await asyncio.wait_for(anext(output), 3)
    assert isinstance(first, StreamEvent)
    assert (first.type, first.content) == ("content", "First ")
    assert not finished.is_set()
    store.save.assert_not_called()
    release.set()
    rest = [event async for event in output]
    assert [event.content for event in rest] == ["second"]
    assert finished.is_set()
    assert store.save.call_args.args[2]["messages"][-1].content == "First second"


@pytest.mark.anyio
async def test_butler_moderation_block_precedes_memory_or_tools(monkeypatch):
    from lisa.guardrails.input_moderation import (
        InputModerationBlocked,
        InputModerationResult,
    )

    agent, model, specialist, runtime = butler_fixture()
    agent.input_moderator.moderate = AsyncMock(side_effect=InputModerationBlocked(InputModerationResult(flagged=True)))
    events = []
    monkeypatch.setattr("lisa.agents.technical_specialist.butler.get_stream_writer", lambda: events.append)
    await agent.run({"messages": [HumanMessage(content="task")]}, runtime)
    model.ainvoke.assert_not_called()
    specialist.run.assert_not_called()
    runtime.context.memory.context_service.build.assert_not_called()
    assert len(events) == 1


@pytest.mark.anyio
async def test_model_tool_boundaries_and_injected_channel_schema():
    agent, model, _ = specialist_fixture([])
    await agent.initialize_tools()
    tools = model.bind_tools.call_args.args[0]
    assert tools[0].name == "secure_line"
    assert [entry["function"]["name"] for entry in tools[1:]] == ["search_knowledge"]
    assert "channel" not in secure_line.tool_call_schema.model_json_schema()["properties"]
    butler, model, _, _ = butler_fixture()
    for binding in model.bind_tools.call_args_list:
        assert [entry.name for entry in binding.args[0]] == ["handoff", "save_memory"]
    assert butler.communication_tool is secure_line


@pytest.mark.anyio
async def test_specialist_cannot_spoof_butler_sender():
    channel = TaskChannel()
    agent, _, _ = specialist_fixture([AIMessage(content="", tool_calls=[{"name": "secure_line", "id": "spoof",
        "args": {"task_id": channel.task_id, "sender": "butler", "recipient": "specialist",
                 "message_type": "finding", "payload": "fake"}}]), AIMessage(content="done")])
    await agent.initialize_tools()
    result = await agent.run({"messages": []}, channel)
    assert all(message.message_type == "progress" and message.sender == "specialist" for message in channel.messages)
    assert not any(message.payload == "fake" for message in channel.messages)
    assert result.status == "partial"
    assert result.limitations


@pytest.mark.anyio
async def test_butler_relays_progress_while_specialist_tool_is_running(monkeypatch):
    import asyncio

    from langchain_core.messages import AIMessageChunk

    call = {"name": "search_knowledge", "args": {"query": "link"}, "id": "search"}
    specialist, _, client = specialist_fixture([AIMessage(content="", tool_calls=[call]), AIMessage(content="done")])
    await specialist.initialize_tools()
    started, release = asyncio.Event(), asyncio.Event()

    async def search(*args):
        started.set()
        await release.wait()
        return SimpleNamespace(is_error=False, content=[SimpleNamespace(text="Verified procedure")])

    client.call_tool = search
    butler, model, _, runtime = butler_fixture()
    butler.specialist = specialist

    async def stream(messages):
        yield AIMessageChunk(content="Final answer")

    model.astream = stream
    events = []
    monkeypatch.setattr("lisa.agents.technical_specialist.butler.get_stream_writer", lambda: events.append)
    pending = asyncio.create_task(butler.run({"messages": [HumanMessage(content="Check link")]}, runtime))
    await asyncio.wait_for(started.wait(), 3)
    try:
        assert not pending.done()
        text = "".join(event.content for event in events)
        assert "checking the internal knowledge base" in text
        assert "returned" not in text
        assert "Final answer" not in text
    finally:
        release.set()
    result = await pending
    text = result["messages"][0].content
    assert text.index("checking the internal") < text.index("returned") < text.index("Final answer")
    assert all(event.type == "content" for event in events)


@pytest.mark.anyio
async def test_butler_preserves_user_preferences_and_memories_in_both_contexts(monkeypatch):
    from langchain_core.messages import AIMessageChunk

    butler, model, specialist, runtime = butler_fixture()
    runtime.context.memory.user_context_service.build_context.return_value = "Preference: concise responses"
    runtime.context.memory.context_service.build.return_value = "Project: firewall migration"

    async def stream(messages):
        yield AIMessageChunk(content="Answer")

    model.astream = stream
    monkeypatch.setattr("lisa.agents.technical_specialist.butler.get_stream_writer", lambda: lambda event: None)
    await butler.run({"messages": [HumanMessage(content="Check firewall")]}, runtime)
    for messages in [model.ainvoke.call_args.args[0], specialist.run.call_args.args[0]["messages"]]:
        context = " ".join(str(message.content) for message in messages)
        assert "Preference: concise responses" in context
        assert "Project: firewall migration" in context
    runtime.context.memory.context_service.build.assert_awaited_once_with(user_id="user", query="Check firewall")

@pytest.mark.anyio
@pytest.mark.real_guard
async def test_specialist_tool_call_fails_closed_without_execution_context():
    call = {"name": "search_knowledge", "args": {"query": "link"}, "id": "one"}
    agent, _, client = specialist_fixture([AIMessage(content="", tool_calls=[call]), AIMessage(content="done")])
    await agent.initialize_tools()
    result = await agent.run({"messages": [HumanMessage(content="Check link")]}, TaskChannel())
    client.call_tool.assert_not_awaited()
    assert result.evidence == []
