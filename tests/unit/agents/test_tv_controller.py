"""TV controller agent: TV-only tools, multi-step execution and failure handling."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage
from langgraph.runtime import Runtime
from mcp.types import CallToolResult, TextContent

from lisa.agents import tv_controller
from lisa.agents.tool_access import TV_CONTROLLER_TOOLS
from lisa.agents.tv_controller import TV_UNAVAILABLE_MESSAGE, TVController
from lisa.context import LISAContext
from lisa.guardrails.hitl.context import ExecutionContext, Principal, execution_scope
from lisa.guardrails.hitl.policy import (
    TV_CONTROL_TOOLS,
    TV_READ_TOOLS,
    GuardError,
    authorize,
)

TV_TOOLS = TV_READ_TOOLS | TV_CONTROL_TOOLS
AGENT_MCP_TOOLS = TV_TOOLS | {"web_search"}
OTHER_TOOLS = {"web_search_with_content", "search_knowledge", "search_jira_tickets", "create_jira_ticket"}


def ok(text="{}"):
    return CallToolResult(content=[TextContent(type="text", text=text)])


def failed(text):
    return CallToolResult(is_error=True, content=[TextContent(type="text", text=text)])


def call(name, args=None, call_id=None):
    return {"name": name, "args": args or {}, "id": call_id or f"{name}-id"}


def turn(*calls, content=""):
    return AIMessageChunk(content=content, tool_calls=list(calls))


def make_agent(tool_names=AGENT_MCP_TOOLS | OTHER_TOOLS, results=None):
    models = [MagicMock() for _ in range(3)]
    for model in models:
        model.bind_tools.return_value = model
    tools = [
        SimpleNamespace(name=name, description=name, input_schema={"type": "object"})
        for name in sorted(tool_names)
    ]
    results = dict(results or {})
    client = SimpleNamespace(
        list_tools=AsyncMock(return_value=SimpleNamespace(tools=tools)),
        call_tool=AsyncMock(side_effect=lambda name, args: results.get(name, ok())),
    )
    moderator = SimpleNamespace(moderate=AsyncMock(return_value=SimpleNamespace(flagged=False)))
    agent = TVController(*models, "identity and role", client, moderator)
    return agent, models, client


@pytest.fixture
def harness(monkeypatch):
    events = []
    monkeypatch.setattr(tv_controller, "get_stream_writer", lambda: events.append)
    principal = Principal(uuid4(), uuid4(), "admin")
    memory = SimpleNamespace(
        user_context_service=SimpleNamespace(build_context=AsyncMock(return_value="")),
        context_service=SimpleNamespace(build=AsyncMock(return_value="")),
    )
    service = SimpleNamespace(
        check=AsyncMock(side_effect=lambda principal, name: authorize(principal.role, name)),
        propose=AsyncMock(),
    )
    return SimpleNamespace(
        events=events,
        principal=principal,
        runtime=Runtime(context=LISAContext(user_id=principal.user_id, memory=memory)),
        scope=lambda principal=principal: execution_scope(
            ExecutionContext(principal, uuid4(), service)
        ),
    )


def scripted(agent, *turns):
    """Replace the model stream with fixed turns and record each turn's input."""
    pending = list(turns)
    inputs = []

    async def stream(messages):
        inputs.append(list(messages))
        yield pending.pop(0)

    agent.model_resilience.stream = stream
    return inputs


def tool_replies(messages):
    return {m.tool_call_id: m.content for m in messages if isinstance(m, ToolMessage)}


async def test_binds_only_tv_tools_and_handoff():
    agent, models, _ = make_agent()
    await agent.initialize_tools()

    assert set(agent.tool_schemas) == AGENT_MCP_TOOLS
    assert set(agent.available_tools) == {"handoff"}
    assert TV_CONTROLLER_TOOLS == AGENT_MCP_TOOLS | {"handoff"}
    bound = models[0].bind_tools.call_args.args[0]
    names = {item.name if hasattr(item, "name") else item["function"]["name"] for item in bound}
    assert names == AGENT_MCP_TOOLS | {"handoff"}


async def test_rejects_tools_outside_the_tv(harness):
    agent, _, client = make_agent()
    await agent.initialize_tools()

    for name in OTHER_TOOLS:
        with pytest.raises(PermissionError):
            await agent._execute_mcp_tool(name, {})
    client.call_tool.assert_not_awaited()


async def test_reports_tv_unavailable_without_calling_the_model(harness):
    agent, _, _ = make_agent(tool_names=OTHER_TOOLS | {"web_search"})
    await agent.initialize_tools()
    inputs = scripted(agent)

    result = await agent.run({"messages": [HumanMessage(content="Turn on the TV")]}, harness.runtime)

    assert result["messages"][-1].content == TV_UNAVAILABLE_MESSAGE
    assert [(e.type, e.content) for e in harness.events] == [("content", TV_UNAVAILABLE_MESSAGE)]
    assert inputs == []


async def test_runs_a_multi_step_request_in_order(harness):
    agent, _, client = make_agent()
    await agent.initialize_tools()
    scripted(
        agent,
        turn(call("tv_get_status")),
        turn(call("tv_power_on")),
        turn(call("tv_launch_app", {"app": "YouTube"})),
        turn(call("tv_set_volume", {"level": 15})),
        turn(call("tv_get_status", call_id="confirm")),
        AIMessageChunk(content="TV on, YouTube open, volume 15."),
    )

    with harness.scope():
        result = await agent.run(
            {"messages": [HumanMessage(content="Turn on the TV, open YouTube and set volume 15")]},
            harness.runtime,
        )

    assert result["messages"][-1].content == "TV on, YouTube open, volume 15."
    assert [c.args for c in client.call_tool.await_args_list] == [
        ("tv_get_status", {}),
        ("tv_power_on", {}),
        ("tv_launch_app", {"app": "YouTube"}),
        ("tv_set_volume", {"level": 15}),
        ("tv_get_status", {}),
    ]


async def test_failed_step_skips_the_rest_of_the_batch_and_reports_the_tv_error(harness):
    error = "The TV is off or unreachable at 192.0.2.10. Use tv_power_on to turn it on."
    agent, _, client = make_agent(results={"tv_launch_app": failed(error)})
    await agent.initialize_tools()
    inputs = scripted(
        agent,
        turn(
            call("tv_launch_app", {"app": "YouTube"}),
            call("tv_set_volume", {"level": 15}),
        ),
        AIMessageChunk(content="The TV is off, so I couldn't open YouTube."),
    )

    with harness.scope():
        result = await agent.run({"messages": [HumanMessage(content="Open YouTube at 15")]}, harness.runtime)

    assert result["messages"][-1].content == "The TV is off, so I couldn't open YouTube."
    client.call_tool.assert_awaited_once_with("tv_launch_app", {"app": "YouTube"})
    replies = tool_replies(inputs[-1])
    assert error in replies["tv_launch_app-id"]
    assert "Do not repeat" in replies["tv_launch_app-id"]
    assert "Skipped" in replies["tv_set_volume-id"]


async def test_plays_a_named_song_by_searching_then_launching_youtube(harness):
    search = ok('{"results": [{"url": "https://www.youtube.com/watch?v=8xg3vE8Ie_E", '
                '"title": "Taylor Swift - Love Story"}]}')
    agent, _, client = make_agent(results={"web_search": search})
    await agent.initialize_tools()
    inputs = scripted(
        agent,
        turn(call("web_search", {"query": "Taylor Swift Love Story official video youtube"})),
        turn(call("tv_launch_app", {"app": "YouTube",
                                    "content_id": "https://www.youtube.com/watch?v=8xg3vE8Ie_E"})),
        AIMessageChunk(content="Playing Taylor Swift - Love Story on YouTube."),
    )

    with harness.scope():
        result = await agent.run(
            {"messages": [HumanMessage(content="play Taylor Swift's love story on the TV")]},
            harness.runtime,
        )

    assert result["messages"][-1].content == "Playing Taylor Swift - Love Story on YouTube."
    assert [c.args[0] for c in client.call_tool.await_args_list] == ["web_search", "tv_launch_app"]
    assert "8xg3vE8Ie_E" in tool_replies(inputs[1])["web_search-id"]


async def test_independent_reads_in_one_turn_all_run(harness):
    agent, _, client = make_agent()
    await agent.initialize_tools()
    scripted(
        agent,
        turn(call("tv_get_status"), call("tv_list_inputs")),
        AIMessageChunk(content="It's on HDMI 1."),
    )

    with harness.scope():
        await agent.run({"messages": [HumanMessage(content="What input is on?")]}, harness.runtime)

    assert [c.args[0] for c in client.call_tool.await_args_list] == ["tv_get_status", "tv_list_inputs"]


async def test_tv_tools_are_admin_only(harness):
    agent, _, client = make_agent()
    await agent.initialize_tools()
    member = Principal(harness.principal.user_id, harness.principal.session_id, "member")

    with harness.scope(member), pytest.raises(GuardError, match="not_authorized"):
        await agent._execute_mcp_tool("tv_power_off", {})
    client.call_tool.assert_not_awaited()


async def test_hands_off_requests_that_are_not_about_the_tv(harness):
    agent, _, client = make_agent()
    await agent.initialize_tools()
    scripted(agent, turn(call("handoff", {"reason": "User wants a web search."})))

    result = await agent.run({"messages": [HumanMessage(content="Search the web")]}, harness.runtime)

    assert result == {"agent_handoff": {"reason": "User wants a web search."}}
    client.call_tool.assert_not_awaited()


async def test_allows_more_tool_steps_than_the_other_agents():
    from lisa.agents import troubleshooting_agent

    assert tv_controller.MAX_TOOL_ITERATIONS > troubleshooting_agent.MAX_TOOL_ITERATIONS


async def test_does_not_repeat_a_completed_action_in_the_same_turn(harness):
    launch = {"app": "YouTube", "content_id": "8xg3vE8Ie_E"}
    agent, _, client = make_agent()
    await agent.initialize_tools()
    inputs = scripted(
        agent,
        turn(call("tv_launch_app", launch, call_id="first")),
        turn(call("tv_launch_app", launch, call_id="again")),
        AIMessageChunk(content="YouTube is asking who's watching."),
    )

    with harness.scope():
        result = await agent.run({"messages": [HumanMessage(content="Play Love Story")]}, harness.runtime)

    assert result["messages"][-1].content == "YouTube is asking who's watching."
    client.call_tool.assert_awaited_once_with("tv_launch_app", launch)
    assert "Not sent again" in tool_replies(inputs[-1])["again"]


async def test_reads_and_volume_steps_may_repeat_in_one_turn(harness):
    agent, _, client = make_agent()
    await agent.initialize_tools()
    up = {"direction": "up", "steps": 2}
    scripted(
        agent,
        turn(call("tv_adjust_volume", up, call_id="a")),
        turn(call("tv_adjust_volume", up, call_id="b")),
        turn(call("tv_get_status", call_id="c")),
        turn(call("tv_get_status", call_id="d")),
        AIMessageChunk(content="Volume raised by 4."),
    )

    with harness.scope():
        await agent.run({"messages": [HumanMessage(content="Up 2, then 2 more")]}, harness.runtime)

    assert [c.args[0] for c in client.call_tool.await_args_list] == [
        "tv_adjust_volume", "tv_adjust_volume", "tv_get_status", "tv_get_status",
    ]


async def test_a_failed_action_can_be_retried(harness):
    agent, _, client = make_agent(results={"tv_set_volume": failed("TV still starting")})
    await agent.initialize_tools()
    scripted(
        agent,
        turn(call("tv_set_volume", {"level": 10}, call_id="a")),
        turn(call("tv_set_volume", {"level": 10}, call_id="b")),
        AIMessageChunk(content="Couldn't set the volume."),
    )

    with harness.scope():
        await agent.run({"messages": [HumanMessage(content="Volume 10")]}, harness.runtime)

    assert client.call_tool.await_count == 2
