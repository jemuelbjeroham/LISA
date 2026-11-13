from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage
from langgraph.runtime import Runtime
from mcp.types import CallToolResult, TextContent

from lisa.agents.general_enquiry import MCP_TOOL_ALLOWLIST, GeneralEnquiry
from lisa.agents.knowledge_assistant import KnowledgeAssistant
from lisa.agents.troubleshooting_agent import TroubleshootingAgent
from lisa.context import LISAContext
from lisa.guardrails.hitl.context import ExecutionContext, Principal, execution_scope
from lisa.guardrails.hitl.policy import GuardError, authorize

KNOWLEDGE = {"web_search", "web_search_with_content", "search_knowledge"}
TROUBLESHOOTING = KNOWLEDGE | {"search_jira_tickets", "onair_finder"}
ALL_TOOLS = MCP_TOOL_ALLOWLIST | {"unregistered_tool"}


@pytest.fixture(params=[(KnowledgeAssistant, KNOWLEDGE), (TroubleshootingAgent, TROUBLESHOOTING)])
async def specialist(request, monkeypatch):
    agent_class, allowed = request.param
    models = [MagicMock() for _ in range(3)]
    for model in models:
        model.bind_tools.return_value = model
    client = SimpleNamespace(
        list_tools=AsyncMock(return_value=SimpleNamespace(tools=[
            SimpleNamespace(name=name, description=name, input_schema={"type": "object"})
            for name in sorted(ALL_TOOLS)
        ])),
        call_tool=AsyncMock(return_value=CallToolResult(content=[TextContent(type="text", text="ok")])),
    )
    moderator = SimpleNamespace(moderate=AsyncMock(return_value=SimpleNamespace(flagged=False)))
    agent = agent_class(*models, "identity and role", client, moderator)
    await agent.initialize_tools()
    events = []
    monkeypatch.setattr(f"{agent_class.__module__}.get_stream_writer", lambda: events.append)
    principal = Principal(uuid4(), uuid4(), "admin")
    memory = SimpleNamespace(
        user_context_service=SimpleNamespace(build_context=AsyncMock(return_value="")),
        context_service=SimpleNamespace(build=AsyncMock(return_value="")),
    )
    runtime = Runtime(context=LISAContext(user_id=principal.user_id, memory=memory))
    service = SimpleNamespace(check=AsyncMock(side_effect=lambda principal, name: authorize(principal.role, name)),
                              propose=AsyncMock())
    return SimpleNamespace(agent=agent, models=models, client=client, principal=principal,
                           runtime=runtime, service=service, allowed=allowed, events=events)


async def test_exact_bound_tools(specialist):
    case = specialist
    assert set(case.agent.available_tools) == {"handoff"}
    assert set(case.agent.tool_schemas) == case.allowed
    for model in case.models:
        bindings = model.bind_tools.call_args.args[0]
        names = {item.name if hasattr(item, "name") else item["function"]["name"] for item in bindings}
        assert names == case.allowed | {"handoff"}
        for call in model.bind_tools.call_args_list:
            assert "save_memory" not in {
                item.name if hasattr(item, "name") else item["function"]["name"]
                for item in call.args[0]
            }


@pytest.mark.parametrize("name", ["save_memory", "create_jira_ticket", "update_jira_ticket", "close_ticket", "health_check", "unregistered_tool"])
async def test_forged_disallowed_model_call_never_dispatches(specialist, name):
    case = specialist
    responses = [AIMessageChunk(content="", tool_calls=[{"name": name, "id": "call-1", "args": {}}]),
                 AIMessageChunk(content="Done")]
    inputs = []

    async def stream(messages):
        inputs.append(list(messages))
        yield responses.pop(0)

    case.agent.model_resilience.stream = stream
    state = {"messages": [HumanMessage(content="Help")], "enable_thinking": False}
    with execution_scope(ExecutionContext(case.principal, uuid4(), case.service)):
        result = await case.agent.run(state, case.runtime)
    assert result["messages"][-1].content == "Done"
    receipt = next(message for message in inputs[-1] if isinstance(message, ToolMessage))
    assert "not authorized" in receipt.content
    case.client.call_tool.assert_not_awaited()
    case.service.propose.assert_not_awaited()


async def test_allowed_tool_dispatch_obeys_role_policy(specialist):
    case = specialist
    with execution_scope(ExecutionContext(case.principal, uuid4(), case.service)):
        assert await case.agent._execute_mcp_tool("search_knowledge", {}) == "ok"
    case.client.call_tool.assert_awaited_once_with("search_knowledge", {})
    case.client.call_tool.reset_mock()
    member = Principal(case.principal.user_id, case.principal.session_id, "member")
    with execution_scope(ExecutionContext(member, uuid4(), case.service)):
        with pytest.raises(GuardError, match="not_authorized"):
            await case.agent._execute_mcp_tool("search_knowledge", {})
        assert await case.agent._execute_mcp_tool("web_search", {}) == "ok"
    case.client.call_tool.assert_awaited_once_with("web_search", {})


async def test_knowledge_agent_rejects_troubleshooting_tools(specialist):
    case = specialist
    if not isinstance(case.agent, KnowledgeAssistant):
        return
    for name in ("search_jira_tickets", "onair_finder"):
        with pytest.raises(PermissionError):
            await case.agent._execute_mcp_tool(name, {})
    case.client.call_tool.assert_not_awaited()


async def test_unavailable_schema_fails_closed(specialist):
    case = specialist
    case.agent.tool_schemas.pop("web_search")
    with pytest.raises(PermissionError, match="No registered schema"):
        await case.agent._execute_mcp_tool("web_search", {})
    case.client.call_tool.assert_not_awaited()


async def test_handoff_takes_precedence_over_other_calls(specialist):
    case = specialist

    async def stream(messages):
        yield AIMessageChunk(content="", tool_calls=[
            {"name": "handoff", "id": "handoff-1", "args": {"reason": "Another agent is needed."}},
            {"name": "search_knowledge", "id": "search-1", "args": {}},
        ])

    case.agent.model_resilience.stream = stream
    result = await case.agent.run({"messages": [HumanMessage(content="Help")]}, case.runtime)
    assert result["agent_handoff"] == {"reason": "Another agent is needed."}
    case.client.call_tool.assert_not_awaited()


async def test_troubleshooting_can_use_ticket_and_onair_tools(specialist):
    case = specialist
    if not isinstance(case.agent, TroubleshootingAgent):
        return
    with execution_scope(ExecutionContext(case.principal, uuid4(), case.service)):
        for name in ("search_jira_tickets", "onair_finder"):
            assert await case.agent._execute_mcp_tool(name, {}) == "ok"
    assert [call.args[0] for call in case.client.call_tool.await_args_list] == ["search_jira_tickets", "onair_finder"]


async def test_general_enquiry_keeps_all_tools():
    model = MagicMock()
    model.bind_tools.return_value = model
    client = SimpleNamespace(list_tools=AsyncMock(return_value=SimpleNamespace(tools=[
        SimpleNamespace(name=name, description=name, input_schema={"type": "object"})
        for name in sorted(ALL_TOOLS)
    ])))
    agent = GeneralEnquiry(model, model, model, "prompt", client, MagicMock())
    await agent.initialize_tools()
    assert set(agent.available_tools) == {"handoff", "save_memory"}
    assert set(agent.tool_schemas) == MCP_TOOL_ALLOWLIST