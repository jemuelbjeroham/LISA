import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    ToolMessage,
    messages_to_dict,
)
from langgraph.runtime import Runtime
from mcp.types import CallToolResult, TextContent

from lisa.agents.general_enquiry import (
    LOCAL_TOOL_ALLOWLIST,
    MCP_TOOL_ALLOWLIST,
    GeneralEnquiry,
)
from lisa.agents.technical_specialist.butler import Butler
from lisa.agents.technical_specialist.models import SpecialistResult
from lisa.context import LISAContext
from lisa.graph import route_after_handoff
from lisa.guardrails.hitl.context import ExecutionContext, Principal, execution_scope
from lisa.guardrails.hitl.policy import authorize


class FakeService:
    def __init__(self):
        self.proposal = {
            "approval_id": str(uuid4()),
            "execution_id": str(uuid4()),
            "revision": 1,
            "status": "pending",
        }
        self.continuation = None
        self.check = AsyncMock(side_effect=self._check)
        self.propose = AsyncMock(side_effect=self._propose)

    async def _check(self, principal, name):
        return authorize(principal.role, name)

    async def _propose(self, principal, conversation_id, name, arguments, logical_id, continuation):
        await self.check(principal, name)
        self.continuation = json.loads(json.dumps(continuation))
        return dict(self.proposal)


@pytest.fixture(params=["butler", "general_enquiry"])
def agent_case(request, monkeypatch):
    name = request.param
    model = MagicMock()
    model.bind_tools.return_value = model
    responses = []
    inputs = []

    async def invoke(messages):
        inputs.append(list(messages))
        return responses.pop(0)

    async def stream(messages):
        inputs.append(list(messages))
        response = responses.pop(0)
        yield AIMessageChunk(content=response.content, tool_calls=response.tool_calls)

    model.ainvoke = AsyncMock(side_effect=invoke)
    model.astream = MagicMock(side_effect=stream)
    specialist = SimpleNamespace(run=AsyncMock(return_value=SpecialistResult(
        status="completed", summary="Checked offline", next_action="present_result",
    )))
    moderator = SimpleNamespace(moderate=AsyncMock(return_value=SimpleNamespace(flagged=False)))
    schema = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
    client = SimpleNamespace(
        list_tools=AsyncMock(return_value=SimpleNamespace(tools=[SimpleNamespace(
            name="web_search", description="Offline search", input_schema=schema,
        )])),
        call_tool=AsyncMock(return_value=CallToolResult(content=[
            TextContent(type="text", text="Offline search receipt"),
        ])),
    )
    if name == "butler":
        agent = Butler(model, model, model, specialist, moderator, system_prompt="Offline test")
        module = "lisa.agents.technical_specialist.butler"
    else:
        agent = GeneralEnquiry(model, model, model, "Offline test", client, moderator)
        module = "lisa.agents.general_enquiry"
    events = []
    monkeypatch.setattr(f"{module}.get_stream_writer", lambda: events.append)
    memory = SimpleNamespace(
        service=SimpleNamespace(
            create=AsyncMock(),
            find_duplicate=AsyncMock(return_value=None),
        ),
        user_context_service=SimpleNamespace(build_context=AsyncMock(return_value="")),
        context_service=SimpleNamespace(build=AsyncMock(return_value="")),
    )
    principal = Principal(uuid4(), uuid4(), "admin")
    runtime = Runtime(context=LISAContext(user_id=principal.user_id, memory=memory))
    return SimpleNamespace(
        name=name, agent=agent, model=model, responses=responses, inputs=inputs,
        specialist=specialist, client=client, events=events, memory=memory,
        principal=principal, runtime=runtime, service=FakeService(), conversation_id=uuid4(),
        state={"messages": [HumanMessage(content="Remember the offline finding", id="user-turn")],
               "enable_thinking": False},
    )


def memory_call(call_id):
    return {"name": "save_memory", "id": call_id, "args": {
        "content": "  Offline finding  ", "memory_type": "fact", "scope": "user",
    }}


def prior_call(case):
    if case.name == "butler":
        return {"name": "handoff", "id": "prior", "args": {"reason": "Check offline"}}
    return {"name": "web_search", "id": "prior", "args": {"query": "offline"}}


async def run_agent(case):
    if case.name == "general_enquiry":
        await case.agent.initialize_tools()
    context = ExecutionContext(case.principal, case.conversation_id, case.service)
    with execution_scope(context):
        return await case.agent.run(case.state, case.runtime)


def assert_no_memory_write(case):
    case.memory.service.create.assert_not_awaited()
    case.memory.service.find_duplicate.assert_not_awaited()


@pytest.mark.parametrize("role", ["admin", "member"])
async def test_general_jira_call_suspends_before_mcp_or_returns_typed_member_denial(agent_case, role):
    case = agent_case
    if case.name != "general_enquiry":
        return
    case.principal = Principal(case.principal.user_id, case.principal.session_id, role)
    case.client.list_tools.return_value.tools.append(SimpleNamespace(name="update_jira_ticket",
        description="Comment", input_schema={"type": "object"}))
    case.responses.append(AIMessage(content="", tool_calls=[{
        "name": "update_jira_ticket", "id": "jira-comment", "args": {"ticket_key": "OPS-123", "comment": "Recovered"}}]))
    if role == "member":
        case.responses.append(AIMessage(content="Not authorized."))
    result = await run_agent(case)
    case.client.call_tool.assert_not_awaited()
    if role == "admin":
        assert result["hitl_suspended"] == case.service.proposal["execution_id"]
        assert [event.type for event in case.events] == ["approval_required"]
        assert case.service.continuation["agent"] == "general_enquiry"
        assert case.service.continuation["call_id"] == "jira-comment"
    else:
        assert not result.get("hitl_suspended")
        receipt = next(message for message in case.inputs[-1] if isinstance(message, ToolMessage))
        assert json.loads(receipt.content) == {"accepted": False, "error": "not_authorized"}


def assert_allowlists_unchanged(case):
    assert set(case.agent.available_tools) == {"handoff", "save_memory"}
    assert LOCAL_TOOL_ALLOWLIST == frozenset({"handoff", "save_memory"})
    from lisa.guardrails.hitl.policy import REGISTRY
    assert MCP_TOOL_ALLOWLIST == frozenset(
        name for name, policy in REGISTRY.items() if policy.namespace == "mcp"
    )
    if case.name == "general_enquiry":
        assert set(case.agent.tool_schemas) == {"web_search"}


@pytest.mark.parametrize("with_prior_receipt", [False, True])
@pytest.mark.parametrize("role", ["admin", "member"])
async def test_actual_agent_saves_memory_without_approval(agent_case, with_prior_receipt, role):
    case = agent_case
    case.principal = Principal(case.principal.user_id, case.principal.session_id, role)
    calls = ([prior_call(case)] if with_prior_receipt else []) + [
        memory_call("memory"),
    ]
    case.responses.extend([AIMessage(content="", tool_calls=calls), AIMessage(content="Memory saved.")])
    if case.name == "butler" and not with_prior_receipt:
        case.responses.append(AIMessage(content="Memory saved."))
    history = messages_to_dict(case.state["messages"])

    result = await run_agent(case)

    assert not result.get("hitl_suspended")
    assert not result.get("agent_handoff")
    case.service.propose.assert_not_awaited()
    case.service.check.assert_any_await(case.principal, "save_memory")
    assert all(event.type != "approval_required" for event in case.events)
    receipts = [message for message in case.inputs[1] if isinstance(message, ToolMessage)]
    assert [receipt.tool_call_id for receipt in receipts] == (["prior"] if with_prior_receipt else []) + ["memory"]
    assert receipts[-1].content == "Memory saved successfully: Offline finding"
    case.memory.service.create.assert_awaited_once()
    saved = case.memory.service.create.call_args.args[0]
    assert saved.user_id == case.principal.user_id
    assert saved.content == "Offline finding"
    assert case.client.call_tool.await_count == int(case.name == "general_enquiry" and with_prior_receipt)
    assert case.responses == []
    assert messages_to_dict(case.state["messages"]) == history
    assert_allowlists_unchanged(case)


async def test_actual_agent_unknown_role_denial_is_a_structured_tool_receipt(agent_case):
    case = agent_case
    case.principal = Principal(case.principal.user_id, case.principal.session_id, "unknown")
    case.responses.extend([
        AIMessage(content="", tool_calls=[prior_call(case), memory_call("denied")]),
        AIMessage(content="Memory was not saved."),
    ])

    result = await run_agent(case)

    case.service.propose.assert_not_awaited()
    assert case.service.continuation is None
    assert any(call.args == (case.principal, "save_memory") for call in case.service.check.await_args_list)
    assert len(case.inputs) == 2
    receipts = [message for message in case.inputs[-1] if isinstance(message, ToolMessage)]
    assert [receipt.tool_call_id for receipt in receipts] == ["prior", "denied"]
    assert json.loads(receipts[-1].content) == {"accepted": False, "error": "not_authorized"}
    assert not result.get("hitl_suspended")
    assert not result.get("agent_handoff")
    assert result["messages"][0].content == "Memory was not saved."
    assert [event.type for event in case.events] == ["content"]
    assert "".join(event.content for event in case.events) == "Memory was not saved."
    assert_no_memory_write(case)
    assert_allowlists_unchanged(case)


@pytest.mark.parametrize("handoff", [None, {"reason": "stale handoff"}])
def test_graph_route_after_handoff_stops_when_suspended(handoff):
    state = {"agent_handoff": handoff, "handoff_count": 0}
    assert route_after_handoff(state) == ("end" if handoff is None else "orchestrator")
    assert route_after_handoff({**state, "hitl_suspended": str(uuid4())}) == "end"