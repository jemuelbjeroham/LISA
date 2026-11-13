import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime
from mcp.types import CallToolResult, TextContent

from lisa.agents.general_enquiry import GeneralEnquiry
from lisa.agents.technical_specialist.communication import TaskChannel
from lisa.agents.technical_specialist.specialist import Specialist
from lisa.context import LISAContext
from lisa.guardrails.hitl.context import (
    ExecutionContext,
    Principal,
    automatic_memory_enabled,
    execution_scope,
    native_runtime,
    native_write,
    require_native_write,
)
from lisa.guardrails.hitl.dispatcher import guard_mcp
from lisa.guardrails.hitl.policy import GuardError, authorize
from lisa.infrastructure.postgres.memory_dependencies import MemoryDependencies
from lisa.memory.service import MemoryService
from lisa.tools import save_memory


class FakeMemoryService:
    def __init__(self):
        self.memories = []
        self.find_duplicate = AsyncMock(return_value=None)

    async def create(self, memory):
        require_native_write(memory.user_id)
        self.memories.append(memory)
        return memory


class TrustedFakeService:
    async def check(self, principal, name):
        return authorize(principal.role, name)


@pytest.fixture
def native_case():
    user_id = uuid4()
    service = FakeMemoryService()
    runtime = native_runtime(user_id, MemoryDependencies(service, None, None, None))
    token = native_write.set(None)
    try:
        principal = Principal(user_id, uuid4(), "admin")
        with execution_scope(ExecutionContext(principal, uuid4(), TrustedFakeService())):
            yield SimpleNamespace(user_id=user_id, service=service, arguments={
                "content": " remember this ", "memory_type": "fact", "scope": "user",
                "runtime": runtime,
            })
    finally:
        native_write.reset(token)


async def test_real_save_memory_without_identity_denies_before_memory_access(native_case):
    native_case.arguments["runtime"] = Runtime(context=LISAContext(
        user_id=native_case.user_id,
        memory=MemoryDependencies(native_case.service, None, None, None),
    ))
    with execution_scope(None), pytest.raises(GuardError, match="identity_unavailable"):
        await save_memory.ainvoke(native_case.arguments)
    native_case.service.find_duplicate.assert_not_awaited()
    assert native_case.service.memories == []


@pytest.mark.parametrize("role", ["admin", "member"])
async def test_role_authorized_real_save_memory_needs_no_approval(native_case, role):
    context = ExecutionContext(Principal(native_case.user_id, uuid4(), role), uuid4(), TrustedFakeService())
    with execution_scope(context):
        result = await save_memory.ainvoke(native_case.arguments)
    assert result == "Memory saved successfully: remember this"
    native_case.service.find_duplicate.assert_awaited_once()
    assert len(native_case.service.memories) == 1
    memory = native_case.service.memories[0]
    assert memory.user_id == native_case.user_id
    assert memory.content == "remember this"
    assert memory.scope.value == "user"
    assert memory.type.value == "fact"
    assert native_write.get() is None


async def test_real_save_memory_rejects_other_users_runtime(native_case):
    native_case.arguments["runtime"] = Runtime(context=LISAContext(
        user_id=native_case.user_id,
        memory=MemoryDependencies(native_case.service, None, None, None),
    ))
    context = ExecutionContext(Principal(uuid4(), uuid4(), "admin"), uuid4(), TrustedFakeService())
    with execution_scope(context), pytest.raises(GuardError, match="not_authorized"):
        await save_memory.ainvoke(native_case.arguments)
    native_case.service.find_duplicate.assert_not_awaited()
    assert native_case.service.memories == []


@pytest.mark.parametrize("missing_user", [False, True])
def test_native_write_guard_checks_user(missing_user):
    user_id = uuid4()
    token = native_write.set(user_id)
    try:
        require_native_write(user_id)
        with pytest.raises(GuardError, match="approval_required"):
            require_native_write(None if missing_user else uuid4())
    finally:
        native_write.reset(token)


def test_automatic_memory_workflow_is_disabled():
    assert automatic_memory_enabled() is False


@pytest.mark.parametrize("role,name,error", [
    ("member", "search_knowledge", "not_authorized"),
    ("member", "create_jira_ticket", "not_authorized"),
    ("member", "update_jira_ticket", "not_authorized"),
    ("member", "close_ticket", "not_authorized"),
    ("admin", "create_jira_ticket", "approval_required"),
    ("admin", "update_jira_ticket", "approval_required"),
    ("admin", "close_ticket", "approval_required"),
])
async def test_mcp_denial_is_structured_actual_error_result(role, name, error):
    principal = Principal(uuid4(), uuid4(), role)
    invoke = AsyncMock()
    with execution_scope(ExecutionContext(principal, uuid4(), TrustedFakeService())):
        result = await guard_mcp(name, {}, invoke)
    assert isinstance(result, CallToolResult)
    assert result.is_error is True
    assert len(result.content) == 1
    assert isinstance(result.content[0], TextContent)
    assert json.loads(result.content[0].text) == {"accepted": False, "error": error}
    invoke.assert_not_awaited()


@pytest.mark.parametrize("role,name", [
    ("member", "web_search"),
    ("member", "web_search_with_content"),
    ("admin", "web_search"),
    ("admin", "web_search_with_content"),
    ("admin", "search_knowledge"),
    ("admin", "health_check"),
    ("admin", "search_jira_tickets"),
])
async def test_mcp_allowed_read_returns_actual_result_with_trusted_scope(role, name):
    principal = Principal(uuid4(), uuid4(), role)
    expected = CallToolResult(is_error=False, content=[TextContent(type="text", text="offline result")])
    invoke = AsyncMock(return_value=expected)
    service = TrustedFakeService()
    service.propose = AsyncMock()
    with execution_scope(ExecutionContext(principal, uuid4(), service)):
        result = await guard_mcp(name, {"query": "offline"}, invoke)
    assert result is expected
    assert result.is_error is False
    invoke.assert_awaited_once_with(name, {"query": "offline"})
    service.propose.assert_not_awaited()


async def test_real_save_memory_with_real_runtime_writes(native_case):
    native_case.arguments["runtime"] = Runtime(context=LISAContext(
        user_id=native_case.user_id,
        memory=MemoryDependencies(native_case.service, None, None, None),
    ))
    result = await save_memory.ainvoke(native_case.arguments)
    assert result == "Memory saved successfully: remember this"
    assert len(native_case.service.memories) == 1
    assert native_case.service.memories[0].user_id == native_case.user_id


@pytest.mark.parametrize("role", ["unknown", "", None])
async def test_save_memory_denies_unknown_roles_before_memory_access(native_case, role):
    context = ExecutionContext(Principal(native_case.user_id, uuid4(), role), uuid4(), TrustedFakeService())
    with execution_scope(context), pytest.raises(GuardError, match="not_authorized"):
        await save_memory.ainvoke(native_case.arguments)
    native_case.service.find_duplicate.assert_not_awaited()
    assert native_case.service.memories == []
    assert native_write.get() is None


async def test_save_memory_rechecks_session_before_memory_access(native_case):
    service = SimpleNamespace(check=AsyncMock(side_effect=GuardError("identity_unavailable", 401)))
    principal = Principal(native_case.user_id, uuid4(), "admin")
    with execution_scope(ExecutionContext(principal, uuid4(), service)):
        with pytest.raises(GuardError, match="identity_unavailable"):
            await save_memory.ainvoke(native_case.arguments)
    service.check.assert_awaited_once_with(principal, "save_memory")
    native_case.service.find_duplicate.assert_not_awaited()
    assert native_case.service.memories == []


async def test_save_memory_restores_sink_capability_after_failure(native_case):
    native_case.service.find_duplicate.side_effect = RuntimeError("storage unavailable")
    with pytest.raises(RuntimeError, match="storage unavailable"):
        await save_memory.ainvoke(native_case.arguments)
    assert native_write.get() is None
    assert native_case.service.memories == []


@pytest.mark.parametrize("capability", ["missing", "other_user", "matching"])
async def test_memory_service_create_checks_user_before_repository(native_case, capability):
    repository = SimpleNamespace(create=AsyncMock(return_value="stored"))
    service = MemoryService(repository)
    memory = SimpleNamespace(user_id=native_case.user_id)
    token = native_write.set({"missing": None, "other_user": uuid4(),
                              "matching": native_case.user_id}[capability])
    try:
        if capability == "matching":
            assert await service.create(memory) == "stored"
            repository.create.assert_awaited_once_with(memory)
        else:
            with pytest.raises(GuardError, match="approval_required"):
                await service.create(memory)
            repository.create.assert_not_awaited()
    finally:
        native_write.reset(token)


async def test_mcp_without_trusted_scope_denies_with_actual_error_result():
    invoke = AsyncMock()
    with execution_scope(None):
        result = await guard_mcp("web_search", {}, invoke)
    assert isinstance(result, CallToolResult)
    assert result.is_error is True
    assert json.loads(result.content[0].text) == {"accepted": False, "error": "identity_unavailable"}
    invoke.assert_not_awaited()


@pytest.fixture
def agent_case():
    model = MagicMock()
    model.bind_tools.return_value = model
    model.ainvoke = AsyncMock(side_effect=[AIMessage(content="", tool_calls=[{
        "name": "search_knowledge", "args": {"query": "offline"}, "id": "search",
    }]), AIMessage(content="done")])
    model.with_structured_output.return_value.ainvoke = AsyncMock(return_value={
        "status": "completed", "summary": "Verified procedure", "findings": ["Check link"],
        "next_action": "present_result",
    })
    schema = {"type": "object", "properties": {"query": {"type": "string"}},
              "required": ["query"]}
    client = SimpleNamespace(
        list_tools=AsyncMock(return_value=SimpleNamespace(tools=[SimpleNamespace(
            name="search_knowledge", description="Offline search", input_schema=schema,
        )])),
        call_tool=AsyncMock(return_value=CallToolResult(is_error=False, content=[
            TextContent(type="text", text="Verified procedure"),
        ])),
    )
    return SimpleNamespace(model=model, client=client, schema=schema)


@pytest.mark.parametrize("role", ["admin", "member", None])
async def test_specialist_dispatch_uses_trusted_scope_and_preserves_denial(agent_case, role):
    agent = Specialist(agent_case.model, agent_case.client)
    await agent.initialize_tools()
    principal = Principal(uuid4(), uuid4(), role or "admin")
    context = ExecutionContext(principal, uuid4(), TrustedFakeService()) if role else None
    with execution_scope(context):
        result = await agent.run({"messages": [HumanMessage(content="Check link")]}, TaskChannel())
    if role == "admin":
        agent_case.client.call_tool.assert_awaited_once_with("search_knowledge", {"query": "offline"})
        assert result.status == "completed"
        assert result.evidence[0].content == "Verified procedure"
    else:
        agent_case.client.call_tool.assert_not_awaited()
        assert result.status == "partial"
        assert result.evidence == []
        messages = agent_case.model.ainvoke.call_args.args[0]
        receipt = next(message for message in messages if message.type == "tool")
        assert json.loads(receipt.content) == {"accepted": False, "error": (
            "not_authorized" if role else "identity_unavailable")}


@pytest.mark.parametrize("trusted", [False, True])
async def test_general_enquiry_dispatch_requires_trusted_scope(agent_case, trusted):
    agent = GeneralEnquiry(
        model=agent_case.model, fallback_model=agent_case.model, thinking_model=agent_case.model,
        system_prompt="Offline test", mcp_client=agent_case.client,
        input_moderator=SimpleNamespace(moderate=AsyncMock()),
    )
    agent.tool_schemas = {"web_search": agent_case.schema}
    principal = Principal(uuid4(), uuid4(), "member")
    context = ExecutionContext(principal, uuid4(), TrustedFakeService()) if trusted else None
    with execution_scope(context):
        if trusted:
            assert await agent._execute_mcp_tool("web_search", {"query": "offline"}) == "Verified procedure"
            agent_case.client.call_tool.assert_awaited_once_with("web_search", {"query": "offline"})
        else:
            with pytest.raises(GuardError, match="identity_unavailable"):
                await agent._execute_mcp_tool("web_search", {"query": "offline"})
            agent_case.client.call_tool.assert_not_awaited()