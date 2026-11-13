from uuid import uuid4

import pytest
from pydantic import ValidationError

from lisa.guardrails.hitl.context import (
    ExecutionContext,
    Principal,
    automatic_memory_enabled,
    current_execution,
    execution_scope,
    native_write,
    require_native_write,
)
from lisa.guardrails.hitl.policy import REGISTRY, GuardError, MemoryArguments, authorize

MCP_TOOLS = {
    "web_search", "web_search_with_content", "search_knowledge", "health_check",
    "search_jira_tickets", "create_jira_ticket", "update_jira_ticket",
    "close_ticket",
}
WRITE_TOOLS = {"create_jira_ticket", "update_jira_ticket", "close_ticket"}


def test_registration_count_eight_mcp_plus_native_memory():
    assert {name for name, policy in REGISTRY.items() if policy.namespace == "mcp"} == MCP_TOOLS
    assert len(MCP_TOOLS) == 8
    assert set(REGISTRY) == MCP_TOOLS | {"save_memory"}
    assert {name for name, policy in REGISTRY.items() if policy.write} == WRITE_TOOLS
    assert REGISTRY["save_memory"].namespace == "native"


@pytest.mark.parametrize("name", sorted(MCP_TOOLS | {"save_memory"}))
@pytest.mark.parametrize("role", ["admin", "member"])
def test_role_matrix(role, name):
    allowed = role == "admin" or name in {"web_search", "web_search_with_content", "save_memory"}
    if allowed:
        assert authorize(role, name) is REGISTRY[name]
    else:
        with pytest.raises(GuardError) as caught:
            authorize(role, name)
        assert caught.value.status == 403
        assert caught.value.result() == {"accepted": False, "error": "not_authorized"}


@pytest.mark.parametrize("role", ["admin", "member", "Admin", "", None, "owner", "administrator"])
@pytest.mark.parametrize("name", ["missing_tool", "SAVE_MEMORY", "mcp.save_memory", "", None])
def test_unknown_tools_and_roles_fail_closed(role, name):
    with pytest.raises(GuardError):
        authorize(role, name)


@pytest.mark.parametrize("role", ["Admin", "", None, "owner", "administrator"])
@pytest.mark.parametrize("name", sorted(MCP_TOOLS | {"save_memory"}))
def test_unknown_role_cannot_use_known_tool(role, name):
    with pytest.raises(GuardError):
        authorize(role, name)


def test_native_sink_requires_matching_user_and_resets_on_error():
    user_id, other_id = uuid4(), uuid4()
    assert native_write.get() is None
    with pytest.raises(GuardError, match="approval_required"):
        require_native_write(user_id)
    token = native_write.set(user_id)
    try:
        require_native_write(user_id)
        with pytest.raises(GuardError, match="approval_required"):
            require_native_write(other_id)
    finally:
        native_write.reset(token)
    with pytest.raises(GuardError, match="approval_required"):
        require_native_write(user_id)
    assert not automatic_memory_enabled()


def test_execution_scope_restores_nested_context_after_exception():
    principal = Principal(uuid4(), uuid4(), "admin")
    outer = ExecutionContext(principal, uuid4(), object())
    inner = ExecutionContext(principal, uuid4(), object())
    assert current_execution.get() is None
    with execution_scope(outer):
        assert current_execution.get() is outer
        with pytest.raises(RuntimeError), execution_scope(inner):
            assert current_execution.get() is inner
            raise RuntimeError("interrupted")
        assert current_execution.get() is outer
    assert current_execution.get() is None


@pytest.mark.parametrize("changes", [
    {"content": " "}, {"content": "x" * 32001}, {"memory_type": "unknown"},
    {"scope": "global"}, {"user_id": str(uuid4())}, {"approved": True},
])
def test_memory_arguments_reject_invalid_or_privileged_fields(changes):
    with pytest.raises(ValidationError):
        MemoryArguments.model_validate({
            "content": "remember", "memory_type": "fact", "scope": "user", **changes,
        })


def test_memory_arguments_normalize_before_hashing():
    arguments = MemoryArguments.model_validate({
        "content": " remember ", "memory_type": "fact", "scope": "user",
    })
    assert arguments.model_dump(mode="json") == {
        "content": "remember", "memory_type": "fact", "scope": "user",
    }


def test_native_sink_rejects_missing_user_identity():
    assert native_write.get() is None
    with pytest.raises(GuardError, match="approval_required"):
        require_native_write(None)