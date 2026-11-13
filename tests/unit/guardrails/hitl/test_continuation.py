import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    ToolMessage,
    messages_from_dict,
    messages_to_dict,
)

from lisa.guardrails.hitl.context import ExecutionContext, Principal, execution_scope
from lisa.guardrails.hitl.continuation import continue_agent
from lisa.guardrails.hitl.dispatcher import propose_native
from lisa.guardrails.hitl.policy import GuardError

ARGUMENTS = {"content": "original", "memory_type": "fact", "scope": "user"}


def call(name, identifier, arguments=None):
    return {"name": name, "id": identifier, "args": dict(ARGUMENTS if arguments is None else arguments)}


@pytest.fixture
def harness():
    history = [HumanMessage(content="Remember this", id="turn-1")]
    inputs = []
    responses = [[AIMessageChunk(content="New answer")]]

    async def stream(messages):
        inputs.append(list(messages))
        for chunk in responses.pop(0):
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    agent = SimpleNamespace(
        model_resilience=SimpleNamespace(stream=stream),
        thinking_model=SimpleNamespace(astream=stream), fallback_model=Mock(),
        tool_schemas={"web_search": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
        _execute_mcp_tool=AsyncMock(return_value="new search receipt"), _delegate=AsyncMock(),
    )
    store = SimpleNamespace(get=AsyncMock(return_value={"messages": history, "route": "keep", "hitl_suspended": "old"}), save=AsyncMock())
    application = SimpleNamespace(butler=agent, general_enquiry_agent=agent, knowledge_assistant_agent=agent,
                                  troubleshooting_agent=agent, conversation_store=store)
    actor = Principal(uuid4(), uuid4(), "admin")
    service = SimpleNamespace(propose=AsyncMock(return_value={"approval_id": str(uuid4()), "execution_id": str(uuid4()), "revision": 1, "status": "pending"}))
    return SimpleNamespace(agent=agent, application=application, actor=actor, conversation_id=uuid4(),
                           history=history, inputs=inputs, responses=responses, service=service)


def snapshot(harness, calls=None, cursor=0, receipts=(), **changes):
    messages = [*harness.history, AIMessage(content="Already visible", tool_calls=calls or [call("save_memory", "memory-1")]), *receipts]
    return {"version": 1, "agent": "general_enquiry", "messages": messages_to_dict(messages),
            "history": messages_to_dict(harness.history), "call_id": "memory-1", "cursor": cursor,
            "iteration": 1, "emitted": "Already visible", "delegated": False, **changes}


async def run(harness, continuation, result=None):
    with execution_scope(ExecutionContext(harness.actor, harness.conversation_id, harness.service)):
        return [event async for event in continue_agent(harness.application, continuation,
                result or {"accepted": True, "result": "saved"}, harness.actor, harness.conversation_id)]


async def test_restores_toolmessage_tail_and_revised_arguments_without_replay(harness):
    previous = ToolMessage(content="completed search", tool_call_id="search-1")
    continuation = snapshot(harness, [call("web_search", "search-1", {"query": "old"}),
                                    call("save_memory", "memory-1")], cursor=1, receipts=[previous])
    revised = {**ARGUMENTS, "content": "revised"}
    events = await run(harness, continuation, {"accepted": True, "result": "saved revised", "approved_arguments": revised})
    assert [event.content for event in events] == ["New answer"]
    harness.agent._execute_mcp_tool.assert_not_awaited()
    restored = harness.inputs[0]
    pending = next(message for message in restored if isinstance(message, AIMessage))
    assert pending.tool_calls[1]["args"] == revised
    receipts = [message for message in restored if isinstance(message, ToolMessage)]
    assert [(message.tool_call_id, message.content) for message in receipts] == [
        ("search-1", "completed search"), ("memory-1", json.dumps({"accepted": True, "result": "saved revised"}))]
    saved = harness.application.conversation_store.save.await_args.args[2]
    assert saved["messages"][-1].content == "Already visibleNew answer"
    assert saved["route"] == "keep" and saved["hitl_suspended"] is None


@pytest.mark.parametrize("agent_name", ["knowledge_assistant", "troubleshooting_agent"])
async def test_new_agent_old_jira_continuation_is_rejected(harness, agent_name):
    arguments = {"ticket_key": "OPS-123", "comment": "Original"}
    continuation = snapshot(harness, [call("update_jira_ticket", "jira", arguments)],
        agent=agent_name, call_id="jira")
    with pytest.raises(GuardError, match="continuation_tool_not_authorized"):
        await run(harness, continuation)
    harness.agent._execute_mcp_tool.assert_not_awaited()
    assert harness.inputs == []


async def test_retired_agent_continuation_is_rejected(harness):
    with pytest.raises(GuardError, match="continuation_version_unsupported"):
        await run(harness, snapshot(harness, agent="technical_clarification"))
    assert harness.inputs == []


@pytest.mark.parametrize("agent_name", ["knowledge_assistant", "troubleshooting_agent"])
async def test_new_agent_old_memory_continuation_is_rejected(harness, agent_name):
    with pytest.raises(GuardError, match="continuation_tool_not_authorized"):
        await run(harness, snapshot(harness, agent=agent_name))
    assert harness.inputs == []


@pytest.mark.parametrize("schema", [None, {}, {"type": "invalid"}])
async def test_unknown_schemas_fail_closed(harness, schema):
    harness.agent.tool_schemas["web_search"] = schema
    events = await run(harness, snapshot(harness, [call("save_memory", "memory-1"), call("web_search", "search-2", {"query": "new"})]))
    assert "stopped safely" in events[-1].content
    harness.agent._execute_mcp_tool.assert_not_awaited()
    assert harness.inputs == []


async def test_subsequent_pending_serializes_same_batch_cursor_and_receipts(harness):
    continuation = snapshot(harness, [call("save_memory", "memory-1"), call("web_search", "search-2", {"query": "new"}), call("save_memory", "memory-2")])
    events = await run(harness, continuation)
    assert [event.type for event in events] == ["approval_required"]
    proposed = harness.service.propose.await_args.args
    restored = proposed[-1]
    json.dumps(restored, allow_nan=False)
    assert restored["cursor"] == 2 and restored["call_id"] == "memory-2" and restored["iteration"] == 1
    assert [message.tool_call_id for message in messages_from_dict(restored["messages"]) if isinstance(message, ToolMessage)] == ["memory-1", "search-2"]
    assert proposed[-2].endswith("/general_enquiry/1/memory-2")
    assert harness.inputs == []
    harness.agent._execute_mcp_tool.assert_awaited_once_with("web_search", {"query": "new"})


async def test_dispatcher_accepts_pending_with_toolmessage_tail(harness):
    messages = [*harness.history, AIMessage(content="", tool_calls=[call("web_search", "search-1", {"query": "old"}), call("save_memory", "memory-1")]), ToolMessage(content="receipt", tool_call_id="search-1")]
    with execution_scope(ExecutionContext(harness.actor, harness.conversation_id, harness.service)):
        await propose_native(ARGUMENTS, agent="general_enquiry", messages=messages, state={"messages": harness.history}, call_id="memory-1", cursor=1, iteration=1)
    assert harness.service.propose.await_count == 1


@pytest.mark.parametrize("agent,iteration,expected_calls", [("butler", 4, 0), ("general_enquiry", 5, 1)])
async def test_restores_agent_iteration_budget(harness, agent, iteration, expected_calls):
    await run(harness, snapshot(harness, agent=agent, iteration=iteration, delegated=True))
    assert len(harness.inputs) == expected_calls


async def test_butler_without_delegation_stops_explicitly(harness):
    events = await run(harness, snapshot(harness, agent="butler", iteration=0))
    assert "delegation cannot be restored safely" in events[-1].content
    harness.agent._delegate.assert_not_awaited()
    assert harness.inputs == []


async def test_thinking_mode_restored(harness):
    harness.agent.model_resilience.stream = Mock(side_effect=AssertionError("wrong model"))
    events = await run(harness, snapshot(harness, enable_thinking=True))
    assert [event.content for event in events] == ["New answer"]


async def test_partial_stream_is_not_replayed(harness):
    harness.responses[:] = [[AIMessageChunk(content="Partial"), RuntimeError("offline")]]
    events = await run(harness, snapshot(harness))
    assert events[0].content == "Partial" and "interrupted" in events[1].content
    assert len(harness.inputs) == 1
    assert harness.application.conversation_store.save.await_args.args[2]["messages"][-1].content == "Already visible" + "".join(event.content for event in events)


async def test_inconsistent_cursor_fails_before_execution(harness):
    with pytest.raises(GuardError):
        await run(harness, snapshot(harness, cursor=1))
    assert harness.inputs == []
    harness.agent._execute_mcp_tool.assert_not_awaited()


@pytest.mark.parametrize("name", ["create_jira_ticket", "secure_line", "search_knowledge"])
async def test_unauthorized_tools_never_dispatch(harness, name):
    events = await run(harness, snapshot(harness, [call("save_memory", "memory-1"), call(name, "unknown")]))
    assert "stopped safely" in events[-1].content
    harness.agent._execute_mcp_tool.assert_not_awaited()
    assert harness.inputs == []


async def test_service_blocked_followup_proposal_stops_without_model_or_replay(harness):
    harness.service.propose.side_effect = GuardError("execution_pending_cancel_required", 409)
    events = await run(harness, snapshot(harness, [call("save_memory", "memory-1"), call("save_memory", "memory-2")]))
    assert [event.type for event in events] == ["content"]
    assert "stopped safely" in events[0].content
    assert harness.inputs == []
    harness.agent._execute_mcp_tool.assert_not_awaited()
    assert harness.service.propose.await_count == 1


async def test_pending_after_new_model_batch_restores_cursor_and_budget(harness):
    harness.responses[:] = [[AIMessageChunk(content="Next", tool_calls=[call("save_memory", "memory-2")])]]
    events = await run(harness, snapshot(harness))
    assert [event.type for event in events] == ["content", "approval_required"]
    continuation = harness.service.propose.await_args.args[-1]
    assert continuation["call_id"] == "memory-2" and continuation["cursor"] == 0
    assert continuation["iteration"] == 2 and continuation["emitted"] == "Already visibleNext"
    harness.responses[:] = [[AIMessageChunk(content="Done")]]
    events = await run(harness, continuation)
    assert [event.content for event in events] == ["Done"]
    receipts = [message for message in harness.inputs[-1] if isinstance(message, ToolMessage)]
    assert [message.tool_call_id for message in receipts] == ["memory-1", "memory-2"]
    assert harness.application.conversation_store.save.await_args.args[2]["messages"][-1].content == "Already visibleNextDone"


async def test_repeated_call_id_cannot_reexecute_completed_tool(harness):
    harness.responses[:] = [[AIMessageChunk(content="", tool_calls=[call("web_search", "search-1", {"query": "repeat"})])]]
    continuation = snapshot(harness, [call("web_search", "search-1", {"query": "old"}), call("save_memory", "memory-1")],
                            cursor=1, receipts=[ToolMessage(content="completed", tool_call_id="search-1")])
    events = await run(harness, continuation)
    assert "No tools were replayed" in events[-1].content
    harness.agent._execute_mcp_tool.assert_not_awaited()


async def test_butler_thinking_uses_existing_resilience_interface(harness, monkeypatch):
    resilience = Mock(return_value=harness.agent.model_resilience)
    monkeypatch.setattr("lisa.guardrails.hitl.continuation.ModelResilience", resilience)
    events = await run(harness, snapshot(harness, agent="butler", iteration=1, delegated=True, enable_thinking=True))
    resilience.assert_called_once_with(harness.agent.thinking_model, harness.agent.fallback_model)
    assert [event.content for event in events] == ["New answer"]
    harness.agent._delegate.assert_not_awaited()


async def test_general_enquiry_last_batch_does_not_execute_new_tools(harness):
    harness.responses[:] = [[AIMessageChunk(content="", tool_calls=[call("web_search", "search-new", {"query": "new"})])]]
    events = await run(harness, snapshot(harness, iteration=5))
    assert "allowed steps" in events[-1].content
    harness.agent._execute_mcp_tool.assert_not_awaited()


async def test_handoff_after_approval_is_explicit_and_not_dispatched(harness):
    events = await run(harness, snapshot(harness, [call("save_memory", "memory-1"), call("handoff", "handoff-1", {"reason": "help"})]))
    assert "no handoff was run" in events[-1].content
    harness.agent._delegate.assert_not_awaited()
    assert harness.inputs == []


@pytest.mark.parametrize("result", [{"accepted": False, "error": "approval_rejected"}, {"accepted": False, "error": "approval_timeout"}])
async def test_negative_outcome_appended_as_receipt_without_saving_memory(harness, result):
    await run(harness, snapshot(harness), result)
    receipt = next(message for message in harness.inputs[0] if isinstance(message, ToolMessage))
    assert receipt.tool_call_id == "memory-1" and json.loads(receipt.content) == result
    harness.service.propose.assert_not_awaited()