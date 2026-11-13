import json

from langchain_core.messages import AIMessage, ToolMessage, messages_to_dict
from mcp.types import CallToolResult, TextContent

from lisa.guardrails.hitl.context import current_execution
from lisa.guardrails.hitl.jira import JIRA_TOOLS
from lisa.guardrails.hitl.policy import GuardError, MemoryArguments
from lisa.streaming.events import StreamEvent


async def check_tool(name):
    context = current_execution.get()
    if context is None:
        raise GuardError("identity_unavailable", 401)
    return await context.service.check(context.principal, name)


async def guard_mcp(name, arguments, invoke):
    try:
        policy = await check_tool(name)
        if policy.write:
            raise GuardError("approval_required", 403)
        return await invoke(name, arguments)
    except GuardError as exc:
        return CallToolResult(is_error=True, content=[TextContent(type="text", text=json.dumps(exc.result()))])


def pending_call(messages, call_id, cursor):
    matches = [(index, message) for index, message in enumerate(messages)
               if isinstance(message, AIMessage)
               and any(call.get("id") == call_id for call in message.tool_calls)]
    if not call_id or len(matches) != 1 or type(cursor) is not int:
        raise GuardError("invalid_continuation_cursor", 409)
    index, message = matches[0]
    calls = message.tool_calls
    identifiers = [call.get("id") for call in calls]
    if (not 0 <= cursor < len(calls) or calls[cursor].get("id") != call_id
            or calls[cursor].get("name") not in JIRA_TOOLS | {"save_memory"} or len(calls) > 4
            or not all(identifiers) or len(set(identifiers)) != len(identifiers)):
        raise GuardError("invalid_continuation_cursor", 409)
    receipts = messages[index + 1:]
    if (any(not isinstance(receipt, ToolMessage) for receipt in receipts)
            or [receipt.tool_call_id for receipt in receipts] != identifiers[:cursor]):
        raise GuardError("invalid_continuation_receipts", 409)
    return index, message


async def propose_native(arguments, *, agent, messages, state, call_id, cursor, iteration, emitted="", delegated=False):
    _, pending = pending_call(messages, call_id, cursor)
    name = pending.tool_calls[cursor]["name"]
    if name == "save_memory":
        arguments = MemoryArguments.model_validate(arguments).model_dump(mode="json")
    context = current_execution.get()
    if context is None:
        raise GuardError("identity_unavailable", 401)
    continuation = {"version": 1, "agent": agent, "messages": messages_to_dict(messages),
        "history": messages_to_dict(state.get("messages", [])), "call_id": call_id, "cursor": cursor,
        "iteration": iteration, "emitted": emitted, "delegated": delegated,
        "enable_thinking": bool(state.get("enable_thinking", False)), "resumed": False}
    last_user = next((message for message in reversed(state.get("messages", [])) if message.type == "human"), None)
    logical_id = f"{getattr(last_user, 'id', None) or context.conversation_id}/{agent}/{iteration}/{call_id}"
    proposal = await context.service.propose(context.principal, context.conversation_id, name, arguments, logical_id, continuation)
    return StreamEvent(type="approval_required", event_id=proposal["approval_id"], execution_id=proposal["execution_id"],
        approval_id=proposal["approval_id"], revision=proposal["revision"], status=proposal["status"])


class ApprovalSuspended(Exception):
    def __init__(self, event):
        self.event = event
        super().__init__("approval_required")