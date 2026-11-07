import json

from jsonschema import validate
from langchain_core.messages import AIMessage, ToolMessage, messages_from_dict
from langchain_core.messages.utils import message_chunk_to_message

from lisa.agents.general_enquiry import (
    LOCAL_TOOL_ALLOWLIST,
    MAX_TOOL_ITERATIONS,
    MCP_TOOL_ALLOWLIST,
    extract_text,
)
from lisa.agents.tool_access import RESTRICTED_AGENT_TOOLS
from lisa.guardrails.hitl.dispatcher import pending_call, propose_native
from lisa.guardrails.hitl.policy import GuardError, MemoryArguments
from lisa.guardrails.hitl.jira import JIRA_TOOLS
from lisa.message_context import with_system_time
from lisa.model_resilience import ModelResilience
from lisa.streaming.events import StreamEvent


async def continue_agent(application, continuation, result, actor, conversation_id):
    if continuation.get("version") != 1 or continuation.get("agent") not in ("butler", "general_enquiry", "knowledge_assistant", "troubleshooting_agent"):
        raise GuardError("continuation_version_unsupported", 409)
    agent_name = continuation["agent"]
    agent = getattr(application, {
        "butler": "butler",
        "general_enquiry": "general_enquiry_agent",
        "knowledge_assistant": "knowledge_assistant_agent",
        "troubleshooting_agent": "troubleshooting_agent",
    }[agent_name])
    messages = messages_from_dict(continuation["messages"])
    history = messages_from_dict(continuation["history"])
    state = {"messages": history, "enable_thinking": continuation.get("enable_thinking", False)}
    index, pending = pending_call(messages, continuation["call_id"], continuation["cursor"])
    restricted = RESTRICTED_AGENT_TOOLS.get(agent_name)
    if restricted is not None and pending.tool_calls[continuation["cursor"]]["name"] not in restricted:
        raise GuardError("continuation_tool_not_authorized", 409)
    iteration = continuation["iteration"]
    minimum = 0 if agent_name == "butler" else 1
    if type(iteration) is not int or not minimum <= iteration < minimum + 5:
        raise GuardError("continuation_iteration_limit", 409)
    approved = result.get("approved_arguments")
    if approved is not None:
        if pending.tool_calls[continuation["cursor"]]["name"] == "save_memory":
            approved = MemoryArguments.model_validate(approved).model_dump(mode="json")
        elif not isinstance(approved, dict):
            raise GuardError("invalid_approved_arguments", 409)
        calls = [dict(call) for call in pending.tool_calls]
        calls[continuation["cursor"]]["args"] = approved
        pending = pending.model_copy(update={"tool_calls": calls})
        messages[index] = pending
    messages.append(ToolMessage(
        content=json.dumps({key: value for key, value in result.items() if key != "approved_arguments"}),
        tool_call_id=continuation["call_id"],
    ))
    remaining = list(enumerate(pending.tool_calls))[continuation["cursor"] + 1:]
    emitted = continuation.get("emitted", "")
    new_text = []
    delegated = continuation.get("delegated", False)

    async def persist(suspended=None):
        stored = await application.conversation_store.get(actor.user_id, conversation_id)
        stored = dict(stored) if stored is not None else dict(state)
        if suspended is None:
            stored["messages"] = [*stored.get("messages", []), AIMessage(content=emitted + "".join(new_text))]
        stored["hitl_suspended"] = suspended
        await application.conversation_store.save(actor.user_id, conversation_id, stored)

    async def stop(text):
        new_text.append(text)
        await persist()
        return StreamEvent(type="content", content=text)

    while True:
        for cursor, call in remaining:
            name, arguments = call["name"], call["args"]
            try:
                allowed = (restricted if restricted is not None else
                    {"handoff", "save_memory"} if agent_name == "butler" else
                    LOCAL_TOOL_ALLOWLIST | MCP_TOOL_ALLOWLIST)
                if name not in allowed or not isinstance(arguments, dict):
                    raise GuardError("continuation_tool_not_authorized", 409)
                if name == "save_memory" or name in JIRA_TOOLS:
                    if name in JIRA_TOOLS:
                        schema = agent.tool_schemas.get(name)
                        if not isinstance(schema, dict) or not schema:
                            raise GuardError("continuation_tool_schema_unavailable", 409)
                        validate(instance=arguments, schema=schema)
                    event = await propose_native(
                        arguments, agent=agent_name, messages=messages, state=state,
                        call_id=call["id"], cursor=cursor, iteration=iteration,
                        emitted=emitted + "".join(new_text), delegated=delegated,
                    )
                    await persist(event.execution_id)
                    yield event
                    return
                if name == "handoff":
                    yield await stop("The approval decision has been processed. Handoff after approval requires a new turn; no handoff was run.")
                    return
                schema = agent.tool_schemas.get(name)
                if not isinstance(schema, dict) or not schema:
                    raise GuardError("continuation_tool_schema_unavailable", 409)
                validate(instance=arguments, schema=schema)
                observation = await agent._execute_mcp_tool(name, arguments)
            except Exception:
                yield await stop("The approval decision has been processed, but continuation stopped safely. No further tool was retried; please start a new turn.")
                return
            messages.append(ToolMessage(content=str(observation), tool_call_id=call["id"]))
        if agent_name == "butler" and not delegated:
            yield await stop("The approval decision has been processed. Specialist delegation cannot be restored safely after approval; please start a new turn.")
            return
        if agent_name == "butler":
            iteration += 1
            exhausted = iteration >= 5
        else:
            exhausted = iteration > MAX_TOOL_ITERATIONS
        if exhausted:
            yield await stop("I couldn't complete this request within the allowed steps. Please narrow the question.")
            return
        response = None
        if state["enable_thinking"]:
            stream = (ModelResilience(agent.thinking_model, agent.fallback_model).stream(with_system_time(messages))
                      if agent_name == "butler" else agent.thinking_model.astream(with_system_time(messages)))
        else:
            stream = agent.model_resilience.stream(with_system_time(messages))
        try:
            async for chunk in stream:
                response = chunk if response is None else response + chunk
                text = extract_text(chunk.content)
                if text:
                    new_text.append(text)
                    yield StreamEvent(type="content", content=text)
        except Exception:
            yield await stop("The response was interrupted. Please start a new turn; no tools were retried.")
            return
        if response is None:
            yield await stop("I couldn't generate a response after the approval decision. Please start a new turn.")
            return
        pending = message_chunk_to_message(response)
        messages.append(pending)
        if not pending.tool_calls:
            await persist()
            return
        if agent_name != "butler":
            iteration += 1
        if len(pending.tool_calls) > 4 or (agent_name != "butler" and iteration > MAX_TOOL_ITERATIONS):
            yield await stop("I couldn't complete this request within the allowed steps. Please narrow the question.")
            return
        identifiers = [call.get("id") for call in pending.tool_calls]
        previous_ids = {call.get("id") for message in messages[:-1] if isinstance(message, AIMessage)
                        for call in message.tool_calls}
        if not all(identifiers) or len(set(identifiers)) != len(identifiers) or previous_ids.intersection(identifiers):
            yield await stop("Continuation stopped because a tool call identifier was missing or repeated. No tools were replayed.")
            return
        remaining = list(enumerate(pending.tool_calls))