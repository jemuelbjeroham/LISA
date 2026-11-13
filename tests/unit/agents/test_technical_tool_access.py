from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from lisa.agents.general_enquiry import MCP_TOOL_ALLOWLIST, GeneralEnquiry
from lisa.agents.technical_clarification import TechnicalClarificationAgent
from lisa.guardrails.hitl.context import ExecutionContext, Principal, execution_scope
from lisa.guardrails.hitl.dispatcher import ApprovalSuspended
from lisa.guardrails.hitl.policy import GuardError, authorize


def make_agent(agent_type):
    model = MagicMock()
    model.bind_tools.return_value = model
    tools = [SimpleNamespace(name=name, description=name, input_schema={"type": "object"})
             for name in MCP_TOOL_ALLOWLIST]
    client = SimpleNamespace(
        list_tools=AsyncMock(return_value=SimpleNamespace(tools=tools)),
        call_tool=AsyncMock(return_value=SimpleNamespace(
            is_error=False, content=[SimpleNamespace(text="complete grouped evidence and URLs")]
        )),
    )
    options = dict(model=model, system_prompt="Engineering", mcp_client=client,
                   input_moderator=AsyncMock())
    if agent_type is TechnicalClarificationAgent:
        options.update(hyde_prompt="HyDE", retriever=AsyncMock(), retrieval_planner=AsyncMock())
    else:
        options.update(fallback_model=model, thinking_model=model)
    return agent_type(**options), client


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_type", [GeneralEnquiry, TechnicalClarificationAgent])
async def test_both_agents_bind_all_registered_mcp_tools(agent_type):
    agent, client = make_agent(agent_type)
    await agent.initialize_tools()
    assert set(agent.tool_schemas) == MCP_TOOL_ALLOWLIST
    assert set(agent.available_tools) == {"handoff", "save_memory"}
    assert agent._tools_initialized


@pytest.mark.asyncio
async def test_technical_agent_executes_search_and_passes_full_result_to_model():
    agent, client = make_agent(TechnicalClarificationAgent)
    await agent.initialize_tools()
    agent._generate_response_with_retries = AsyncMock(side_effect=[
        AIMessage(content="", tool_calls=[{"name": "search_knowledge", "id": "kb", "args": {}}]),
        AIMessage(content="Grounded answer"),
    ])
    principal = Principal(uuid4(), uuid4(), "admin")
    service = SimpleNamespace(check=AsyncMock())
    messages = []
    with execution_scope(ExecutionContext(principal, uuid4(), service)):
        answer, handoff = await agent._generate_response_with_tool_corrections(messages, writer=MagicMock())
    assert answer.content == "Grounded answer"
    assert handoff is None
    client.call_tool.assert_awaited_once_with("search_knowledge", {})
    receipt = next(message for message in messages if isinstance(message, ToolMessage))
    assert receipt.content == "complete grouped evidence and URLs"


@pytest.mark.parametrize("role", ["admin", "member"])
async def test_technical_jira_call_suspends_before_any_mcp_write(role):
    import json

    agent, client = make_agent(TechnicalClarificationAgent)
    await agent.initialize_tools()
    principal = Principal(uuid4(), uuid4(), role)
    arguments = {"ticket_id": "123"}
    event_id = str(uuid4())

    async def propose(actor, conversation, name, values, logical, continuation):
        authorize(actor.role, name)
        assert name == "close_ticket" and values == arguments
        assert continuation["agent"] == "technical_clarification"
        assert continuation["cursor"] == 0
        return {"approval_id": event_id, "execution_id": event_id, "revision": 1, "status": "pending"}

    service = SimpleNamespace(propose=AsyncMock(side_effect=propose))
    agent._generate_response_with_retries = AsyncMock(side_effect=[
        AIMessage(content="", tool_calls=[{"name": "close_ticket", "id": "close", "args": arguments}]),
        AIMessage(content="Denied"),
    ])
    messages = []
    writer = MagicMock()
    with execution_scope(ExecutionContext(principal, uuid4(), service)):
        if role == "admin":
            with pytest.raises(ApprovalSuspended) as suspended:
                await agent._generate_response_with_tool_corrections(messages, writer)
            assert suspended.value.event.execution_id == event_id
            assert writer.call_args.args[0].type == "approval_required"
        else:
            await agent._generate_response_with_tool_corrections(messages, writer)
            receipt = next(message for message in messages if isinstance(message, ToolMessage))
            assert json.loads(receipt.content) == {"accepted": False, "error": "not_authorized"}
    client.call_tool.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_type", [GeneralEnquiry, TechnicalClarificationAgent])
async def test_tool_visibility_does_not_bypass_write_guard(agent_type):
    agent, client = make_agent(agent_type)
    await agent.initialize_tools()
    principal = Principal(uuid4(), uuid4(), "admin")
    service = SimpleNamespace(check=AsyncMock(side_effect=GuardError("verified_preparation_unavailable")))
    with execution_scope(ExecutionContext(principal, uuid4(), service)):
        with pytest.raises(GuardError):
            await agent._execute_mcp_tool("create_jira_ticket", {})
    client.call_tool.assert_not_awaited()


@pytest.mark.asyncio
async def test_technical_retriever_keeps_grouped_evidence_and_urls():
    import json

    from lisa.knowledge.mcp_retriever import MCPKnowledgeRetriever

    context = "## Text\nText\n## Images\nOCR\n## Macros\nMacro\n## Source URLs\nhttps://example.com/42"
    client = SimpleNamespace(call_tool=AsyncMock(return_value=SimpleNamespace(
        is_error=False, content=[SimpleNamespace(text=json.dumps({
            "context": context, "results": [{"content": "Text"}]
        }))]
    )))
    assert await MCPKnowledgeRetriever(client).retrieve("Skypath") == [context]