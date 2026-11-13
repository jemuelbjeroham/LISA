import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from lisa.agents.technical_specialist.communication import TaskChannel
from lisa.agents.technical_specialist.specialist import Specialist
from lisa.guardrails.hitl.context import ExecutionContext, Principal, execution_scope
from lisa.guardrails.hitl.policy import authorize


@pytest.mark.asyncio
async def test_specialist_receives_three_groups_and_urls_alongside_existing_context():
    context = ("## Chunks from Confluence documents (text content)\nText evidence\n"
        "## Image data (OCR content)\nOCR evidence\n## Macro data\nMacro evidence\n"
        "## Source URLs\nhttps://confluence.example/pages/42")
    payload = json.dumps({"context": context, "results": [{"content": f"Chunk {index}"} for index in range(15)]})
    model = MagicMock()
    model.bind_tools.return_value = model
    model.ainvoke = AsyncMock(side_effect=[AIMessage(content="", tool_calls=[{
        "name": "search_knowledge", "id": "knowledge", "args": {"query": "routing", "top_k": 5}}]),
        AIMessage(content="Evidence reviewed")])
    model.with_structured_output.return_value.ainvoke = AsyncMock(return_value={
        "status": "completed", "summary": "Reviewed grouped evidence", "next_action": "present_result"})
    client = SimpleNamespace(list_tools=AsyncMock(return_value=SimpleNamespace(tools=[SimpleNamespace(
        name="search_knowledge", description="Grouped Confluence retrieval", input_schema={"type": "object",
            "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}}, "required": ["query"]})])),
        call_tool=AsyncMock(return_value=SimpleNamespace(is_error=False, content=[SimpleNamespace(text=payload)])))
    agent = Specialist(model, client)
    await agent.initialize_tools()
    principal = Principal(uuid4(), uuid4(), "admin")
    service = SimpleNamespace(check=AsyncMock(side_effect=lambda actor, name: authorize(actor.role, name)))
    history = HumanMessage(content="Existing troubleshooting context")
    with execution_scope(ExecutionContext(principal, uuid4(), service)):
        result = await agent.run({"messages": [history]}, TaskChannel())
    assert result.status == "completed"
    assert result.evidence[0].content == payload
    messages = model.ainvoke.call_args.args[0]
    assert any(history.content in message.content for message in messages if isinstance(message.content, str))
    receipts = [message for message in messages if isinstance(message, ToolMessage)]
    assert len(receipts) == 1
    assert json.loads(receipts[0].content)["context"] == context
    assert len(json.loads(receipts[0].content)["results"]) == 15
    service.check.assert_awaited_once_with(principal, "search_knowledge")