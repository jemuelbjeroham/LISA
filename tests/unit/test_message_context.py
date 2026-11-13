from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from lisa.message_context import RECEIVED_AT, user_message, with_system_time


def test_user_time_context_preserves_originals_and_is_idempotent():
    user = user_message("Original question", received_at=datetime(2026, 10, 26, 12, 30, tzinfo=UTC))
    user.id = "original-id"
    history = [SystemMessage(content="system"), user, AIMessage(content="answer"),
               ToolMessage(content="result", tool_call_id="call")]
    prepared = with_system_time(history)
    assert prepared[1].content == "at system time: 2026-10-26T12:30:00Z\nOriginal question"
    assert prepared[1].id == "original-id"
    assert user.content == "Original question"
    assert user.additional_kwargs == {RECEIVED_AT: "2026-10-26T12:30:00Z"}
    assert all(prepared[index] is history[index] for index in [0, 2, 3])
    assert with_system_time(prepared)[1].content == prepared[1].content


def test_block_content_and_legacy_timing_remain_explicit():
    blocks = [{"type": "text", "text": "Inspect image"},
              {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}]
    message = HumanMessage(content=blocks)
    prepared = with_system_time([message])[0]
    assert "message receipt time unavailable" in prepared.content[0]["text"]
    assert prepared.content[1:] == blocks
    assert message.content == blocks


def test_each_turn_keeps_its_own_receipt_time():
    messages = [user_message("first", received_at=datetime(2026, 10, 25, tzinfo=UTC)),
                user_message("second", received_at=datetime(2026, 10, 26, tzinfo=UTC))]
    prepared = with_system_time(messages)
    assert "2026-10-25T00:00:00Z" in prepared[0].content
    assert "2026-10-26T00:00:00Z" in prepared[1].content


def test_receipt_metadata_survives_conversation_serialization():
    from langchain_core.messages import messages_from_dict, messages_to_dict

    original = user_message("unchanged question", received_at=datetime(2026, 10, 26, tzinfo=UTC))
    restored = messages_from_dict(messages_to_dict([original]))[0]
    assert restored.content == "unchanged question"
    assert restored.additional_kwargs[RECEIVED_AT] == "2026-10-26T00:00:00Z"
    assert with_system_time([restored])[0].content == "at system time: 2026-10-26T00:00:00Z\nunchanged question"


@pytest.mark.anyio
async def test_memory_and_retrieval_planner_get_time_without_mutating_history():
    from lisa.agents.mem_agent import MemoryAgent
    from lisa.retrieval.planning import RetrievalPlanner

    history = [user_message("Remember concise answers")]
    for agent_class, method in [(MemoryAgent, "analyze"), (RetrievalPlanner, "plan")]:
        model = MagicMock()
        model.with_structured_output.return_value = model
        model.ainvoke = AsyncMock(return_value=SimpleNamespace(retrieval_query="query", use_hyde=False, reasoning="reason"))
        agent = agent_class(model=model, prompt="unchanged prompt")
        await getattr(agent, method)(history)
        sent = model.ainvoke.call_args.args[0]
        assert sent[0].content == "unchanged prompt"
        assert sent[1].content.startswith("at system time: ")
        assert sent[1].content.endswith("\nRemember concise answers")
    assert history[0].content == "Remember concise answers"


@pytest.mark.anyio
async def test_general_enquiry_preserves_memory_and_raw_moderation_input(monkeypatch):
    from langchain_core.messages import AIMessageChunk

    from lisa.agents.general_enquiry import GeneralEnquiry

    sent = []
    model = MagicMock()
    model.bind_tools.return_value = model

    async def stream(messages):
        sent.extend(messages)
        yield AIMessageChunk(content="answer")

    model.astream = stream
    moderator = SimpleNamespace(moderate=AsyncMock(return_value=SimpleNamespace(flagged=False)))
    agent = GeneralEnquiry(model, model, model, "original system", MagicMock(), moderator)
    agent._tools_initialized = True
    memory = SimpleNamespace(user_context_service=SimpleNamespace(build_context=AsyncMock(return_value="Preference: brief")),
        context_service=SimpleNamespace(build=AsyncMock(return_value="Relevant memory")))
    runtime = SimpleNamespace(context=SimpleNamespace(user_id="user", memory=memory))
    monkeypatch.setattr("lisa.agents.general_enquiry.get_stream_writer", lambda: lambda event: None)
    history = [user_message("question")]
    await agent.run({"messages": history}, runtime)
    assert "Preference: brief" in sent[0].content
    assert "Relevant memory" in sent[0].content
    assert sent[-1].content.startswith("at system time: ")
    assert sent[-1].content.endswith("\nquestion")
    assert history[0].content == "question"
    moderator.moderate.assert_awaited_once_with("question")


@pytest.mark.anyio
async def test_retained_technical_agent_scope_hyde_and_stream_use_time():
    from langchain_core.messages import AIMessageChunk

    from lisa.agents.technical_clarification import TechnicalClarificationAgent

    model = MagicMock()
    model.bind_tools.return_value = model
    model.ainvoke = AsyncMock(return_value=AIMessage(content="hypothesis"))
    agent = TechnicalClarificationAgent(model, "technical system", "hyde system", MagicMock(), MagicMock(), MagicMock())
    history = [user_message("question")]
    assert await agent._check_scope("question", history) is None
    assert model.ainvoke.call_args.args[0][-1].content.startswith("at system time: ")
    assert await agent._generate_hypothetical_document("search query") == "hypothesis"
    assert model.ainvoke.call_args.args[0][-1].content.endswith("\nsearch query")
    sent = []

    async def stream(messages):
        sent.extend(messages)
        yield AIMessageChunk(content="answer")

    model.astream = stream
    await agent._generate_response_with_retries(history, writer=lambda event: None)
    assert sent[-1].content.startswith("at system time: ")
    assert history[0].content == "question"