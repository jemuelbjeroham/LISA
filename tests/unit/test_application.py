from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from conversation.test_in_memory import InMemoryConversationStore
from langchain_core.messages import AIMessage, HumanMessage

from lisa.application import LISA
from lisa.routing import Route


@pytest.mark.anyio
async def test_chat_persists_conversation_state():
    conversation_id = uuid4()

    store = InMemoryConversationStore()

    graph = AsyncMock()
    graph.ainvoke.return_value = {
        "messages": [
            AIMessage(content="Hello!")
        ]
    }

    lisa = LISA(
        conversation_store=store,
    )

    lisa.graph = graph

    await lisa.chat(
        conversation_id=conversation_id,
        message="Hello",
    )

    saved_state = await store.get(conversation_id)

    assert saved_state is not None
    assert saved_state["messages"][0].content == "Hello!"


@pytest.mark.anyio
async def test_chat_retrieves_existing_conversation():
    conversation_id = uuid4()

    store = InMemoryConversationStore()

    existing_state = {
        "messages": [
            HumanMessage(content="Previous message"),
        ],
        "route": None,
    }

    await store.save(conversation_id, existing_state)

    graph = AsyncMock()
    graph.ainvoke.return_value = {
        "messages": [
            HumanMessage(content="Previous message"),
            HumanMessage(content="New message"),
        ],
        "route": None,
    }

    lisa = LISA(
        conversation_store=store,
    )

    lisa.graph = graph

    await lisa.chat(
        conversation_id=conversation_id,
        message="New message",
    )

    graph.ainvoke.assert_awaited_once()

    state_sent_to_graph = graph.ainvoke.call_args.args[0]

    assert state_sent_to_graph["messages"][0].content == "Previous message"
    assert state_sent_to_graph["messages"][1].content == "New message"

@pytest.mark.anyio
async def test_chat_maintains_state_across_two_turns():
    conversation_id = uuid4()

    store = InMemoryConversationStore()

    graph = AsyncMock()
    graph.ainvoke.side_effect = [
        {
            "messages": [
                HumanMessage(content="Hello"),
                AIMessage(content="Hi there!"),
            ],
            "route": None,
        },
        {
            "messages": [
                HumanMessage(content="Hello"),
                AIMessage(content="Hi there!"),
                HumanMessage(content="How are you?"),
                AIMessage(content="I'm doing well!"),
            ],
            "route": None,
        },
    ]

    lisa = LISA(
        conversation_store=store,
    )

    lisa.graph = graph

    await lisa.chat(
        conversation_id=conversation_id,
        message="Hello",
    )

    await lisa.chat(
        conversation_id=conversation_id,
        message="How are you?",
    )

    assert graph.ainvoke.await_count == 2

    second_call_state = graph.ainvoke.call_args_list[1].args[0]

    assert second_call_state["messages"][0].content == "Hello"
    assert second_call_state["messages"][1].content == "Hi there!"
    assert second_call_state["messages"][2].content == "How are you?"

@pytest.mark.anyio
async def test_graph_stream_output():
    async with LISA() as lisa:
        async for chunk in lisa.graph.astream(
            {
                "messages": [HumanMessage(content="How do I troubleshoot BGP?")],
                "route": None,
            }
        ):
            print("\nSTREAM CHUNK:")
            print(chunk)

@pytest.mark.anyio
async def test_model_streams_tokens():
    async with LISA() as lisa:
        async for chunk in lisa.model.astream(
            [HumanMessage(content="Explain what BGP is in one sentence.")]
        ):
            print("\nMODEL CHUNK:")
            print(repr(chunk))

@pytest.mark.anyio
async def test_stream_chat_yields_chunks_and_persists_response():
    conversation_id = uuid4()
    store = InMemoryConversationStore()

    model = AsyncMock()

    lisa = LISA(
        model=model,
        conversation_store=store,
    )

    orchestrator = MagicMock()
    orchestrator.route.return_value = {
        "route": Route.TECHNICAL_CLARIFICATION,
    }

    technical_agent = MagicMock()

    async def stream(state):
        yield AIMessage(content="Hello")
        yield AIMessage(content=" there")
        yield AIMessage(content="!")

    technical_agent.stream = stream

    lisa.orchestrator = orchestrator
    lisa.technical_clarification_agent = technical_agent

    chunks = []

    async for chunk in lisa.stream_chat(
        conversation_id=conversation_id,
        message="Hi",
    ):
        chunks.append(chunk)

    assert chunks == [
        "Hello",
        " there",
        "!",
    ]

    saved_state = await store.get(conversation_id)

    assert saved_state is not None
    assert saved_state["messages"][0].content == "Hi"
    assert saved_state["messages"][1].content == "Hello there!"