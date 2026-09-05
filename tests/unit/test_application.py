from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from conversation.test_in_memory import InMemoryConversationStore
from langchain_core.messages import AIMessage, HumanMessage

from lisa.application import LISA


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