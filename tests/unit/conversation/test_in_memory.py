import pytest

from lisa.conversation.in_memory import InMemoryConversationStore


@pytest.mark.anyio
async def test_save_and_get_conversation():
    store = InMemoryConversationStore()

    state = {
        "messages": [],
        "route": None,
    }

    await store.save("conversation-1", state)

    result = await store.get("conversation-1")

    assert result == state


@pytest.mark.anyio
async def test_get_unknown_conversation_returns_none():
    store = InMemoryConversationStore()

    result = await store.get("unknown")

    assert result is None