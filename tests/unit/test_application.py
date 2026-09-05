from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from conversation.test_in_memory import InMemoryConversationStore
from langchain_core.messages import AIMessage

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
