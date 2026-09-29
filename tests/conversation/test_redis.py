from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from redis.asyncio import Redis

from lisa.conversation.redis import RedisConversationStore
from lisa.routing import Route
from lisa.state import LISAState


@pytest.fixture
async def redis_client():
    client = Redis(
        host="localhost",
        port=6379,
        decode_responses=True,
    )

    yield client

    await client.aclose()


@pytest.mark.asyncio
async def test_save_and_get(redis_client: Redis):

    store = RedisConversationStore(
        redis=redis_client,
        key_prefix="test:lisa:conversation:",
    )

    conversation_id = uuid4()

    state: LISAState = {
        "messages": [
            HumanMessage(
                content="How do I troubleshoot a firewall?"
            ),
            AIMessage(
                content="Check the firewall logs first."
            ),
        ],
        "route": Route.TECHNICAL_CLARIFICATION,
        "active_route": Route.TECHNICAL_CLARIFICATION,
        "agent_handoff": None,
        "knowledge_context": [
            "Firewall troubleshooting procedure."
        ],
        "enable_thinking": False,
    }

    await store.save(
        conversation_id,
        state,
    )

    restored = await store.get(
        conversation_id
    )

    assert restored is not None

    assert restored["route"] == (
        Route.TECHNICAL_CLARIFICATION
    )

    assert restored["active_route"] == (
        Route.TECHNICAL_CLARIFICATION
    )

    assert restored["enable_thinking"] is False

    assert restored["knowledge_context"] == [
        "Firewall troubleshooting procedure."
    ]

    assert len(restored["messages"]) == 2

    assert isinstance(
        restored["messages"][0],
        HumanMessage,
    )

    assert restored["messages"][0].content == (
        "How do I troubleshoot a firewall?"
    )

    assert isinstance(
        restored["messages"][1],
        AIMessage,
    )

    assert restored["messages"][1].content == (
        "Check the firewall logs first."
    )


@pytest.mark.asyncio
async def test_get_missing_conversation(
    redis_client: Redis,
):

    store = RedisConversationStore(
        redis=redis_client,
        key_prefix="test:lisa:conversation:",
    )

    conversation_id = uuid4()

    result = await store.get(
        conversation_id
    )

    assert result is None