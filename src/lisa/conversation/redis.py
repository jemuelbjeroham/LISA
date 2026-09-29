import json
import logging
from uuid import UUID

from redis.asyncio import Redis

from lisa.conversation.serialization import ConversationStateSerializer
from lisa.conversation.store import ConversationStore
from lisa.state import LISAState

logger = logging.getLogger(__name__)

class RedisConversationStore(ConversationStore):
    def __init__(
        self,
        redis: Redis,
        key_prefix: str = "lisa:conversation:",
    ) -> None:
        self.redis = redis
        self.key_prefix = key_prefix

    def _key(self, conversation_id: UUID) -> str:
        return f"{self.key_prefix}{conversation_id}"

    async def get(
        self,
        conversation_id: UUID,
    ) -> LISAState | None:
        key = self._key(conversation_id)

        logger.info(
            "Reading conversation state from Redis: key=%s",
            key,
        )
        data = await self.redis.get(key)

        if data is None:
            logger.info(
                "Conversation state not found in Redis: key=%s",
                key,
            )
            return None

        logger.info(
            "Conversation state retrived from Redis: key=%s",
            key,
        )
        serialized_state = json.loads(data)

        return ConversationStateSerializer.deserialize(
            serialized_state
        )

    async def save(
        self,
        conversation_id: UUID,
        state: LISAState,
    ) -> None:
        key = self._key(conversation_id)

        logger.info(
            "Saving conversation state to Redis: key=%s",
            key,
        )
        serialized_state = ConversationStateSerializer.serialize(
            state
        )

        await self.redis.set(
            key,
            json.dumps(serialized_state),
        )

        logger.info(
            "Conversation state saved to Redis: key=%s",
            key,
        )