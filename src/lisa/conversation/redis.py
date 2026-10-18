
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

    def _key(self, user_id: UUID, conversation_id: UUID) -> str:
        return f"{self.key_prefix}{user_id}:{conversation_id}"

    def _index_key(self, user_id: UUID) -> str:
        return f"{self.key_prefix}index:{user_id}"

    async def get(
        self,
        user_id: UUID,
        conversation_id: UUID,
    ) -> LISAState | None:
        key = self._key(user_id, conversation_id)
        data = await self.redis.get(key)

        if data is None:
            return None

        serialized_state = json.loads(data)
        return ConversationStateSerializer.deserialize(serialized_state)

    async def save(
        self,
        user_id: UUID,
        conversation_id: UUID,
        state: LISAState,
    ) -> None:
        key = self._key(user_id, conversation_id)
        serialized_state = ConversationStateSerializer.serialize(state)

        async with self.redis.pipeline(transaction=True) as pipeline:
            pipeline.set(key, json.dumps(serialized_state))
            pipeline.sadd(
                self._index_key(user_id),
                str(conversation_id),
            )
            await pipeline.execute()

    async def list_conversations(self, user_id: UUID) -> list[UUID]:
        conversation_ids = await self.redis.smembers(
            self._index_key(user_id)
        )

        return sorted(
            (UUID(conversation_id) for conversation_id in conversation_ids),
            reverse=True,
        )
