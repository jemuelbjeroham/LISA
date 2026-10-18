
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

    def _title_key(
        self,
        user_id: UUID,
        conversation_id: UUID,
    ) -> str:
        return (
            f"{self.key_prefix}metadata:"
            f"{user_id}:{conversation_id}"
        )

    async def get(
        self,
        user_id: UUID,
        conversation_id: UUID,
    ) -> LISAState | None:
        data = await self.redis.get(
            self._key(user_id, conversation_id)
        )

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

    async def get_title(
        self,
        user_id: UUID,
        conversation_id: UUID,
    ) -> str | None:
        title = await self.redis.get(
            self._title_key(user_id, conversation_id)
        )

        if title is None:
            return None

        if isinstance(title, bytes):
            return title.decode("utf-8")

        return title

    async def set_title(
        self,
        user_id: UUID,
        conversation_id: UUID,
        title: str,
    ) -> None:
        await self.redis.set(
            self._title_key(user_id, conversation_id),
            title,
        )

    async def delete(
        self,
        user_id: UUID,
        conversation_id: UUID,
    ) -> None:
        async with self.redis.pipeline(transaction=True) as pipeline:
            pipeline.delete(
                self._key(user_id, conversation_id)
            )
            pipeline.delete(
                self._title_key(user_id, conversation_id)
            )
            pipeline.srem(
                self._index_key(user_id),
                str(conversation_id),
            )
            await pipeline.execute()
