import json
from uuid import UUID

from redis.asyncio import Redis

from lisa.conversation.serialization import ConversationStateSerializer
from lisa.conversation.store import ConversationStore
from lisa.state import LISAState


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

        data = await self.redis.get(key)

        if data is None:
            return None

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

        serialized_state = ConversationStateSerializer.serialize(
            state
        )

        await self.redis.set(
            key,
            json.dumps(serialized_state),
        )