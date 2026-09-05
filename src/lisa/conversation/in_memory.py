from uuid import UUID

from lisa.state import LISAState


class InMemoryConversationStore:
    def __init__(self) -> None:
        self._conversations: dict[str, LISAState] = {}

    async def get(self, conversation_id: UUID) -> LISAState | None:
        return self._conversations.get(conversation_id)

    async def save(
            self,
            conversation_id: UUID,
            state: LISAState,
    ) -> None:
        self._conversations[conversation_id] = state