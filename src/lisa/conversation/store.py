from typing import Protocol
from uuid import UUID

from lisa.state import LISAState


class ConversationStore(Protocol):
    async def get(self, conversation_id: UUID) -> LISAState | None:
        ...

    async def save(
            self,
            conversation_id: UUID,
            state: LISAState,
    ) -> None:
        ...