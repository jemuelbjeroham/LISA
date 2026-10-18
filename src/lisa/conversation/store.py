
from typing import Protocol
from uuid import UUID

from lisa.state import LISAState


class ConversationStore(Protocol):
    async def get(
        self,
        user_id: UUID,
        conversation_id: UUID,
    ) -> LISAState | None:
        ...

    async def save(
        self,
        user_id: UUID,
        conversation_id: UUID,
        state: LISAState,
    ) -> None:
        ...

    async def list_conversations(
        self,
        user_id: UUID,
    ) -> list[UUID]:
        ...

    async def get_title(
        self,
        user_id: UUID,
        conversation_id: UUID,
    ) -> str | None:
        ...

    async def set_title(
        self,
        user_id: UUID,
        conversation_id: UUID,
        title: str,
    ) -> None:
        ...

    async def delete(
        self,
        user_id: UUID,
        conversation_id: UUID,
    ) -> None:
        ...
