
from typing import Any
from uuid import UUID

from lisa.memory.user_context.models import UserPreference, UserProfile
from lisa.memory.user_context.repository import UserContextRepository


class UserContextService:
    def __init__(self, repository: UserContextRepository) -> None:
        self.repository = repository

    async def get_profile(self, user_id: UUID) -> UserProfile | None:
        return await self.repository.get_profile(user_id)

    async def upsert_profile(
        self,
        profile: UserProfile,
    ) -> UserProfile:
        return await self.repository.upsert_profile(profile)

    async def get_preferences(
        self,
        user_id: UUID,
    ) -> list[UserPreference]:
        return await self.repository.list_preferences(user_id)

    async def upsert_preference(
        self,
        preference: UserPreference,
    ) -> UserPreference:
        return await self.repository.upsert_preference(preference)

    async def delete_preference(
        self,
        user_id: UUID,
        category: str,
        key: str,
    ) -> bool:
        return await self.repository.delete_preference(
            user_id=user_id,
            category=category,
            key=key,
        )

    async def build_context(self, user_id: UUID) -> dict[str, Any]:
        profile = await self.get_profile(user_id)
        preferences = await self.get_preferences(user_id)

        return {
            "profile": (
                {
                    "display_name": profile.display_name,
                    "details": profile.details,
                }
                if profile is not None
                else None
            ),
            "preferences": [
                {
                    "category": preference.category,
                    "key": preference.key,
                    "value": preference.value,
                }
                for preference in preferences
            ],
        }
