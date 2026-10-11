
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from lisa.memory.user_context.models import UserPreference, UserProfile
from lisa.memory.user_context.repository_models import (
    UserPreferenceModel,
    UserProfileModel,
)


class UserContextRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_profile(self, user_id: UUID) -> UserProfile | None:
        result = await self.session.get(UserProfileModel, user_id)

        if result is None:
            return None

        return UserProfile(
            user_id=result.user_id,
            display_name=result.display_name,
            details=result.details,
            created_at=result.created_at,
            updated_at=result.updated_at,
        )

    async def upsert_profile(
        self,
        profile: UserProfile,
    ) -> UserProfile:
        now = datetime.now(UTC)
        result = await self.session.get(UserProfileModel, profile.user_id)

        if result is None:
            result = UserProfileModel(
                user_id=profile.user_id,
                display_name=profile.display_name,
                details=profile.details,
                created_at=now,
                updated_at=now,
            )
            self.session.add(result)
        else:
            result.display_name = profile.display_name
            result.details = profile.details
            result.updated_at = now

        await self.session.flush()

        return UserProfile(
            user_id=result.user_id,
            display_name=result.display_name,
            details=result.details,
            created_at=result.created_at,
            updated_at=result.updated_at,
        )

    async def list_preferences(
        self,
        user_id: UUID,
    ) -> list[UserPreference]:
        statement = (
            select(UserPreferenceModel)
            .where(UserPreferenceModel.user_id == user_id)
            .order_by(
                UserPreferenceModel.category,
                UserPreferenceModel.key,
            )
        )
        result = await self.session.scalars(statement)

        return [
            UserPreference(
                id=row.id,
                user_id=row.user_id,
                category=row.category,
                key=row.key,
                value=row.value,
                created_at=row.created_at,
                updated_at=row.updated_at,
            )
            for row in result.all()
        ]

    async def upsert_preference(
        self,
        preference: UserPreference,
    ) -> UserPreference:
        now = datetime.now(UTC)

        statement = select(UserPreferenceModel).where(
            UserPreferenceModel.user_id == preference.user_id,
            UserPreferenceModel.category == preference.category,
            UserPreferenceModel.key == preference.key,
        )
        result = await self.session.scalars(statement)
        row = result.first()

        if row is None:
            row = UserPreferenceModel(
                id=preference.id,
                user_id=preference.user_id,
                category=preference.category,
                key=preference.key,
                value=preference.value,
                created_at=now,
                updated_at=now,
            )
            self.session.add(row)
        else:
            row.value = preference.value
            row.updated_at = now

        await self.session.flush()

        return UserPreference(
            id=row.id,
            user_id=row.user_id,
            category=row.category,
            key=row.key,
            value=row.value,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    async def delete_preference(
        self,
        user_id: UUID,
        category: str,
        key: str,
    ) -> bool:
        statement = select(UserPreferenceModel).where(
            UserPreferenceModel.user_id == user_id,
            UserPreferenceModel.category == category,
            UserPreferenceModel.key == key,
        )
        result = await self.session.scalars(statement)
        row = result.first()

        if row is None:
            return False

        await self.session.delete(row)
        await self.session.flush()
        return True
