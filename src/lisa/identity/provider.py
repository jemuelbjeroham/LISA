from typing import Protocol

from lisa.identity.models import User


class IdentityProvider(Protocol):
    async def get_current_user(self) -> User:
        ...
        