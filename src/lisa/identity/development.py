
from lisa.identity.models import User


class DevelopmentIdentityProvider:
    def __init__(self, user: User) -> None:
        self.user = user

    async def get_current_user(self) -> User:
        return self.user