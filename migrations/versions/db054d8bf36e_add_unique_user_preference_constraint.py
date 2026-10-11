"""add unique user preference constraint

Revision ID: db054d8bf36e
Revises: 7c6a7dcf39c0
Create Date: 2026-10-12 00:01:59.377949

"""
from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'db054d8bf36e'
down_revision: str | Sequence[str] | None = '7c6a7dcf39c0'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None



def upgrade() -> None:
    op.create_unique_constraint(
        "uq_user_preference_user_category_key",
        "user_preferences",
        ["user_id", "category", "key"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_user_preference_user_category_key",
        "user_preferences",
        type_="unique",
    )
