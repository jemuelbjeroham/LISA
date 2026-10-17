
"""add admin role to user profiles

Revision ID: 7f4a89b99ea1
Revises: f0069c95796f
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "7f4a89b99ea1"
down_revision: Union[str, Sequence[str], None] = "f0069c95796f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "user_profiles",
        sa.Column(
            "is_admin",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )


def downgrade() -> None:
    op.drop_column("user_profiles", "is_admin")
