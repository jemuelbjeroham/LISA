"""Add HITL approval records and PostgreSQL immutability guards.

Revision ID: a41c9d72e830
Revises: f0069c95796f
"""

import sqlalchemy as sa
from alembic import op

revision = "a41c9d72e830"
down_revision = "f0069c95796f"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "hitl_executions",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("session_id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("tool", sa.String(100), nullable=False),
        sa.Column("operation", sa.String(100), nullable=False),
        sa.Column("policy_version", sa.String(32), nullable=False),
        sa.Column("logical_key", sa.String(64), nullable=False, unique=True),
        sa.Column("state", sa.String(32), nullable=False),
        sa.Column("continuation", sa.JSON(), nullable=False),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_hitl_executions_user_id", "hitl_executions", ["user_id"])
    op.create_index("ix_hitl_executions_conversation_id", "hitl_executions", ["conversation_id"])
    op.create_index("ix_hitl_executions_state", "hitl_executions", ["state"])
    op.create_index(
        "uq_hitl_pending_conversation", "hitl_executions", ["user_id", "conversation_id"],
        unique=True,
        postgresql_where=sa.text("state IN ('pending', 'approved', 'claimed', 'executing', 'resuming')"),
    )
    op.create_table(
        "hitl_approvals",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("execution_id", sa.Uuid(), sa.ForeignKey("hitl_executions.id"), nullable=False, unique=True),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("state", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("decision_actor", sa.Uuid(), nullable=True),
        sa.Column("decision_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("decision_key", sa.String(128), nullable=True),
    )
    op.create_index("ix_hitl_approvals_state", "hitl_approvals", ["state"])
    op.create_index("ix_hitl_approvals_expires_at", "hitl_approvals", ["expires_at"])
    op.create_table(
        "hitl_revisions",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("approval_id", sa.Uuid(), sa.ForeignKey("hitl_approvals.id"), nullable=False),
        sa.Column("number", sa.Integer(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("payload_hash", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("approval_id", "number", name="uq_hitl_revision"),
    )
    op.create_table(
        "hitl_claims",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("execution_id", sa.Uuid(), sa.ForeignKey("hitl_executions.id"), nullable=False, unique=True),
        sa.Column("approval_id", sa.Uuid(), sa.ForeignKey("hitl_approvals.id"), nullable=False, unique=True),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("payload_hash", sa.String(64), nullable=False),
        sa.Column("idempotency_key", sa.String(64), nullable=False, unique=True),
        sa.Column("state", sa.String(32), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("result", sa.JSON(), nullable=True),
    )
    op.create_table(
        "hitl_events",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("execution_id", sa.Uuid(), nullable=True),
        sa.Column("approval_id", sa.Uuid(), nullable=True),
        sa.Column("actor_id", sa.Uuid(), nullable=False),
        sa.Column("event", sa.String(64), nullable=False),
        sa.Column("details", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    for column in ("execution_id", "approval_id", "actor_id", "created_at"):
        op.create_index(f"ix_hitl_events_{column}", "hitl_events", [column])
    op.execute("""
        CREATE FUNCTION hitl_reject_mutation() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION 'HITL records in % are immutable', TG_TABLE_NAME
                USING ERRCODE = '55000';
        END;
        $$
    """)
    for table in ("hitl_events", "hitl_revisions"):
        op.execute(f"""
            CREATE TRIGGER {table}_immutable
            BEFORE UPDATE OR DELETE ON {table}
            FOR EACH ROW EXECUTE FUNCTION hitl_reject_mutation()
        """)


def downgrade() -> None:
    for table in ("hitl_events", "hitl_revisions"):
        op.execute(f"DROP TRIGGER {table}_immutable ON {table}")
    op.execute("DROP FUNCTION hitl_reject_mutation()")
    for table in ("hitl_events", "hitl_claims", "hitl_revisions", "hitl_approvals", "hitl_executions"):
        op.drop_table(table)