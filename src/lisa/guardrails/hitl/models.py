from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from lisa.infrastructure.postgres.models import Base


class Execution(Base):
    __tablename__ = "hitl_executions"
    id: Mapped[UUID] = mapped_column(primary_key=True)
    user_id: Mapped[UUID] = mapped_column(index=True)
    session_id: Mapped[UUID] = mapped_column()
    conversation_id: Mapped[UUID] = mapped_column(index=True)
    role: Mapped[str] = mapped_column(String(16))
    tool: Mapped[str] = mapped_column(String(100))
    operation: Mapped[str] = mapped_column(String(100))
    policy_version: Mapped[str] = mapped_column(String(32))
    logical_key: Mapped[str] = mapped_column(String(64), unique=True)
    state: Mapped[str] = mapped_column(String(32), index=True)
    continuation: Mapped[dict] = mapped_column(JSON)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    __table_args__ = (Index("uq_hitl_pending_conversation", "user_id", "conversation_id", unique=True,
                            postgresql_where=text("state IN ('pending', 'approved', 'claimed', 'executing', 'resuming')")),)


class Approval(Base):
    __tablename__ = "hitl_approvals"
    id: Mapped[UUID] = mapped_column(primary_key=True)
    execution_id: Mapped[UUID] = mapped_column(ForeignKey("hitl_executions.id"), unique=True)
    revision: Mapped[int] = mapped_column(Integer)
    state: Mapped[str] = mapped_column(String(32), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    decision_actor: Mapped[UUID | None] = mapped_column(nullable=True)
    decision_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    decision_key: Mapped[str | None] = mapped_column(String(128), nullable=True)


class Revision(Base):
    __tablename__ = "hitl_revisions"
    id: Mapped[UUID] = mapped_column(primary_key=True)
    approval_id: Mapped[UUID] = mapped_column(ForeignKey("hitl_approvals.id"))
    number: Mapped[int] = mapped_column(Integer)
    payload: Mapped[dict] = mapped_column(JSON)
    payload_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    __table_args__ = (UniqueConstraint("approval_id", "number", name="uq_hitl_revision"),)


class Claim(Base):
    __tablename__ = "hitl_claims"
    id: Mapped[UUID] = mapped_column(primary_key=True)
    execution_id: Mapped[UUID] = mapped_column(ForeignKey("hitl_executions.id"), unique=True)
    approval_id: Mapped[UUID] = mapped_column(ForeignKey("hitl_approvals.id"), unique=True)
    revision: Mapped[int] = mapped_column(Integer)
    payload_hash: Mapped[str] = mapped_column(String(64))
    idempotency_key: Mapped[str] = mapped_column(String(64), unique=True)
    state: Mapped[str] = mapped_column(String(32))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class AuditEvent(Base):
    __tablename__ = "hitl_events"
    id: Mapped[UUID] = mapped_column(primary_key=True)
    execution_id: Mapped[UUID | None] = mapped_column(index=True, nullable=True)
    approval_id: Mapped[UUID | None] = mapped_column(index=True, nullable=True)
    actor_id: Mapped[UUID] = mapped_column(index=True)
    event: Mapped[str] = mapped_column(String(64))
    details: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)