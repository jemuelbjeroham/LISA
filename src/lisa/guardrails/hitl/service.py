import asyncio
import hashlib
import json
import logging
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sqlalchemy import select

from lisa.guardrails.hitl.context import (
    Principal,
    current_execution,
    native_runtime,
    native_write,
)
from lisa.guardrails.hitl.jira import JIRA_TOOLS, JiraAdapter, verify_capability
from lisa.guardrails.hitl.models import Approval, AuditEvent, Claim, Execution, Revision
from lisa.guardrails.hitl.policy import (
    POLICY_VERSION,
    GuardError,
    MemoryArguments,
    authorize,
)
from lisa.memory.user_context.repository_models import (
    UserProfileModel,
    UserSessionModel,
)

logger = logging.getLogger(__name__)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()).hexdigest()


class ApprovalService:
    def __init__(self, database, clock=None):
        self.database = database
        self.clock = clock or (lambda: datetime.now(UTC))
        self.jira = JiraAdapter()

    async def identity(self, session, principal):
        now = self.clock()
        row = (await session.execute(select(UserProfileModel, UserSessionModel).join(
            UserSessionModel, UserSessionModel.user_id == UserProfileModel.user_id).where(
            UserProfileModel.user_id == principal.user_id, UserProfileModel.is_active.is_(True),
            UserSessionModel.id == principal.session_id, UserSessionModel.revoked_at.is_(None),
            UserSessionModel.expires_at > now))).first()
        if row is None:
            raise GuardError("identity_unavailable", 401)
        return Principal(principal.user_id, principal.session_id, "admin" if row[0].is_admin else "member")

    def event(self, session, actor, event, execution=None, approval=None, **details):
        session.add(AuditEvent(id=uuid4(), actor_id=actor.user_id, event=event,
            execution_id=execution.id if execution else None, approval_id=approval.id if approval else None,
            details={"role": actor.role, "policy_version": POLICY_VERSION, **details}, created_at=self.clock()))

    async def check(self, principal, name):
        denied = None
        async with self.database.session() as session:
            actor = await self.identity(session, principal)
            try:
                policy = authorize(actor.role, name)
            except GuardError as exc:
                denied = exc
                self.event(session, actor, "authorization_denied", tool=name)
            else:
                self.event(session, actor, "authorization_allowed", tool=name)
        if denied:
            raise denied
        return policy

    async def assert_turn_available(self, actor, conversation_id):
        await self.recover()
        async with self.database.session() as session:
            await self.identity(session, actor)
            exists = await session.scalar(select(Execution.id).where(Execution.user_id == actor.user_id,
                Execution.conversation_id == conversation_id, Execution.state.in_(["pending", "approved", "claimed", "executing", "resuming"])))
            if exists:
                raise GuardError("execution_pending_cancel_required", 409)

    async def propose(self, actor, conversation_id, name, arguments, logical_id, continuation):
        policy = await self.check(actor, name)
        if not policy.write:
            raise GuardError("approval_not_required", 409)
        effective = None
        if name in JIRA_TOOLS:
            effective = await self.jira.prepare(name, arguments)
            arguments = effective["arguments"]
        elif name == "save_memory":
            arguments = MemoryArguments.model_validate(arguments).model_dump(mode="json")
        else:
            raise GuardError("verified_preparation_unavailable", 503)
        now = self.clock()
        key = digest({"actor": str(actor.user_id), "conversation": str(conversation_id), "logical_id": logical_id,
                      "tool": name, "arguments": arguments})
        async with self.database.session() as session:
            actor = await self.identity(session, actor)
            authorize(actor.role, name)
            await session.scalar(select(UserProfileModel).where(UserProfileModel.user_id == actor.user_id).with_for_update())
            existing = await session.scalar(select(Execution).where(Execution.logical_key == key))
            if existing:
                approval = await session.scalar(select(Approval).where(Approval.execution_id == existing.id))
                return await self.view(session, existing, approval)
            context = current_execution.get()
            exclusion = (context.resuming_execution_id if context is not None
                and context.principal.user_id == actor.user_id and context.conversation_id == conversation_id else None)
            active = await session.scalar(select(Execution.id).where(Execution.user_id == actor.user_id,
                Execution.conversation_id == conversation_id, Execution.state.in_(["pending", "approved", "claimed", "executing", "resuming"]),
                ~((Execution.id == exclusion) & (Execution.state == "resuming")) if exclusion else True))
            if active:
                raise GuardError("execution_pending_cancel_required", 409)
            execution = Execution(id=uuid4(), user_id=actor.user_id, session_id=actor.session_id,
                conversation_id=conversation_id, role=actor.role, tool=name, operation=policy.operation,
                policy_version=POLICY_VERSION, logical_key=key, state="pending", continuation=continuation,
                created_at=now, updated_at=now)
            approval = Approval(id=uuid4(), execution_id=execution.id, revision=1, state="pending",
                created_at=now, expires_at=now + timedelta(minutes=10))
            payload = {"tool": name, "operation": policy.operation, "arguments": arguments,
                "target": str(actor.user_id), "user_id": str(actor.user_id), "execution_id": str(execution.id),
                "conversation_id": str(conversation_id), "approval_id": str(approval.id), "revision": 1,
                "policy_version": POLICY_VERSION, "schema_version": policy.schema_version}
            if effective is not None:
                payload["effective_operation"] = effective
            session.add(execution)
            await session.flush()
            session.add(approval)
            await session.flush()
            session.add(Revision(id=uuid4(), approval_id=approval.id, number=1, payload=payload,
                payload_hash=digest(payload), created_at=now))
            self.event(session, actor, "approval_created", execution, approval, tool=name, revision=1, payload_hash=digest(payload))
            await session.flush()
            return await self.view(session, execution, approval)

    async def owned(self, session, actor, approval_id):
        row = (await session.execute(select(Execution, Approval).join(Approval, Approval.execution_id == Execution.id)
            .where(Approval.id == approval_id, Execution.user_id == actor.user_id).with_for_update())).first()
        if row is None:
            raise GuardError("not_found", 404)
        return row

    async def view(self, session, execution, approval):
        revision = await session.scalar(select(Revision).where(Revision.approval_id == approval.id, Revision.number == approval.revision))
        return {"approval_id": str(approval.id), "execution_id": str(execution.id),
            "conversation_id": str(execution.conversation_id), "tool": execution.tool, "operation": execution.operation,
            "status": approval.state, "execution_status": execution.state, "revision": approval.revision,
            "arguments_hash": revision.payload_hash, "arguments": revision.payload["arguments"],
            "target": revision.payload["target"], "created_at": approval.created_at.isoformat(),
            "effective_operation": revision.payload.get("effective_operation"),
            "expires_at": approval.expires_at.isoformat(), "result": execution.result,
            "editable_fields": list(authorize("admin", execution.tool).editable)}

    async def get(self, actor, approval_id):
        await self.recover()
        async with self.database.session() as session:
            await self.identity(session, actor)
            execution, approval = await self.owned(session, actor, approval_id)
            return await self.view(session, execution, approval)

    async def list(self, actor, conversation_id):
        await self.recover()
        async with self.database.session() as session:
            await self.identity(session, actor)
            rows = (await session.execute(select(Execution, Approval).join(Approval, Approval.execution_id == Execution.id)
                .where(Execution.user_id == actor.user_id, Execution.conversation_id == conversation_id)
                .order_by(Approval.created_at.desc()).limit(100))).all()
            return [await self.view(session, execution, approval) for execution, approval in rows]

    async def edit(self, actor, approval_id, expected_revision, expected_hash, changes):
        await self.recover()
        async with self.database.session() as session:
            actor = await self.identity(session, actor)
            execution, approval = await self.owned(session, actor, approval_id)
            policy = authorize(actor.role, execution.tool)
            if self.clock() >= approval.expires_at:
                raise GuardError("approval_expired", 410)
            if approval.state != "pending":
                raise GuardError("approval_not_pending", 409)
            revision = await session.scalar(select(Revision).where(Revision.approval_id == approval.id, Revision.number == approval.revision))
            if approval.revision != expected_revision or revision.payload_hash != expected_hash:
                raise GuardError("stale_revision", 409)
            if not changes or set(changes) - set(policy.editable):
                raise GuardError("invalid_edit", 422)
            effective = None
            if execution.tool in JIRA_TOOLS:
                effective = await self.jira.prepare(execution.tool, {**revision.payload["arguments"], **changes})
                arguments = effective["arguments"]
            else:
                arguments = MemoryArguments.model_validate({**revision.payload["arguments"], **changes}).model_dump(mode="json")
            approval.revision += 1
            payload = {**revision.payload, "arguments": arguments, "revision": approval.revision}
            if effective is not None:
                payload["effective_operation"] = effective
            session.add(Revision(id=uuid4(), approval_id=approval.id, number=approval.revision,
                payload=payload, payload_hash=digest(payload), created_at=self.clock()))
            self.event(session, actor, "proposal_revised", execution, approval, revision=approval.revision, payload_hash=digest(payload))
            await session.flush()
            return await self.view(session, execution, approval)

    async def decide(self, actor, approval_id, revision_number, payload_hash, decision, decision_key):
        if decision not in ("approve", "reject") or not isinstance(decision_key, str) or not 1 <= len(decision_key) <= 128:
            raise GuardError("invalid_decision", 422)
        await self.recover()
        async with self.database.session() as session:
            actor = await self.identity(session, actor)
            execution, approval = await self.owned(session, actor, approval_id)
            authorize(actor.role, execution.tool)
            revision = await session.scalar(select(Revision).where(Revision.approval_id == approval.id, Revision.number == approval.revision))
            if approval.state == "expired":
                raise GuardError("approval_expired", 410)
            if approval.revision != revision_number or revision.payload_hash != payload_hash:
                raise GuardError("stale_revision", 409)
            target = "approved" if decision == "approve" else "rejected"
            if approval.state != "pending":
                if approval.decision_key == decision_key and approval.decision_actor == actor.user_id and approval.state == target:
                    return await self.view(session, execution, approval)
                raise GuardError("decision_conflict", 409)
            if self.clock() >= approval.expires_at:
                raise GuardError("approval_expired", 410)
            approval.state = target
            approval.decision_actor = actor.user_id
            approval.decision_at = self.clock()
            approval.decision_key = decision_key
            execution.state = target
            if target == "rejected":
                execution.result = {"accepted": False, "error": "approval_rejected"}
            self.event(session, actor, target, execution, approval, revision=approval.revision, payload_hash=payload_hash)
            return await self.view(session, execution, approval)

    async def cancel(self, actor, execution_id):
        async with self.database.session() as session:
            actor = await self.identity(session, actor)
            execution = await session.scalar(select(Execution).where(Execution.id == execution_id,
                Execution.user_id == actor.user_id).with_for_update())
            if execution is None:
                raise GuardError("not_found", 404)
            approval = await session.scalar(select(Approval).where(Approval.execution_id == execution.id).with_for_update())
            if execution.state not in ("pending", "approved", "cancelled"):
                raise GuardError("execution_already_claimed", 409)
            execution.state = approval.state = "cancelled"
            execution.result = {"accepted": False, "error": "execution_cancelled"}
            self.event(session, actor, "cancelled", execution, approval)
            return await self.view(session, execution, approval)

    async def recover(self):
        async with self.database.session() as session:
            rows = (await session.execute(select(Execution, Approval).join(Approval, Approval.execution_id == Execution.id)
                .where(Approval.state.in_(["pending", "approved"]), Approval.expires_at <= self.clock()).with_for_update(skip_locked=True))).all()
            for execution, approval in rows:
                if execution.state not in ("pending", "approved"):
                    continue
                execution.state = approval.state = "expired"
                execution.result = {"accepted": False, "error": "approval_timeout"}
                self.event(session, Principal(execution.user_id, execution.session_id, execution.role), "expired", execution, approval)
            stale = (await session.execute(select(Execution, Claim).outerjoin(Claim, Claim.execution_id == Execution.id)
                .where(Execution.state.in_(["claimed", "executing", "resuming"]),
                    Execution.updated_at <= self.clock() - timedelta(minutes=10))
                .with_for_update(of=Execution, skip_locked=True))).all()
            for execution, claim in stale:
                execution.state = "manual_review"
                execution.result = {"accepted": False, "error": "execution_interrupted_no_retry"}
                execution.updated_at = self.clock()
                self.event(session, Principal(execution.user_id, execution.session_id, execution.role),
                    "manual_review", execution, claim_id=str(claim.id) if claim else None)

    async def recovery_loop(self):
        while True:
            try:
                await self.recover()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.error("HITL recovery unavailable; protected execution remains closed")
            await asyncio.sleep(10)

    async def execution(self, actor, execution_id):
        await self.recover()
        async with self.database.session() as session:
            await self.identity(session, actor)
            approval_id = await session.scalar(select(Approval.id).join(Execution, Execution.id == Approval.execution_id)
                .where(Execution.id == execution_id, Execution.user_id == actor.user_id))
            if approval_id is None:
                raise GuardError("not_found", 404)
        return await self.get(actor, approval_id)

    async def consume_jira(self, execution_capability, name, arguments):
        from uuid import UUID

        claims = verify_capability(execution_capability)
        try:
            claim_id = UUID(claims["claim_id"])
        except (ValueError, KeyError, TypeError):
            raise GuardError("invalid_execution_capability", 403) from None
        async with self.database.session() as session:
            claim = await session.scalar(select(Claim).where(Claim.id == claim_id).with_for_update())
            if claim is None or claim.state != "claimed" or claim.payload_hash != claims.get("hash"):
                raise GuardError("execution_capability_consumed_or_invalid", 409)
            execution = await session.scalar(select(Execution).where(Execution.id == claim.execution_id).with_for_update())
            approval = await session.scalar(select(Approval).where(Approval.id == claim.approval_id).with_for_update())
            revision = await session.scalar(select(Revision).where(Revision.approval_id == approval.id,
                Revision.number == claim.revision))
            actor = await self.identity(session, Principal(execution.user_id, execution.session_id, execution.role))
            authorize(actor.role, name)
            if (name not in JIRA_TOOLS or execution.tool != name or execution.state != "claimed"
                    or approval.execution_id != execution.id or approval.state != "approved"
                    or approval.revision != claim.revision or self.clock() >= approval.expires_at
                    or revision is None or revision.payload_hash != claim.payload_hash
                    or digest(revision.payload) != claim.payload_hash
                    or execution.policy_version != POLICY_VERSION
                    or revision.payload["arguments"] != arguments):
                raise GuardError("execution_binding_invalid", 403)
            effective = revision.payload.get("effective_operation")
            if not isinstance(effective, dict) or effective.get("tool_name") != name or effective.get("arguments") != arguments:
                raise GuardError("execution_binding_invalid", 403)
            claim.state = "executing"
            execution.state = "executing"
            execution.updated_at = self.clock()
            self.event(session, actor, "jira_capability_consumed", execution, approval, claim_id=str(claim.id))
            return {"authorized": True, "effective_operation": effective}

    async def resume(self, actor, execution_id):

        from lisa.agents.tool_access import RESTRICTED_AGENT_TOOLS
        from lisa.guardrails.hitl.context import ExecutionContext, execution_scope
        from lisa.infrastructure.postgres.memory_dependencies import memory_dependencies
        from lisa.streaming.events import StreamEvent
        from lisa.tools import save_memory

        await self.recover()
        async with self.database.session() as session:
            actor = await self.identity(session, actor)
            execution = await session.scalar(select(Execution).where(Execution.id == execution_id,
                Execution.user_id == actor.user_id).with_for_update())
            if execution is None:
                raise GuardError("not_found", 404)
            approval = await session.scalar(select(Approval).where(Approval.execution_id == execution.id).with_for_update())
            revision = await session.scalar(select(Revision).where(Revision.approval_id == approval.id, Revision.number == approval.revision))
            if execution.continuation.get("resumed"):
                yield StreamEvent(type="execution_completed", execution_id=str(execution.id), status=execution.state)
                return
            if execution.continuation.get("agent") == "technical_clarification":
                raise GuardError("continuation_version_unsupported", 409)
            allowed = RESTRICTED_AGENT_TOOLS.get(execution.continuation.get("agent"))
            if allowed is not None and execution.tool not in allowed:
                raise GuardError("continuation_tool_not_authorized", 409)
            if execution.state in ("pending", "claimed", "executing", "resuming"):
                raise GuardError("execution_not_resumable", 409)
            continuation = dict(execution.continuation)
            conversation_id = execution.conversation_id
            if execution.state == "approved":
                original = Principal(execution.user_id, execution.session_id, execution.role)
                actor = await self.identity(session, original)
                policy = authorize(actor.role, execution.tool)
                if self.clock() >= approval.expires_at:
                    raise GuardError("approval_expired", 410)
                if digest(revision.payload) != revision.payload_hash or execution.policy_version != POLICY_VERSION:
                    raise GuardError("payload_integrity_failed", 409)
                bindings = {"tool": execution.tool, "operation": execution.operation, "user_id": str(execution.user_id),
                    "target": str(execution.user_id), "execution_id": str(execution.id), "conversation_id": str(execution.conversation_id),
                    "approval_id": str(approval.id), "revision": approval.revision, "policy_version": POLICY_VERSION,
                    "schema_version": policy.schema_version}
                if any(revision.payload.get(key) != value for key, value in bindings.items()):
                    raise GuardError("payload_integrity_failed", 409)
                if policy.name not in JIRA_TOOLS | {"save_memory"} or revision.payload["target"] != str(actor.user_id):
                    raise GuardError("execution_adapter_unavailable", 503)
                if policy.name in JIRA_TOOLS:
                    effective = revision.payload.get("effective_operation")
                    if not isinstance(effective, dict) or effective.get("tool_name") != policy.name or effective.get("arguments") != revision.payload["arguments"]:
                        raise GuardError("payload_integrity_failed", 409)
                    arguments = revision.payload["arguments"]
                else:
                    arguments = MemoryArguments.model_validate(revision.payload["arguments"]).model_dump(mode="json")
                claim = Claim(id=uuid4(), execution_id=execution.id, approval_id=approval.id, revision=approval.revision,
                    payload_hash=revision.payload_hash, idempotency_key=digest({"execution": str(execution.id), "revision": approval.revision,
                    "hash": revision.payload_hash}), state="claimed", started_at=self.clock())
                session.add(claim)
                execution.state = "claimed"
                execution.updated_at = self.clock()
                self.event(session, actor, "execution_claimed", execution, approval, revision=approval.revision, payload_hash=revision.payload_hash)
                await session.flush()
                claim_id = claim.id
                result = None
            else:
                claim_id = None
                arguments = None
                result = execution.result or {"accepted": False, "error": execution.state}
            if claim_id is None:
                execution.state = "resuming"
                execution.updated_at = self.clock()
        if claim_id is not None:
            try:
                jira_result = None
                if execution.tool in JIRA_TOOLS:
                    jira_result = await self.jira.execute(execution.tool, arguments, claim_id, revision.payload_hash)
                async with self.database.session() as session:
                    execution = await session.scalar(select(Execution).where(Execution.id == execution_id).with_for_update())
                    claim = await session.get(Claim, claim_id)
                    approval = await session.scalar(select(Approval).where(Approval.execution_id == execution_id).with_for_update())
                    actor = await self.identity(session, Principal(execution.user_id, execution.session_id, execution.role))
                    authorize(actor.role, execution.tool)
                    if self.clock() >= approval.expires_at or approval.state != "approved":
                        raise GuardError("approval_expired_or_cancelled", 410)
                    expected_state = "executing" if execution.tool in JIRA_TOOLS else "claimed"
                    if execution.state != expected_state or claim.revision != approval.revision or claim.payload_hash != revision.payload_hash:
                        raise GuardError("claim_integrity_failed", 409)
                    if execution.tool in JIRA_TOOLS:
                        if claim.state != "executing":
                            raise GuardError("claim_integrity_failed", 409)
                        result = {"accepted": True, "result": jira_result}
                    else:
                        token = native_write.set(actor.user_id)
                        try:
                            runtime = native_runtime(actor.user_id, memory_dependencies(session))
                            with execution_scope(ExecutionContext(actor, conversation_id, self)):
                                text = await asyncio.wait_for(save_memory.ainvoke({**arguments, "runtime": runtime}), timeout=60)
                        finally:
                            native_write.reset(token)
                        result = {"accepted": True, "result": text}
                    execution.state = "resuming"
                    execution.updated_at = self.clock()
                    execution.result = result
                    claim.state = "succeeded"
                    claim.result = result
                    claim.completed_at = self.clock()
                    self.event(session, actor, "execution_succeeded", execution, approval, claim_id=str(claim.id))
            except BaseException:
                async with self.database.session() as session:
                    execution = await session.get(Execution, execution_id)
                    execution.state = "manual_review"
                    execution.result = {"accepted": False, "error": "execution_interrupted_no_retry"}
                    self.event(session, actor, "manual_review", execution)
                raise
        yield StreamEvent(type="tool_execution_status", execution_id=str(execution_id), status="succeeded" if result.get("accepted") else result.get("error"))
        runner = getattr(self, "continuation_runner", None)
        if runner is None:
            raise GuardError("continuation_unavailable", 503)
        with execution_scope(ExecutionContext(actor, conversation_id, self, execution_id)):
            continuation_result = {**result, "approved_arguments": arguments} if result.get("accepted") else result
            try:
                async with asyncio.timeout(120):
                    async for event in runner(continuation, continuation_result, actor, conversation_id):
                        yield event
            except BaseException:
                async with self.database.session() as session:
                    execution = await session.get(Execution, execution_id)
                    execution.state = "manual_review"
                    execution.updated_at = self.clock()
                    execution.continuation = {**continuation, "resumed": True}
                    self.event(session, actor, "continuation_interrupted", execution)
                raise
        async with self.database.session() as session:
            execution = await session.scalar(select(Execution).where(Execution.id == execution_id).with_for_update())
            execution.continuation = {**continuation, "resumed": True}
            execution.state = "completed"
            self.event(session, actor, "continuation_completed", execution)
        yield StreamEvent(type="execution_completed", execution_id=str(execution_id), status="completed")