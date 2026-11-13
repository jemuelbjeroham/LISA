import asyncio
import copy
import sys
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from lisa.guardrails.hitl.context import Principal, native_write, require_native_write
from lisa.guardrails.hitl.models import Approval, Claim, Execution, Revision
from lisa.guardrails.hitl.policy import POLICY_VERSION, GuardError
from lisa.guardrails.hitl.service import ApprovalService, digest
from lisa.infrastructure.postgres.memory_dependencies import MemoryDependencies


@pytest.fixture
def execution_case(monkeypatch):
    actor = Principal(uuid4(), uuid4(), "admin")
    now = datetime(2026, 10, 26, 12, tzinfo=UTC)
    execution = Execution(
        id=uuid4(), user_id=actor.user_id, session_id=actor.session_id,
        conversation_id=uuid4(), role="admin", tool="save_memory",
        operation="memory.create", policy_version=POLICY_VERSION,
        logical_key=digest(str(uuid4())), state="approved",
        continuation={"route": "general_enquiry"}, created_at=now, updated_at=now,
    )
    approval = Approval(
        id=uuid4(), execution_id=execution.id, revision=1, state="approved",
        created_at=now, expires_at=now + timedelta(minutes=10),
    )
    payload = {
        "tool": execution.tool, "operation": execution.operation,
        "arguments": {"content": "remember", "memory_type": "fact", "scope": "user"},
        "target": str(actor.user_id), "user_id": str(actor.user_id),
        "execution_id": str(execution.id), "conversation_id": str(execution.conversation_id),
        "approval_id": str(approval.id), "revision": 1,
        "policy_version": POLICY_VERSION, "schema_version": "1",
    }
    revision = Revision(id=uuid4(), approval_id=approval.id, number=1,
                        payload=payload, payload_hash=digest(payload), created_at=now)
    case = SimpleNamespace(
        actor=actor, execution=execution, approval=approval, revision=revision,
        now=now, active=True, admin=True, identity_calls=0, before_identity=None,
        records=[execution, approval, revision], sessions=[],
    )
    lock = asyncio.Lock()

    @asynccontextmanager
    async def database_session():
        async with lock:
            snapshots = [(record, {column.name: copy.deepcopy(getattr(record, column.name))
                                  for column in record.__table__.columns})
                         for record in case.records]
            added = []
            session = MagicMock()
            session.add.side_effect = added.append
            session.flush = AsyncMock()

            async def scalar(statement):
                entity = statement.column_descriptions[0]["entity"]
                assert statement._for_update_arg is not None or entity is Revision
                if entity is Claim:
                    return next((record for record in case.records if isinstance(record, Claim)), None)
                return {Execution: execution, Approval: approval, Revision: revision}[entity]

            async def execute(statement):
                entities = [description["entity"] for description in statement.column_descriptions]
                if entities == [Execution, Approval]:
                    assert statement._for_update_arg.skip_locked
                    rows = [(execution, approval)] if (
                        approval.state in {"pending", "approved"} and approval.expires_at <= case.now
                    ) else []
                    return SimpleNamespace(all=lambda: rows)
                if entities == [Execution, Claim]:
                    assert statement._for_update_arg.skip_locked
                    rows = [(execution, record) for record in case.records if (
                        isinstance(record, Claim) and record.execution_id == execution.id
                        and execution.state in {"claimed", "executing", "resuming"}
                        and execution.updated_at <= case.now - timedelta(minutes=10)
                    )]
                    return SimpleNamespace(all=lambda: rows)
                case.identity_calls += 1
                if case.before_identity:
                    case.before_identity(case.identity_calls)
                row = (SimpleNamespace(is_admin=case.admin), object()) if case.active else None
                return SimpleNamespace(first=lambda: row)

            async def get(model, identifier):
                return next(record for record in case.records
                            if isinstance(record, model) and record.id == identifier)

            session.scalar = AsyncMock(side_effect=scalar)
            session.execute = AsyncMock(side_effect=execute)
            session.get = AsyncMock(side_effect=get)
            case.sessions.append(session)
            try:
                yield session
            except BaseException:
                for record, values in snapshots:
                    for name, value in values.items():
                        setattr(record, name, value)
                raise
            else:
                case.records.extend(added)

    async def dispatch(arguments):
        require_native_write(actor.user_id)
        assert arguments["runtime"].context.user_id == actor.user_id
        assert arguments["content"] == revision.payload["arguments"]["content"]
        return "stored locally"

    case.dispatch = AsyncMock(side_effect=dispatch)
    tools = ModuleType("lisa.tools")
    tools.save_memory = SimpleNamespace(ainvoke=case.dispatch)
    dependencies = ModuleType("lisa.infrastructure.postgres.memory_dependencies")
    dependencies.MemoryDependencies = MemoryDependencies
    dependencies.memory_dependencies = MagicMock(return_value=object())
    monkeypatch.setitem(sys.modules, "lisa.tools", tools)
    monkeypatch.setitem(sys.modules, "lisa.infrastructure.postgres.memory_dependencies", dependencies)
    case.service = ApprovalService(SimpleNamespace(session=database_session), clock=lambda: case.now)
    case.results = []

    async def continuation_runner(continuation, result, principal, conversation_id):
        case.results.append(result)
        assert principal.user_id == actor.user_id
        assert conversation_id == execution.conversation_id
        if False:
            yield

    case.service.continuation_runner = continuation_runner
    return case


async def consume(case):
    return [event async for event in case.service.resume(case.actor, case.execution.id)]


async def test_retired_agent_approval_cannot_execute(execution_case):
    case = execution_case
    case.execution.continuation = {"agent": "technical_clarification"}
    with pytest.raises(GuardError, match="continuation_version_unsupported"):
        await consume(case)
    case.dispatch.assert_not_awaited()
    assert case.execution.state == "approved"
    assert not any(isinstance(record, Claim) for record in case.records)


def jira_case(case, monkeypatch, name="update_jira_ticket"):
    import secrets

    from lisa.guardrails.hitl.jira import capability
    from lisa.guardrails.hitl.policy import REGISTRY

    monkeypatch.setenv("LISA_JIRA_SIGNING_KEY", secrets.token_hex(32))
    case.execution.tool = name
    case.execution.operation = REGISTRY[name].operation
    arguments = ({"ticket_key": "OPS-123", "comment": "Approved"} if name == "update_jira_ticket" else
        {"ticket_id": "123"} if name == "close_ticket" else {"subject": "Approved"})
    effective = {"tool_name": name, "arguments": arguments, "destination": "https://jira.example",
        "ticket_key": "OPS-123", "payload": {"exact": "resolved-before-approval"}}
    case.revision.payload = {**case.revision.payload, "tool": name, "operation": REGISTRY[name].operation,
        "arguments": arguments, "effective_operation": effective}
    case.revision.payload_hash = digest(case.revision.payload)
    case.writes = []
    case.capabilities = []

    async def execute(tool_name, values, claim_id, payload_hash):
        token = capability(claim_id, payload_hash)
        case.capabilities.append(token)
        authorized = await case.service.consume_jira(token, tool_name, values)
        case.writes.append(authorized["effective_operation"])
        return {"success": True, "ticket_key": "OPS-123", "ticket_url": "https://jira.example/browse/OPS-123"}

    case.service.jira.execute = AsyncMock(side_effect=execute)
    return effective


async def test_retired_agent_jira_approval_cannot_execute(execution_case, monkeypatch):
    case = execution_case
    jira_case(case, monkeypatch)
    case.execution.continuation = {"agent": "technical_clarification"}
    with pytest.raises(GuardError, match="continuation_version_unsupported"):
        await consume(case)
    case.service.jira.execute.assert_not_awaited()
    assert case.writes == []
    assert case.execution.state == "approved"


@pytest.mark.parametrize("agent_name", ["knowledge_assistant", "troubleshooting_agent"])
@pytest.mark.parametrize("jira", [False, True])
async def test_restricted_agent_old_approval_cannot_execute(execution_case, monkeypatch, agent_name, jira):
    case = execution_case
    if jira:
        jira_case(case, monkeypatch)
    case.execution.continuation = {"agent": agent_name}
    with pytest.raises(GuardError, match="continuation_tool_not_authorized"):
        await consume(case)
    case.dispatch.assert_not_awaited()
    if jira:
        case.service.jira.execute.assert_not_awaited()
    assert case.execution.state == "approved"
    assert not any(isinstance(record, Claim) for record in case.records)


@pytest.mark.parametrize("name", ["create_jira_ticket", "update_jira_ticket", "close_ticket"])
async def test_jira_pending_then_approved_revision_executes_once(execution_case, monkeypatch, name):
    case = execution_case
    effective = jira_case(case, monkeypatch, name)
    case.execution.state = case.approval.state = "pending"
    with pytest.raises(GuardError, match="execution_not_resumable"):
        await consume(case)
    assert case.writes == []
    case.execution.state = case.approval.state = "approved"
    await consume(case)
    await consume(case)
    assert case.writes == [effective]
    case.service.jira.execute.assert_awaited_once()
    case.dispatch.assert_not_awaited()
    assert case.results[0]["approved_arguments"] == effective["arguments"]
    assert case.execution.state == "completed"
    with pytest.raises(GuardError, match="execution_capability_consumed_or_invalid"):
        await case.service.consume_jira(case.capabilities[0], name, effective["arguments"])
    assert case.writes == [effective]


@pytest.mark.parametrize("state", ["rejected", "expired", "cancelled"])
async def test_jira_negative_decisions_never_write(execution_case, monkeypatch, state):
    case = execution_case
    jira_case(case, monkeypatch)
    case.execution.state = case.approval.state = state
    await consume(case)
    case.service.jira.execute.assert_not_awaited()
    assert case.writes == []


@pytest.mark.parametrize("change", ["arguments", "hash", "member", "revoked", "revision", "expires"])
async def test_jira_consume_revalidates_exact_claim_before_write(execution_case, monkeypatch, change):
    case = execution_case
    jira_case(case, monkeypatch)
    original = case.service.jira.execute.side_effect

    async def changed(tool_name, values, claim_id, payload_hash):
        if change == "arguments":
            values = {**values, "comment": "Unapproved"}
        elif change == "hash":
            payload_hash = "wrong"
        elif change == "member":
            case.admin = False
        elif change == "revoked":
            case.active = False
        elif change == "revision":
            case.approval.revision += 1
        elif change == "expires":
            case.now = case.approval.expires_at
        return await original(tool_name, values, claim_id, payload_hash)

    case.service.jira.execute.side_effect = changed
    with pytest.raises(GuardError):
        await consume(case)
    assert case.writes == []
    assert case.execution.state == "manual_review"


async def test_jira_timeout_after_write_is_never_retried(execution_case, monkeypatch):
    case = execution_case
    jira_case(case, monkeypatch)
    original = case.service.jira.execute.side_effect

    async def interrupted(*arguments):
        await original(*arguments)
        raise TimeoutError("mock transport interruption")

    case.service.jira.execute.side_effect = interrupted
    with pytest.raises(TimeoutError):
        await consume(case)
    await consume(case)
    assert len(case.writes) == 1
    case.service.jira.execute.assert_awaited_once()


async def test_success_commits_unique_claim_and_repeated_resume_does_not_dispatch(execution_case):
    case = execution_case
    events = await consume(case)
    claims = [record for record in case.records if isinstance(record, Claim)]
    assert len(claims) == 1
    claim = claims[0]
    assert claim.execution_id == case.execution.id
    assert claim.approval_id == case.approval.id
    assert claim.revision == 1
    assert claim.payload_hash == case.revision.payload_hash
    assert claim.state == "succeeded"
    assert claim.completed_at == case.now
    assert claim.result == case.execution.result == {"accepted": True, "result": "stored locally"}
    assert case.execution.state == "completed"
    assert case.execution.continuation["resumed"] is True
    assert [event.type for event in events] == ["tool_execution_status", "execution_completed"]
    assert len(await consume(case)) == 1
    case.dispatch.assert_awaited_once()
    assert case.results == [{
        "accepted": True, "result": "stored locally",
        "approved_arguments": case.revision.payload["arguments"],
    }]
    assert native_write.get() is None


async def test_concurrent_resume_cannot_dispatch_twice_under_serialized_transactions(execution_case):
    case = execution_case
    entered = asyncio.Event()
    release = asyncio.Event()
    original = case.dispatch.side_effect

    async def paused_dispatch(arguments):
        entered.set()
        await release.wait()
        return await original(arguments)

    case.dispatch.side_effect = paused_dispatch
    first = asyncio.create_task(consume(case))
    second = None
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        second = asyncio.create_task(consume(case))
        release.set()
        outcomes = await asyncio.wait_for(asyncio.gather(first, second, return_exceptions=True), timeout=5)
    finally:
        release.set()
        for task in (first, second):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(*(task for task in (first, second) if task is not None), return_exceptions=True)
    assert any(isinstance(outcome, list) for outcome in outcomes)
    assert all(isinstance(outcome, list) or (
        isinstance(outcome, GuardError) and outcome.code == "execution_not_resumable"
    ) for outcome in outcomes)
    case.dispatch.assert_awaited_once()
    assert sum(isinstance(record, Claim) for record in case.records) == 1
    assert len(case.results) == 1


@pytest.mark.parametrize("state", ["rejected", "expired", "cancelled"])
async def test_terminal_negative_outcomes_continue_without_dispatch(execution_case, state):
    case = execution_case
    case.execution.state = case.approval.state = state
    error = {"rejected": "approval_rejected", "expired": "approval_timeout",
             "cancelled": "execution_cancelled"}[state]
    case.execution.result = {"accepted": False, "error": error}
    await consume(case)
    case.dispatch.assert_not_awaited()
    assert case.results == [{"accepted": False, "error": error}]
    assert not any(isinstance(record, Claim) for record in case.records)


@pytest.mark.parametrize("state", ["pending", "approved"])
@pytest.mark.parametrize("offset", [0, 1])
async def test_expiry_at_ttl_boundary_never_dispatches(execution_case, state, offset):
    case = execution_case
    case.execution.state = case.approval.state = state
    case.now = case.approval.expires_at + timedelta(seconds=offset)
    await consume(case)
    assert case.approval.state == "expired"
    assert case.results == [{"accepted": False, "error": "approval_timeout"}]
    case.dispatch.assert_not_awaited()


async def test_one_microsecond_before_ttl_can_dispatch(execution_case):
    case = execution_case
    case.now = case.approval.expires_at - timedelta(microseconds=1)
    await consume(case)
    case.dispatch.assert_awaited_once()


@pytest.mark.parametrize("phase", [1, 2, 3])
async def test_identity_revalidated_before_claim_and_before_dispatch(execution_case, phase):
    case = execution_case

    def change_identity(call):
        if call == phase:
            case.active = False

    case.before_identity = change_identity
    with pytest.raises(GuardError, match="identity_unavailable"):
        await consume(case)
    assert case.identity_calls == phase
    case.dispatch.assert_not_awaited()
    assert case.results == []
    assert case.execution.state == ("manual_review" if phase == 3 else "approved")
    assert native_write.get() is None


@pytest.mark.parametrize("phase", [1, 2, 3])
async def test_legacy_approved_memory_remains_authorized_after_admin_becomes_member(execution_case, phase):
    case = execution_case

    def demote(call):
        if call == phase:
            case.admin = False

    case.before_identity = demote
    await consume(case)
    case.dispatch.assert_awaited_once()
    assert case.execution.state == "completed"
    assert native_write.get() is None


async def test_expiry_between_claim_and_dispatch_fails_closed(execution_case):
    case = execution_case

    def expire_after_claim(call):
        if call == 3:
            case.now = case.approval.expires_at

    case.before_identity = expire_after_claim
    with pytest.raises(GuardError, match="approval_expired_or_cancelled"):
        await consume(case)
    case.dispatch.assert_not_awaited()
    assert case.execution.state == "manual_review"
    assert case.execution.result == {"accepted": False, "error": "execution_interrupted_no_retry"}


@pytest.mark.parametrize("failure", [RuntimeError("interrupted"), asyncio.CancelledError()])
async def test_interrupted_dispatch_is_not_retried_and_capability_is_reset(execution_case, failure):
    case = execution_case
    case.dispatch.side_effect = failure
    with pytest.raises(type(failure)):
        await consume(case)
    assert native_write.get() is None
    assert case.execution.state == "manual_review"
    assert case.execution.result == {"accepted": False, "error": "execution_interrupted_no_retry"}
    claim = next(record for record in case.records if isinstance(record, Claim))
    assert claim.state == "claimed"
    await consume(case)
    case.dispatch.assert_awaited_once()
    assert case.results == [{"accepted": False, "error": "execution_interrupted_no_retry"}]


@pytest.mark.parametrize("state", ["claimed", "executing", "resuming"])
async def test_persisted_inflight_state_cannot_be_replayed(execution_case, state):
    case = execution_case
    case.execution.state = state
    case.now = case.approval.expires_at + timedelta(days=1)
    await case.service.recover()
    assert case.execution.state == state
    assert case.execution.result is None
    with pytest.raises(GuardError, match="execution_not_resumable"):
        await consume(case)
    assert case.execution.state == state
    assert case.execution.result is None
    case.dispatch.assert_not_awaited()
    assert case.results == []


async def test_abandoned_claim_recovery_requires_manual_review_without_retry(execution_case):
    case = execution_case
    case.execution.state = "claimed"
    case.records.append(Claim(
        id=uuid4(), execution_id=case.execution.id, approval_id=case.approval.id,
        revision=1, payload_hash=case.revision.payload_hash,
        idempotency_key=digest(str(case.execution.id)), state="claimed", started_at=case.now,
    ))
    case.now = case.approval.expires_at + timedelta(days=1)
    await case.service.recover()
    case.dispatch.assert_not_awaited()
    assert case.execution.state == "manual_review"
    assert case.execution.result == {"accepted": False, "error": "execution_interrupted_no_retry"}


@pytest.mark.parametrize("tamper", ["payload", "policy"])
async def test_integrity_failure_does_not_claim_or_dispatch(execution_case, tamper):
    case = execution_case
    if tamper == "payload":
        case.revision.payload = {**case.revision.payload, "target": str(uuid4())}
    else:
        case.execution.policy_version = "obsolete"
    with pytest.raises(GuardError, match="payload_integrity_failed"):
        await consume(case)
    case.dispatch.assert_not_awaited()
    assert not any(isinstance(record, Claim) for record in case.records)