import hashlib
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from lisa.guardrails.hitl.context import Principal
from lisa.guardrails.hitl.models import Approval, AuditEvent, Claim, Execution, Revision
from lisa.guardrails.hitl.policy import REGISTRY, GuardError
from lisa.guardrails.hitl.service import ApprovalService, digest


@pytest.fixture
def lifecycle(monkeypatch):
    records = []
    actor = Principal(uuid4(), uuid4(), "admin")
    clock = SimpleNamespace(now=datetime(2026, 10, 26, 12, tzinfo=UTC))
    session = MagicMock()
    session.add.side_effect = records.append
    session.flush = AsyncMock()
    session.execute = AsyncMock(return_value=MagicMock(all=list))

    async def execute(statement):
        entities = [description["entity"] for description in statement.column_descriptions]
        if statement._for_update_arg is None or not statement._for_update_arg.skip_locked:
            return session.execute.return_value
        if entities == [Execution, Approval]:
            rows = [(execution, approval) for execution in records if isinstance(execution, Execution)
                    for approval in records if isinstance(approval, Approval)
                    and approval.execution_id == execution.id
                    and approval.state in {"pending", "approved"}
                    and approval.expires_at <= clock.now]
        elif entities == [Execution, Claim]:
            rows = [(execution, claim) for execution in records if isinstance(execution, Execution)
                    and execution.state in {"claimed", "executing", "resuming"}
                    and execution.updated_at <= clock.now - timedelta(minutes=10)
                    for claim in records if isinstance(claim, Claim)
                    and claim.execution_id == execution.id]
        else:
            return session.execute.return_value
        return SimpleNamespace(all=lambda: rows)

    session.execute.side_effect = execute

    async def scalar(statement):
        entity = statement.column_descriptions[0].get("entity")
        if entity is Revision:
            return next(record for record in reversed(records) if isinstance(record, Revision))
        return None

    session.scalar = AsyncMock(side_effect=scalar)

    @asynccontextmanager
    async def database_session():
        yield session

    service = ApprovalService(SimpleNamespace(session=database_session), clock=lambda: clock.now)
    service.identity = AsyncMock(return_value=actor)
    return SimpleNamespace(service=service, session=session, records=records, actor=actor, monkeypatch=monkeypatch,
                           clock=clock, conversation_id=uuid4())


async def propose(harness):
    harness.monkeypatch.setitem(REGISTRY, "save_memory", replace(
        REGISTRY["save_memory"], write=True, member=False,
        editable=("content", "memory_type", "scope"),
    ))
    return await harness.service.propose(
        harness.actor, harness.conversation_id, "save_memory",
        {"content": " remember ", "memory_type": "fact", "scope": "user"},
        "turn-1/tool-1", {"route": "general_enquiry"},
    )


def bind_owned(harness):
    execution = next(record for record in harness.records if isinstance(record, Execution))
    approval = next(record for record in harness.records if isinstance(record, Approval))
    harness.service.owned = AsyncMock(return_value=(execution, approval))
    return execution, approval


@pytest.mark.parametrize("name,arguments", [
    ("create_jira_ticket", {"subject": "Alarm", "description": "Loss", "core": "1", "side": "x"}),
    ("update_jira_ticket", {"ticket_key": "OPS-123", "comment": "Recovered"}),
    ("close_ticket", {"ticket_id": "123"}),
])
async def test_jira_proposal_is_pending_with_effective_payload_and_no_write(lifecycle, name, arguments):
    effective = {"tool_name": name, "arguments": arguments, "destination": "https://jira.example",
        "ticket_key": "OPS-123", "payload": {"resolved": "original"}}
    lifecycle.service.jira.prepare = AsyncMock(return_value=effective)
    lifecycle.service.jira.execute = AsyncMock()
    result = await lifecycle.service.propose(lifecycle.actor, lifecycle.conversation_id, name, arguments, "jira-turn", {})
    assert result["status"] == "pending"
    assert result["effective_operation"] == effective
    revision = next(record for record in lifecycle.records if isinstance(record, Revision))
    assert revision.payload["effective_operation"] == effective
    assert revision.payload_hash == digest(revision.payload)
    lifecycle.service.jira.execute.assert_not_awaited()


async def test_jira_edit_reprepares_and_invalidates_old_revision(lifecycle):
    arguments = {"ticket_key": "OPS-123", "comment": "First"}

    async def prepare(name, values):
        return {"tool_name": name, "arguments": values, "destination": "https://jira.example",
            "ticket_key": "OPS-123", "payload": {"comment": values["comment"]}}

    lifecycle.service.jira.prepare = AsyncMock(side_effect=prepare)
    first = await lifecycle.service.propose(lifecycle.actor, lifecycle.conversation_id, "update_jira_ticket", arguments, "jira-turn", {})
    _, approval = bind_owned(lifecycle)
    revised = await lifecycle.service.edit(lifecycle.actor, approval.id, 1, first["arguments_hash"], {"comment": "Edited"})
    assert revised["revision"] == 2
    assert revised["effective_operation"]["payload"] == {"comment": "Edited"}
    assert revised["arguments_hash"] != first["arguments_hash"]
    assert lifecycle.service.jira.prepare.await_count == 2
    with pytest.raises(GuardError, match="stale_revision"):
        await lifecycle.service.decide(lifecycle.actor, approval.id, 1, first["arguments_hash"], "approve", "old")
    assert (await lifecycle.service.decide(lifecycle.actor, approval.id, 2, revised["arguments_hash"], "approve", "new"))["status"] == "approved"


async def test_create_core_edit_reprepares_department_and_disallows_department_edit(lifecycle):
    # Cross-repo test: needs the sibling net-tools-app package installed.
    pytest.importorskip("net_tools_app")
    from net_tools_app.jira.service import JiraService
    from net_tools_app.onair_finder.service import OnAirResult

    finder = SimpleNamespace(find=AsyncMock(side_effect=lambda request: OnAirResult(
        core=request.core, side=request.side, found=True, jira_core=f"X{request.core}g")))
    jira = JiraService(SimpleNamespace(base_url="https://jira.example"))

    async def prepare(name, values):
        return await jira.prepare(name, values, finder)

    lifecycle.service.jira.prepare = AsyncMock(side_effect=prepare)
    arguments = {"subject": "Alarm", "description": "Loss", "core": "1", "side": "x"}
    first = await lifecycle.service.propose(lifecycle.actor, lifecycle.conversation_id, "create_jira_ticket", arguments, "jira-turn", {})
    _, approval = bind_owned(lifecycle)
    assert "department" not in first["editable_fields"]
    assert first["effective_operation"]["payload"]["requestFieldValues"]["customfield_10483"] == [{"value": "DC NOC"}]
    with pytest.raises(GuardError):
        await lifecycle.service.edit(lifecycle.actor, approval.id, 1, first["arguments_hash"], {"department": "DNOC"})
    revised = await lifecycle.service.edit(lifecycle.actor, approval.id, 1, first["arguments_hash"], {"core": "3"})
    assert revised["arguments"]["core"] == "3"
    assert revised["effective_operation"]["payload"]["requestFieldValues"]["customfield_10483"] == [{"value": "Stamford Ops"}]


@pytest.mark.parametrize("name", ["create_jira_ticket", "update_jira_ticket", "close_ticket"])
async def test_member_cannot_prepare_jira(lifecycle, name):
    lifecycle.service.identity.return_value = Principal(lifecycle.actor.user_id, lifecycle.actor.session_id, "member")
    lifecycle.service.jira.prepare = AsyncMock()
    with pytest.raises(GuardError, match="not_authorized"):
        await lifecycle.service.propose(lifecycle.actor, lifecycle.conversation_id, name, {}, "member", {})
    lifecycle.service.jira.prepare.assert_not_awaited()


def test_digest_is_canonical_and_known_sha256():
    assert digest({"second": 2, "first": {"b": 2, "a": 1}}) == digest(
        {"first": {"a": 1, "b": 2}, "second": 2},
    )
    assert digest({"a": 1}) == hashlib.sha256(b'{"a":1}').hexdigest()
    assert digest({"a": 1}) != digest({"a": "1"})
    assert digest({"a": [1, 2]}) != digest({"a": [2, 1]})
    assert digest({"text": "\u00e9"}) == hashlib.sha256(b'{"text":"\\u00e9"}').hexdigest()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_digest_rejects_non_json_numbers(value):
    with pytest.raises(ValueError):
        digest({"value": value})


@pytest.mark.parametrize("role", ["admin", "member"])
async def test_current_policy_refuses_memory_approval_without_creating_execution(lifecycle, role):
    lifecycle.service.identity.return_value = Principal(lifecycle.actor.user_id, lifecycle.actor.session_id, role)
    with pytest.raises(GuardError, match="approval_not_required"):
        await lifecycle.service.propose(lifecycle.actor, lifecycle.conversation_id, "save_memory",
            {"content": "Remember", "memory_type": "fact", "scope": "user"}, "turn", {})
    assert not any(isinstance(record, (Execution, Approval, Revision)) for record in lifecycle.records)


async def test_propose_binds_identity_payload_and_ten_minute_utc_ttl(lifecycle):
    result = await propose(lifecycle)
    execution, approval = bind_owned(lifecycle)
    revision = next(record for record in lifecycle.records if isinstance(record, Revision))
    assert execution.state == approval.state == result["status"] == "pending"
    assert approval.created_at == lifecycle.clock.now
    assert approval.expires_at == lifecycle.clock.now + timedelta(minutes=10)
    assert datetime.fromisoformat(result["expires_at"]).utcoffset() == timedelta(0)
    assert revision.payload_hash == digest(revision.payload) == result["arguments_hash"]
    assert revision.payload["user_id"] == revision.payload["target"] == str(lifecycle.actor.user_id)
    assert revision.payload["conversation_id"] == str(lifecycle.conversation_id)
    assert revision.payload["execution_id"] == str(execution.id)
    assert revision.payload["approval_id"] == str(approval.id)
    assert revision.payload["arguments"]["content"] == "remember"
    assert revision.payload["revision"] == approval.revision == 1
    assert {record.event for record in lifecycle.records if isinstance(record, AuditEvent)} == {
        "authorization_allowed", "approval_created",
    }


async def test_duplicate_proposal_returns_existing_without_new_revision(lifecycle):
    first = await propose(lifecycle)
    execution, approval = bind_owned(lifecycle)
    revision = next(record for record in lifecycle.records if isinstance(record, Revision))
    lifecycle.session.scalar = AsyncMock(side_effect=[None, execution, approval, revision])
    second = await propose(lifecycle)
    assert second == first
    assert sum(isinstance(record, Revision) for record in lifecycle.records) == 1


async def test_active_conversation_blocks_new_proposal(lifecycle):
    lifecycle.session.scalar = AsyncMock(side_effect=[None, None, uuid4()])
    with pytest.raises(GuardError, match="execution_pending_cancel_required"):
        await propose(lifecycle)
    assert not any(isinstance(record, Execution) for record in lifecycle.records)


@pytest.mark.parametrize("name,error", [
    ("web_search", "approval_not_required"),
    ("create_jira_ticket", "jira_signing_unavailable"),
    ("update_jira_ticket", "jira_signing_unavailable"),
    ("close_ticket", "jira_signing_unavailable"),
])
async def test_unsupported_proposals_do_not_create_executions(lifecycle, name, error):
    with pytest.raises(GuardError, match=error):
        await lifecycle.service.propose(lifecycle.actor, uuid4(), name, {}, "logical", {})
    assert not any(isinstance(record, Execution) for record in lifecycle.records)


async def test_edit_appends_revision_without_mutating_original_or_extending_ttl(lifecycle):
    first = await propose(lifecycle)
    execution, approval = bind_owned(lifecycle)
    original = next(record for record in lifecycle.records if isinstance(record, Revision))
    original_payload, original_hash = dict(original.payload), original.payload_hash
    expires_at = approval.expires_at
    lifecycle.clock.now += timedelta(minutes=1)
    edited = await lifecycle.service.edit(lifecycle.actor, approval.id, 1, first["arguments_hash"],
                                          {"content": "revised"})
    assert original.payload == original_payload
    assert original.payload_hash == original_hash
    assert approval.expires_at == expires_at
    assert edited["revision"] == 2
    assert edited["arguments_hash"] != first["arguments_hash"]
    assert edited["arguments"]["content"] == "revised"
    assert execution.state == "pending"
    assert sum(isinstance(record, Revision) for record in lifecycle.records) == 2


@pytest.mark.parametrize("revision_number,payload_hash", [(0, None), (1, "wrong")])
async def test_stale_edits_and_decisions_fail_closed(lifecycle, revision_number, payload_hash):
    first = await propose(lifecycle)
    execution, approval = bind_owned(lifecycle)
    payload_hash = payload_hash or first["arguments_hash"]
    with pytest.raises(GuardError, match="stale_revision"):
        await lifecycle.service.edit(lifecycle.actor, approval.id, revision_number, payload_hash,
                                     {"content": "revised"})
    with pytest.raises(GuardError, match="stale_revision"):
        await lifecycle.service.decide(lifecycle.actor, approval.id, revision_number, payload_hash,
                                       "approve", "decision-1")
    assert execution.state == approval.state == "pending"


@pytest.mark.parametrize("changes", [{}, {"target": "other"}, {"tool": "close_ticket"}, {"revision": 5}])
async def test_edit_rejects_noneditable_fields(lifecycle, changes):
    first = await propose(lifecycle)
    _, approval = bind_owned(lifecycle)
    with pytest.raises(GuardError, match="invalid_edit"):
        await lifecycle.service.edit(lifecycle.actor, approval.id, 1, first["arguments_hash"], changes)
    assert approval.revision == 1


@pytest.mark.parametrize("decision,state", [("approve", "approved"), ("reject", "rejected")])
async def test_decision_is_repeatable_only_with_same_key(lifecycle, decision, state):
    first = await propose(lifecycle)
    execution, approval = bind_owned(lifecycle)
    arguments = (lifecycle.actor, approval.id, 1, first["arguments_hash"], decision)
    decided = await lifecycle.service.decide(*arguments, "decision-1")
    assert execution.state == approval.state == decided["status"] == state
    assert approval.decision_actor == lifecycle.actor.user_id
    assert approval.decision_at == lifecycle.clock.now
    assert await lifecycle.service.decide(*arguments, "decision-1") == decided
    with pytest.raises(GuardError, match="decision_conflict"):
        await lifecycle.service.decide(*arguments, "different-key")


@pytest.mark.parametrize("offset_seconds", [0, 1])
async def test_decision_rejects_at_and_after_expiry_using_injected_clock(lifecycle, offset_seconds):
    first = await propose(lifecycle)
    _, approval = bind_owned(lifecycle)
    lifecycle.clock.now = approval.expires_at + timedelta(seconds=offset_seconds)
    with pytest.raises(GuardError, match="approval_expired"):
        await lifecycle.service.decide(lifecycle.actor, approval.id, 1, first["arguments_hash"],
                                       "approve", "decision-1")


@pytest.mark.parametrize("state", ["pending", "approved"])
async def test_recovery_expires_rows_returned_by_expiry_query(lifecycle, state):
    await propose(lifecycle)
    execution, approval = bind_owned(lifecycle)
    execution.state = approval.state = state
    lifecycle.clock.now = approval.expires_at
    await lifecycle.service.recover()
    assert execution.state == approval.state == "expired"
    assert execution.result == {"accepted": False, "error": "approval_timeout"}
    event = lifecycle.records[-1]
    assert isinstance(event, AuditEvent)
    assert event.event == "expired"
    assert event.created_at == lifecycle.clock.now
    statement = lifecycle.session.execute.call_args_list[-2].args[0]
    assert [description["entity"] for description in statement.column_descriptions] == [Execution, Approval]
    assert lifecycle.clock.now in statement.compile().params.values()
    assert statement._for_update_arg.skip_locked


@pytest.mark.parametrize("state", ["claimed", "executing", "completed", "cancelled"])
async def test_recovery_does_not_expire_nonpending_executions(lifecycle, state):
    await propose(lifecycle)
    execution, approval = bind_owned(lifecycle)
    execution.state = state
    lifecycle.clock.now = approval.expires_at
    await lifecycle.service.recover()
    assert execution.state == state
    assert approval.state == "pending"


@pytest.mark.parametrize("state", ["claimed", "executing", "resuming", "completed", "cancelled"])
@pytest.mark.parametrize("age", [timedelta(minutes=10) - timedelta(microseconds=1),
                                 timedelta(minutes=10), timedelta(minutes=10, microseconds=1)])
@pytest.mark.parametrize("has_claim", [False, True])
async def test_recovery_only_marks_stale_inflight_executions_with_claims(lifecycle, state, age, has_claim):
    await propose(lifecycle)
    execution, approval = bind_owned(lifecycle)
    execution.state = state
    execution.updated_at = lifecycle.clock.now - age
    claim = Claim(
        id=uuid4(), execution_id=execution.id, approval_id=approval.id,
        revision=approval.revision, payload_hash=next(
            record.payload_hash for record in lifecycle.records if isinstance(record, Revision)),
        idempotency_key=digest(str(execution.id)), state="claimed", started_at=execution.updated_at,
    )
    if has_claim:
        lifecycle.records.append(claim)
    await lifecycle.service.recover()
    statement = lifecycle.session.execute.call_args.args[0]
    assert [description["entity"] for description in statement.column_descriptions] == [Execution, Claim]
    cutoff = lifecycle.clock.now - timedelta(minutes=10)
    assert any(criterion.compare(Execution.updated_at <= cutoff) for criterion in statement._where_criteria)
    assert any(criterion.compare(Execution.state.in_(["claimed", "executing", "resuming"]))
               for criterion in statement._where_criteria)
    assert statement.get_final_froms()[0].onclause.compare(Claim.execution_id == Execution.id)
    assert statement._for_update_arg.skip_locked
    events = [record for record in lifecycle.records
              if isinstance(record, AuditEvent) and record.event == "manual_review"]
    if has_claim and state in {"claimed", "executing", "resuming"} and age >= timedelta(minutes=10):
        assert execution.state == "manual_review"
        assert execution.result == {"accepted": False, "error": "execution_interrupted_no_retry"}
        assert execution.updated_at == lifecycle.clock.now
        assert len(events) == 1
        assert events[0].execution_id == execution.id
        assert events[0].details["claim_id"] == str(claim.id)
        assert events[0].created_at == lifecycle.clock.now
    else:
        assert execution.state == state
        assert execution.result is None
        assert execution.updated_at == lifecycle.clock.now - age
        assert events == []
    assert approval.state == "pending"


async def test_cancel_pending_execution_without_dispatch(lifecycle):
    await propose(lifecycle)
    execution, approval = bind_owned(lifecycle)
    lifecycle.session.scalar = AsyncMock(side_effect=[execution, approval,
        next(record for record in lifecycle.records if isinstance(record, Revision))])
    result = await lifecycle.service.cancel(lifecycle.actor, execution.id)
    assert execution.state == approval.state == result["status"] == "cancelled"
    assert result["result"] == {"accepted": False, "error": "execution_cancelled"}


@pytest.mark.parametrize("state", ["claimed", "executing", "completed"])
async def test_cancel_rejects_claimed_execution(lifecycle, state):
    await propose(lifecycle)
    execution, approval = bind_owned(lifecycle)
    execution.state = state
    lifecycle.session.scalar = AsyncMock(side_effect=[execution, approval])
    with pytest.raises(GuardError, match="execution_already_claimed"):
        await lifecycle.service.cancel(lifecycle.actor, execution.id)
    assert execution.state == state


@pytest.mark.parametrize("is_admin,expected_role", [(False, "member"), (True, "admin")])
async def test_identity_uses_database_role_not_claimed_role(lifecycle, is_admin, expected_role):
    del lifecycle.service.identity
    lifecycle.session.execute.return_value = MagicMock(first=lambda: (SimpleNamespace(is_admin=is_admin), object()))
    actor = await lifecycle.service.identity(lifecycle.session, lifecycle.actor)
    assert actor.role == expected_role
    statement = lifecycle.session.execute.call_args.args[0]
    params = statement.compile().params
    assert lifecycle.clock.now in params.values()
    assert lifecycle.actor.session_id in params.values()
    assert lifecycle.actor.user_id in params.values()


async def test_unavailable_identity_fails_closed(lifecycle):
    del lifecycle.service.identity
    lifecycle.session.execute.return_value = MagicMock(first=lambda: None)
    with pytest.raises(GuardError, match="identity_unavailable") as caught:
        await lifecycle.service.check(lifecycle.actor, "save_memory")
    assert caught.value.status == 401
    assert not lifecycle.records


async def test_denial_records_audit_without_creating_execution(lifecycle):
    lifecycle.service.identity.return_value = Principal(lifecycle.actor.user_id, lifecycle.actor.session_id, "member")
    with pytest.raises(GuardError):
        await lifecycle.service.check(lifecycle.actor, "create_jira_ticket")
    assert len(lifecycle.records) == 1
    event = lifecycle.records[0]
    assert isinstance(event, AuditEvent)
    assert event.event == "authorization_denied"
    assert event.details["role"] == "member"


async def test_owned_query_binds_user_and_approval_and_locks_rows(lifecycle):
    approval_id = uuid4()
    row = (object(), object())
    lifecycle.session.execute.return_value = MagicMock(first=lambda: row)
    assert await lifecycle.service.owned(lifecycle.session, lifecycle.actor, approval_id) == row
    statement = lifecycle.session.execute.call_args.args[0]
    params = statement.compile().params
    assert lifecycle.actor.user_id in params.values()
    assert approval_id in params.values()
    assert statement._for_update_arg is not None


async def test_owned_query_does_not_disclose_missing_or_other_users_approval(lifecycle):
    lifecycle.session.execute.return_value = MagicMock(first=lambda: None)
    with pytest.raises(GuardError, match="not_found") as caught:
        await lifecycle.service.owned(lifecycle.session, lifecycle.actor, uuid4())
    assert caught.value.status == 404


async def test_edit_rechecks_ttl_when_recovery_did_not_return_row(lifecycle):
    first = await propose(lifecycle)
    _, approval = bind_owned(lifecycle)
    lifecycle.clock.now = approval.expires_at
    lifecycle.session.execute.side_effect = None
    with pytest.raises(GuardError, match="approval_expired|approval_not_pending"):
        await lifecycle.service.edit(lifecycle.actor, approval.id, 1, first["arguments_hash"],
                                     {"content": "too late"})