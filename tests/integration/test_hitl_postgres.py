import asyncio
import importlib.util
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from io import StringIO
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from alembic.migration import MigrationContext
from alembic.operations import Operations

pytestmark = pytest.mark.skipif(
    os.environ.get("LISA_HITL_POSTGRES_TESTS") != "1",
    reason="Opt in with LISA_HITL_POSTGRES_TESTS=1; uses only a new network-none postgres:17 container",
)
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def postgres():
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker CLI unavailable; PostgreSQL behavior unverified")

    def command(*arguments, input=None, timeout=30):
        return subprocess.run([docker, *arguments], input=input, capture_output=True,
                              text=True, timeout=timeout, check=False)

    image = command("image", "inspect", "postgres:17")
    if image.returncode:
        pytest.skip("Local postgres:17 image unavailable; no pull attempted; PostgreSQL unverified")
    name = f"lisa-hitl-test-{uuid4().hex}"
    started = command(
        "run", "--detach", "--rm", "--pull=never", "--network=none", "--name", name,
        "--tmpfs", "/var/lib/postgresql/data:rw", "--env", "POSTGRES_HOST_AUTH_METHOD=trust",
        "--env", "POSTGRES_DB=hitl_test", "postgres:17",
    )
    assert started.returncode == 0, started.stderr
    container = started.stdout.strip()

    def sql(statement):
        return command(
            "exec", "--interactive", container, "psql", "--no-psqlrc", "--username=postgres",
            "--dbname=hitl_test", "--set=ON_ERROR_STOP=1", "--set=VERBOSITY=verbose",
            "--tuples-only", "--no-align", input=statement,
        )

    try:
        deadline = time.monotonic() + 30
        while True:
            ready = command("exec", container, "pg_isready", "--host=127.0.0.1", "--username=postgres", "--dbname=hitl_test")
            if ready.returncode == 0:
                break
            if time.monotonic() >= deadline:
                pytest.fail(f"Disposable PostgreSQL did not become ready: {ready.stderr}")
        spec = importlib.util.spec_from_file_location(
            "isolated_hitl_postgres_migration",
            ROOT / "migrations/versions/a41c9d72e830_add_hitl_approvals.py",
        )
        migration = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(migration)
        buffer = StringIO()
        migration.op = Operations(MigrationContext.configure(
            dialect_name="postgresql", dialect_opts={"paramstyle": "named"},
            opts={"as_sql": True, "output_buffer": buffer},
        ))
        migration.upgrade()
        migrated = sql(buffer.getvalue())
        assert migrated.returncode == 0, migrated.stderr
        sql.container = container
        yield sql
    finally:
        removed = command("rm", "--force", "--volumes", container)
        assert removed.returncode == 0, removed.stderr


@pytest.fixture
def rows(postgres):
    execution_id, approval_id, revision_id, event_id = [str(uuid4()) for _ in range(4)]
    user_id, session_id, conversation_id = [str(uuid4()) for _ in range(3)]
    result = postgres(f"""
        INSERT INTO hitl_executions
            (id, user_id, session_id, conversation_id, role, tool, operation, policy_version,
             logical_key, state, continuation, created_at, updated_at)
        VALUES ('{execution_id}', '{user_id}', '{session_id}', '{conversation_id}', 'admin',
                'save_memory', 'memory.create', 'hitl-v1', '{uuid4().hex}', 'approved', '{{}}', now(), now());
        INSERT INTO hitl_approvals (id, execution_id, revision, state, created_at, expires_at)
        VALUES ('{approval_id}', '{execution_id}', 1, 'approved', now(), now() + interval '10 minutes');
        INSERT INTO hitl_revisions (id, approval_id, number, payload, payload_hash, created_at)
        VALUES ('{revision_id}', '{approval_id}', 1, '{{"arguments":{{"content":"original"}}}}', 'original-hash', now());
        INSERT INTO hitl_events (id, execution_id, approval_id, actor_id, event, details, created_at)
        VALUES ('{event_id}', '{execution_id}', '{approval_id}', '{user_id}', 'approved', '{{}}', now());
    """)
    assert result.returncode == 0, result.stderr
    return {"execution": execution_id, "approval": approval_id, "revision": revision_id,
            "event": event_id, "user": user_id, "session": session_id, "conversation": conversation_id}


def claim_sql(rows, *, execution_id=None, approval_id=None, key=None):
    return f"""
        INSERT INTO hitl_claims
            (id, execution_id, approval_id, revision, payload_hash, idempotency_key, state, started_at)
        VALUES ('{uuid4()}', '{execution_id or rows['execution']}', '{approval_id or rows['approval']}',
                1, 'original-hash', '{key or uuid4().hex}', 'claimed', now());
    """


@pytest.mark.parametrize("table,identifier", [("hitl_revisions", "revision"), ("hitl_events", "event")])
@pytest.mark.parametrize("operation", ["update", "delete"])
def test_real_migration_rejects_revision_and_audit_mutation(postgres, rows, table, identifier, operation):
    before = postgres(f"SELECT row_to_json(record) FROM {table} AS record WHERE id = '{rows[identifier]}';")
    statement = (f"UPDATE {table} SET id = '{uuid4()}'" if operation == "update" else f"DELETE FROM {table}")
    rejected = postgres(f"{statement} WHERE id = '{rows[identifier]}';")
    assert rejected.returncode != 0
    assert "55000" in rejected.stderr
    assert "immutable" in rejected.stderr
    after = postgres(f"SELECT row_to_json(record) FROM {table} AS record WHERE id = '{rows[identifier]}';")
    assert before.returncode == after.returncode == 0
    assert before.stdout == after.stdout


def test_revision_append_preserves_original_and_rejects_duplicate_number(postgres, rows):
    original = postgres(f"SELECT payload::text, payload_hash FROM hitl_revisions WHERE id = '{rows['revision']}';")
    appended_id = str(uuid4())
    append = f"""
        INSERT INTO hitl_revisions (id, approval_id, number, payload, payload_hash, created_at)
        VALUES ('{appended_id}', '{rows['approval']}', 2, '{{"arguments":{{"content":"edited"}}}}', 'edited-hash', now());
    """
    assert postgres(append).returncode == 0
    duplicate = postgres(append.replace(appended_id, str(uuid4())).replace("2,", "1,"))
    assert duplicate.returncode != 0
    assert "23505" in duplicate.stderr
    assert "uq_hitl_revision" in duplicate.stderr
    unchanged = postgres(f"SELECT payload::text, payload_hash FROM hitl_revisions WHERE id = '{rows['revision']}';")
    assert unchanged.stdout == original.stdout


@pytest.mark.parametrize("dimension", ["execution", "approval", "idempotency_key"])
def test_claim_constraints_are_independently_unique(postgres, rows, dimension):
    other = {"execution": str(uuid4()), "approval": str(uuid4())}
    seeded = postgres(f"""
        INSERT INTO hitl_executions
        SELECT '{other['execution']}'::uuid, user_id, session_id, '{uuid4()}'::uuid, role, tool,
               operation, policy_version, '{uuid4().hex}', state, continuation, result, created_at, updated_at
        FROM hitl_executions WHERE id = '{rows['execution']}';
        INSERT INTO hitl_approvals (id, execution_id, revision, state, created_at, expires_at)
        VALUES ('{other['approval']}', '{other['execution']}', 1, 'approved', now(), now() + interval '10 minutes');
    """)
    assert seeded.returncode == 0, seeded.stderr
    key = uuid4().hex
    assert postgres(claim_sql(rows, key=key)).returncode == 0
    duplicate = postgres(claim_sql(
        rows, execution_id=rows["execution"] if dimension == "execution" else other["execution"],
        approval_id=rows["approval"] if dimension == "approval" else other["approval"],
        key=key if dimension == "idempotency_key" else uuid4().hex,
    ))
    assert duplicate.returncode != 0
    assert "23505" in duplicate.stderr
    column = dimension if dimension == "idempotency_key" else f"{dimension}_id"
    assert f"hitl_claims_{column}_key" in duplicate.stderr


def test_concurrent_transactions_create_exactly_one_claim(postgres, rows):
    barrier = Barrier(2)

    def compete():
        barrier.wait(timeout=10)
        return postgres(claim_sql(rows))

    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [workers.submit(compete) for _ in range(2)]
        results = [future.result(timeout=30) for future in futures]
    assert sorted(result.returncode == 0 for result in results) == [False, True]
    failure = next(result for result in results if result.returncode)
    assert "23505" in failure.stderr
    count = postgres(f"SELECT count(*) FROM hitl_claims WHERE execution_id = '{rows['execution']}';")
    assert count.returncode == 0
    assert count.stdout.strip() == "1"


@pytest_asyncio.fixture
async def service_case(request):
    if os.environ.get("LISA_HITL_POSTGRES_WORKER") != "1":
        postgres = request.getfixturevalue("postgres")
        database_name = f"hitl_service_{uuid4().hex}"
        created = postgres(f'CREATE DATABASE "{database_name}";')
        assert created.returncode == 0, created.stderr
        image = "lisa-cicd-lisa-api:latest"
        inspected = subprocess.run(
            ["docker", "image", "inspect", image], capture_output=True, text=True, check=False,
        )
        if inspected.returncode:
            pytest.skip(f"Local {image} unavailable; no pull attempted")
        worker_name = f"lisa-hitl-worker-{uuid4().hex}"
        try:
            completed = subprocess.run([
                "docker", "run", "--rm", "--pull=never", "--name", worker_name,
                "--network", f"container:{postgres.container}", "--read-only",
                "--tmpfs", "/tmp:rw", "--mount", f"type=bind,source={ROOT},target=/lisa,readonly",
                "--env", "PYTHONPATH=/lisa/src", "--env", "LISA_HITL_POSTGRES_TESTS=1",
                "--env", "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
                "--env", "LISA_HITL_POSTGRES_WORKER=1",
                "--env", f"LISA_HITL_SERVICE_DATABASE={database_name}",
                "--entrypoint", "/app/.venv/bin/python", image,
                "-m", "pytest", f"/lisa/tests/integration/test_hitl_postgres.py::{request.node.name}",
                "-q", "-p", "pytest_asyncio.plugin", "-p", "no:cacheprovider",
            ], capture_output=True, text=True, timeout=180, check=False)
            assert completed.returncode == 0, completed.stdout + completed.stderr
        finally:
            subprocess.run(["docker", "rm", "--force", "--volumes", worker_name],
                           capture_output=True, text=True, check=False)
        yield None
        return

    from lisa.guardrails.hitl.context import Principal
    from lisa.guardrails.hitl.service import ApprovalService
    from lisa.infrastructure.postgres.database import Database
    from lisa.infrastructure.postgres.memory_dependencies import memory_dependencies
    from lisa.infrastructure.postgres.models import Base
    from lisa.memory.user_context.repository_models import (
        UserProfileModel,
        UserSessionModel,
    )

    database_name = os.environ["LISA_HITL_SERVICE_DATABASE"]
    assert database_name.startswith("hitl_service_")
    assert len(database_name.removeprefix("hitl_service_")) == 32
    UUID(hex=database_name.removeprefix("hitl_service_"))
    database = Database(SimpleNamespace(
        database_url=f"postgresql+asyncpg://postgres:hitl-test-only@127.0.0.1/{database_name}",
    ))
    case = SimpleNamespace(
        database=database, actor=Principal(uuid4(), uuid4(), "admin"),
        now=datetime.now(UTC), conversation_id=uuid4(), continuations=[],
    )
    try:
        async with database.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with database.session() as session:
            session.add(UserProfileModel(
                user_id=case.actor.user_id, display_name="Isolated HITL test",
                email=f"{case.actor.user_id}@example.invalid", is_active=True, is_admin=True,
                details={}, created_at=case.now, updated_at=case.now,
            ))
            await session.flush()
            session.add(UserSessionModel(
                id=case.actor.session_id, user_id=case.actor.user_id,
                token_hash=uuid4().hex + uuid4().hex, created_at=case.now,
                expires_at=case.now + timedelta(hours=1),
            ))
            assert memory_dependencies(session).service is not None

        async def continuation_runner(continuation, result, actor, conversation_id):
            assert actor == case.actor
            assert conversation_id == case.conversation_id
            case.continuations.append((continuation, result))
            for event in ():
                yield event

        case.service = ApprovalService(database, clock=lambda: case.now)
        case.service.continuation_runner = continuation_runner
        yield case
    finally:
        await database.close()


async def propose_memory(case):
    from dataclasses import replace

    from lisa.guardrails.hitl.policy import REGISTRY

    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(REGISTRY, "save_memory", replace(
            REGISTRY["save_memory"], write=True, member=False,
            editable=("content", "memory_type", "scope"),
        ))
        return await _propose_legacy_memory(case)


async def _propose_legacy_memory(case):
    return await case.service.propose(
        case.actor, case.conversation_id, "save_memory",
        {"content": "Original isolated memory", "memory_type": "fact", "scope": "user"},
        uuid4().hex, {"route": "general_enquiry"},
    )


async def approve_memory(case, proposal):
    return await case.service.decide(
        case.actor, UUID(proposal["approval_id"]), proposal["revision"],
        proposal["arguments_hash"], "approve", uuid4().hex,
    )


async def resume_memory(case, proposal, service=None):
    return [event async for event in (service or case.service).resume(
        case.actor, UUID(proposal["execution_id"]),
    )]


async def persisted_service_rows(case, proposal):
    from sqlalchemy import select

    from lisa.guardrails.hitl.models import (
        Approval,
        AuditEvent,
        Claim,
        Execution,
        Revision,
    )
    from lisa.infrastructure.postgres.models import MemoryModel

    execution_id = UUID(proposal["execution_id"])
    approval_id = UUID(proposal["approval_id"])
    async with case.database.session() as session:
        return SimpleNamespace(
            execution=await session.get(Execution, execution_id),
            approval=await session.get(Approval, approval_id),
            revisions=(await session.scalars(select(Revision).where(
                Revision.approval_id == approval_id).order_by(Revision.number))).all(),
            claims=(await session.scalars(select(Claim).where(Claim.execution_id == execution_id))).all(),
            events=(await session.scalars(select(AuditEvent).where(
                AuditEvent.execution_id == execution_id))).all(),
            memories=(await session.scalars(select(MemoryModel).where(
                MemoryModel.user_id == case.actor.user_id))).all(),
        )


@pytest.mark.asyncio
async def test_service_postgres_propose_edit_decide_and_native_memory(service_case):
    if service_case is None:
        return
    from dataclasses import replace

    from lisa.guardrails.hitl.policy import REGISTRY, GuardError
    from lisa.guardrails.hitl.service import digest

    case = service_case
    proposal = await propose_memory(case)
    assert proposal["status"] == proposal["execution_status"] == "pending"
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(REGISTRY, "save_memory", replace(
            REGISTRY["save_memory"], editable=("content", "memory_type", "scope"),
        ))
        revised = await case.service.edit(
            case.actor, UUID(proposal["approval_id"]), 1, proposal["arguments_hash"],
            {"content": "Edited isolated memory"},
        )
    assert revised["revision"] == 2
    assert revised["arguments_hash"] != proposal["arguments_hash"]
    with pytest.raises(GuardError, match="stale_revision"):
        await approve_memory(case, proposal)
    approved = await approve_memory(case, revised)
    assert approved["status"] == "approved"
    before = await persisted_service_rows(case, revised)
    assert before.memories == before.claims == []
    events = await resume_memory(case, revised)
    assert [event.type for event in events] == ["tool_execution_status", "execution_completed"]
    assert [event.type for event in await resume_memory(case, revised)] == ["execution_completed"]
    rows = await persisted_service_rows(case, revised)
    assert rows.execution.state == "completed"
    assert rows.execution.continuation["resumed"] is True
    assert rows.approval.state == "approved"
    assert rows.approval.decision_actor == case.actor.user_id
    assert [revision.number for revision in rows.revisions] == [1, 2]
    assert rows.revisions[0].payload["arguments"] == proposal["arguments"]
    assert all(digest(revision.payload) == revision.payload_hash for revision in rows.revisions)
    assert len(rows.memories) == len(rows.claims) == len(case.continuations) == 1
    memory, claim = rows.memories[0], rows.claims[0]
    assert (memory.content, memory.scope, memory.type, memory.source) == (
        "Edited isolated memory", "user", "fact", "explicit",
    )
    assert memory.user_id == case.actor.user_id
    assert claim.state == "succeeded" and claim.completed_at is not None
    assert claim.approval_id == rows.approval.id and claim.revision == 2
    assert claim.payload_hash == revised["arguments_hash"]
    assert claim.result == rows.execution.result == {
        "accepted": True, "result": "Memory saved successfully: Edited isolated memory",
    }
    assert case.continuations[0][1] == {
        **claim.result, "approved_arguments": revised["arguments"],
    }
    assert sorted(event.event for event in rows.events) == sorted([
        "approval_created", "proposal_revised", "approved", "execution_claimed",
        "execution_succeeded", "continuation_completed",
    ])
    claimed = next(event for event in rows.events if event.event == "execution_claimed")
    assert claimed.details["revision"] == 2
    assert claimed.details["payload_hash"] == claim.payload_hash
    succeeded = next(event for event in rows.events if event.event == "execution_succeeded")
    assert succeeded.details["claim_id"] == str(claim.id)


@pytest.mark.asyncio
async def test_service_postgres_concurrent_resume_at_most_once(service_case):
    if service_case is None:
        return
    from lisa.guardrails.hitl.policy import GuardError
    from lisa.guardrails.hitl.service import ApprovalService

    case = service_case
    proposal = await approve_memory(case, await propose_memory(case))
    competitor = ApprovalService(case.database, clock=lambda: case.now)
    competitor.continuation_runner = case.service.continuation_runner
    barrier = asyncio.Barrier(2)

    async def compete(service):
        await barrier.wait()
        return await resume_memory(case, proposal, service)

    outcomes = await asyncio.wait_for(asyncio.gather(
        compete(case.service), compete(competitor), return_exceptions=True,
    ), timeout=30)
    assert any(isinstance(outcome, list) for outcome in outcomes)
    assert all(isinstance(outcome, list) or (
        isinstance(outcome, GuardError) and outcome.code == "execution_not_resumable"
    ) for outcome in outcomes), outcomes
    await resume_memory(case, proposal)
    rows = await persisted_service_rows(case, proposal)
    assert len(rows.memories) == len(rows.claims) == len(case.continuations) == 1
    assert rows.memories[0].content == proposal["arguments"]["content"]
    assert rows.execution.state == "completed"
    claim = rows.claims[0]
    assert claim.state == "succeeded" and claim.completed_at is not None
    assert claim.result == rows.execution.result == {
        "accepted": True, "result": "Memory saved successfully: Original isolated memory",
    }
    for event_name in ("execution_claimed", "execution_succeeded", "continuation_completed"):
        assert sum(event.event == event_name for event in rows.events) == 1
    succeeded = next(event for event in rows.events if event.event == "execution_succeeded")
    assert succeeded.details["claim_id"] == str(claim.id)
    assert claim.payload_hash == proposal["arguments_hash"]


@pytest.mark.asyncio
async def test_service_postgres_revoked_session_blocks_native_memory(service_case):
    if service_case is None:
        return
    from lisa.guardrails.hitl.policy import GuardError
    from lisa.memory.user_context.repository_models import UserSessionModel

    case = service_case
    proposal = await approve_memory(case, await propose_memory(case))
    async with case.database.session() as session:
        identity = await session.get(UserSessionModel, case.actor.session_id)
        identity.revoked_at = case.now
    with pytest.raises(GuardError, match="identity_unavailable"):
        await resume_memory(case, proposal)
    rows = await persisted_service_rows(case, proposal)
    assert rows.execution.state == rows.approval.state == "approved"
    assert rows.memories == rows.claims == case.continuations == []
    assert not any(event.event.startswith("execution_") for event in rows.events)


@pytest.mark.asyncio
async def test_service_postgres_stale_recovery_with_and_without_claim_never_retries(service_case):
    if service_case is None:
        return
    from lisa.guardrails.hitl.models import Claim, Execution

    case = service_case
    for state in ("resuming", "claimed"):
        proposal = await propose_memory(case)
        if state == "claimed":
            proposal = await approve_memory(case, proposal)
        else:
            proposal = await case.service.decide(
                case.actor, UUID(proposal["approval_id"]), proposal["revision"],
                proposal["arguments_hash"], "reject", uuid4().hex,
            )
        stale_at = case.now - timedelta(minutes=11)
        claim_id = uuid4() if state == "claimed" else None
        async with case.database.session() as session:
            execution = await session.get(Execution, UUID(proposal["execution_id"]))
            execution.state = state
            execution.updated_at = stale_at
            if claim_id is not None:
                session.add(Claim(
                    id=claim_id, execution_id=execution.id,
                    approval_id=UUID(proposal["approval_id"]), revision=proposal["revision"],
                    payload_hash=proposal["arguments_hash"], idempotency_key=uuid4().hex,
                    state="claimed", started_at=stale_at,
                ))
            else:
                assert execution.result == {"accepted": False, "error": "approval_rejected"}

        await case.service.recover()
        for recovery_pass in range(2):
            rows = await persisted_service_rows(case, proposal)
            assert rows.execution.state == "manual_review"
            assert rows.execution.result == {
                "accepted": False, "error": "execution_interrupted_no_retry",
            }
            assert rows.execution.updated_at == case.now
            assert not rows.execution.continuation.get("resumed")
            assert rows.approval.state == ("approved" if claim_id else "rejected")
            assert rows.memories == case.continuations == []
            assert len(rows.claims) == (1 if claim_id else 0)
            if claim_id is not None:
                claim = rows.claims[0]
                assert claim.id == claim_id and claim.state == "claimed"
                assert claim.started_at == stale_at
                assert claim.completed_at is None and claim.result is None
                assert claim.payload_hash == proposal["arguments_hash"]
            recovered = [event for event in rows.events if event.event == "manual_review"]
            assert len(recovered) == 1
            assert recovered[0].details["claim_id"] == (str(claim_id) if claim_id else None)
            assert not any(event.event in (
                "execution_claimed", "execution_succeeded", "continuation_completed",
            ) for event in rows.events)
            if recovery_pass == 0:
                await case.service.recover()


@pytest.mark.parametrize("approved", [False, True])
@pytest.mark.asyncio
async def test_service_postgres_expiry_blocks_native_memory(service_case, approved):
    if service_case is None:
        return
    from lisa.guardrails.hitl.policy import GuardError

    case = service_case
    proposal = await propose_memory(case)
    if approved:
        proposal = await approve_memory(case, proposal)
    case.now = datetime.fromisoformat(proposal["expires_at"])
    with pytest.raises(GuardError, match="approval_expired"):
        await approve_memory(case, proposal)
    expired = await persisted_service_rows(case, proposal)
    assert expired.execution.state == expired.approval.state == "expired"
    assert expired.execution.result == {"accepted": False, "error": "approval_timeout"}
    assert [event.event for event in expired.events].count("expired") == 1
    await resume_memory(case, proposal)
    await resume_memory(case, proposal)
    rows = await persisted_service_rows(case, proposal)
    assert rows.memories == rows.claims == []
    assert rows.approval.state == "expired" and rows.execution.state == "completed"
    assert len(case.continuations) == 1
    assert case.continuations[0][1] == {"accepted": False, "error": "approval_timeout"}