"""Live end-to-end check of LISA's three routed agents.

Runs the real application (real models, MCP server, Redis, Postgres, HITL
guardrails) for a temporary user and checks that each prompt is routed to the
expected agent and produces an answer. The temporary user, session and
conversations are removed afterwards (append-only HITL audit rows remain).

Usage (from the LISA directory):
    .venv/bin/python scripts/test_live_main_agents.py [--admin]
"""

import argparse
import asyncio
import hashlib
import json
import secrets
import sys
import time
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from redis.asyncio import Redis
from sqlalchemy import delete

from lisa.application import LISA
from lisa.config import Settings
from lisa.conversation.redis import RedisConversationStore
from lisa.guardrails.hitl.context import ExecutionContext, Principal, execution_scope
from lisa.infrastructure.postgres.database import Database
from lisa.memory.user_context.repository_models import (
    UserProfileModel,
    UserSessionModel,
)

NO_SIDE_EFFECTS = " Do not create tickets or save memories."
DEGRADED_PREFIXES = (
    "I'm unable to process this request right now",
    "I'm sorry, I couldn't generate a response",
    "The response was interrupted",
)

# (conversation index, expected route, prompt)
TURNS = [
    (0, "general_enquiry", "Hi! In one sentence, what kinds of things can you help me with?"),
    (1, "knowledge_assistant",
     "What does our internal documentation say about SkyPath? Summarise briefly and cite sources."
     + NO_SIDE_EFFECTS),
    (2, "troubleshooting_agent",
     "The Amagi playout feed for one of our channels is showing black video but audio continues. "
     "Help me troubleshoot it step by step." + NO_SIDE_EFFECTS),
    # Follow-up in the same conversation should stay with the troubleshooting agent.
    (2, "troubleshooting_agent", "I've confirmed the source feed looks fine. What should I check next?"
     + NO_SIDE_EFFECTS),
]


async def create_user(database: Database, admin: bool):
    user_id, session_id = uuid4(), uuid4()
    now = datetime.now(UTC)
    async with database.session() as session:
        session.add(UserProfileModel(
            user_id=user_id, email=f"lisa-live-test-{user_id.hex}@example.com",
            display_name="Temporary live validation", is_active=True, is_admin=admin,
            details={}, created_at=now, updated_at=now,
        ))
        await session.flush()
        session.add(UserSessionModel(
            id=session_id, user_id=user_id,
            token_hash=hashlib.sha256(secrets.token_bytes(32)).hexdigest(),
            created_at=now, expires_at=now + timedelta(hours=1),
        ))
    return user_id, session_id


async def main(admin: bool) -> int:
    settings = Settings()
    database = Database(settings)
    redis = Redis(host="localhost", port=6379, decode_responses=True)
    store = RedisConversationStore(redis, key_prefix="test:lisa-live:conversation:")
    user_id, session_id = await create_user(database, admin)
    conversation_ids = [uuid4() for _ in range(3)]
    principal = Principal(user_id, session_id, "admin" if admin else "member")
    report = {"role": principal.role, "turns": []}
    failures = 0

    try:
        async with LISA(conversation_store=store) as app:
            for index, expected, prompt in TURNS:
                conversation_id = conversation_ids[index]
                started = time.monotonic()
                answer, event_types, error = [], [], None
                try:
                    with execution_scope(ExecutionContext(principal, conversation_id, app.hitl)):
                        async for event in app.stream_chat(conversation_id, prompt, user_id):
                            event_types.append(event.type)
                            if event.type == "content":
                                answer.append(event.content)
                except Exception as exc:  # noqa: BLE001 - report every turn, not just the first failure
                    error = f"{type(exc).__name__}: {exc}"

                state = await store.get(user_id, conversation_id) or {}
                route = state.get("active_route") or state.get("route")
                route = getattr(route, "value", route)
                text = "".join(answer).strip()
                # Agents degrade to a canned message on model failure; that is not a pass.
                degraded = text.startswith(DEGRADED_PREFIXES)
                ok = error is None and route == expected and bool(text) and not degraded
                failures += not ok
                turn = {
                    "ok": ok, "expected_route": expected, "route": route,
                    "seconds": round(time.monotonic() - started, 1),
                    "event_types": sorted(set(event_types)), "error": error,
                    "prompt": prompt, "answer": text,
                }
                report["turns"].append(turn)
                print(json.dumps(turn, indent=2), flush=True)
    finally:
        for conversation_id in conversation_ids:
            await store.delete(user_id, conversation_id)
        await redis.delete(store._index_key(user_id))
        async with database.session() as session:
            await session.execute(delete(UserProfileModel).where(UserProfileModel.user_id == user_id))
        await redis.aclose()
        await database.close()

    passed = len(TURNS) - failures
    print(f"\nLIVE RESULT ({principal.role}): {passed}/{len(TURNS)} turns passed; temporary user removed.")
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--admin", action="store_true", help="run as an admin instead of a member")
    sys.exit(asyncio.run(main(parser.parse_args().admin)))
