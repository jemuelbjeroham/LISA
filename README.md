# LISA — Level 1 Intelligent System and Assistant

LISA is the agent engine behind the LISA assistant: an AI assistant that supports
a Level 1 Engineering team with knowledge lookup, troubleshooting, ticket
operations and general questions. This repository contains the orchestration
graph, the specialist agents, prompts, long-term memory, human-in-the-loop (HITL)
approval guardrails and database migrations.

LISA is a Python library. It is served over HTTP by
[`lisa-api`](https://github.com/jemuelbjeroham/lisa-api), which embeds it, and it
reaches external tools through [`lisa-mcp-server`](https://github.com/jemuelbjeroham/lisa-mcp-server).

---

## Table of contents

- [System overview](#system-overview)
- [How a request flows](#how-a-request-flows)
- [Features](#features)
- [Project layout](#project-layout)
- [Requirements](#requirements)
- [Getting started](#getting-started)
- [Configuration](#configuration)
- [Database migrations](#database-migrations)
- [Human-in-the-loop approvals](#human-in-the-loop-approvals)
- [Prompts](#prompts)
- [Testing and evaluation](#testing-and-evaluation)
- [Continuous integration](#continuous-integration)
- [Security notes](#security-notes)

---

## System overview

LISA is one of seven repositories that are developed side by side in a common
parent directory:

| Repository | Role |
| --- | --- |
| **LISA** (this repo) | Agent engine: routing, agents, prompts, memory, HITL guardrails, migrations |
| [`lisa-api`](https://github.com/jemuelbjeroham/lisa-api) | FastAPI service exposing LISA: auth, chat streaming, conversations, approvals, admin |
| [`lisa-ui`](https://github.com/jemuelbjeroham/lisa-ui) | Next.js web client: chat workspace, approval panels, settings, user admin |
| [`lisa-mcp-server`](https://github.com/jemuelbjeroham/lisa-mcp-server) | MCP server providing LISA's tools (knowledge search, web search, Jira) |
| [`net-tools-app`](https://github.com/jemuelbjeroham/net-tools-app) | FastAPI tools backend: knowledge-base retrieval and Jira integration |
| [`net-ops-kb-rag`](https://github.com/jemuelbjeroham/net-ops-kb-rag) | Ingestion library and CLIs that build the vector knowledge base |
| [`lisa-cicd`](https://github.com/jemuelbjeroham/lisa-cicd) | Docker Compose deployment and reusable CI workflows |

```text
 Browser
    │
    ▼
 lisa-ui (Next.js, :3000) ──cookie session──►  lisa-api (FastAPI, :8008)
                                                   │  embeds LISA (this repo)
                                                   │  ├─ PostgreSQL: users, sessions, memories, approvals
                                                   │  └─ Redis: conversation state
                                                   ▼  spawns over stdio
                                              lisa-mcp-server (MCP)
                                                   │
                         ┌─────────────────────────┴───────────────┐
                         ▼                                         ▼
                net-tools-app (FastAPI, :8000)              Tavily web search
                 ├─ knowledge retrieval (Chroma + reranker)
                 └─ Jira (search; writes only after approval)
                         ▲
                net-ops-kb-rag builds the Chroma knowledge base
```

## How a request flows

1. **Input moderation.** The user message is screened with OpenAI's moderation
   model before any agent runs.
2. **Routing.** The `Orchestrator` picks one of three routes. It asks the
   [TypeSafe](https://typesafe.ai) **Jev** decision model first (when
   `TYPESAFE_API_KEY` is set) and falls back to an LLM router when Jev is
   unavailable or unsure. Follow-up turns can continue on the active route
   without re-routing.
3. **Specialist agent.** The chosen agent answers, calling MCP tools from its
   own allowlist:

   | Route | Purpose | Tools |
   | --- | --- | --- |
   | `general_enquiry` | General questions, explanations, everyday help | Every registered MCP tool, plus `save_memory` and `handoff` (still subject to the HITL policy) |
   | `knowledge_assistant` | Answers grounded in internal documentation | `search_knowledge`, `web_search`, `web_search_with_content`, `handoff` |
   | `troubleshooting_agent` | Diagnosing and resolving operational issues | Knowledge tools plus `search_jira_tickets` |

4. **Handoffs.** An agent that decides the request belongs elsewhere calls
   `handoff`; the graph returns to the orchestrator. Handoffs are capped at three
   per request to prevent loops.
5. **Guarded writes.** Write tools (Jira create/comment/close, memory saves) are
   intercepted by the HITL guardrail and turned into approval proposals instead
   of executing. The turn suspends until a person decides (see
   [Human-in-the-loop approvals](#human-in-the-loop-approvals)).
6. **Memory.** After the turn, the memory workflow decides whether anything the
   user said is worth remembering (facts, preferences, instructions, project
   context) and de-duplicates it before storing.
7. **Streaming.** Responses stream as typed events (`content`, `reasoning`,
   `approval_required`, `approval_updated`, `tool_execution_status`,
   `execution_suspended`, `execution_completed`).

## Features

- **LangGraph orchestration** with an explicit routing policy, bounded handoffs and
  suspend/resume support for approvals.
- **Hybrid routing**: calibrated Jev decision model first, LLM router fallback.
- **Model resilience**: primary model with automatic fallback (default OpenAI
  `gpt-5-nano`, fallback NVIDIA Nemotron), configurable per role.
- **Per-agent tool allowlists** enforced in code, not just in prompts.
- **Long-term memory** in PostgreSQL with temporal validity and duplicate detection.
- **User context**: profiles, preferences and roles (`admin` / `member`).
- **HITL approvals** with immutable revisions, payload hashing, expiry,
  claims and crash recovery.
- **Observability** via [Langfuse](https://langfuse.com) (optional).
- **Evaluations** with DeepEval / RAGAS golden sets.

## Project layout

```text
src/lisa/
├── application.py          # LISA facade: startup, chat(), stream_chat()
├── graph.py                # LangGraph wiring: orchestrator → agents → handoff
├── orchestrator.py         # Route selection (Jev first, LLM fallback)
├── jev_router.py           # TypeSafe Jev routing questions
├── routing.py              # Route enum and routing policy
├── state.py / context.py   # Graph state and per-request context
├── config.py               # Pydantic settings (environment variables)
├── model.py                # Model factory (OpenAI / NVIDIA)
├── model_resilience.py     # Primary → fallback model handling
├── agents/                 # general_enquiry, knowledge_assistant, troubleshooting_agent,
│                           # memory agent, tool allowlists
├── guardrails/
│   ├── input_moderation.py # OpenAI moderation gate
│   └── hitl/               # Approval policy, service, Jira adapter, models, continuation
├── memory/                 # Long-term memory and user context (repositories, services)
├── conversation/           # Conversation stores (Redis, in-memory) and serialization
├── infrastructure/postgres # Async SQLAlchemy database access
├── mcp/client.py           # MCP client used to call lisa-mcp-server
├── retrieval/              # Retrieval planning
├── streaming/events.py     # Typed stream events
├── observability.py        # Langfuse integration
└── prompts/                # Versioned prompt files (see Prompts)
migrations/                 # Alembic migrations
evals/                      # DeepEval / RAGAS evaluation suites and golden data
scripts/                    # Manual diagnostics against live models (not run in CI)
tests/                      # unit, conversation, integration and infrastructure tests
```

## Requirements

- Python **3.14+** and [uv](https://docs.astral.sh/uv/)
- PostgreSQL 17 (users, sessions, memories, approvals)
- Redis 7 (conversation state; used by `lisa-api`)
- A sibling checkout of `lisa-mcp-server` (LISA starts it as a subprocess)
- API keys: OpenAI (required), Tavily (required by the web-search tool),
  NVIDIA (fallback model), TypeSafe (optional, enables Jev routing)

## Getting started

LISA is normally run through `lisa-api`. For local development, clone the
repositories into one parent directory:

```sh
mkdir lisa-workspace && cd lisa-workspace
for repo in LISA lisa-api lisa-ui lisa-mcp-server net-tools-app net-ops-kb-rag lisa-cicd; do
  git clone "https://github.com/jemuelbjeroham/${repo}.git"
done
```

Install LISA's dependencies and run the unit tests:

```sh
cd LISA
uv sync
uv run pytest tests/unit
```

Create a `.env` file in the directory you run from (it is git-ignored) with at
least the required settings below, apply the migrations, then start `lisa-api`
as described in its README. For a full stack in containers, use
[`lisa-cicd`](https://github.com/jemuelbjeroham/lisa-cicd).

## Configuration

Settings are read from environment variables and from a local `.env` file by
[`lisa.config.Settings`](src/lisa/config.py). Fields without an explicit alias
use the `LISA_` prefix (for example `model_name` → `LISA_MODEL_NAME`).

### Required

| Variable | Description |
| --- | --- |
| `OPENAI_API_KEY` | OpenAI key for the primary model, router, moderation and memory agent |
| `TAVILY_API_KEY` | Tavily key, passed to the MCP web-search tools |
| `LISA_DATABASE_URL` | Async SQLAlchemy URL, e.g. `postgresql+asyncpg://lisa:<password>@localhost:5432/lisa` |

### Models

| Variable | Default | Description |
| --- | --- | --- |
| `LISA_MODEL_PROVIDER` / `LISA_MODEL_NAME` | `openai` / `gpt-5-nano` | Primary agent model |
| `LISA_FALLBACK_MODEL_PROVIDER` / `LISA_FALLBACK_MODEL_NAME` | `nvidia` / `nvidia/nemotron-3-ultra-550b-a55b` | Fallback model |
| `NVIDIA_API_KEY` | — | Required when a provider is `nvidia` |
| `LISA_ROUTER_MODEL_PROVIDER` / `LISA_ROUTER_MODEL_NAME` | `openai` / `gpt-5-nano` | LLM router fallback |
| `LISA_TECHNICAL_MODEL_PROVIDER` / `LISA_TECHNICAL_MODEL_NAME` | `openai` / `gpt-5-nano` | Technical clarification model |
| `TYPESAFE_API_KEY` | — | Enables Jev routing; without it the LLM router is used |
| `LISA_JEV_MODEL_NAME` | `jev-1.14.0` | Pinned Jev version (routing thresholds are tuned against it) |
| `LISA_JEV_TIMEOUT_SECONDS` | `10.0` | Jev request timeout |

### MCP server

| Variable | Default | Description |
| --- | --- | --- |
| `LISA_MCP_SERVER_COMMAND` | `uv` | Command that starts `lisa-mcp-server` |
| `LISA_MCP_SERVER_ARGS` | `["run","python","-m","lisa_mcp_server.server"]` | JSON list of arguments |
| `LISA_MCP_SERVER_CWD` | `../lisa-mcp-server` | Working directory for the MCP server |

### Jira approvals (optional)

| Variable | Description |
| --- | --- |
| `NET_TOOLS_API_URL` | Base URL of `net-tools-app`, e.g. `http://net-tools-app:8000` |
| `NET_TOOLS_SERVICE_TOKEN` | Shared secret LISA presents to `net-tools-app` prepare/execute endpoints |
| `LISA_JIRA_SIGNING_KEY` | At least 32 random characters; signs short-lived execution capabilities |

If these are missing, Jira write proposals fail closed; read-only tools still work.

### Observability (optional)

| Variable | Default | Description |
| --- | --- | --- |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | — | Leave blank to disable tracing |
| `LANGFUSE_BASE_URL` | `http://localhost:3001` | Langfuse endpoint |

## Database migrations

Schema changes are managed with Alembic. The URL comes from `LISA_DATABASE_URL`.

```sh
uv run alembic upgrade head      # apply all migrations
uv run alembic current           # show the applied revision
uv run alembic revision -m "describe change" --autogenerate
```

Migrations cover memories (with temporal validity), user profiles and
preferences, local authentication, admin roles and the HITL approval tables.
In Docker, `lisa-cicd` runs them with the one-off `db-migrate` service.

## Human-in-the-loop approvals

Write operations never run directly from a model's tool call. The policy registry
in [`guardrails/hitl/policy.py`](src/lisa/guardrails/hitl/policy.py) declares
every tool, its operation, whether it writes, whether members may use it, and
which fields an approver may edit.

| Tool | Write | Who | Editable fields |
| --- | --- | --- | --- |
| `web_search`, `web_search_with_content` | No | Members and admins | — |
| `search_knowledge`, `health_check`, `search_jira_tickets` | No | Admins | — |
| `create_jira_ticket` | Yes | Admins | `subject`, `description`, `project_key`, `issue_type`, `priority` |
| `update_jira_ticket` | Yes | Admins | `comment` |
| `close_ticket` | Yes | Admins | — |
| `save_memory` (native) | Yes | Members and admins | — |

The lifecycle for a write:

1. The agent calls a write tool. LISA creates an **execution** and a **pending
   approval** (expires after 10 minutes). For Jira, `net-tools-app` prepares the
   exact request first, and that frozen payload is what gets approved.
2. The user sees the proposal in `lisa-ui`, and can **edit** allowed fields
   (creating a new revision with a new payload hash), **approve**, **reject** or
   **cancel**.
3. Approval records a decision only. The user must explicitly **resume**
   ("Run approved action") for the operation to execute.
4. For Jira, LISA issues a signed, single-use, 90-second capability.
   `net-tools-app` calls back to `lisa-api` to consume it, and LISA re-checks
   identity, role, expiry, revision and payload hash before the write is allowed.
5. Rejected, cancelled or expired proposals resume the conversation with a
   negative tool result. Uncertain failures move to `manual_review`.

## Prompts

Prompts live in [`src/lisa/prompts`](src/lisa/prompts) as versioned text files and
are loaded with `lisa.prompts.loader.load_prompt`. The active set is:

| Prompt | Used by |
| --- | --- |
| `Identity.txt` | Shared identity for every agent |
| `orchestrator/routing_level1_engineering_v2.txt` | LLM router |
| `orchestrator/jev_routing_v1.json` | Jev routing questions and background |
| `general_enquiry/general_enquiry_v2.txt` | General enquiry agent |
| `knowledge_assistant/knowledge_assistant_v1.txt` | Knowledge assistant |
| `troubleshooting_agent/troubleshooting_agent_v1.txt` | Troubleshooting agent |
| `memory/memory_agent_v2.txt` | Memory agent |

When changing behaviour, add a new versioned file rather than editing an old one
in place, and update the loader call in `application.py`.
`tests/unit/prompts` checks that the application loads exactly the expected
prompt set.

## Testing and evaluation

```sh
uv run pytest tests/unit                 # fast, offline unit tests
uv run pytest tests/conversation         # conversation store and serialization
uv run ruff check src tests              # lint
```

PostgreSQL integration tests for HITL are opt-in and start a disposable,
network-isolated `postgres:17` container:

```sh
LISA_HITL_POSTGRES_TESTS=1 uv run pytest tests/integration
```

The test in `tests/unit/guardrails/hitl/test_service.py` that exercises the real
Jira preparation code is skipped unless the sibling `net-tools-app` package is
importable.

Model-quality evaluations live in [`evals/`](evals) and call real models; run
them deliberately, not in CI:

```sh
uv run pytest evals/tests
```

Scripts in [`scripts/`](scripts) are manual diagnostics against live providers.

## Continuous integration

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) calls the shared
`python-ci.yml` workflow from `lisa-cicd` (pinned to a commit SHA) on every pull
request and on pushes to `main`. It installs the locked dependencies with uv and
builds the package on Python 3.14. Lint and test steps are currently disabled in
the shared workflow, so run them locally before pushing.

## Security notes

- Never commit `.env` files or API keys; `.env` is git-ignored.
- Roles and ownership come from the authenticated session, never from model
  output or request bodies.
- Tool access is enforced twice: per-agent allowlists and the HITL policy registry.
- Approval payloads are hashed; any change after approval is rejected.
- Treat retrieved documents and Jira ticket text as untrusted input in prompts.
