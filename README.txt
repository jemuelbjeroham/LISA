LISA — Lightweight Intent/Agent Orchestrator

Summary:
- LISA is a small orchestration scaffold that builds a state-driven graph to route user messages to specialized agents.
- It uses `langchain-core` + `langchain-nvidia-ai-endpoints` for chat models and `langgraph` for state graph wiring.
- Current implemented pieces: model factory, orchestrator (routing), a technical clarification agent, a simple knowledge retrieval protocol, and a state graph that connects the orchestrator to the technical clarification agent.

Status (what's implemented):
- `Settings` (pydantic) to configure model provider and model name.
- `create_chat_model` returns a `ChatNVIDIA` instance when provider is `nvidia`.
- `Orchestrator` uses a structured output model to make routing decisions via a prompt.
- `TechnicalClarificationAgent` composes retrieved knowledge and user input to produce a model response.
- `build_graph` composes a `StateGraph` with a start -> orchestrator -> conditional routing -> technical clarification -> end flow.

Requirements:
- Python >= 3.14
- See `pyproject.toml` for exact dependency pins. Key deps include:
  - langchain-nvidia-ai-endpoints
  - langgraph
  - pydantic / pydantic-settings
  - pytest (for tests)

Configuration:
- Settings are read from environment variables prefixed with `LISA_` (see `lisa.config.Settings`).
  - Example vars: `LISA_MODEL_PROVIDER`, `LISA_MODEL_NAME`.
- When using NVIDIA endpoints you may also need provider-specific credentials (e.g. NVIDIA API keys) in your environment as required by the `langchain-nvidia-ai-endpoints` client.

Quick start (interactive):
- Install project (editable) and dependencies, then run tests:

  python -m pip install -e .
  pytest -q

- Example Python usage (simple):

  from lisa.application import LISA
  lisa = LISA()
  # `lisa.graph` is a compiled StateGraph instance; wire an initial state with messages
  state = {"messages": [], "route": None}
  # call into the graph/runner as appropriate for your chosen run-time (see `langgraph` usage)

Project layout (key files):
- src/lisa/config.py — `Settings` for env-configured model settings
- src/lisa/model.py — `create_chat_model` factory (NVIDIA currently supported)
- src/lisa/orchestrator.py — `Orchestrator` that returns routing decisions
- src/lisa/routing.py — `Route` enum and `RoutingDecision` structured model
- src/lisa/agents/technical_clarification.py — agent that enriches user queries with knowledge
- src/lisa/knowledge/protocol.py — `KnowledgeRetriever` protocol
- src/lisa/graph.py — builds the `StateGraph` that wires the orchestrator and agents

Tests:
- Unit and integration tests are present under `tests/` (run with `pytest`).

Next steps / TODOs:
- Add a concrete `KnowledgeRetriever` implementation.
- Wire a runnable CLI or ASGI app entrypoint for receiving user messages.
- Add more agents (general enquiries, issue reporting) and expand routing prompt coverage.

If you want, I can:
- Convert this into `README.md` (Markdown) and update `pyproject.toml:readme` if desired.
- Add example tests or a small CLI runner for manual testing.
