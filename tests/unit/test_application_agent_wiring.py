from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from mcp.types import Tool

from lisa.agents.general_enquiry import GeneralEnquiry
from lisa.agents.knowledge_assistant import KnowledgeAssistant
from lisa.agents.troubleshooting_agent import TroubleshootingAgent
from lisa.agents.tv_controller import TVController
from lisa.application import LISA
from lisa.prompts.loader import load_prompt


async def test_application_initializes_the_routed_agents(monkeypatch):
    settings = SimpleNamespace(
        openai_token="test",
        model_provider="test",
        router_model_provider="test",
        mcp_server_command="test",
        mcp_server_args=[],
        mcp_server_cwd=None,
        tavily_api_key=None,
        typesafe_api_key=None,
    )
    monkeypatch.setattr("lisa.application.Settings", lambda: settings)
    monkeypatch.setattr("lisa.application.create_langfuse_client", lambda settings: None)
    monkeypatch.setattr("lisa.application.AsyncOpenAI", MagicMock())
    monkeypatch.setattr("lisa.application.OpenAIInputModerator", MagicMock())
    database = SimpleNamespace(close=AsyncMock())
    monkeypatch.setattr("lisa.application.Database", lambda settings: database)
    approvals = SimpleNamespace(recover=AsyncMock(), recovery_loop=AsyncMock())
    monkeypatch.setattr("lisa.application.ApprovalService", lambda database: approvals)
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.list_tools = AsyncMock(return_value=SimpleNamespace(tools=[]))
    monkeypatch.setattr("lisa.application.MCPClient", lambda **kwargs: client)

    model = MagicMock()
    model.bind_tools.return_value = model
    app = LISA(
        model=model,
        fallback_model=model,
        thinking_model=model,
        router_model=model,
        conversation_store=MagicMock(),
    )
    async with app:
        assert isinstance(app.general_enquiry_agent, GeneralEnquiry)
        assert isinstance(app.knowledge_assistant_agent, KnowledgeAssistant)
        assert isinstance(app.troubleshooting_agent, TroubleshootingAgent)
        assert isinstance(app.tv_controller_agent, TVController)
        identity = load_prompt("Identity.txt").rstrip() + "\n\n"
        for agent, prompt_path in (
            (app.general_enquiry_agent, "general_enquiry/general_enquiry_v2.txt"),
            (app.knowledge_assistant_agent, "knowledge_assistant/knowledge_assistant_v1.txt"),
            (app.troubleshooting_agent, "troubleshooting_agent/troubleshooting_agent_v1.txt"),
            (app.tv_controller_agent, "tv_controller/tv_controller_v1.txt"),
        ):
            assert agent.system_prompt == identity + load_prompt(prompt_path)
        assert not hasattr(app, "technical_clarification_agent")
        assert client.list_tools.await_count == 4
        assert {
            "general_enquiry", "knowledge_assistant", "troubleshooting_agent", "tv_controller",
        } <= set(app.graph.nodes)
        assert "technical_clarification" not in app.graph.nodes
        assert app.orchestrator.jev_router is None

async def test_application_wires_jev_router_with_versioned_prompt(monkeypatch):
    settings = SimpleNamespace(
        openai_token="test",
        model_provider="test",
        router_model_provider="test",
        mcp_server_command="test",
        mcp_server_args=[],
        mcp_server_cwd=None,
        tavily_api_key=None,
        typesafe_api_key=None,
    )
    monkeypatch.setattr("lisa.application.Settings", lambda: settings)
    monkeypatch.setattr("lisa.application.create_langfuse_client", lambda settings: None)
    monkeypatch.setattr("lisa.application.AsyncOpenAI", MagicMock())
    monkeypatch.setattr("lisa.application.OpenAIInputModerator", MagicMock())
    monkeypatch.setattr("lisa.application.Database", lambda settings: SimpleNamespace(close=AsyncMock()))
    approvals = SimpleNamespace(recover=AsyncMock(), recovery_loop=AsyncMock())
    monkeypatch.setattr("lisa.application.ApprovalService", lambda database: approvals)
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.list_tools = AsyncMock(return_value=SimpleNamespace(tools=[]))
    monkeypatch.setattr("lisa.application.MCPClient", lambda **kwargs: client)

    model = MagicMock()
    model.bind_tools.return_value = model
    jev_model = MagicMock()
    app = LISA(
        model=model,
        fallback_model=model,
        thinking_model=model,
        router_model=model,
        jev_model=jev_model,
        conversation_store=MagicMock(),
    )
    async with app:
        jev_router = app.orchestrator.jev_router
        assert jev_router.client is jev_model
        assert set(jev_router.questions) == {
            "wants_ticket",
            "wants_tv_control",
            "wants_problem_solved",
            "needs_internal_docs",
        }


def _app_settings(**overrides):
    return SimpleNamespace(
        openai_token="test", model_provider="test", router_model_provider="test",
        mcp_server_command="test", mcp_server_args=[], mcp_server_cwd=None,
        tavily_api_key=None, typesafe_api_key=None, **overrides,
    )


def _patch_app(monkeypatch, settings, clients):
    monkeypatch.setattr("lisa.application.Settings", lambda: settings)
    monkeypatch.setattr("lisa.application.create_langfuse_client", lambda settings: None)
    monkeypatch.setattr("lisa.application.AsyncOpenAI", MagicMock())
    monkeypatch.setattr("lisa.application.OpenAIInputModerator", MagicMock())
    monkeypatch.setattr("lisa.application.Database", lambda settings: SimpleNamespace(close=AsyncMock()))
    approvals = SimpleNamespace(recover=AsyncMock(), recovery_loop=AsyncMock())
    monkeypatch.setattr("lisa.application.ApprovalService", lambda database: approvals)
    factory = iter(clients)
    monkeypatch.setattr("lisa.application.MCPClient", lambda **kwargs: next(factory))


def _mcp_client(*tool_names, fail=False):
    client = MagicMock()
    client.__aenter__ = AsyncMock(side_effect=RuntimeError("spawn failed") if fail else None,
                                  return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    tools = [Tool(name=name, description=name, input_schema={"type": "object"}) for name in tool_names]
    client.list_tools = AsyncMock(return_value=SimpleNamespace(tools=tools))
    return client


def _lisa():
    model = MagicMock()
    model.bind_tools.return_value = model
    return LISA(model=model, fallback_model=model, thinking_model=model, router_model=model,
                conversation_store=MagicMock())


async def test_lg_mcp_server_tools_are_combined_with_lisa_tools(monkeypatch):
    from lisa.mcp.client import MultiMCPClient

    lisa_client, tv_client = _mcp_client("web_search"), _mcp_client("tv_get_status")
    settings = _app_settings(lg_mcp_server_enabled=True, lg_mcp_server_command="uv",
                             lg_mcp_server_args=["run", "lg-mcp-server"], lg_mcp_server_cwd=None)
    _patch_app(monkeypatch, settings, [lisa_client, tv_client])
    async with _lisa() as app:
        assert isinstance(app.mcp_client, MultiMCPClient)
        assert "tv_get_status" in app.troubleshooting_agent.tool_schemas
        assert "tv_get_status" in app.knowledge_assistant_agent.tool_schemas
        assert set(app.tv_controller_agent.tool_schemas) == {"tv_get_status", "web_search"}
    tv_client.__aexit__.assert_awaited()


async def test_lisa_starts_without_lg_mcp_server_when_it_fails(monkeypatch):
    lisa_client, tv_client = _mcp_client("web_search"), _mcp_client(fail=True)
    settings = _app_settings(lg_mcp_server_enabled=True, lg_mcp_server_command="uv",
                             lg_mcp_server_args=[], lg_mcp_server_cwd=None)
    _patch_app(monkeypatch, settings, [lisa_client, tv_client])
    async with _lisa() as app:
        assert app.mcp_client is lisa_client
        assert not any(name.startswith("tv_") for name in app.tv_controller_agent.tool_schemas)
    tv_client.__aexit__.assert_awaited_once()
