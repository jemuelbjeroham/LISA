"""Behaviour shared by LISA's three routed agents.

General Enquiry, Knowledge Assistant and Troubleshooting Agent all follow the same
moderate -> build context -> stream -> (handoff | tools | answer) loop, so every
test here runs against all three.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage

from lisa.agents import general_enquiry, knowledge_assistant, troubleshooting_agent
from lisa.agents.general_enquiry import GeneralEnquiry
from lisa.agents.knowledge_assistant import KnowledgeAssistant
from lisa.agents.troubleshooting_agent import TroubleshootingAgent
from lisa.guardrails.input_moderation import (
    InputModerationBlocked,
    InputModerationResult,
    InputModerationUnavailable,
)

AGENTS = [
    pytest.param(GeneralEnquiry, general_enquiry, id="general_enquiry"),
    pytest.param(KnowledgeAssistant, knowledge_assistant, id="knowledge_assistant"),
    pytest.param(TroubleshootingAgent, troubleshooting_agent, id="troubleshooting_agent"),
]


def streaming_model(*turns):
    """A chat model whose successive astream() calls yield the given chunk lists."""
    model = MagicMock()
    model.bind_tools.return_value = model
    calls = []
    pending = list(turns)

    async def astream(messages, *args, **kwargs):
        calls.append(messages)
        for chunk in pending.pop(0):
            yield chunk

    model.astream = astream
    model.calls = calls
    return model


def mcp_client(tool_names=()):
    tools = [SimpleNamespace(name=name, description=name, input_schema={"type": "object", "properties": {}})
             for name in tool_names]
    return SimpleNamespace(list_tools=AsyncMock(return_value=SimpleNamespace(tools=tools)), call_tool=AsyncMock())


def runtime(user_context="", memories=""):
    memory = SimpleNamespace(
        user_context_service=SimpleNamespace(build_context=AsyncMock(return_value=user_context)),
        context_service=SimpleNamespace(build=AsyncMock(return_value=memories)),
    )
    return SimpleNamespace(context=SimpleNamespace(user_id="user-1", memory=memory))


def allowed_moderator():
    return SimpleNamespace(moderate=AsyncMock(return_value=SimpleNamespace(flagged=False)))


@pytest.fixture
def events(monkeypatch):
    captured = []
    for module in (general_enquiry, knowledge_assistant, troubleshooting_agent):
        monkeypatch.setattr(module, "get_stream_writer", lambda: captured.append)
    return captured


def build(agent_cls, model, *, moderator=None, client=None):
    return agent_cls(
        model=model,
        fallback_model=model,
        thinking_model=model,
        system_prompt="You are LISA.",
        mcp_client=client or mcp_client(),
        input_moderator=moderator or allowed_moderator(),
    )


@pytest.mark.parametrize("agent_cls,module", AGENTS)
async def test_agent_streams_answer_and_returns_it(agent_cls, module, events):
    model = streaming_model([AIMessageChunk(content="Hello "), AIMessageChunk(content="there.")])
    agent = build(agent_cls, model)
    await agent.initialize_tools()

    result = await agent.run({"messages": [HumanMessage(content="Hi")]}, runtime())

    assert result["messages"][-1].content == "Hello there."
    assert [(e.type, e.content) for e in events] == [("content", "Hello "), ("content", "there.")]
    sent = model.calls[0]
    assert isinstance(sent[0], SystemMessage) and sent[0].content.startswith("You are LISA.")
    assert sent[-1].content.endswith("Hi")


@pytest.mark.parametrize("agent_cls,module", AGENTS)
async def test_agent_injects_user_context_and_memories_as_data(agent_cls, module, events):
    model = streaming_model([AIMessageChunk(content="ok")])
    agent = build(agent_cls, model)
    await agent.initialize_tools()

    await agent.run({"messages": [HumanMessage(content="Hi")]}, runtime("team=Playout", "prefers short answers"))

    system = model.calls[0][0].content
    assert "<user_context>\nteam=Playout\n</user_context>" in system
    assert "<retrieved_memories>\nprefers short answers\n</retrieved_memories>" in system


@pytest.mark.parametrize("agent_cls,module", AGENTS)
async def test_agent_hands_off_to_orchestrator(agent_cls, module, events):
    handoff = AIMessageChunk(content="", tool_call_chunks=[
        {"name": "handoff", "args": '{"reason": "Needs troubleshooting"}', "id": "h1", "index": 0}])
    agent = build(agent_cls, streaming_model([handoff]))
    await agent.initialize_tools()

    result = await agent.run({"messages": [HumanMessage(content="The feed is down")]}, runtime())

    assert result == {"agent_handoff": {"reason": "Needs troubleshooting"}}
    assert events == []


@pytest.mark.parametrize("agent_cls,module", AGENTS)
@pytest.mark.parametrize("failure", [InputModerationBlocked(InputModerationResult(flagged=True)), InputModerationUnavailable(), None])
async def test_agent_blocks_flagged_or_unmoderated_input_before_model(agent_cls, module, failure, events):
    if failure is None:
        moderator = SimpleNamespace(moderate=AsyncMock(return_value=SimpleNamespace(flagged=True, categories=["x"])))
    else:
        moderator = SimpleNamespace(moderate=AsyncMock(side_effect=failure))
    model = streaming_model()
    agent = build(agent_cls, model, moderator=moderator)
    await agent.initialize_tools()
    rt = runtime()

    result = await agent.run({"messages": [HumanMessage(content="bad")]}, rt)

    assert result["messages"][-1].content
    assert model.calls == []
    rt.context.memory.context_service.build.assert_not_awaited()


@pytest.mark.parametrize("agent_cls,module", AGENTS)
async def test_agent_refuses_to_run_with_uninitialized_tools(agent_cls, module, events):
    model = streaming_model()
    agent = build(agent_cls, model)

    result = await agent.run({"messages": [HumanMessage(content="Hi")]}, runtime())

    assert result["messages"][-1].content == module.SAFE_ERROR_MESSAGE
    assert model.calls == []


@pytest.mark.parametrize("agent_cls,module", AGENTS)
async def test_agent_binds_only_its_allowlisted_mcp_tools(agent_cls, module):
    client = mcp_client([*sorted(module.MCP_TOOL_ALLOWLIST), "delete_everything"])
    agent = build(agent_cls, streaming_model(), client=client)

    await agent.initialize_tools()

    assert set(agent.tool_schemas) == set(module.MCP_TOOL_ALLOWLIST)
    with pytest.raises(PermissionError):
        await agent._execute_mcp_tool("delete_everything", {})
    client.call_tool.assert_not_awaited()


@pytest.mark.parametrize("agent_cls,module", AGENTS)
async def test_agent_empty_conversation_is_handled(agent_cls, module, events):
    agent = build(agent_cls, streaming_model())
    result = await agent.run({"messages": []}, runtime())
    assert result["messages"][-1].content
