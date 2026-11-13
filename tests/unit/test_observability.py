from unittest.mock import MagicMock, Mock

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableLambda

from lisa.observability import LISAObservability, create_langfuse_handler


class RecordingCallback(BaseCallbackHandler):
    def __init__(self):
        self.starts = 0

    def on_chain_start(self, serialized, inputs, **kwargs):
        self.starts += 1


class AsyncAgent:
    async def run(self, value):
        return await RunnableLambda(lambda item: item + 1).ainvoke(value)


class SyncAgent:
    def route(self, value):
        return RunnableLambda(lambda item: item + 1).invoke(value)


@pytest.mark.asyncio
async def test_async_agent_traces_direct_calls_and_child_runnables():
    client = MagicMock()
    handler = RecordingCallback()
    agent = AsyncAgent()
    observability = LISAObservability(client)
    observability.instrument(agent, "run", handler)

    assert await agent.run(4) == 5
    assert handler.starts == 1
    client.start_as_current_observation.assert_called_once_with(
        as_type="span", name="lisa.agent.AsyncAgent.run", input=None
    )


def test_sync_agent_traces_router_and_child_runnables():
    client = MagicMock()
    handler = RecordingCallback()
    agent = SyncAgent()
    LISAObservability(client).instrument(agent, "route", handler)

    assert agent.route(4) == 5
    assert handler.starts == 1


@pytest.mark.asyncio
async def test_disabled_tracing_leaves_agent_unchanged():
    agent = AsyncAgent()
    original = agent.run
    LISAObservability(None).instrument(agent, "run", RecordingCallback())
    assert agent.run == original
    assert await agent.run(4) == 5


@pytest.mark.asyncio
async def test_nested_graph_callback_not_duplicated_or_leaked():
    handler = RecordingCallback()
    agent = AsyncAgent()
    LISAObservability(MagicMock()).instrument(agent, "run", handler)
    assert await RunnableLambda(agent.run).ainvoke(1, config={"callbacks": [handler]}) == 2
    assert handler.starts == 2
    await RunnableLambda(lambda value: value).ainvoke(1)
    assert handler.starts == 2


def test_handler_uses_configured_project_key(monkeypatch):
    factory = Mock()
    monkeypatch.setattr("lisa.observability.CallbackHandler", factory)
    assert create_langfuse_handler("test-public-key") is factory.return_value
    factory.assert_called_once_with(public_key="test-public-key")


@pytest.mark.asyncio
async def test_streaming_continuation_inherits_callbacks():
    class Continuation:
        async def resume(self):
            yield await RunnableLambda(lambda value: value + 1).ainvoke(1)

    handler = RecordingCallback()
    continuation = Continuation()
    LISAObservability(MagicMock()).instrument(continuation, "resume", handler)
    assert [event async for event in continuation.resume()] == [2]
    assert handler.starts == 1
    await RunnableLambda(lambda value: value).ainvoke(1)
    assert handler.starts == 1