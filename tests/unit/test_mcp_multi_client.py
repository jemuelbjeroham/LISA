from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mcp.types import Tool

from lisa.mcp.client import MultiMCPClient


def server(*names):
    tools = [Tool(name=name, input_schema={"type": "object"}) for name in names]
    return SimpleNamespace(
        list_tools=AsyncMock(return_value=SimpleNamespace(tools=tools)),
        call_tool=AsyncMock(side_effect=lambda name, arguments: f"{name}:{arguments}"),
    )


async def test_lists_tools_from_every_server():
    client = MultiMCPClient({"lisa": server("web_search"), "tv": server("tv_get_status")})
    assert [tool.name for tool in (await client.list_tools()).tools] == ["web_search", "tv_get_status"]


async def test_routes_each_call_to_the_owning_server():
    lisa, tv = server("web_search"), server("tv_set_volume")
    client = MultiMCPClient({"lisa": lisa, "tv": tv})
    assert await client.call_tool("tv_set_volume", {"level": 5}) == "tv_set_volume:{'level': 5}"
    tv.call_tool.assert_awaited_once_with("tv_set_volume", {"level": 5})
    lisa.call_tool.assert_not_awaited()


async def test_unknown_tool_goes_to_first_server_whose_guard_rejects_it():
    lisa, tv = server("web_search"), server("tv_get_status")
    await MultiMCPClient({"lisa": lisa, "tv": tv}).call_tool("missing", {})
    lisa.call_tool.assert_awaited_once_with("missing", {})


async def test_duplicate_tool_names_are_rejected():
    client = MultiMCPClient({"lisa": server("health_check"), "tv": server("health_check")})
    with pytest.raises(ValueError, match="registered by both"):
        await client.list_tools()


def test_requires_at_least_one_server():
    with pytest.raises(ValueError):
        MultiMCPClient({})
