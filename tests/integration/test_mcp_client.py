from pathlib import Path

import pytest

from lisa.mcp.client import MCPClient


@pytest.mark.anyio
async def test_mcp_client_initializes_session():
    mcp_server_path = Path(__file__).resolve().parents[2] / ".." / "lisa-mcp-server"

    async with MCPClient(
        command="uv",
        args=[
            "run",
            "python",
            "-m",
            "lisa_mcp_server.server",
        ],
        cwd=mcp_server_path.resolve(),
    ) as client:
        assert client.session is not None
        tools = await client.list_tools()
        tool_names = {tool.name for tool in tools.tools}

        assert "health_check" in tool_names
        assert "search_knowledge" in tool_names

        result = await client.call_tool(
            "search_knowledge",
            {
                "query": "How do I troubleshoot firewall connectivity",
                "top_k": 5
            }
        )

        print("\nMCP RESULT:")
        print(result)
        assert not result.is_error