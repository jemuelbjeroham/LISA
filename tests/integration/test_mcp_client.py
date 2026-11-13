from pathlib import Path
from uuid import uuid4

import pytest

from lisa.guardrails.hitl.context import ExecutionContext, Principal, execution_scope
from lisa.guardrails.hitl.policy import authorize
from lisa.mcp.client import MCPClient


class RolePolicyService:
    """Applies the real role policy without the approval database."""

    async def check(self, principal, name):
        return authorize(principal.role, name)


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
        env={},
    ) as client:
        assert client.session is not None
        tools = await client.list_tools()
        tool_names = {tool.name for tool in tools.tools}

        assert "health_check" in tool_names
        assert "search_knowledge" in tool_names

        arguments = {"query": "How do I troubleshoot firewall connectivity", "top_k": 5}

        # Without an authenticated execution the guardrail must refuse the call.
        denied = await client.call_tool("search_knowledge", arguments)
        assert denied.is_error
        assert "identity_unavailable" in denied.content[0].text

        context = ExecutionContext(
            principal=Principal(user_id=uuid4(), session_id=uuid4(), role="admin"),
            conversation_id=uuid4(),
            service=RolePolicyService(),
        )
        with execution_scope(context):
            result = await client.call_tool("search_knowledge", arguments)

        print("\nMCP RESULT:")
        print(result)
        assert not result.is_error