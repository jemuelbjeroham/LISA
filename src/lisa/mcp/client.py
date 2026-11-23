import os
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Self

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import ListToolsResult, Tool

from lisa.guardrails.hitl.dispatcher import guard_mcp


class MCPClient:
    def __init__(self, command: str, args: list[str], cwd: Path, env: dict[str,str]) -> None:
        self.server_params = StdioServerParameters(command=command, args=args, cwd=str(cwd), env=env or os.environ.copy())
        self.exit_stack = AsyncExitStack()
        self.session: ClientSession | None = None

    async def __aenter__(self) -> Self:
        read, write = await self.exit_stack.enter_async_context(
            stdio_client(self.server_params)
        )

        self.session = await self.exit_stack.enter_async_context(
            ClientSession(read, write)
        )

        await self.session.initialize()

        return self

    async def __aexit__(
            self,
            exc_type,
            exc_value,
            traceback,
    ) -> None:
        await self.exit_stack.aclose()

    async def list_tools(self):
        if self.session is None:
            raise RuntimeError("MCP client is not connected")

        return await self.session.list_tools()

    async def call_tool(self, name: str, arguments: dict):
        if self.session is None:
            raise RuntimeError("MCP client is not connected")

        return await guard_mcp(name, arguments, self.session.call_tool)
    

class MultiMCPClient:
    """Presents several connected MCP servers to the agents as one client.

    Tools from every server are listed together and each call goes to the
    server that registered the tool. A tool name registered by two servers is
    rejected, because a call could otherwise reach the wrong server. Each
    underlying MCPClient still applies the HITL guard to its calls.
    """

    def __init__(self, clients: dict[str, MCPClient]) -> None:
        if not clients:
            raise ValueError("at least one MCP client is required")
        self.clients = clients
        self.owners: dict[str, str] = {}

    async def list_tools(self) -> ListToolsResult:
        tools: list[Tool] = []
        owners: dict[str, str] = {}
        for server, client in self.clients.items():
            for tool in (await client.list_tools()).tools:
                if tool.name in owners:
                    raise ValueError(
                        f"MCP tool '{tool.name}' is registered by both "
                        f"'{owners[tool.name]}' and '{server}'"
                    )
                owners[tool.name] = server
                tools.append(tool)
        self.owners = owners
        return ListToolsResult(tools=tools)

    async def call_tool(self, name: str, arguments: dict):
        if name not in self.owners:
            await self.list_tools()
        # Unknown tools go to the first server, whose guard rejects them.
        server = self.owners.get(name, next(iter(self.clients)))
        return await self.clients[server].call_tool(name, arguments)
