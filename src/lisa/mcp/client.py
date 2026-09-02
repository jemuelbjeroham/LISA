from contextlib import AsyncExitStack
from pathlib import Path
from typing import Self

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClient:
    def __init__(self, command: str, args: list[str], cwd: Path) -> None:
        self.server_params = StdioServerParameters(command=command, args=args, cwd=str(cwd))
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