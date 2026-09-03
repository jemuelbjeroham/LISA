import json

from lisa.mcp.client import MCPClient


class MCPKnowledgeRetriever:
    def __init__(self, client: MCPClient, top_k: int = 5):
        self.client = client
        self.top_k = top_k

    async def retrieve(self, query: str) -> list[str]:
        result = await self.client.call_tool(
            "search_knowledge",
            {
                "query": query,
                "top_k": self.top_k,
            },
        )

        if result.is_error:
            raise RuntimeError("MCP knowledge search failed")

        response = json.loads(result.content[0].text)

        return [
            item["content"] for item in response["results"]
        ]