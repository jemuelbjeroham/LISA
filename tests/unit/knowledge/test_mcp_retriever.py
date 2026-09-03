import json
from unittest.mock import AsyncMock, Mock

import pytest
from mcp.types import CallToolResult, TextContent

from lisa.knowledge.mcp_retriever import MCPKnowledgeRetriever


@pytest.mark.anyio
async def test_retrieve_returns_knowledge_content():
    client = Mock()

    client.call_tool = AsyncMock(
        return_value=CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "results": [
                                {
                                    "content": "Firewall troubleshooting guide.",
                                    "source": "firewall.txt",
                                    "chunk_index": 0,
                                },
                                {
                                    "content": "Check ACL rules.",
                                    "source": "network_security.txt",
                                    "chunk_index": 1,
                                },
                            ]
                        }
                    )
                )
            ],
            is_error=False,
        )
    )

    retriever = MCPKnowledgeRetriever(
        client=client,
        top_k=5,
    )

    result = await retriever.retrieve(
        "How do I troubleshoot firewall connectivity?"
    )

    assert result == [
        "Firewall troubleshooting guide.",
        "Check ACL rules.",
    ]

    client.call_tool.assert_awaited_once_with(
        "search_knowledge",
        {
            "query": "How do I troubleshoot firewall connectivity?",
            "top_k": 5,
        }
    )