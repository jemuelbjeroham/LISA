from unittest.mock import Mock

import pytest
from langchain_core.messages import HumanMessage

from lisa.agents.technical_clarification import TechnicalClarificationAgent
from lisa.routing import Route


class TestKnowledgeRetriever:
    async def retrieve(self, query: str) -> list[str]:
        return [
            "The firewall is unreachable after commiting a firewall rule"
        ]


@pytest.mark.anyio
async def test_technical_clarification_retrieves_knowledge():
    model = Mock()

    model.invoke.return_value = Mock(
        content="Technical explanation"
    )

    retriever = TestKnowledgeRetriever()

    agent = TechnicalClarificationAgent(
        model=model,
        retriever=retriever,
        system_prompt="You are a technical assistant.",
    )

    state = {
        "messages": [
            HumanMessage(
                content="Why the firewall is unreachable?"
            )
        ],
        "route": Route.TECHNICAL_CLARIFICATION,
    }

    result = await agent.run(state)

    assert result["messages"][0].content == "Technical explanation"