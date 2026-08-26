from unittest.mock import Mock

from langchain_core.messages import HumanMessage

from lisa.agents.technical_clarification import (
    TechnicalClarificationAgent,
)
from lisa.routing import Route


class TestKnowledgeRetriever:
    def retrieve(self, query: str) -> list[str]:
        return ["Relevant technical documentation"]


def test_technical_clarification_retrieves_knowledge():
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
                content="Why is the channel stuck?"
            )
        ],
        "route": Route.TECHNICAL_CLARIFICATION,
    }

    result = agent.run(state)

    assert result["messages"][0].content == "Technical explanation"