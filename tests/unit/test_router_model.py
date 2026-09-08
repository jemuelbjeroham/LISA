import pytest
from langchain_core.messages import HumanMessage

from lisa.config import Settings
from lisa.model import create_router_model


@pytest.mark.anyio
async def test_router_model_works():
    settings = Settings()

    model = create_router_model(settings)

    response = await model.ainvoke(
        [
            HumanMessage(
                content=(
                    "Classify this request as exactly one of: "
                    "general_enquiry, technical_clarification, issue_reporting.\n\n"
                    "What is DNS?"
                )
            )
        ]
    )

    print(f"\nRouter response: {response.content}")

    assert response.content