from langchain_core.messages import HumanMessage

from lisa.config import Settings
from lisa.model import create_router_model
from lisa.routing import RoutingDecision


def test_router_model_supports_structured_output():
    settings = Settings()

    model = create_router_model(settings)

    structured_model = model.with_structured_output(RoutingDecision)

    response = structured_model.invoke(
        [
            HumanMessage(
                content=(
                    "Classify the following request into exactly one of "
                    "these routes: general_enquiry, "
                    "technical_clarification, issue_reporting.\n\n"
                    "What is DNS?"
                )
            )
        ]
    )

    print(f"\nRouter decision: {response}")

    assert isinstance(response, RoutingDecision)
    assert response.route.value in {
        "general_enquiry",
        "technical_clarification",
        "issue_reporting",
    }