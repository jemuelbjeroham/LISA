from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from lisa.agents.general_enquiry import GeneralEnquiry


@pytest.mark.anyio
async def test_general_enquiry_returns_model_response():
    model = MagicMock()

    model.invoke.return_value = AIMessage(
        content="Hello! How can I help you?"
    )

    agent = GeneralEnquiry(
        model=model,
        system_prompt="You are a helpful general assistant.",
    )

    state = {
        "messages": [
            HumanMessage(content="Hello"),
        ],
        "route": None,
    }

    result = await agent.run(state)

    assert result == {
        "messages": [
            AIMessage(content="Hello! How can I help you?")
        ]
    }

    model.invoke.assert_called_once()

    messages = model.invoke.call_args.args[0]

    assert messages == [
        SystemMessage(
            content="You are a helpful general assistant."
        ),
        HumanMessage(content="Hello"),
    ]