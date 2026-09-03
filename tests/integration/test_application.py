from unittest.mock import Mock, patch

import pytest

from lisa.application import LISA


@pytest.mark.anyio
async def test_lisa_application_initializes_with_injected_model():
    model = Mock()

    async with LISA(model) as lisa:
        assert lisa.model is model
        assert lisa.graph is not None
        assert lisa.mcp_client is not None


@pytest.mark.anyio
async def test_lisa_application_creates_model_when_not_provided():
    model = Mock()

    with patch(
        "lisa.application.create_chat_model",
        return_value=model,
    ) as create_model:
        async with LISA() as lisa:
            assert lisa.model is model
            assert lisa.graph is not None
            assert lisa.mcp_client is not None

        create_model.assert_called_once()