from unittest.mock import AsyncMock, Mock, patch

import pytest

from lisa.application import LISA


@pytest.mark.anyio
async def test_lisa_application_initializes_with_injected_model():
    model = Mock()
    router_model = Mock()
    mcp_client = AsyncMock()
    mcp_client.__aenter__.return_value = mcp_client
    mcp_client.__aexit__.return_value = None

    with patch("lisa.application.MCPClient", return_value=mcp_client), patch(
        "lisa.application.create_router_model",
        return_value=router_model,
    ):
        async with LISA(model) as lisa:
            assert lisa.model is model
            assert lisa.router_model is router_model
            assert lisa.graph is not None
            assert lisa.mcp_client is mcp_client
            assert lisa.orchestrator is not None
            assert lisa.technical_clarification_agent is not None
            assert lisa.general_enquiry_agent is not None


@pytest.mark.anyio
async def test_lisa_application_creates_model_when_not_provided():
    model = Mock()
    router_model = Mock()
    mcp_client = AsyncMock()
    mcp_client.__aenter__.return_value = mcp_client
    mcp_client.__aexit__.return_value = None

    with patch(
        "lisa.application.create_chat_model",
        return_value=model,
    ) as create_model, patch(
        "lisa.application.create_router_model",
        return_value=router_model,
    ) as create_router_model, patch(
        "lisa.application.MCPClient",
        return_value=mcp_client,
    ):
        async with LISA() as lisa:
            assert lisa.model is model
            assert lisa.router_model is router_model
            assert lisa.graph is not None
            assert lisa.mcp_client is mcp_client

        create_model.assert_called_once()
        create_router_model.assert_called_once()