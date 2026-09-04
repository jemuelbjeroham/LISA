import logging
from contextlib import AsyncExitStack
from typing import Self

from langchain_core.language_models import BaseChatModel

from lisa.agents.technical_clarification import TechnicalClarificationAgent
from lisa.config import Settings
from lisa.graph import build_graph
from lisa.knowledge.mcp_retriever import MCPKnowledgeRetriever
from lisa.mcp.client import MCPClient
from lisa.model import create_chat_model
from lisa.orchestrator import Orchestrator
from lisa.prompts.loader import load_prompt

logger = logging.getLogger(__name__)

class LISA:
    def __init__(self, model: BaseChatModel | None = None):
        self.model = model
        self.graph = None
        self.mcp_client: MCPClient | None = None
        self.exit_stack = AsyncExitStack()

    async def __aenter__(self) -> Self:
        logger.info("Initializing LISA (Level1 Intelligent System and Assistant)")
        settings = Settings()

        if self.model is None:
            self.model = create_chat_model(settings)

        self.mcp_client = await self.exit_stack.enter_async_context(
            MCPClient(
                command=settings.mcp_server_command,
                args=settings.mcp_server_args,
                cwd=settings.mcp_server_cwd,
            )
        )

        retriever = MCPKnowledgeRetriever(
            client=self.mcp_client,
        )

        routing_prompt = load_prompt("orchestrator/routing_v1.txt")
        technical_prompt = load_prompt("technical_clarification/technical_clarification_v1.txt")

        orchestrator = Orchestrator(
            model = self.model,
            routing_prompt=routing_prompt,
        )

        technical_clarification_agent = TechnicalClarificationAgent(
            model=self.model,
            retriever=retriever,
            system_prompt=technical_prompt,
        )

        self.graph = build_graph(
            orchestrator=orchestrator,
            technical_clarification_agent=technical_clarification_agent
        )

        logger.info("LISA has been initialized")
        return self

    async def __aexit__(
            self,
            exc_type,
            exc_value,
            traceback,
    ) -> None:
        logger.info("Application LISA is shutting down")
        await self.exit_stack.aclose()
        logger.info("Application LISA shutdown complete")