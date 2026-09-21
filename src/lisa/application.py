import logging
from contextlib import AsyncExitStack
from typing import Self
from uuid import UUID

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from lisa.agents.general_enquiry import GeneralEnquiry
from lisa.agents.technical_clarification import TechnicalClarificationAgent
from lisa.config import Settings
from lisa.conversation.in_memory import InMemoryConversationStore
from lisa.conversation.store import ConversationStore
from lisa.graph import build_graph
from lisa.knowledge.mcp_retriever import MCPKnowledgeRetriever
from lisa.mcp.client import MCPClient
from lisa.model import (
    create_chat_model,
    create_fallback_model,
    create_router_model,
    create_thinking_chat_model,
    create_technical_model,
)
from lisa.orchestrator import Orchestrator
from lisa.prompts.loader import load_prompt
from lisa.routing import RoutingPolicy
from lisa.streaming.events import StreamEvent

logger = logging.getLogger(__name__)


class LISA:

    def __init__(self, model: BaseChatModel | None = None, 
                 thinking_model: BaseChatModel | None = None, 
                 router_model: BaseChatModel | None = None,
                 fallback_model: BaseChatModel | None = None,
                 technical_model: BaseChatModel | None = None,
                 conversation_store: ConversationStore | None = None
                ):
        self.model = model
        self.fallback_model = fallback_model
        self.thinking_model = thinking_model
        self.router_model = router_model
        self.technical_model = technical_model
        self.graph = None
        self.mcp_client: MCPClient | None = None
        self.conversation_store = (
            conversation_store or InMemoryConversationStore()
        )
        self.orchestrator = None
        self.technical_clarification_agent = None
        self.general_enquiry_agent = None
        self.exit_stack = AsyncExitStack()
        self.routing_policy = None
        logger.debug(
            "LISA instance created: custom_model=%s, custom_thinking_model=%s, "
            "custom_router_model=%s, custom_fallback_model=%s, "
            "custom_conversation_store=%s",
            model is not None,
            thinking_model is not None,
            router_model is not None,
            fallback_model is not None,
            conversation_store is not None,
        )

    async def chat(self, conversation_id: UUID, message: str, enable_thinking: bool = False) -> str:
        logger.info(
            "Starting chat request: conversation_id=%s, thinking_enabled=%s",
            conversation_id,
            enable_thinking,
        )
        state = await self.conversation_store.get(conversation_id)
        logger.debug(
            "Loaded conversation state: conversation_id=%s, exists=%s, messages=%d",
            conversation_id,
            state is not None,
            len(state["messages"]) if state else 0,
        )

        if state is None:
            state = {
                "messages": [],
                "route": None,
                "active_route": None,
                "agent_handoff": None,
                "knowledge_context": [],
                "enable_thinking": enable_thinking,
            }
        else:
            state["enable_thinking"] = enable_thinking

        state["messages"].append(
            HumanMessage(content=message)
        )

        logger.debug(
            "Invoking conversation graph: conversation_id=%s, messages=%d",
            conversation_id,
            len(state["messages"]),
        )
        try:
            result = await self.graph.ainvoke(state)
        except Exception:
            logger.exception(
                "Conversation graph failed: conversation_id=%s",
                conversation_id,
            )
            raise

        try:
            await self.conversation_store.save(
                conversation_id,
                result,
            )
        except Exception:
            logger.exception(
                "Failed to persist chat result: conversation_id=%s",
                conversation_id,
            )
            raise

        response = result["messages"][-1]
        logger.info(
            "Completed chat request: conversation_id=%s, response_type=%s",
            conversation_id,
            type(response).__name__,
        )

        return response.content

    async def stream_chat(self, conversation_id: UUID, message: str, enable_thinking: bool = False):
        logger.info(
            "Starting streaming chat request: conversation_id=%s, thinking_enabled=%s",
            conversation_id,
            enable_thinking,
        )
        state = await self.conversation_store.get(conversation_id)
        logger.info(
            "Loaded conversation state: conversation_id=%s, active_route=%s, messages=%d",
            conversation_id,
            state.get("active_route") if state else None,
            len(state["messages"]) if state else 0,
        )

        if state is None:
            state = {
                "messages": [],
                "route": None,
                "active_route": None,
                "agent_handoff": None,
                "knowledge_context": [],
                "enable_thinking": enable_thinking,
            }
        else:
            state["enable_thinking"] = enable_thinking

        state["messages"].append(
            HumanMessage(content=message)
        )

        final_state = None
        event_count = 0

        async for mode, chunk in self.graph.astream(state, stream_mode=["custom", "values"]):

            if mode == "custom":

                if not isinstance(chunk, StreamEvent):
                    logger.warning(
                        "Received unexpected custom stream event: type=%s",
                        type(chunk).__name__,
                    )
                    continue

                logger.info(
                    "Received stream event: conversation_id=%s, type=%s",
                    conversation_id,
                    chunk.type,
                )

                event_count += 1
                yield chunk

            elif mode == "values":
                final_state = chunk

        if final_state is not None:
            try:
                await self.conversation_store.save(
                    conversation_id,
                    final_state,
                )
            except Exception:
                logger.exception(
                    "Failed to persist streaming result: conversation_id=%s",
                    conversation_id,
                )
                raise
            logger.info(
                "Completed streaming chat request: conversation_id=%s, events=%d",
                conversation_id,
                event_count,
            )
        else:
            logger.warning(
                "Streaming chat completed without a final state: conversation_id=%s, events=%d",
                conversation_id,
                event_count,
            )

        # # async for chunk in self.technical_clarification_agent.stream(state):
        # #     content = chunk.content

        # #     if content:
        # #         response_chunks.append(content)
        # #         yield content

        # final_response = "".join(response_chunks)

        # state["messages"].append(
        #     AIMessage(content=final_response)
        # )

        # await self.conversation_store.save(
        #     conversation_id,
        #     state,
        # )

    async def __aenter__(self) -> Self:
        logger.info("Initializing LISA (Level1 Intelligent System and Assistant)")
        settings = Settings()
        logger.debug(
            "Loaded application settings: model_provider=%s, router_model_provider=%s, "
            "mcp_server_command=%s",
            settings.model_provider,
            settings.router_model_provider,
            settings.mcp_server_command,
        )

        if self.model is None:
            self.model = create_chat_model(settings)

        if self.thinking_model is None:
            self.thinking_model = create_thinking_chat_model(settings)

        if self.router_model is None:
            self.router_model = create_router_model(settings)

        if self.fallback_model is None:
            logger.info(
                "Initializing fallback model: provider=%s, model_name=%s",
                settings.fallback_model_provider,
                settings.fallback_model_name,
            )
            try:
                self.fallback_model = create_fallback_model(settings)
                logger.info(
                    "Fallback model initialized: %s",
                    type(self.fallback_model).__name__,
                )
            except Exception:
                logger.exception("Failed to initialize fallback model")
                raise

        if self.technical_model is None:
            logger.info(
                "Initializing technical model: provider=%s, model_name=%s",
                settings.technical_model_provider,
                settings.technical_model_name,
            )
            try:
                self.technical_model = create_technical_model(settings)
                logger.info(
                    "Technical model has been initialized: %s",
                    type(self.technical_model).__name__,
                )
            except Exception:
                logger.exception("Failed to initialize Technical Model")

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
        general_enquiry_prompt = load_prompt("general_enquiry/general_enquiry_v1.txt")

        self.orchestrator = Orchestrator(
            model=self.router_model,
            routing_prompt=routing_prompt,
        )

        self.technical_clarification_agent = TechnicalClarificationAgent(
            model=self.model,
            retriever=retriever,
            system_prompt=technical_prompt,
        )

        self.general_enquiry_agent = GeneralEnquiry(
            model=self.model,
            fallback_model=self.fallback_model,
            thinking_model=self.thinking_model,
            system_prompt=general_enquiry_prompt,
        )

        self.routing_policy = RoutingPolicy()

        self.graph = build_graph(
            orchestrator=self.orchestrator,
            technical_clarification_agent=self.technical_clarification_agent,
            general_enquiry_agent=self.general_enquiry_agent,
            routing_policy=self.routing_policy,
        )

        logger.info(
            "LISA has been initialized: model=%s, thinking_model=%s, router_model=%s, "
            "fallback_model=%s",
            type(self.model).__name__,
            type(self.thinking_model).__name__,
            type(self.router_model).__name__,
            type(self.fallback_model).__name__,
        )
        return self

    async def __aexit__(
            self,
            exc_type,
            exc_value,
            traceback,
    ) -> None:
        logger.info(
            "Application LISA is shutting down: exception_type=%s",
            exc_type.__name__ if exc_type else None,
        )
        try:
            await self.exit_stack.aclose()
        except Exception:
            logger.exception("Application LISA shutdown failed")
            raise
        logger.info("Application LISA shutdown complete")
