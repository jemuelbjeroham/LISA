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
from lisa.model import create_chat_model, create_thinking_chat_model, create_router_model
from lisa.orchestrator import Orchestrator
from lisa.prompts.loader import load_prompt

logger = logging.getLogger(__name__)

class LISA:
    def __init__(self, model: BaseChatModel | None = None, thinking_model: BaseChatModel | None = None, router_model: BaseChatModel | None = None, conversation_store: ConversationStore | None = None):
        self.model = model
        self.thinking_model = thinking_model
        self.router_model = router_model
        self.graph = None
        self.mcp_client: MCPClient | None = None
        self.conversation_store = (
            conversation_store or InMemoryConversationStore()
        )
        self.orchestrator = None
        self.technical_clarification_agent = None
        self.general_enquiry_agent = None
        self.exit_stack = AsyncExitStack()

    async def chat(self, conversation_id: UUID, message: str, enable_thinking: bool = False) -> str:
        state = await self.conversation_store.get(conversation_id)

        if state is None:
            state = {
                "messages": [],
                "route": None,
                "active_route": None,
                "knowledge_context": [],
                "enable_thinking": enable_thinking,
            }
        else:
            state["enable_thinking"] = enable_thinking

        state["messages"].append(
            HumanMessage(content=message)
        )

        result = await self.graph.ainvoke(state)

        await self.conversation_store.save(
            conversation_id,
            result,
        )

        response = result["messages"][-1]

        return response.content

    async def stream_chat(self, conversation_id: UUID, message: str, enable_thinking: bool = False):
        state = await self.conversation_store.get(conversation_id)
        logger.info(
            "Loaded conversation state: active_route=%s, messages=%d",
            state.get("active_route") if state else None,
            len(state["messages"]) if state else 0,
        )

        if state is None:
            state = {
                "messages": [],
                "route": None,
                "active_route": None,
                "knowledge_context": [],
                "enable_thinking": enable_thinking,
            }
        else:
            state["enable_thinking"] = enable_thinking

        state["messages"].append(
            HumanMessage(content=message)
        )

        final_state = None

        async for mode, chunk in self.graph.astream(state, stream_mode=["messages", "values"]):

            if mode == "messages":
                message_chunk, metadata = chunk

                logger.info(
                    "LLM chunk: content=%r reasoning=%r",
                    message_chunk.content,
                    message_chunk.additional_kwargs.get("reasoning_content"),
                )
                
                if message_chunk.content:
                    yield message_chunk.content

            elif mode == "values":
                final_state = chunk

        if final_state is not None:
            await self.conversation_store.save(
                conversation_id,
                final_state,
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

        if self.model is None:
            self.model = create_chat_model(settings)

        if self.thinking_model is None:
            self.thinking_model = create_thinking_chat_model(settings)

        if self.router_model is None:
            self.router_model = create_router_model(settings)

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
            thinking_model=self.thinking_model,
            system_prompt=general_enquiry_prompt,
        )

        self.graph = build_graph(
            orchestrator=self.orchestrator,
            technical_clarification_agent=self.technical_clarification_agent,
            general_enquiry_agent=self.general_enquiry_agent,
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