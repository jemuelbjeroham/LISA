import asyncio
import logging
import os
from contextlib import AsyncExitStack
from typing import Self
from uuid import UUID

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from openai import AsyncOpenAI
from typesafe_sdk import TypeSafeClient

from lisa.agents.general_enquiry import GeneralEnquiry
from lisa.agents.knowledge_assistant import KnowledgeAssistant
from lisa.agents.mem_agent import MemoryAgent
from lisa.agents.troubleshooting_agent import TroubleshootingAgent
from lisa.config import Settings
from lisa.context import LISAContext
from lisa.conversation.store import ConversationStore
from lisa.graph import build_graph
from lisa.guardrails.input_moderation import OpenAIInputModerator
from lisa.guardrails.hitl.service import ApprovalService
from lisa.guardrails.hitl.context import automatic_memory_enabled
from lisa.infrastructure.postgres.database import Database
from lisa.infrastructure.postgres.memory_dependencies import (
    memory_dependencies,
)
from lisa.jev_router import JevRouter, parse_jev_routing_prompt
from lisa.mcp.client import MCPClient
from lisa.message_context import user_message
from lisa.memory.workflow import MemoryWorkflow
from lisa.model import (
    create_chat_model,
    create_fallback_model,
    create_jev_model,
    create_router_model,
    create_thinking_chat_model,
)
from lisa.observability import (
    LISAObservability,
    create_langfuse_client,
    create_langfuse_handler,
)
from lisa.orchestrator import Orchestrator
from lisa.prompts.loader import load_prompt
from lisa.retrieval.planning import RetrievalPlanner
from lisa.routing import RoutingPolicy
from lisa.streaming.events import StreamEvent

logger = logging.getLogger(__name__)

class LISA:
    def __init__(self, model: BaseChatModel | None = None, 
                 thinking_model: BaseChatModel | None = None, 
                 router_model: BaseChatModel | None = None,
                 jev_model: TypeSafeClient | None = None,
                 fallback_model: BaseChatModel | None = None,
                 technical_model: BaseChatModel | None = None,
                 retrieval_planner: RetrievalPlanner | None = None,
                 conversation_store: ConversationStore | None = None
                ):
        self.model = model
        self.fallback_model = fallback_model
        self.thinking_model = thinking_model
        self.router_model = router_model
        self.jev_model = jev_model
        self.technical_model = technical_model
        self.retrieval_planner = retrieval_planner
        self.graph = None
        self.mcp_client: MCPClient | None = None
        self.database: Database | None = None
        self.hitl = None
        self._hitl_recovery_task = None
        self.input_moderator = None
        if conversation_store is None:
            raise ValueError("conversation_store must be provided")
        self.conversation_store = conversation_store
        self.orchestrator = None
        self.knowledge_assistant_agent = None
        self.troubleshooting_agent = None
        self.butler = None
        self.specialist = None
        self.general_enquiry_agent = None
        self.memory_agent = None
        self.exit_stack = AsyncExitStack()
        self.routing_policy = None
        self._memory_tasks: set[asyncio.Task[None]] = set()
        self.observability = LISAObservability(client=None)
        self.langfuse_handler = None
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

    def _schedule_memory_processing(
            self,
            user_id: UUID,
            messages: list[BaseMessage],
            conversation_id: UUID,
    ) -> None:
        if not automatic_memory_enabled():
            return
        if self.memory_agent is None or self.database is None:
            logger.warning("Memory processing is unavailable")
            return
        task = asyncio.create_task(
            self._process_memory(
                user_id=user_id,
                messages=messages,
                conversation_id=conversation_id,
            ),
            name=f"memory-{conversation_id}",
        )
        self._memory_tasks.add(task)
        task.add_done_callback(self._memory_tasks.discard)

 
    async def _process_memory(
        self,
        user_id: UUID,
        messages: list[BaseMessage],
        conversation_id: UUID,
    ) -> None:
        operation = "background_memory_processing"

        try:
            if self.database is None or self.memory_agent is None:
                logger.warning(
                    "Skipping background memory processing: "
                    "conversation_id=%s, reason=dependencies_unavailable",
                    conversation_id,
                )
                return

            async with self.database.session() as session:
                dependencies = memory_dependencies(session)

                workflow = MemoryWorkflow(
                    agent=self.memory_agent,
                    service=dependencies.service,
                    user_context_service=dependencies.user_context_service,
                )

                saved = await workflow.process(
                    user_id=user_id,
                    messages=messages,
                )

            logger.info(
                "Background memory processing completed: "
                "conversation_id=%s, operation=%s, saved=%d",
                conversation_id,
                operation,
                len(saved),
            )


        except TimeoutError:
            logger.exception(
                "Background memory processing timed out: "
                "conversation_id=%s, operation=%s",
                conversation_id,
                operation,
            )

        except asyncio.CancelledError:
            logger.info(
                "Background memory processing cancelled: "
                "conversation_id=%s, operation=%s",
                conversation_id,
                operation,
            )
            raise

        except Exception:
            logger.exception(
                "Unexpected background memory processing failure: "
                "conversation_id=%s, operation=%s",
                conversation_id,
                operation,
            )

    async def list_conversations(self, user_id: UUID) -> list[UUID]:
            """Return the IDs of conversations belonging to a user."""
            return await self.conversation_store.list_conversations(user_id)

    async def chat(
        self,
        conversation_id: UUID,
        message: str,
        user_id: UUID,
        enable_thinking: bool = False,
    ) -> str:
        logger.info(
            "Starting chat request: conversation_id=%s, thinking_enabled=%s",
            conversation_id,
            enable_thinking,
        )

        state = await self.conversation_store.get(
            user_id,
            conversation_id,
        )

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

        current_message = user_message(message)
        state["hitl_suspended"] = None
        state["messages"].append(current_message)

        logger.debug(
            "Invoking conversation graph: conversation_id=%s, messages=%d",
            conversation_id,
            len(state["messages"]),
        )

        if self.database is None:
            raise RuntimeError("Database has not been initialized")

        config = {
            "recursion_limit": 12,
            "callbacks": (
                [self.langfuse_handler]
                if self.langfuse_handler is not None
                else []
            ),
        }

        try:
            async with self.database.session() as session:
                context = LISAContext(
                    user_id=user_id,
                    memory=memory_dependencies(session),
                )

                with self.observability.span(
                    "lisa.graph",
                    input={
                        "enable_thinking": enable_thinking,
                        "message_count": len(state["messages"]),
                    },
                ) as span:
                    result = await self.graph.ainvoke(
                        state,
                        config=config,
                        context=context,
                    )

                    if span is not None:
                        response = result["messages"][-1]

                        span.update(
                            output={
                                "route": str(
                                    result.get("active_route")
                                    or result.get("route")
                                ),
                                "response_type": type(response).__name__,
                            }
                        )

        except Exception:
            logger.exception(
                "Conversation graph failed: conversation_id=%s",
                conversation_id,
            )
            raise

        try:
            await self.conversation_store.save(
                user_id,
                conversation_id,
                result,
            )
        except Exception:
            logger.exception(
                "Failed to persist chat result: conversation_id=%s",
                conversation_id,
            )
            raise

        if result.get("hitl_suspended"):
            return "Approval required. Retrieve the pending approval before continuing this task."
        response = result["messages"][-1]

        logger.info(
            "Completed chat request: conversation_id=%s, response_type=%s",
            conversation_id,
            type(response).__name__,
        )

        self._schedule_memory_processing(
            user_id=user_id,
            messages=[
                current_message,
                response,
            ],
            conversation_id=conversation_id,
        )

        return response.content



    async def stream_chat(
        self,
        conversation_id: UUID,
        message: str,
        user_id: UUID,
        enable_thinking: bool = False,
    ):
        logger.info(
            "Starting streaming chat request: conversation_id=%s, thinking_enabled=%s",
            conversation_id,
            enable_thinking,
        )

        state = await self.conversation_store.get(
            user_id,
            conversation_id,
        )

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
                "handoff_count": 0,
            }
        else:
            state["enable_thinking"] = enable_thinking
            state["handoff_count"] = 0

        current_message = user_message(message)
        state["hitl_suspended"] = None
        state["messages"].append(current_message)

        if self.database is None:
            raise RuntimeError("Database is not initialized")

        config = {
            "recursion_limit": 12,
            "callbacks": (
                [self.langfuse_handler]
                if self.langfuse_handler is not None
                else []
            ),
        }

        final_state = None
        event_count = 0

        try:
            with self.observability.span(
                "lisa.graph.stream",
                input={
                    "enable_thinking": enable_thinking,
                    "message_count": len(state["messages"]),
                    "stream_mode": ["custom", "values"],
                },
            ) as span:
                async with self.database.session() as session:
                    context = LISAContext(
                        user_id=user_id,
                        memory=memory_dependencies(session),
                    )

                    async for mode, chunk in self.graph.astream(
                        state,
                        config=config,
                        context=context,
                        stream_mode=["custom", "values"],
                    ):
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

                if span is not None:
                    final_messages = (
                        final_state.get("messages", [])
                        if final_state is not None
                        else []
                    )

                    final_response = (
                        final_messages[-1]
                        if final_messages
                        else None
                    )

                    span.update(
                        output={
                            "event_count": event_count,
                            "final_state_received": final_state is not None,
                            "active_route": (
                                str(
                                    final_state.get("active_route")
                                    or final_state.get("route")
                                )
                                if final_state is not None
                                else None
                            ),
                            "response_type": (
                                type(final_response).__name__
                                if final_response is not None
                                else None
                            ),
                        }
                    )

        except Exception:
            logger.exception(
                "Streaming graph failed: conversation_id=%s",
                conversation_id,
            )
            raise

        if final_state is not None:
            try:
                await self.conversation_store.save(
                    user_id,
                    conversation_id,
                    final_state,
                )

                messages = final_state["messages"]

                if messages and messages[-1].type == "ai":
                    self._schedule_memory_processing(
                        user_id=user_id,
                        messages=[
                            current_message,
                            messages[-1],
                        ],
                        conversation_id=conversation_id,
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
        self.observability = LISAObservability(client=create_langfuse_client(settings))
        self.exit_stack.callback(self.observability.flush)
        self.langfuse_handler = (
            create_langfuse_handler(public_key=settings.langfuse_public_key)
            if self.observability.client is not None
            else None
        )
        moderation_client = AsyncOpenAI(
            api_key=settings.openai_token,
        )
        self.input_moderator = OpenAIInputModerator(
            client=moderation_client,
        )
        self.database = Database(settings)
        self.exit_stack.push_async_callback(self.database.close)
        from functools import partial
        from lisa.guardrails.hitl.continuation import continue_agent
        self.hitl = ApprovalService(self.database)
        self.hitl.continuation_runner = partial(continue_agent, self)
        await self.hitl.recover()
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
        if self.jev_model is None:
            if settings.typesafe_api_key:
                self.jev_model = create_jev_model(settings)
                self.exit_stack.callback(self.jev_model.close)
                logger.info(
                    "Jev model initialized: model_name=%s",
                    settings.jev_model_name,
                )
            else:
                logger.warning(
                    "TYPESAFE_API_KEY not set; orchestrator will route with "
                    "the LLM router only"
                )
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
        self.mcp_client = await self.exit_stack.enter_async_context(
            MCPClient(
                command=settings.mcp_server_command,
                args=settings.mcp_server_args,
                cwd=settings.mcp_server_cwd,
                env={
                    **os.environ,
                    **(
                        {"TAVILY_API_KEY": settings.tavily_api_key}
                        if settings.tavily_api_key
                        else {}
                    ),
                }
            )
        )
        routing_prompt = load_prompt("orchestrator/routing_distribution_engineering_v2.txt")
        identity = load_prompt("Identity.txt").rstrip()
        general_enquiry_prompt = identity + "\n\n" + load_prompt("general_enquiry/general_enquiry_v2.txt")
        knowledge_assistant_prompt = identity + "\n\n" + load_prompt("knowledge_assistant/knowledge_assistant_v1.txt")
        troubleshooting_prompt = identity + "\n\n" + load_prompt("troubleshooting_agent/troubleshooting_agent_v1.txt")
        memory_agent_prompt = load_prompt("memory/memory_agent_v2.txt")
        jev_router = None
        if self.jev_model is not None:
            jev_routing_prompt = parse_jev_routing_prompt(
                load_prompt("orchestrator/jev_routing_v1.json")
            )
            jev_router = JevRouter(self.jev_model, jev_routing_prompt)
        self.orchestrator = Orchestrator(
            model=self.router_model,
            routing_prompt=routing_prompt,
            jev_router=jev_router,
        )
        self.memory_agent = MemoryAgent(
            model=self.model,
            prompt=memory_agent_prompt,
        )
        self.knowledge_assistant_agent = KnowledgeAssistant(
            model=self.model,
            fallback_model=self.fallback_model,
            thinking_model=self.thinking_model,
            system_prompt=knowledge_assistant_prompt,
            input_moderator=self.input_moderator,
            mcp_client=self.mcp_client,
        )
        self.troubleshooting_agent = TroubleshootingAgent(
            model=self.model,
            fallback_model=self.fallback_model,
            thinking_model=self.thinking_model,
            system_prompt=troubleshooting_prompt,
            input_moderator=self.input_moderator,
            mcp_client=self.mcp_client,
        )
        self.general_enquiry_agent = GeneralEnquiry(
            model=self.model,
            fallback_model=self.fallback_model,
            thinking_model=self.thinking_model,
            system_prompt=general_enquiry_prompt,
            mcp_client=self.mcp_client,
            input_moderator=self.input_moderator,
        )
        await self.knowledge_assistant_agent.initialize_tools()
        await self.troubleshooting_agent.initialize_tools()
        await self.general_enquiry_agent.initialize_tools()
        for agent, method_name in (
            (self.orchestrator, "route"),
            (self.memory_agent, "analyze"),
            (self.knowledge_assistant_agent, "run"),
            (self.troubleshooting_agent, "run"),
            (self.general_enquiry_agent, "run"),
            (self.hitl, "continuation_runner"),
        ):
            self.observability.instrument(agent, method_name, self.langfuse_handler)
        self.routing_policy = RoutingPolicy()
        self.graph = build_graph(
            orchestrator=self.orchestrator,
            knowledge_assistant_agent=self.knowledge_assistant_agent,
            troubleshooting_agent=self.troubleshooting_agent,
            general_enquiry_agent=self.general_enquiry_agent,
            routing_policy=self.routing_policy,
        )
        logger.info(
            "LISA has been initialized: model=%s, thinking_model=%s, router_model=%s, "
            "jev_model=%s, fallback_model=%s",
            type(self.model).__name__,
            type(self.thinking_model).__name__,
            type(self.router_model).__name__,
            type(self.jev_model).__name__ if self.jev_model is not None else None,
            type(self.fallback_model).__name__,
        )
        self._hitl_recovery_task = asyncio.create_task(self.hitl.recovery_loop(), name="hitl-recovery")
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
            if self._hitl_recovery_task is not None:
                self._hitl_recovery_task.cancel()
                await asyncio.gather(self._hitl_recovery_task, return_exceptions=True)
            if self._memory_tasks:
                logger.info(
                    "Waiting for background memory tasks: count=%d",
                    len(self._memory_tasks),
                )
                await asyncio.gather(
                    *self._memory_tasks,
                    return_exceptions=True,
                )
            await self.exit_stack.aclose()
        except Exception:
            logger.exception("Application LISA shutdown failed")
            raise
        logger.info("Application LISA shutdown complete")
