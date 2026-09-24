import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.config import get_stream_writer

from lisa.agents.base import BaseAgent
from lisa.knowledge.protocol import KnowledgeRetriever
from lisa.state import LISAState
from lisa.streaming.events import StreamEvent

logger = logging.getLogger(__name__)

class TechnicalClarificationAgent(BaseAgent):
    def __init__(
            self,
            model: BaseChatModel,
            system_prompt: str,
            hyde_prompt: str,
            retriever: KnowledgeRetriever,
    ):
        self.hyde_model = model
        super().__init__(
            model=model,
            system_prompt=system_prompt,
        )

        self.hyde_prompt = hyde_prompt
        self.retriever = retriever

    async def run(self, state: LISAState):
        user_message = state["messages"][-1]

        if not isinstance(user_message, HumanMessage):
            raise TypeError(
                "TechnicalClarificationAgent expected the latest "
                "message to be a HumanMessage."
            )

        user_query = user_message.content

        if not isinstance(user_query, str):
            raise TypeError(
                "TechnicalClarificationAgent currently supports "
                "text-based user messages only."
            )

        handoff = await self._check_scope(user_query)

        if handoff is not None:
            return {
                "agent_handoff": handoff,
            }
        
        logger.info(
            "Technical clarification started: query=%r",
            user_query,
        )

        hyde_messages = [
            SystemMessage(content=self.hyde_prompt),
            HumanMessage(content=user_query),
        ]

        logger.info("Generating HyDE retrival document")

        hyde_response = await self.hyde_model.ainvoke(
            hyde_messages
        )

        hypothetical_document = hyde_response.content

        if not isinstance(hypothetical_document, str):
            raise TypeError(
                "HyDE model returned an empty retrievel document."
            )

        hypothetical_document = hypothetical_document.strip()

        if not hypothetical_document:
            raise RuntimeError(
                "HyDE model returned an empty retrival document."
            )

        logger.debug(
            "HyDE document generated: %s",
            hypothetical_document,
        )

        """
        Retrival of actual knowledge using HyDE document
        """

        logger.info(
            "Retrieving technical knowledge using HyDE document"
        )

        knowledge = await self.retriever.retrieve(
            hypothetical_document
        )

        logger.info(
            "Technical knowledge retrievel completed: chunks=%d",
            len(knowledge),
        )

        knowledge_context = "\n\n".join(knowledge)

        # state["knowledge_content"] = knowledge


        system_message = SystemMessage(
            content=(
                f"{self.system_prompt}\n\n"
                "Use the following retrieved technical knowledge as "
                "the source of truth for your answer.\n\n"
                "Important grounding rules:\n"
                "- Base technical claims on the retrieved knowledge.\n"
                "- Do not treat the user's question as evidence.\n"
                "- Do not treat the hypothetical retrieval document "
                "as evidence.\n"
                "- Do not invent internal configurations, logs, "
                "procedures, or root causes.\n"
                "- If the retrieved knowledge does not contain enough "
                "information to answer confidently, say so clearly.\n\n"
                f"Retrieved technical knowledge:\n\n"
                f"{knowledge_context}"
            )
        )

        messages = [
            system_message,
            *state["messages"],
        ]

        writer = get_stream_writer()
        response_chunks: list[str] = []

        logger.info(
            "Generating grounded technical response"
        )

        async for chunk in self.model.astream(messages):

            reasoning = chunk.additional_kwargs.get(
                "reasoning_content"
            )

            if reasoning:
                writer(
                    StreamEvent(
                        type="reasoning",
                        content=reasoning,
                    )
                )

            if chunk.tool_calls:
                for tool_call in chunk.tool_calls:
                    if tool_call["name"] == "handoff":
                        logger.info(
                            "Technical clarification requested handoff: %s",
                            tool_call["args"],
                        )

                        return {
                            "agent_handoff": tool_call["args"],
                            "knowledge_context": knowledge,
                        }

            if chunk.content:
                writer(
                    StreamEvent(
                        type="content",
                        content=chunk.content,
                    )
                )

                response_chunks.append(chunk.content)

        final_response = "".join(response_chunks)

        logger.info(
            "Technical clarification completed"
        )

        return {
            "messages": [
                AIMessage(content=final_response)
            ],
            "knowledge_context": knowledge,
        }

    async def _check_scope(self, user_query: str):
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_query),
        ]

        response = await self.model.ainvoke(messages)

        if response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call["name"] == "handoff":
                    return tool_call["args"]

        return None

        

