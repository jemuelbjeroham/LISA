from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage

from lisa.agents.base import BaseAgent
from lisa.knowledge.protocol import KnowledgeRetriever
from lisa.state import LISAState


class TechnicalClarificationAgent(BaseAgent):
    def __init__(self, model: BaseChatModel, system_prompt: str, retriever: KnowledgeRetriever):

        super().__init__(model=model, system_prompt=system_prompt)
        self.retriever = retriever

    async def run(self, state: LISAState):
        user_message = state["messages"][-1]

        knowledge = await self.retriever.retrieve(user_message.content)

        state["knowledge_context"] = knowledge
        knowledge_context = "\n\n".join(knowledge)

        system_message = SystemMessage(
            content=(
                f"{self.system_prompt}\n\n"
                f"Relevant technical knowledge:\n\n"
                f"{knowledge_context}"
            )
        )
        messages = [
            system_message,
            *state["messages"],
        ]

        response_chunks = []

        async for chunk in self.model.astream(messages):
            response_chunks.append(chunk)

        final_response = "".join(
            chunk.content
            for chunk in response_chunks
            if chunk.content
        )

        return {
            "messages": [AIMessage(content=final_response)],
            "knowledge_context": knowledge,
        }

    async def stream(self, state: LISAState):
        user_message = state["messages"][-1]

        knowledge = await self.retriever.retrieve(user_message.content)
        state["knowledge_context"] = knowledge
        knowledge_context = "\n\n".join(state["knowledge_context"])

        system_message = SystemMessage(
            content=(
                f"{self.system_prompt}\n\n"
                f"Relevant technical knowledge:\n\n"
                f"{knowledge_context}"
            )
        )

        messages = [
            system_message,
            *state["messages"],
        ]

        async for chunk in self.model.astream(messages):
            yield chunk
