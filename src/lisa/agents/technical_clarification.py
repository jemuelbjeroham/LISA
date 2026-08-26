from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage

from lisa.knowledge.protocol import KnowledgeRetriever
from lisa.state import LISAState


class TechnicalClarificationAgent:
    def __init__(self, model: BaseChatModel, retriever: KnowledgeRetriever, system_prompt: str):
        self.model = model
        self.retriever = retriever
        self.system_prompt = system_prompt

    def run(self, state: LISAState):
        user_message = state["messages"][-1]

        knowledge = self.retriever.retrieve(user_message.content)
        messages = [
            SystemMessage(
                content=self.system_prompt
            ),
            *state["messages"]
        ]
        response = self.model.invoke(messages)
        return {
            "messages": [response]
        }