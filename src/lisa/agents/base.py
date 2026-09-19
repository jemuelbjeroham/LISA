from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool

from lisa.state import LISAState


@tool
def handoff(reason: str) -> str:
    """Handoff the current request back to the orchestrator"""
    return reason

class BaseAgent(ABC):
    def __init__(self, model: BaseChatModel, system_prompt: str):
        self.model = self.bind_tools(model)
        self.system_prompt = system_prompt

    @staticmethod
    def bind_tools(model: BaseChatModel) -> BaseChatModel:
        return model.bind_tools([handoff], tool_choice="auto")
    
    @abstractmethod
    async def run(self, state: LISAState):
        pass
    