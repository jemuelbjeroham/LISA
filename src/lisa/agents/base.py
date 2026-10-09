from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel

from lisa.state import LISAState
from lisa.tools import handoff, save_memory


class BaseAgent(ABC):
    def __init__(self, model: BaseChatModel, system_prompt: str):
        self.model = self.bind_tools(model)
        self.system_prompt = system_prompt

    @staticmethod
    def bind_tools(model: BaseChatModel) -> BaseChatModel:
        return model.bind_tools([handoff, save_memory], tool_choice="auto")
    
    @abstractmethod
    async def run(self, state: LISAState):
        pass
    