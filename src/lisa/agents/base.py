from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel

from lisa.state import LISAState


class BaseAgent(ABC):
    def __init__(self, model: BaseChatModel, system_prompt: str):
        self.model = model
        self.system_prompt = system_prompt

    @abstractmethod
    async def run(self, state: LISAState):
        pass
    