from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel

from lisa.state import LISAState


class BaseAgent(ABC):
    def __init__(self, model: BaseChatModel):
        self.model = model

    @abstractmethod
    def run(self, state: LISAState):
        pass
    