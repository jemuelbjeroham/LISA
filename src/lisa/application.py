from langchain_core.language_models import BaseChatModel

from lisa.config import Settings
from lisa.graph import build_graph
from lisa.model import create_chat_model
from lisa.orchestrator import Orchestrator
from lisa.prompts.loader import load_prompt


class LISA:
    def __init__(self, model: BaseChatModel | None = None):
        if model is None:
            settings = Settings()
            model = create_chat_model(settings)

        routing_prompt = load_prompt("orchestrator/routing_v1.txt")
        orchestrator = Orchestrator(model=model, routing_prompt=routing_prompt)
        self.graph = build_graph(orchestrator)

        

    