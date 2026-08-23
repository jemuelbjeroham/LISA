from langchain_core.language_models import BaseChatModel

from lisa.graph import build_graph
from lisa.orchestrator import Orchestrator
from lisa.prompts.loader import load_prompt


class LISA:
    def __init__(self, model: BaseChatModel):
        routing_prompt = load_prompt("orchestrator/routing_v1.txt")

        orchestrator = Orchestrator(model=model, routing_prompt=routing_prompt)

        self.graph = build_graph(orchestrator)

    