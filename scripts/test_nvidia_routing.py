from langchain_core.messages import HumanMessage

from lisa.config import Settings
from lisa.model import create_chat_model
from lisa.orchestrator import Orchestrator
from lisa.prompts.loader import load_prompt


def main():
    settings = Settings()

    model = create_chat_model(settings)

    routing_prompt = load_prompt("orchestrator/routing_v1.txt")

    orechestrator = Orchestrator(model=model, routing_prompt=routing_prompt)

    state = {
        "messages": [
            HumanMessage(content="Unable to delete a firewall rule it is stuck in deleting state")
        ],
        "route": None
    }

    result = orechestrator.route(state)

    print("Routing result: ", result)

if __name__ == "__main__":
    main()
