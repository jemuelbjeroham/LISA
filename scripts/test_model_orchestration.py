import asyncio

from langchain_core.messages import HumanMessage

from lisa.config import Settings
from lisa.model import create_chat_model
from lisa.orchestrator import Orchestrator
from lisa.prompts.loader import load_prompt
from lisa.state import LISAState


async def main():
    settings = Settings()

    model = create_chat_model(settings)

    prompt = load_prompt(
        "orchestrator/routing_v1.txt"
    )

    orchestrator = Orchestrator(
        model=model,
        routing_prompt=prompt,
    )

    test_messages = [
        "Hi, how are you?",
        "What is your name?",
        "Who is the president of India?",
        "What is Python?",
        "How does DNS work?",
        "How do I troubleshoot a BGP session?",
        "The BGP session keeps going down.",
    ]

    for message in test_messages:
        state: LISAState = {
            "messages": [
                HumanMessage(content=message)
            ],
            "route": None,
        }

        result = orchestrator.route(state)

        print(
            f"{message!r} -> {result['route']}"
        )


asyncio.run(main())