import asyncio

from langchain_core.messages import HumanMessage

from lisa.config import Settings
from lisa.model import (
    create_chat_model,
    create_fallback_model,
)
from lisa.model_resilience import ModelResilience


async def main():
    settings = Settings()

    primary = create_chat_model(settings)
    fallback = create_fallback_model(settings)

    resilient_model = ModelResilience(
        primary=primary,
        fallback=fallback,
        inactivity_timeout_seconds=10.0,
    )

    messages = [
        HumanMessage(
            content="Explain what a firewall is in two sentences."
        )
    ]

    print("=" * 80)
    print("Testing resilient model")
    print("=" * 80)

    async for chunk in resilient_model.stream(messages):

        if chunk.content:
            print(
                "CONTENT:",
                repr(chunk.content),
            )

        reasoning = chunk.additional_kwargs.get(
            "reasoning_content"
        )

        if reasoning:
            print(
                "REASONING:",
                repr(reasoning),
            )

        if chunk.tool_calls:
            print(
                "TOOL CALL:",
                chunk.tool_calls,
            )


if __name__ == "__main__":
    asyncio.run(main())