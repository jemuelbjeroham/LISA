import asyncio

from langchain_core.tools import tool
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from lisa.config import Settings


@tool
def handoff(reason: str) -> str:
    """Hand the current request back to the orchestrator."""
    return reason


async def main():
    settings = Settings()

    model = ChatNVIDIA(
        model=settings.model_name,
    )

    model_with_tools = model.bind_tools([handoff])

    messages = [
        (
            "system",
            """
You are a technical clarification agent.

You are responsible for answering technical questions about
networking, streaming, video engineering, and infrastructure.

If the user's request is within your responsibility, answer it normally.

If the request is outside your responsibility, do NOT answer it.
Instead, call the handoff tool and provide a short reason.

Never answer a request that should be handed off.
""",
        ),
        (
            "human",
            "What is the difference between HLS and DASH?",
        ),
    ]

    print("Starting stream...\n")

    async for chunk in model_with_tools.astream(messages):

        print("=" * 80)
        print("CONTENT:")
        print(repr(chunk.content))

        print("\nTOOL CALLS:")
        print(chunk.tool_calls)

        print("\nINVALID TOOL CALLS:")
        print(chunk.invalid_tool_calls)

        print("\nADDITIONAL KWARGS:")
        print(chunk.additional_kwargs)


if __name__ == "__main__":
    asyncio.run(main())