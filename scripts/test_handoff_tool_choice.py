import asyncio

from langchain_core.tools import tool
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from lisa.config import Settings


@tool
def handoff(reason: str) -> str:
    """Hand the current request back to the orchestrator."""
    return reason


async def test_model(tool_choice=None):
    settings = Settings()

    model = ChatNVIDIA(
        model=settings.model_name,
    )

    if tool_choice is None:
        model_with_tools = model.bind_tools([handoff])
        label = "default"
    else:
        model_with_tools = model.bind_tools(
            [handoff],
            tool_choice=tool_choice,
        )
        label = f"tool_choice={tool_choice}"

    messages = [
        (
            "system",
            """
You are the General Enquiry agent in LISA.

You can answer general questions directly.

However, specialized technical troubleshooting such as firewall,
network, infrastructure, and video engineering troubleshooting
must be handled by a specialized agent.

For specialized technical troubleshooting, you MUST call the
handoff tool.

Do not provide a user-facing response before calling the tool.
Do not explain that you are going to call the tool.
""",
        ),
        (
            "human",
            "I am working on an incident where a particular source "
            "is unable to reach a destination because of a firewall. "
            "I need help troubleshooting it.",
        ),
    ]

    print("=" * 80)
    print(f"TEST: {label}")
    print("=" * 80)

    async for chunk in model_with_tools.astream(messages):

        if chunk.content:
            print("\nCONTENT:")
            print(repr(chunk.content))

        if chunk.tool_calls:
            print("\nTOOL CALL:")
            print(chunk.tool_calls)

        if chunk.invalid_tool_calls:
            print("\nINVALID TOOL CALL:")
            print(chunk.invalid_tool_calls)

        reasoning = chunk.additional_kwargs.get("reasoning_content")

        if reasoning:
            print("\nREASONING:")
            print(repr(reasoning))


async def main():
    await test_model()
    await test_model("auto")


if __name__ == "__main__":
    asyncio.run(main())