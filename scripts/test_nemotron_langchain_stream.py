import asyncio
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA


OUTPUT_FILE = Path("nemotron_langchain_stream.txt")


async def main():
    # model = ChatNVIDIA(
    #     model="nvidia/nemotron-3.5-lightning-30b-a3b",
    # )

    model = ChatNVIDIA(
        model="nvidia/nemotron-3.5-lightning-30b-a3b",
        temperature=1,
        top_p=0.95,
        max_completion_tokens=16384,
        model_kwargs={
            "chat_template_kwargs": {
                "enable_thinking": True,
            },
            "reasoning_budget": 16384,
        },
    )
    messages = [
        HumanMessage(
            content="Explain what an MLP is in the context of Transformers."
        )
    ]

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:

        file.write("=" * 80 + "\n")
        file.write("STREAMING CHUNKS\n")
        file.write("=" * 80 + "\n")

        index = 0

        async for chunk in model.astream(messages):

            file.write("\n" + "-" * 80 + "\n")
            file.write(f"CHUNK #{index}\n")
            file.write("-" * 80 + "\n")

            file.write("\nTYPE:\n")
            file.write(f"{type(chunk)}\n")

            file.write("\nCONTENT:\n")
            file.write(repr(chunk.content) + "\n")

            file.write("\nADDITIONAL KWARGS:\n")
            file.write(repr(chunk.additional_kwargs) + "\n")

            file.write("\nRESPONSE METADATA:\n")
            file.write(repr(chunk.response_metadata) + "\n")

            file.write("\nFULL CHUNK:\n")
            file.write(repr(chunk) + "\n")

            index += 1

    print(f"Streaming output written to: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())