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

    output_lines = [
        "=" * 80 + "\n",
        "STREAMING CHUNKS\n",
        "=" * 80 + "\n",
    ]

    index = 0

    async for chunk in model.astream(messages):
        output_lines.extend(
            [
                "\n" + "-" * 80 + "\n",
                f"CHUNK #{index}\n",
                "-" * 80 + "\n",
                "\nTYPE:\n",
                f"{type(chunk)}\n",
                "\nCONTENT:\n",
                repr(chunk.content) + "\n",
                "\nADDITIONAL KWARGS:\n",
                repr(chunk.additional_kwargs) + "\n",
                "\nRESPONSE METADATA:\n",
                repr(chunk.response_metadata) + "\n",
                "\nFULL CHUNK:\n",
                repr(chunk) + "\n",
            ]
        )

        index += 1

    await asyncio.to_thread(
        OUTPUT_FILE.write_text,
        "".join(output_lines),
        encoding="utf-8",
    )

    print(f"Streaming output written to: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())