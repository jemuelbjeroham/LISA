from lisa.config import Settings
from lisa.model import create_thinking_chat_model

settings = Settings()

model = create_thinking_chat_model(settings)

async for chunk in model.astream("Explain how a transformer works."):
    print(
        "CONTENT:",
        repr(chunk.content),
        "REASONING:",
        repr(
            chunk.additional_kwargs.get("reasoning_content")
        ),
    )