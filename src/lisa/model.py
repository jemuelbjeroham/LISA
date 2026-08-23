from langchain_core.language_models import BaseChatModel
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from lisa.config import Settings


def create_chat_model(settings: Settings) -> BaseChatModel:
    if settings.model_provider == "nvidia":
        return ChatNVIDIA(
            model=settings.model_name
        )
    raise ValueError(f"Unsupported Model Provider: {settings.model_provider}")