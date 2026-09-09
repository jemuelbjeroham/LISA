from langchain_core.language_models import BaseChatModel
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai import ChatOpenAI

from lisa.config import Settings


def create_chat_model(settings: Settings) -> BaseChatModel:
    if settings.model_provider == "nvidia":
        return ChatNVIDIA(
            model=settings.model_name
        )
    raise ValueError(f"Unsupported Model Provider: {settings.model_provider}")

def create_router_model(settings: Settings) -> BaseChatModel:
    if settings.router_model_provider =="openai":
        return ChatOpenAI(
            model=settings.router_model_name,
            api_key=settings.openai_token,
        )

    raise ValueError(
        f"Unsupported Router Model Provider: "
        f"{settings.router_model_provider}"
    )

# def create_router_model(settings: Settings) -> BaseChatModel:
#     if settings.router_model_provider == "huggingface":
#         endpoint = HuggingFaceEndpoint(
#             repo_id=settings.router_model_name,
#             huggingfacehub_api_token=settings.hf_token,
#             task="text-generation"
#         )

#         return ChatHuggingFace(
#             llm=endpoint
#         )
#     raise ValueError(
#         f"Unsupported Router Model Provider: "
#         f"{settings.router_model_provider}"
#     )