from langchain_core.language_models import BaseChatModel
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai import ChatOpenAI
from typesafe_sdk import TypeSafeClient

from lisa.config import Settings


def create_chat_model(settings: Settings) -> BaseChatModel:
    if settings.model_provider == "nvidia":
        return ChatNVIDIA(
            model=settings.model_name,
            model_kwargs={
                "chat_template_kwargs": {
                    "enable_thinking": False,
                },
            },
        )
    
    if settings.model_provider == "openai":
        return ChatOpenAI(
            model=settings.model_name,
            api_key=settings.openai_token,
        )
    
    raise ValueError(
        f"Unsupported Fallback Model Provider: "
        f"{settings.fallback_model_provider}"
    )


def create_fallback_model(settings: Settings) -> BaseChatModel:
    if settings.fallback_model_provider == "nvidia":
        return ChatNVIDIA(
            model=settings.fallback_model_name,
            model_kwargs={
                "chat_template_kwargs": {
                    "enable_thinking": False,
                }
            },
        )
    if settings.fallback_model_provider == "openai":
        return ChatOpenAI(
            model=settings.fallback_model_name,
            api_key=settings.openai_token,
        )
    raise ValueError(
        f"Unsupported Fallback Model Provider: "
        f"{settings.fallback_model_provider}"
    )

def create_thinking_chat_model(settings: Settings) -> BaseChatModel:
    if settings.model_provider == "nvidia":
        return ChatNVIDIA(
            model=settings.model_name,
            model_kwargs={
                "chat_template_kwargs": {
                    "enable_thinking": True,
                },
                "reasoning_budget": 16384,
            },
        )

    if settings.model_provider == "openai":
        if not _is_openai_reasoning_model(settings.model_name):
            # Non-reasoning models (e.g. gpt-4.1-nano) reject the `reasoning` param.
            return ChatOpenAI(
                model=settings.model_name,
                api_key=settings.openai_token,
                use_responses_api=True,
            )
        return ChatOpenAI(
            model=settings.model_name,
            api_key=settings.openai_token,
            reasoning={
                "effort": "high",
                "summary": "auto",
            },
            use_responses_api=True,
        )

    raise ValueError(f"Unsupported Model Provider: {settings.model_provider}")


def _is_openai_reasoning_model(model_name: str) -> bool:
    name = model_name.lower()
    return name.startswith(("gpt-5", "o1", "o3", "o4"))

def create_jev_model(settings: Settings) -> TypeSafeClient:
    if not settings.typesafe_api_key:
        raise ValueError("TYPESAFE_API_KEY is required to use the Jev model")

    return TypeSafeClient(
        api_key=settings.typesafe_api_key,
        model=settings.jev_model_name,
        timeout=settings.jev_timeout_seconds,
    )

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

def create_technical_model(settings: Settings) -> BaseChatModel:
    if settings.technical_model_provider == "openai":
        return ChatOpenAI(
            model=settings.technical_model_name,
            api_key=settings.openai_token,
        )

    if settings.technical_model_provider == "nvidia":
        return ChatNVIDIA(
            model=settings.technical_model_name,
            model_kwargs={
                "chat_template_kwargs": {
                    "enable_thinking": False,
                },
            }
        )

    raise ValueError(
        f"Unsupported Technical Model Provider: "
        f"{settings.technical_model_provider}"
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