from unittest.mock import patch

import pytest

from lisa.config import Settings
from lisa.model import create_chat_model, create_jev_model


def test_create_chat_model_with_nvidia():
    settings = Settings(
        model_provider="nvidia",
        model_name="nvidia/nemotron-3.5-lightning-30b-a3b",
    )

    with patch("lisa.model.ChatNVIDIA") as chat_nvidia:
        create_chat_model(settings)

        chat_nvidia.assert_called_once_with(
            model=settings.model_name,
            model_kwargs={"chat_template_kwargs": {"enable_thinking": False}},
        )

def test_create_chat_model_rejects_unsupported_provider():
    settings = Settings(
        model_provider="unknown",
        model_name="some-model",
    )

    with pytest.raises(ValueError):
        create_chat_model(settings)

def test_create_jev_model_builds_typesafe_client_from_settings():
    settings = Settings(typesafe_api_key="ts-key", jev_model_name="jev-1.14.0")

    with patch("lisa.model.TypeSafeClient") as typesafe_client:
        create_jev_model(settings)

        typesafe_client.assert_called_once_with(
            api_key="ts-key",
            model="jev-1.14.0",
            timeout=settings.jev_timeout_seconds,
        )


def test_create_jev_model_requires_api_key():
    settings = Settings(typesafe_api_key=None)

    with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
        create_jev_model(settings)
