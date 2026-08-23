from unittest.mock import patch

import pytest

from lisa.config import Settings
from lisa.model import create_chat_model


def test_create_chat_model_with_nvidia():
    settings = Settings(
        model_provider="nvidia",
        model_name="nvidia/nemotron-3.5-lightning-30b-a3b",
    )

    with patch("lisa.model.ChatNVIDIA") as chat_nvidia:
        create_chat_model(settings)

        chat_nvidia.assert_called_once_with(
            model=settings.model_name,
        )

def test_create_chat_model_rejects_unsupported_provider():
    settings = Settings(
        model_provider="unknown",
        model_name="some-model",
    )

    with pytest.raises(ValueError):
        create_chat_model(settings)