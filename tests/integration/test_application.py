from unittest.mock import Mock, patch

from lisa.application import LISA


def test_lisa_application_initializes_with_injected_model():
    model = Mock()

    lisa = LISA(model)

    assert lisa.graph is not None
    assert hasattr(lisa.graph, "invoke")


def test_lisa_application_creates_model_when_not_provided():
    model = Mock()

    with patch(
        "lisa.application.create_chat_model",
        return_value=model,
    ) as create_model:
        lisa = LISA()

    create_model.assert_called_once()
    assert lisa.graph is not None