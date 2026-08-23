from unittest.mock import Mock

from lisa.application import LISA


def test_lisa_application_initializes():
    model = Mock()

    lisa = LISA(model)

    assert lisa.graph is not None
    assert hasattr(lisa.graph, "invoke")