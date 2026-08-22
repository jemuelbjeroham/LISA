from lisa.application import LISA


def test_lisa_appplication_initializes():
    lisa = LISA()

    assert lisa.graph is not None

    