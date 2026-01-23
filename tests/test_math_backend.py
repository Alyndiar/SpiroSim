import importlib.util

from spirosim_math import list_backends


def test_numba_backend_availability():
    backends = list_backends(available_only=True)
    names = {backend.name for backend in backends}
    assert "python" in names

    numba_available = importlib.util.find_spec("numba") is not None
    if numba_available:
        assert "numba" in names
    else:
        assert "numba" not in names
