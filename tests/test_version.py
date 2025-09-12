import graft


def test_version_is_dotted_string():
    assert isinstance(graft.__version__, str)
    parts = graft.__version__.split(".")
    assert len(parts) >= 2
    assert all(part.isdigit() for part in parts[:2])
