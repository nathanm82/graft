import pytest
import torch


@pytest.fixture(autouse=True)
def _seed():
    """Make every test deterministic."""
    torch.manual_seed(0)
