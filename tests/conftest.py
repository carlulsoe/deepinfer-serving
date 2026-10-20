import pytest
import torch


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="CUDA unavailable")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
