import argparse

import pytest

from scripts.run_repository_overview_experiment import _repositories


def test_repositories_group_related_sources() -> None:
    assert _repositories(["sample=https://one", "sample=https://two"]) == {
        "sample": ["https://one", "https://two"]
    }
    with pytest.raises(argparse.ArgumentTypeError):
        _repositories(["missing-url"])
