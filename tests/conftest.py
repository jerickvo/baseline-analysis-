"""Shared test setup.

The MotionSense dataset is fetched, never committed. Without it, every test
that loads a real trial used to ERROR from deep inside the loader rather than
skip, so a clean checkout could not tell a broken algorithm from a missing
download. This turns dataset access into a skip when the data is absent; the
synthetic tests -- the ones that check the mathematics -- run regardless.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import loader  # noqa: E402


def _dataset_present() -> bool:
    try:
        loader.resolve_data_root(None)
        return True
    except FileNotFoundError:
        return False


DATASET_PRESENT = _dataset_present()


@pytest.fixture(autouse=True)
def _skip_without_dataset(monkeypatch):
    """Make any real-trial access skip, not fail, when the dataset is absent."""
    if DATASET_PRESENT:
        return

    def _skip(*_args, **_kwargs):
        pytest.skip(
            "MotionSense dataset not found: run scripts/fetch_data.py or set MOTIONSENSE_ROOT"
        )

    for name in ("load_trial", "discover_trials", "summarize_trials", "load_all"):
        monkeypatch.setattr(loader, name, _skip)
