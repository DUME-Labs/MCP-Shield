"""Shared fixtures. The code under test lives in ../project/src; nothing is installed."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

sys.dont_write_bytecode = True
TESTS = Path(__file__).resolve().parent
# The code is next door (../project) locally; in CI the tests branch is checked out beside
# the main branch, which points MCP_SHIELD_PROJECT at it.
PROJECT = Path(os.environ.get("MCP_SHIELD_PROJECT") or TESTS.parent / "project").resolve()
PROJECT_SRC = PROJECT / "src"
PROJECT_TRAINING = PROJECT / "training"
for p in (str(PROJECT_SRC), str(PROJECT_TRAINING), str(TESTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from helpers import tiny_model  # noqa: E402


@pytest.fixture(autouse=True)
def shield_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Every test gets its own MCP-Shield home (cache, data, config, runtime)."""
    home = tmp_path / "shield-home"
    monkeypatch.setenv("MCP_SHIELD_HOME", str(home))
    return home


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tiny_model.build(tmp_path_factory.mktemp("tiny-model"))


@pytest.fixture
def classifier(tiny_model_dir: Path):
    from mcp_shield.cascade.tier3_classifier import OnnxClassifier

    return OnnxClassifier(tiny_model_dir, fpr_target=0.005)


@pytest.fixture
def store(shield_home: Path):
    from mcp_shield.store import Store

    s = Store(shield_home / "data" / "test.db")
    yield s
    s.close()


class Clock:
    def __init__(self, start: float = 1_800_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> Clock:
    return Clock()


def run(coro):
    """Run a coroutine on the platform's loop (Proactor on Windows)."""
    from mcp_shield.engine.server import new_event_loop

    loop = new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.fixture
def arun():
    return run


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-training-code",
        action="store_true",
        default=False,
        help="run the opt-in check of the train/export code on a toy model",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        if "training_code" in item.keywords and not config.getoption("--run-training-code"):
            item.add_marker(pytest.mark.skip(reason="opt-in: pytest --run-training-code"))
        if "windows" in item.keywords and sys.platform != "win32":
            item.add_marker(pytest.mark.skip(reason="Windows-only"))
        if "posix" in item.keywords and sys.platform == "win32":
            item.add_marker(pytest.mark.skip(reason="POSIX-only"))


_ = asyncio
