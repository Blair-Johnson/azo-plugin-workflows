import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest


@pytest.fixture(autouse=True)
def isolated_storage_environment(tmp_path, monkeypatch):
    # Never let a live harness redirect test fixtures into the user's state home.
    monkeypatch.delenv("AGENT_ZOO_HOME", raising=False)
    monkeypatch.setenv("AGENT_ZOO_STATE_ROOT", str(tmp_path / "state"))
    monkeypatch.setenv("AGENT_ZOO_LOCAL_STATE_ROOT", str(tmp_path / "local"))
