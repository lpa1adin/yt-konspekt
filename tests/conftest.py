from __future__ import annotations

import pytest

from tests.fakes import FakeLLMServer


@pytest.fixture
def llm_server():
    with FakeLLMServer() as s:
        yield s


@pytest.fixture(autouse=True)
def _data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("YT_KONSPEKT_DATA", str(tmp_path / "data"))
    monkeypatch.delenv("YT_KONSPEKT_CONFIG", raising=False)
