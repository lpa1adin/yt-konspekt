from __future__ import annotations

from app.cli import process
from app.config import effective_config
from app.pipeline.llm import LLMClient
from tests.fakes import FakeTranscriber, FakeYDLFactory, long_segments, make_info


def test_cli_process_subtitles(tmp_path, llm_server):
    f = FakeYDLFactory().add(make_info("abcdefghijk"), segments=long_segments(3))
    cfg = effective_config(None, {"out_dir": str(tmp_path), "format": "md"})
    out = process("https://youtu.be/abcdefghijk", cfg, ydl_factory=f,
                  llm=LLMClient("ollama", "qwen3:8b", llm_server.url), save_transcript=True)
    text = out.read_text(encoding="utf-8")
    assert out.name == "Тестовое видео.md"
    assert "текст — ручные субтитры (ru)" in text
    assert "[00:00](https://youtu.be/abcdefghijk?t=0)" in text
    assert (tmp_path / "Тестовое видео — расшифровка.txt").exists()
    assert llm_server.unloads == 1


def test_cli_process_whisper(tmp_path, llm_server):
    f = FakeYDLFactory().add(make_info("abcdefghijk", subs=None))
    tr = FakeTranscriber(long_segments(2))
    cfg = effective_config(None, {"out_dir": str(tmp_path), "format": "md"})
    out = process("abcdefghijk", cfg, ydl_factory=f, transcriber=tr,
                  llm=LLMClient("ollama", "qwen3:8b", llm_server.url))
    assert tr.calls == 1 and tr.unloaded == 1 and f.count("audio") == 1
    assert "распознавание Whisper" in out.read_text(encoding="utf-8")
