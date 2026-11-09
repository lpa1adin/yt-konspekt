"""Агенты по подписке: фейковые `claude`, `codex`, `gemini` — маленькие скрипты вместо настоящих CLI."""

from __future__ import annotations

import json
import os
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from app.pipeline import Cancelled, LLMUnavailable, PipelineError
from app.pipeline.agent_cli import agent_for_command, find_executable
from app.pipeline.llm import LLMClient, provider_for

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="фейковые CLI — shell-скрипты")

FAKE = r'''
import json, os, sys, time
from pathlib import Path
name = Path(sys.argv[0]).name
args = sys.argv[1:]
log = Path(os.environ["FAKE_LOG"])
mode = os.environ.get("FAKE_MODE", "ok")
if args[:2] in (["auth", "status"], ["login", "status"]):
    ok = os.environ.get("FAKE_LOGGED", "1") == "1"
    print(json.dumps({"loggedIn": ok, "authMethod": "claude.ai" if ok else "none"}))
    sys.exit(0 if ok else 1)
data = sys.stdin.read()
system = ""
if "--system-prompt-file" in args:
    system = Path(args[args.index("--system-prompt-file") + 1]).read_text(encoding="utf-8")
log.write_text(json.dumps({"name": name, "args": args, "stdin": data, "system": system, "cwd": os.getcwd(),
                           "env_key": os.environ.get("ANTHROPIC_API_KEY" if name == "claude" else "OPENAI_API_KEY")}),
               encoding="utf-8")
if mode == "slow":
    time.sleep(30)
if mode == "notlogged":
    print("Not logged in · Please run /login", file=sys.stderr)
    sys.exit(1)
if mode == "limit":
    if name == "claude":
        print(json.dumps({"type": "result", "subtype": "success", "is_error": True,
                          "result": "You've hit your session limit · resets 5pm"}))
        sys.exit(1)
    print("ERROR: You've hit your usage limit.", file=sys.stderr)
    sys.exit(1)
answer = "## Кратко\nКонспект от агента [00:05]."
if name == "claude":
    for i in range(0, len(answer), 6):
        print(json.dumps({"type": "stream_event",
                          "event": {"type": "content_block_delta",
                                    "delta": {"type": "text_delta", "text": answer[i:i + 6]}}}), flush=True)
    stop = "max_tokens" if mode == "max_tokens" else "end_turn"
    print(json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": answer,
                      "stop_reason": stop}))
elif name == "gemini":
    print(json.dumps({"response": answer, "stats": {}}))
else:
    print("progress...", file=sys.stderr)
    print(answer)
'''


@pytest.fixture
def fake_bin(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name in ("claude", "codex", "gemini"):
        p = bindir / name
        p.write_text(f"#!{sys.executable}\n" + textwrap.dedent(FAKE), encoding="utf-8")
        p.chmod(0o755)
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "call.json"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
    return bindir


def last_call(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "call.json").read_text(encoding="utf-8"))


def test_agent_detection(fake_bin):
    assert agent_for_command("/usr/local/bin/claude") == "claude_code"
    assert agent_for_command("codex") == "codex" and agent_for_command("gemini") == "gemini_cli"
    assert provider_for("cli", "codex") == "codex"
    assert find_executable(str(fake_bin / "claude")) == str(fake_bin / "claude")
    assert find_executable(str(fake_bin / "nope")) is None


def test_claude_code_streams_with_subscription(fake_bin, tmp_path):
    llm = LLMClient("cli", "sonnet", str(fake_bin / "claude"))
    assert llm.provider == "claude_code" and llm.display_model == "claude (sonnet)"
    assert llm.list_models() == ["sonnet", "opus", "haiku"]
    llm.check()
    seen: list[str] = []
    res = llm.chat("Системный промпт", "Расшифровка [00:05] текст", on_text=seen.append)
    assert res.text == "## Кратко\nКонспект от агента [00:05]." and not res.truncated
    assert len(seen) > 2  # текст приходит по мере написания
    call = last_call(tmp_path)
    assert call["stdin"] == "Расшифровка [00:05] текст" and call["system"] == "Системный промпт"
    args = call["args"]
    assert args[args.index("--tools") + 1] == "" and "--no-session-persistence" in args
    assert args[args.index("--model") + 1] == "sonnet"
    assert call["env_key"] is None  # API-ключ не передаётся — работает подписка
    assert Path(call["cwd"]).name.startswith("yt-konspekt-agent-") and not Path(call["cwd"]).exists()


def test_claude_code_errors(fake_bin, monkeypatch):
    llm = LLMClient("cli", "", str(fake_bin / "claude"))
    assert llm.display_model == "claude (модель по умолчанию)"
    monkeypatch.setenv("FAKE_MODE", "max_tokens")
    assert llm.chat("s", "u").truncated
    monkeypatch.setenv("FAKE_MODE", "limit")
    with pytest.raises(PipelineError, match="лимит подписки"):
        llm.chat("s", "u")
    monkeypatch.setenv("FAKE_MODE", "notlogged")
    with pytest.raises(PipelineError, match="не выполнен вход"):
        llm.chat("s", "u")
    monkeypatch.setenv("FAKE_LOGGED", "0")
    with pytest.raises(PipelineError, match="вход не выполнен"):
        llm.check()


def test_codex_and_gemini(fake_bin, tmp_path):
    codex = LLMClient("cli", "", str(fake_bin / "codex"))
    assert codex.chat("Система", "Текст").text.startswith("## Кратко")
    call = last_call(tmp_path)
    assert call["args"][0] == "exec" and call["args"][-1] == "-" and "--model" not in call["args"]
    assert call["stdin"].startswith("Система\n\n---\n\nТекст") and call["env_key"] is None
    gem = LLMClient("cli", "flash", str(fake_bin / "gemini"))
    assert gem.chat("Система", "Текст").text == "## Кратко\nКонспект от агента [00:05]."
    assert last_call(tmp_path)["args"][-2:] == ["--model", "flash"]


def test_missing_agent_and_cancel(fake_bin, monkeypatch):
    with pytest.raises(PipelineError, match="Не найдена программа"):
        LLMClient("cli", "", str(fake_bin.parent / "nowhere" / "claude")).chat("s", "u")
    monkeypatch.setenv("FAKE_MODE", "slow")
    llm = LLMClient("cli", "", str(fake_bin / "codex"))
    started = time.monotonic()
    flag = threading.Event()
    threading.Timer(0.5, flag.set).start()
    with pytest.raises(Cancelled):
        llm.chat("s", "u", cancel=flag.is_set)
    assert time.monotonic() - started < 10


def test_busy_error_waits(fake_bin, monkeypatch, tmp_path):
    script = fake_bin / "claude"
    script.write_text(f"#!{sys.executable}\nimport sys\nprint('API Error: 529 overloaded', file=sys.stderr)\n"
                      "sys.exit(1)\n", encoding="utf-8")
    with pytest.raises(LLMUnavailable):
        LLMClient("cli", "", str(script)).chat("s", "u")
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-should-not-leak"  # своё окружение не трогаем
