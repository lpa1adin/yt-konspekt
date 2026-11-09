"""Агенты по подписке: официальные CLI (Claude Code, Codex, Gemini CLI) как бэкенд модели.

Пользователь сам ставит CLI и входит в свой аккаунт (Claude Pro/Max, ChatGPT Plus/Pro, Google). Приложение
не видит и не хранит токенов: оно только запускает установленную программу в неинтерактивном режиме,
передаёт промпт через stdin и читает ответ. Запуск — в пустой временной папке, без инструментов, где CLI
позволяет их отключить, и без API-ключей в окружении, чтобы расходовалась именно подписка.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path

from . import CancelFn, Cancelled, LLMUnavailable, PipelineError

TextFn = Callable[[str], None]

INSTRUCTION = "Выполни задание, приведённое во входных данных. Ответь только итоговым текстом в Markdown."

AGENTS: dict[str, dict] = {
    "claude_code": {
        "title": "Claude Code (подписка Claude Pro / Max)", "command": "claude",
        "login": "claude auth login", "status": ["auth", "status"],
        "install": "npm install -g @anthropic-ai/claude-code (или установщик с claude.com/product/claude-code)",
        "models": ["sonnet", "opus", "haiku"],
        "drop_env": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
    },
    "codex": {
        "title": "Codex CLI (подписка ChatGPT Plus / Pro)", "command": "codex",
        "login": "codex login", "status": ["login", "status"],
        "install": "npm install -g @openai/codex (macOS: brew install codex)",
        "models": [],
        "drop_env": ("OPENAI_API_KEY", "CODEX_API_KEY"),
    },
    "gemini_cli": {
        "title": "Gemini CLI (аккаунт Google)", "command": "gemini",
        "login": "gemini (выберите «Sign in with Google»)", "status": None,
        "install": "npm install -g @google/gemini-cli (macOS: brew install gemini-cli)",
        "models": ["auto", "pro", "flash", "flash-lite"],
        "drop_env": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    },
}

_EXTRA_DIRS = ("~/.local/bin", "~/.claude/local", "/opt/homebrew/bin", "/usr/local/bin", "~/.npm-global/bin",
               "~/.bun/bin", "~/.volta/bin")

_AUTH_MARKERS = ("not logged in", "please run /login", "/login", "log in", "login required", "sign in",
                 "unauthorized", "authentication", "not authenticated", "codex login")
_LIMIT_MARKERS = ("session limit", "weekly limit", "usage limit", "rate limit", "quota", "resets", "limit reached",
                  "too many requests", "429")
_BUSY_MARKERS = ("overloaded", "529", "503", "502", "network", "enotfound", "econnreset", "timed out",
                 "connection error", "unable to connect")


def agent_for_command(command: str) -> str | None:
    name = Path(command or "").name.lower()
    for aid, a in AGENTS.items():
        if a["command"] in name:
            return aid
    return None


def find_executable(command: str) -> str | None:
    """Путь к CLI: явный путь, PATH или типичные папки установки (приложение может стартовать с урезанным PATH)."""
    command = (command or "").strip()
    if not command:
        return None
    p = Path(command).expanduser()
    if p.is_absolute() or os.sep in command:
        return str(p) if p.is_file() and os.access(p, os.X_OK) else None
    found = shutil.which(command)
    if found:
        return found
    for d in _EXTRA_DIRS:
        cand = Path(d).expanduser() / command
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    return None


def child_env(agent: str) -> dict[str, str]:
    """Окружение без API-ключей: иначе CLI возьмёт ключ вместо подписки (у Claude Code ключ важнее входа)."""
    drop = set(AGENTS[agent]["drop_env"])
    env = {k: v for k, v in os.environ.items() if k not in drop}
    extra = [str(Path(d).expanduser()) for d in _EXTRA_DIRS]
    env["PATH"] = os.pathsep.join([env.get("PATH", ""), *extra])
    env.setdefault("NO_COLOR", "1")
    return env


def build_command(agent: str, exe: str, model: str, system_file: str | None) -> list[str]:
    if agent == "claude_code":
        cmd = [exe, "-p", INSTRUCTION, "--output-format", "stream-json", "--verbose", "--include-partial-messages",
               "--tools", "", "--max-turns", "1", "--no-session-persistence"]
        if system_file:
            cmd += ["--system-prompt-file", system_file]
    elif agent == "codex":
        cmd = [exe, "exec", "--skip-git-repo-check", "--ephemeral", "--sandbox", "read-only"]
    else:  # gemini_cli
        cmd = [exe, "-p", INSTRUCTION, "--output-format", "json"]
    if model:
        cmd += ["--model", model]
    if agent == "codex":
        cmd.append("-")  # промпт целиком из stdin
    return cmd


def login_status(agent: str, exe: str, timeout: float = 20) -> tuple[bool | None, str]:
    """(True/False, пояснение) по команде статуса CLI; None — у CLI нет такой команды."""
    args = AGENTS[agent]["status"]
    if not args:
        return None, "вход проверяется при первом запросе"
    try:
        r = subprocess.run([exe, *args], capture_output=True, text=True, timeout=timeout, env=child_env(agent),
                           stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"не удалось запустить: {e}"
    out = (r.stdout or r.stderr or "").strip()
    if agent == "claude_code":
        try:
            method = json.loads(out).get("authMethod", "")
        except ValueError:
            method = ""
        if r.returncode == 0 and method in ("api_key", "api_key_helper"):
            return True, "вход по API-ключу, а не по подписке"
        return r.returncode == 0, ("вход выполнен" if r.returncode == 0 else "вход не выполнен")
    return r.returncode == 0, ("вход выполнен" if r.returncode == 0 else (out[:160] or "вход не выполнен"))


def classify_error(agent: str, text: str, code: int | None) -> PipelineError:
    a = AGENTS[agent]
    low = text.lower()
    tail = " ".join(text.strip().split())[-300:]
    if any(m in low for m in _LIMIT_MARKERS):
        return PipelineError(f"{a['title']}: исчерпан лимит подписки.",
                             "Нажмите «Повторить», когда лимит обновится, — готовые части конспекта сохранены. "
                             "Или выберите в «Настройках» другую модель.")
    if any(m in low for m in _AUTH_MARKERS):
        return PipelineError(f"{a['title']}: не выполнен вход в аккаунт.",
                             f"В терминале выполните «{a['login']}», войдите в аккаунт с подпиской "
                             "и нажмите «Повторить».")
    if any(m in low for m in _BUSY_MARKERS):
        return LLMUnavailable(f"{a['title']}: сервер временно недоступен.", "Задача продолжится сама.")
    return PipelineError(f"{a['title']} завершился с ошибкой (код {code}): {tail}" if tail
                         else f"{a['title']} завершился с ошибкой (код {code}).")


class AgentRun:
    """Один запуск CLI. kill() из другого потока прерывает его (кнопка «Отменить»)."""

    def __init__(self, agent: str, command: str, model: str, timeout: int):
        self.agent = agent
        self.command = command
        self.model = model
        self.timeout = timeout
        self.proc: subprocess.Popen | None = None
        self._killed = False

    def kill(self) -> None:
        self._killed = True
        proc = self.proc
        if proc and proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass

    def run(self, system: str, user: str, on_text: TextFn | None, cancel: CancelFn | None) -> tuple[str, bool]:
        a = AGENTS[self.agent]
        exe = find_executable(self.command)
        if not exe:
            raise PipelineError(f"Не найдена программа {a['command']} ({a['title']}).",
                                f"Установите: {a['install']}. Затем войдите: «{a['login']}».")
        workdir = tempfile.mkdtemp(prefix="yt-konspekt-agent-")
        try:
            system_file = None
            if self.agent == "claude_code":
                system_file = str(Path(workdir) / "system.txt")
                Path(system_file).write_text(system, encoding="utf-8")
                stdin_text = user
            else:  # у Codex и Gemini CLI нет отдельного системного промпта — передаём его в начале
                stdin_text = f"{system}\n\n---\n\n{user}"
            cmd = build_command(self.agent, exe, self.model, system_file)
            return self._spawn(cmd, stdin_text, workdir, on_text, cancel)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def _spawn(self, cmd: list[str], stdin_text: str, workdir: str, on_text: TextFn | None,
               cancel: CancelFn | None) -> tuple[str, bool]:
        kwargs: dict = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
        try:
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                         cwd=workdir, env=child_env(self.agent), text=True, encoding="utf-8",
                                         errors="replace", **kwargs)
        except OSError as e:
            raise PipelineError(f"Не удалось запустить {cmd[0]}: {e}") from e
        proc = self.proc
        err_buf: list[str] = []

        def feed():
            try:
                proc.stdin.write(stdin_text)
                proc.stdin.close()
            except (BrokenPipeError, OSError, ValueError):
                pass

        def drain_err():
            for line in proc.stderr:
                err_buf.append(line)
                del err_buf[:-200]

        def watch():
            while proc.poll() is None:
                if cancel and cancel():
                    self.kill()
                    return
                threading.Event().wait(0.3)

        timer = threading.Timer(self.timeout, self.kill)
        for t in (threading.Thread(target=feed, daemon=True), threading.Thread(target=drain_err, daemon=True),
                  threading.Thread(target=watch, daemon=True), timer):
            t.daemon = True
            t.start()
        try:
            text, truncated, result_error = self._read(proc, on_text)
            code = proc.wait()
        finally:
            timer.cancel()
            self.proc = None
        if cancel and cancel():
            raise Cancelled()
        if self._killed:
            raise PipelineError(f"{AGENTS[self.agent]['title']} не ответил за {self.timeout // 60} мин.",
                                "Нажмите «Повторить» или уменьшите «Размер части» в настройках.")
        if result_error is not None or code != 0 or not text.strip():
            detail = "\n".join(filter(None, [result_error or "", text if code != 0 else "", "".join(err_buf)]))
            raise classify_error(self.agent, detail or "пустой ответ", code)
        return text, truncated

    def _read(self, proc: subprocess.Popen, on_text: TextFn | None) -> tuple[str, bool, str | None]:
        """Читает stdout. Возвращает (текст, обрезан ли, текст ошибки из итогового события или None)."""
        if self.agent == "claude_code":
            acc: list[str] = []
            final, truncated, error = None, False, None
            for line in proc.stdout:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                if ev.get("type") == "stream_event":
                    delta = (ev.get("event") or {}).get("delta") or {}
                    if delta.get("type") == "text_delta" and delta.get("text"):
                        acc.append(delta["text"])
                        if on_text:
                            on_text("".join(acc))
                elif ev.get("type") == "result":
                    final = ev.get("result")
                    truncated = ev.get("stop_reason") == "max_tokens"
                    if ev.get("is_error"):
                        error = str(ev.get("result") or ev.get("errors") or ev.get("subtype") or "ошибка")
            text = final if isinstance(final, str) and final.strip() else "".join(acc)
            return text, truncated, error
        raw = proc.stdout.read()
        if self.agent == "gemini_cli":
            try:
                obj = json.loads(raw[raw.find("{"):]) if "{" in raw else {}
            except ValueError:
                obj = {}
            if obj.get("error"):
                err = obj["error"]
                return "", False, str(err.get("message") if isinstance(err, dict) else err)
            text = str(obj.get("response") or "") if obj else raw
        else:  # codex: в stdout только итоговое сообщение агента
            text = raw
        if on_text and text:
            on_text(text)
        return text, False, None
