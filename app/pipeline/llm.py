"""Клиент локальной LLM: Ollama (/api/chat) и OpenAI-совместимый сервер (LM Studio, llama.cpp).

Только стандартная библиотека (urllib), потоковый ответ, отмена закрытием соединения.
"""

from __future__ import annotations

import json
import logging
import re
import socket
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

from . import CancelFn, Cancelled, LLMUnavailable, PipelineError

log = logging.getLogger(__name__)

TextFn = Callable[[str], None]


@dataclass
class ChatResult:
    text: str
    truncated: bool = False


def clean_llm_output(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"^.*?</think>", "", text, flags=re.S) if "</think>" in text else text
    text = text.strip()
    m = re.fullmatch(r"```(?:markdown|md)?[ \t]*\n(.*?)\n?```", text, flags=re.S)
    if m:
        text = m.group(1)
    return text.strip()


class LLMClient:
    def __init__(self, backend: str = "ollama", model: str = "qwen3:8b",
                 base_url: str = "http://localhost:11434", num_ctx: int = 16384,
                 temperature: float = 0.3, timeout: int = 1800):
        self.backend = backend
        self.model = model
        self.base = base_url.rstrip("/")
        self.num_ctx = int(num_ctx)
        self.temperature = float(temperature)
        self.timeout = int(timeout)
        self._think_supported = True
        self._resp = None
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, cfg: dict, model: str | None = None) -> LLMClient:
        llm = cfg["llm"]
        base = llm.get("base_url") or ("http://localhost:11434" if llm.get("backend") == "ollama"
                                       else "http://localhost:1234/v1")
        return cls(llm.get("backend", "ollama"), model or llm.get("model", "qwen3:8b"), base,
                   llm.get("num_ctx", 16384), llm.get("temperature", 0.3), llm.get("timeout", 1800))

    # ───────── служебное ─────────

    def _unavailable(self, err: Exception) -> LLMUnavailable:
        if self.backend == "ollama":
            hint = "Запустите Ollama (приложение Ollama или команда `ollama serve`) — задача продолжится сама."
        else:
            hint = "Запустите локальный сервер (LM Studio: Developer → Start Server) — задача продолжится сама."
        reason = getattr(err, "reason", err)
        return LLMUnavailable(f"Нет связи с сервером модели {self.base} ({reason}).", hint)

    def _get(self, path: str, timeout: float = 10) -> dict:
        try:
            with urllib.request.urlopen(f"{self.base}{path}", timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, OSError) as e:
            raise self._unavailable(e) from e

    def _open(self, path: str, payload: dict, timeout: float | None = None):
        req = urllib.request.Request(
            f"{self.base}{path}", data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            return urllib.request.urlopen(req, timeout=timeout or self.timeout)
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, OSError) as e:
            raise self._unavailable(e) from e

    def abort(self) -> None:
        """Прервать текущий потоковый ответ из другого потока (кнопка «Отменить»)."""
        with self._lock:
            resp = self._resp
        if resp is None:
            return
        try:
            sock = resp.fp.raw._sock  # type: ignore[attr-defined]
            sock.shutdown(socket.SHUT_RDWR)
        except Exception:  # noqa: BLE001
            pass
        try:
            resp.close()
        except Exception:  # noqa: BLE001
            pass

    # ───────── модели ─────────

    def list_models(self) -> list[str]:
        if self.backend == "ollama":
            return [m["name"] for m in self._get("/api/tags").get("models", [])]
        return [m.get("id", "") for m in self._get("/models").get("data", [])]

    def has_model(self, names: list[str] | None = None) -> bool:
        names = self.list_models() if names is None else names
        if self.backend != "ollama":
            return not names or self.model in names
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        return wanted in names or self.model in names

    def check(self) -> None:
        """LLMUnavailable, если сервер не запущен или модели нет."""
        names = self.list_models()
        if not self.has_model(names):
            if self.backend == "ollama":
                raise LLMUnavailable(f"В Ollama нет модели {self.model}.",
                                     f"Скачайте её на экране «Состояние» или командой: ollama pull {self.model}")
            raise LLMUnavailable(f"На сервере нет модели {self.model}.", "Загрузите модель в LM Studio.")

    def unload(self) -> None:
        """Выгрузить модель из памяти Ollama (keep_alive: 0)."""
        if self.backend != "ollama":
            return
        try:
            with self._open("/api/generate", {"model": self.model, "keep_alive": 0}, timeout=15) as r:
                r.read()
        except Exception as e:  # noqa: BLE001
            log.debug("unload: %s", e)

    def pull(self, name: str, on_progress: Callable[[dict], None]) -> None:
        """Скачать модель через Ollama /api/pull, передавая прогресс построчно."""
        with self._open("/api/pull", {"model": name, "name": name, "stream": True}, timeout=3600) as r:
            for line in r:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("error"):
                    raise PipelineError(f"Ollama: {obj['error']}")
                on_progress(obj)

    # ───────── генерация ─────────

    def chat(self, system: str, user: str, on_text: TextFn | None = None,
             cancel: CancelFn | None = None) -> ChatResult:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if self.backend == "ollama":
            raw, truncated = self._chat_ollama(messages, on_text, cancel)
        else:
            raw, truncated = self._chat_openai(messages, on_text, cancel)
        if truncated:
            log.warning("Ответ модели обрезан по длине контекста")
        return ChatResult(clean_llm_output(raw), truncated)

    def _stream(self, path: str, payload: dict, handle_line: Callable[[str], bool],
                cancel: CancelFn | None) -> None:
        resp = self._open(path, payload)
        with self._lock:
            self._resp = resp
        try:
            for raw_line in resp:
                if cancel and cancel():
                    raise Cancelled()
                line = raw_line.decode("utf-8", "replace").strip()
                if line and handle_line(line):
                    break
        except (OSError, ValueError, AttributeError) as e:
            if cancel and cancel():
                raise Cancelled() from e
            raise self._unavailable(e) from e
        finally:
            with self._lock:
                self._resp = None
            try:
                resp.close()
            except Exception:  # noqa: BLE001
                pass
        if cancel and cancel():
            raise Cancelled()

    def _chat_ollama(self, messages, on_text, cancel) -> tuple[str, bool]:
        payload = {
            "model": self.model, "messages": messages, "stream": True,
            "options": {"num_ctx": self.num_ctx, "temperature": self.temperature},
        }
        if self._think_supported:
            payload["think"] = False  # отключаем «размышления» у reasoning-моделей (Qwen3 и др.)
        acc: list[str] = []
        state = {"reason": None}

        def handle(line: str) -> bool:
            obj = json.loads(line)
            if obj.get("error"):
                raise PipelineError(f"Ошибка Ollama: {obj['error']}")
            piece = (obj.get("message") or {}).get("content", "")
            if piece:
                acc.append(piece)
                if on_text:
                    on_text("".join(acc))
            if obj.get("done"):
                state["reason"] = obj.get("done_reason")
                return True
            return False

        try:
            self._stream("/api/chat", payload, handle, cancel)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:300]
            if e.code == 400 and self._think_supported:
                self._think_supported = False
                return self._chat_ollama(messages, on_text, cancel)
            if e.code == 404 and "not found" in body:
                raise LLMUnavailable(f"В Ollama нет модели {self.model}.",
                                     f"Скачайте её на экране «Состояние» или: ollama pull {self.model}") from e
            raise PipelineError(f"Ошибка Ollama: {e.code} {body}") from e
        return "".join(acc), state["reason"] == "length"

    def _chat_openai(self, messages, on_text, cancel) -> tuple[str, bool]:
        payload = {"model": self.model, "messages": messages, "temperature": self.temperature,
                   "stream": True}
        acc: list[str] = []
        state = {"reason": None}

        def handle(line: str) -> bool:
            if not line.startswith("data:"):
                return False
            data = line[5:].strip()
            if data == "[DONE]":
                return True
            obj = json.loads(data)
            ch = (obj.get("choices") or [{}])[0]
            piece = (ch.get("delta") or {}).get("content") or ""
            if piece:
                acc.append(piece)
                if on_text:
                    on_text("".join(acc))
            if ch.get("finish_reason"):
                state["reason"] = ch["finish_reason"]
            return False

        try:
            self._stream("/chat/completions", payload, handle, cancel)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:300]
            raise PipelineError(f"Ошибка сервера модели: {e.code} {body}") from e
        return "".join(acc), state["reason"] == "length"
