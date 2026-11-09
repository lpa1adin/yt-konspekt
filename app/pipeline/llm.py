"""Клиент LLM: Ollama (/api/chat), OpenAI-совместимые серверы (LM Studio, llama.cpp, ChatGPT, Gemini,
OpenRouter, DeepSeek, Mistral…) и Claude через официальный SDK Anthropic.

Локальные серверы и OpenAI-совместимые API — стандартная библиотека (urllib), потоковый ответ,
отмена закрытием соединения. Claude — пакет `anthropic` (ставится вместе с приложением).
"""

from __future__ import annotations

import json
import logging
import os
import re
import socket
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlparse

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


# Провайдеры для выбора в «Настройках». backend — протокол: ollama | openai (OpenAI-совместимый) | anthropic.
PROVIDERS: dict[str, dict] = {
    "ollama": {"title": "Ollama (локально)", "backend": "ollama", "base_url": "http://localhost:11434",
               "model": "qwen3:8b", "local": True},
    "lmstudio": {"title": "LM Studio / llama.cpp (локально)", "backend": "openai",
                 "base_url": "http://localhost:1234/v1", "model": "", "local": True},
    "anthropic": {"title": "Claude (Anthropic)", "backend": "anthropic", "base_url": "https://api.anthropic.com",
                  "model": "claude-opus-5-5", "env": "ANTHROPIC_API_KEY",
                  "keys_url": "https://console.anthropic.com/settings/keys"},
    "openai": {"title": "ChatGPT (OpenAI)", "backend": "openai", "base_url": "https://api.openai.com/v1",
               "model": "gpt-5-mini", "env": "OPENAI_API_KEY",
               "keys_url": "https://platform.openai.com/api-keys"},
    "gemini": {"title": "Gemini (Google)", "backend": "openai",
               "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
               "model": "gemini-2.5-flash", "env": "GEMINI_API_KEY",
               "keys_url": "https://aistudio.google.com/apikey"},
    "openrouter": {"title": "OpenRouter (сотни моделей)", "backend": "openai",
                   "base_url": "https://openrouter.ai/api/v1", "model": "", "env": "OPENROUTER_API_KEY",
                   "keys_url": "https://openrouter.ai/keys"},
    "deepseek": {"title": "DeepSeek", "backend": "openai", "base_url": "https://api.deepseek.com/v1",
                 "model": "deepseek-chat", "env": "DEEPSEEK_API_KEY",
                 "keys_url": "https://platform.deepseek.com/api_keys"},
    "mistral": {"title": "Mistral", "backend": "openai", "base_url": "https://api.mistral.ai/v1",
                "model": "mistral-large-latest", "env": "MISTRAL_API_KEY",
                "keys_url": "https://console.mistral.ai/api-keys"},
    "custom": {"title": "Другой OpenAI-совместимый API", "backend": "openai", "base_url": "", "model": ""},
}

# Модели Claude, у которых есть уровень effort и серверный запасной вариант при отказе (fallbacks).
_CLAUDE_EFFORT = re.compile(r"^claude-(?:(?:fable|mythos|opus|sonnet|haiku)-5|opus-4-[5-8]|sonnet-4-6)")
_CLAUDE_FALLBACKS = re.compile(r"^claude-(?:fable-5-1|opus-5-5|opus-5|sonnet-5-5)$")


def _host(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def provider_for(backend: str, base_url: str) -> str:
    """Какой провайдер из PROVIDERS соответствует настройкам (для ключа API и подписей)."""
    if backend in ("ollama", "anthropic"):
        return backend
    host = _host(base_url)
    for pid, p in PROVIDERS.items():
        if p["backend"] == "openai" and p["base_url"] and _host(p["base_url"]) == host:
            return pid
    return "lmstudio" if host in ("localhost", "127.0.0.1", "::1", "") else "custom"


def resolve_api_key(cfg_llm: dict, provider: str) -> str:
    """Ключ из «Настроек» или переменной окружения (ANTHROPIC_API_KEY, OPENAI_API_KEY…)."""
    saved = (cfg_llm.get("api_keys") or {}).get(provider) or ""
    if saved:
        return saved
    env = PROVIDERS.get(provider, {}).get("env")
    if provider == "gemini":
        return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
    if provider == "custom":
        return os.environ.get("YT_KONSPEKT_API_KEY", "")
    return os.environ.get(env, "") if env else ""


def mask_key(key: str) -> str:
    return f"…{key[-4:]}" if len(key) > 8 else "сохранён"


class LLMClient:
    def __init__(self, backend: str = "ollama", model: str = "qwen3:8b",
                 base_url: str = "http://localhost:11434", num_ctx: int = 16384,
                 temperature: float = 0.3, timeout: int = 1800, api_key: str = "",
                 effort: str = "medium", provider: str | None = None):
        self.backend = backend
        self.model = model
        self.base = base_url.rstrip("/")
        self.num_ctx = int(num_ctx)
        self.temperature = float(temperature)
        self.timeout = int(timeout)
        self.api_key = api_key or ""
        self.effort = effort or ""
        self.provider = provider or provider_for(backend, self.base)
        self._think_supported = True
        self._temperature_supported = True
        self._fallbacks_supported = True
        self._resp = None
        self._astream = None
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, cfg: dict, model: str | None = None) -> LLMClient:
        llm = cfg["llm"]
        backend = llm.get("backend", "ollama")
        base = llm.get("base_url") or {"ollama": "http://localhost:11434",
                                       "anthropic": "https://api.anthropic.com"}.get(backend,
                                                                                    "http://localhost:1234/v1")
        provider = provider_for(backend, base)
        return cls(backend, model or llm.get("model", "qwen3:8b"), base,
                   llm.get("num_ctx", 16384), llm.get("temperature", 0.3), llm.get("timeout", 1800),
                   api_key=resolve_api_key(llm, provider), effort=llm.get("effort", "medium"), provider=provider)

    @property
    def is_local(self) -> bool:
        return bool(PROVIDERS.get(self.provider, {}).get("local"))

    @property
    def title(self) -> str:
        return PROVIDERS.get(self.provider, PROVIDERS["custom"])["title"]

    # ───────── служебное ─────────

    def _unavailable(self, err: Exception) -> LLMUnavailable:
        if self.backend == "ollama":
            hint = "Запустите Ollama (приложение Ollama или команда `ollama serve`) — задача продолжится сама."
        elif self.is_local:
            hint = "Запустите локальный сервер (LM Studio: Developer → Start Server) — задача продолжится сама."
        else:
            hint = "Проверьте интернет — задача продолжится сама, когда связь появится."
        reason = getattr(err, "reason", err)
        return LLMUnavailable(f"Нет связи с сервером модели {self.base} ({reason}).", hint)

    def _no_key(self) -> PipelineError:
        p = PROVIDERS.get(self.provider, {})
        where = f" Получить ключ: {p['keys_url']}." if p.get("keys_url") else ""
        return PipelineError(f"Не указан API-ключ для {self.title}.",
                             f"Вставьте ключ в «Настройки → Модель».{where}")

    def _http_error(self, e: urllib.error.HTTPError, body: str) -> Exception:
        """Ошибки облачных и локальных OpenAI-совместимых серверов → понятные сообщения."""
        if e.code in (401, 403):
            return PipelineError(f"{self.title}: API-ключ не подошёл ({e.code}).",
                                 "Проверьте ключ в «Настройки → Модель» и баланс аккаунта у провайдера.")
        if e.code == 404:
            return PipelineError(f"{self.title}: модель {self.model} не найдена.",
                                 "Выберите модель из списка в «Настройки → Модель».")
        if e.code == 429 or e.code >= 500:
            return LLMUnavailable(f"{self.title}: сервер занят или лимит запросов ({e.code}).",
                                  "Задача продолжится сама через минуту. Если повторяется — проверьте лимиты "
                                  "и баланс у провайдера.")
        return PipelineError(f"Ошибка сервера модели: {e.code} {body}")

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key and self.backend == "openai":
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _get(self, path: str, timeout: float = 10) -> dict:
        req = urllib.request.Request(f"{self.base}{path}", headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, OSError) as e:
            raise self._unavailable(e) from e

    def _open(self, path: str, payload: dict, timeout: float | None = None):
        req = urllib.request.Request(
            f"{self.base}{path}", data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(), method="POST")
        try:
            return urllib.request.urlopen(req, timeout=timeout or self.timeout)
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, OSError) as e:
            raise self._unavailable(e) from e

    def abort(self) -> None:
        """Прервать текущий потоковый ответ из другого потока (кнопка «Отменить»)."""
        with self._lock:
            resp, astream = self._resp, self._astream
        if astream is not None:
            try:
                astream.close()
            except Exception:  # noqa: BLE001
                pass
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
        if self.backend == "anthropic":
            if not self.api_key:
                raise self._no_key()
            anthropic = _anthropic()
            try:
                return [m.id for m in self._anthropic_client().models.list()]
            except anthropic.AuthenticationError as e:
                raise PipelineError(f"{self.title}: API-ключ не подошёл.",
                                    "Проверьте ключ в «Настройки → Модель».") from e
            except anthropic.APIConnectionError as e:
                raise self._unavailable(e) from e
        if not self.is_local and not self.api_key:
            raise self._no_key()
        try:
            data = self._get("/models").get("data", [])
        except urllib.error.HTTPError as e:
            if e.code == 404:  # провайдер не отдаёт список моделей — не страшно
                return []
            raise self._http_error(e, e.read().decode("utf-8", "replace")[:300]) from e
        ids = [str(m.get("id", "")) for m in data]
        return sorted({i.removeprefix("models/") for i in ids if i})  # Gemini отдаёт «models/…»

    def has_model(self, names: list[str] | None = None) -> bool:
        names = self.list_models() if names is None else names
        if self.backend != "ollama":
            return not names or self.model.removeprefix("models/") in names
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        return wanted in names or self.model in names

    def check(self) -> None:
        """LLMUnavailable, если сервер не запущен или модели нет."""
        names = self.list_models()
        if not self.has_model(names):
            if self.backend == "ollama":
                raise LLMUnavailable(f"В Ollama нет модели {self.model}.",
                                     f"Скачайте её на экране «Состояние» или командой: ollama pull {self.model}")
            if self.is_local:
                raise LLMUnavailable(f"На сервере нет модели {self.model}.", "Загрузите модель в LM Studio.")
            raise PipelineError(f"{self.title}: нет модели {self.model}.",
                                "Выберите модель из списка в «Настройки → Модель».")

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
        elif self.backend == "anthropic":
            raw, truncated = self._chat_anthropic(system, user, on_text, cancel)
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
        if not self.is_local and not self.api_key:
            raise self._no_key()
        payload = {"model": self.model, "messages": messages, "stream": True}
        if self._temperature_supported:
            payload["temperature"] = self.temperature
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
            if e.code == 400 and "temperature" in body.lower() and self._temperature_supported:
                self._temperature_supported = False  # reasoning-модели (GPT-5 и др.) не принимают temperature
                return self._chat_openai(messages, on_text, cancel)
            raise self._http_error(e, body) from e
        return "".join(acc), state["reason"] == "length"

    # ───────── Claude (официальный SDK Anthropic) ─────────

    def _anthropic_client(self):
        anthropic = _anthropic()
        kwargs: dict = {"api_key": self.api_key, "timeout": float(self.timeout)}
        if self.base and _host(self.base) != "api.anthropic.com":
            kwargs["base_url"] = self.base  # прокси или шлюз
        return anthropic.Anthropic(**kwargs)

    def _chat_anthropic(self, system: str, user: str, on_text, cancel) -> tuple[str, bool]:
        if not self.api_key:
            raise self._no_key()
        anthropic = _anthropic()
        client = self._anthropic_client()
        params: dict = {"model": self.model, "max_tokens": 64000, "system": system,
                        "messages": [{"role": "user", "content": user}]}
        if self.effort and _CLAUDE_EFFORT.match(self.model):
            params["output_config"] = {"effort": self.effort}
        use_fallbacks = self._fallbacks_supported and bool(_CLAUDE_FALLBACKS.match(self.model))
        acc: list[str] = []
        try:
            try:
                manager = (client.beta.messages.stream(**params, betas=["server-side-fallback-2026-07-01"],
                                                       fallbacks="default")
                           if use_fallbacks else client.messages.stream(**params))
            except TypeError:  # старая версия SDK без параметра fallbacks
                self._fallbacks_supported = False
                manager = client.messages.stream(**params)
            with manager as stream:
                with self._lock:
                    self._astream = stream
                for piece in stream.text_stream:
                    if cancel and cancel():
                        raise Cancelled()
                    acc.append(piece)
                    if on_text:
                        on_text("".join(acc))
                final = stream.get_final_message()
        except Cancelled:
            raise
        except anthropic.AuthenticationError as e:
            raise PipelineError("Claude: API-ключ не подошёл.",
                                "Проверьте ключ в «Настройки → Модель» и баланс в console.anthropic.com.") from e
        except anthropic.PermissionDeniedError as e:
            raise PipelineError("Claude: у ключа нет доступа к этой модели.", str(e.message)[:300]) from e
        except anthropic.NotFoundError as e:
            raise PipelineError(f"Claude: модель {self.model} не найдена.",
                                "Выберите модель из списка в «Настройки → Модель».") from e
        except anthropic.RateLimitError as e:
            raise LLMUnavailable("Claude: превышен лимит запросов.", "Задача продолжится сама через минуту.") from e
        except anthropic.BadRequestError as e:
            raise PipelineError(f"Claude: запрос отклонён — {e.message}"[:400]) from e
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                raise LLMUnavailable(f"Claude: сервер временно недоступен ({e.status_code}).",
                                     "Задача продолжится сама.") from e
            raise PipelineError(f"Claude: ошибка {e.status_code} — {e.message}"[:400]) from e
        except anthropic.APIConnectionError as e:
            if cancel and cancel():
                raise Cancelled() from e
            raise self._unavailable(e) from e
        except Exception as e:  # обрыв соединения кнопкой «Отменить»
            if cancel and cancel():
                raise Cancelled() from e
            raise
        finally:
            with self._lock:
                self._astream = None
        if cancel and cancel():
            raise Cancelled()
        if final.stop_reason == "refusal":
            raise PipelineError("Claude отказался обрабатывать этот фрагмент.",
                                "Попробуйте другую модель или провайдера в «Настройки → Модель».")
        return "".join(acc), final.stop_reason == "max_tokens"


def _anthropic():
    try:
        import anthropic
    except ImportError as e:
        raise PipelineError("Не установлен пакет anthropic (нужен для Claude).",
                            'В папке приложения выполните: uv pip install -e ".[dev]"') from e
    return anthropic
