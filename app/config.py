"""Настройки: значения по умолчанию → config.toml → таблица settings → параметры задачи."""

from __future__ import annotations

import copy
import logging
import logging.handlers
import os
import sys
import tomllib
from pathlib import Path
from typing import Any

APP_NAME = "yt-konspekt"
HOST = "127.0.0.1"
PORT = 8765

DEFAULTS: dict[str, Any] = {
    "output_dir": "~/Documents/Конспекты",
    "format": "docx",            # md | docx | pages (pages — только macOS)
    "auto_export": True,         # сохранять файл сразу после готовности
    "filename_template": "{title}",  # {title} {channel} {date} {id}; «/» — подпапка
    "keep_markdown": False,
    "docx_template": "",         # reference.docx для pandoc, необязательно
    "language": "ru",
    "detail": "full",            # full | short
    "keep_audio": False,
    "cookies_from_browser": "",  # safari | chrome | firefox — при «Sign in to confirm you're not a bot»
    "llm": {
        "backend": "ollama",     # ollama | openai (LM Studio, llama.cpp server)
        "model": "qwen3:8b",
        "base_url": "http://localhost:11434",
        "num_ctx": 16384,
        "temperature": 0.3,
        "chunk_chars": 18000,
        "timeout": 1800,
    },
    "whisper": {
        "engine": "auto",        # auto | mlx | faster
        "model": "large-v3-turbo",
        "language": "",          # пусто — автоопределение
        "compute_type": "int8",
    },
    "subtitles": {
        "use": True,
        "langs": ["ru", "en"],
    },
}


def data_dir() -> Path:
    """Системная папка данных (macOS: ~/Library/Application Support/yt-konspekt)."""
    env = os.environ.get("YT_KONSPEKT_DATA")
    if env:
        p = Path(env).expanduser()
    else:
        try:
            from platformdirs import user_data_dir
            p = Path(user_data_dir(APP_NAME, appauthor=False))
        except ImportError:  # запасной вариант без platformdirs
            if sys.platform == "darwin":
                p = Path.home() / "Library" / "Application Support" / APP_NAME
            elif os.name == "nt":
                p = Path(os.environ.get("LOCALAPPDATA", Path.home())) / APP_NAME
            else:
                p = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / APP_NAME
    p.mkdir(parents=True, exist_ok=True)
    return p


def deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def config_toml_candidates(base: Path | None = None) -> list[Path]:
    paths = []
    if os.environ.get("YT_KONSPEKT_CONFIG"):
        paths.append(Path(os.environ["YT_KONSPEKT_CONFIG"]).expanduser())
    paths.append((base or data_dir()) / "config.toml")
    paths.append(Path.home() / ".config" / APP_NAME / "config.toml")
    return paths


def load_toml(explicit: str | None = None, base: Path | None = None) -> dict:
    candidates = [Path(explicit).expanduser()] if explicit else config_toml_candidates(base)
    for p in candidates:
        if p.is_file():
            with open(p, "rb") as f:
                return tomllib.load(f)
    return {}


def set_dotted(d: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def get_dotted(d: dict, key: str, default: Any = None) -> Any:
    cur: Any = d
    for p in key.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def effective_config(ui_settings: dict[str, Any] | None = None, job_params: dict | None = None,
                     toml: dict | None = None) -> dict:
    """Собирает итоговый конфиг. ui_settings — плоский словарь «llm.model» → значение."""
    cfg = deep_merge(DEFAULTS, toml or {})
    for k, v in (ui_settings or {}).items():
        if not k.startswith("prompt."):
            set_dotted(cfg, k, v)
    p = job_params or {}
    if p.get("detail"):
        cfg["detail"] = p["detail"]
    if p.get("language"):
        cfg["language"] = p["language"]
    if p.get("model"):
        cfg["llm"]["model"] = p["model"]
    if p.get("force_whisper"):
        cfg["subtitles"]["use"] = False
    if "format" in p:
        cfg["format"] = p["format"] or ""
    if p.get("out_dir"):
        cfg["output_dir"] = p["out_dir"]
    return cfg


def setup_logging(base: Path | None = None, level: int = logging.INFO) -> Path:
    logs = (base or data_dir()) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / "app.log"
    root = logging.getLogger()
    if not any(isinstance(h, logging.handlers.RotatingFileHandler) for h in root.handlers):
        fh = logging.handlers.RotatingFileHandler(path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        root.addHandler(sh)
    root.setLevel(level)
    return path
