"""Шаги пайплайна: источник → расшифровка → LLM → конспект → экспорт.

Модули не знают о базе данных и вебе: прогресс и отмена передаются колбэками.
"""

from __future__ import annotations

from collections.abc import Callable

ProgressFn = Callable[[float, str], None]   # (доля 0..1, сообщение)
CancelFn = Callable[[], bool]               # True — пользователь нажал «Отменить»


class Cancelled(Exception):
    """Задачу отменили."""


class PipelineError(Exception):
    """Понятная пользователю ошибка с подсказкой."""

    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        return f"{self.message} {self.hint}".strip()


class LLMUnavailable(PipelineError):
    """Сервер модели не запущен или нет модели — задача ждёт, а не падает."""


def check_cancel(cancel: CancelFn | None) -> None:
    if cancel and cancel():
        raise Cancelled()


def noop_progress(_frac: float, _msg: str) -> None:
    pass
