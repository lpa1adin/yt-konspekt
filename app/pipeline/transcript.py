"""Таймкоды, блоки расшифровки по ~30 секунд и нарезка длинного текста."""

from __future__ import annotations

Segment = tuple[float, str]  # (начало в секундах, текст)


def fmt_ts(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def ts_to_seconds(ts: str) -> int:
    total = 0
    for p in ts.split(":"):
        total = total * 60 + int(p)
    return total


def build_blocks(segs: list[Segment], block_seconds: int = 30) -> list[tuple[float, str]]:
    """Склеивает фрагменты в блоки по ~block_seconds секунд."""
    blocks: list[tuple[float, str]] = []
    cur_start: float | None = None
    cur: list[str] = []
    for start, text in segs:
        if cur_start is None:
            cur_start = start
        if start - cur_start >= block_seconds and cur:
            blocks.append((cur_start, " ".join(cur)))
            cur_start, cur = start, []
        cur.append(text)
    if cur:
        blocks.append((cur_start or 0.0, " ".join(cur)))
    return blocks


def build_transcript(segs: list[Segment], block_seconds: int = 30) -> str:
    """Строки вида «[мм:сс] текст» (или [ч:мм:сс] для видео длиннее часа)."""
    return "\n".join(f"[{fmt_ts(t)}] {text}" for t, text in build_blocks(segs, block_seconds))


def split_chunks(text: str, max_chars: int) -> list[str]:
    """Режет текст по границам строк на куски не длиннее max_chars (если строка не длиннее)."""
    chunks: list[str] = []
    cur: list[str] = []
    size = 0
    for line in text.splitlines():
        if size + len(line) > max_chars and cur:
            chunks.append("\n".join(cur))
            cur, size = [], 0
        cur.append(line)
        size += len(line) + 1
    if cur:
        chunks.append("\n".join(cur))
    return chunks
