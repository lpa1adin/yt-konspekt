"""Старый CLI без базы и интерфейса: python -m app.cli URL -o ПАПКА -f md|docx|pages."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from .config import effective_config, load_toml
from .pipeline import Cancelled, PipelineError
from .pipeline.export import export_markdown, file_base, open_path
from .pipeline.llm import LLMClient
from .pipeline.source import YouTubeSource, video_id, video_meta, watch_url
from .pipeline.summarize import build_document, linkify_timecodes, load_prompts, summarize
from .pipeline.transcribe import make_transcriber
from .pipeline.transcript import build_transcript


def say(msg: str) -> None:
    print(f"• {msg}", file=sys.stderr, flush=True)


def process(url: str, cfg: dict, *, ydl_factory=None, transcriber=None, llm: LLMClient | None = None,
            save_transcript: bool = False) -> Path:
    vid = video_id(url)
    if not vid:
        raise PipelineError(f"Не похоже на ссылку YouTube: {url}")
    source = YouTubeSource(cfg.get("cookies_from_browser", ""), ydl_factory)
    llm = llm or LLMClient.from_config(cfg)
    llm.check()

    say(f"Получаю данные о видео {vid}…")
    info = source.extract_info(watch_url(vid))
    meta = video_meta(info, vid)
    say(f"«{meta['title']}»")

    got = None
    if cfg["subtitles"].get("use", True):
        got = source.fetch_subtitles(info, cfg["subtitles"].get("langs", ["ru", "en"]))
    if got:
        segs, track = got
        source_desc = track.desc
        say(f"Текст: {source_desc}, {len(segs)} фрагментов")
    else:
        say("Субтитров нет — распознаю речь")
        if transcriber is None:
            transcriber, source_desc, hint = make_transcriber(cfg["whisper"])
            if hint:
                say(hint)
        else:
            source_desc = f"распознавание {getattr(transcriber, 'name', 'whisper')}"
        with tempfile.TemporaryDirectory() as tmp:
            audio = source.download_audio(watch_url(vid), Path(tmp))
            segs = transcriber.transcribe(audio, lambda f, m: print(f"\r  {m}   ", end="", file=sys.stderr))
            print(file=sys.stderr)
        transcriber.unload()
        source_desc = f"распознавание Whisper ({source_desc})"
        if not segs:
            raise PipelineError("Не удалось получить текст видео.")

    transcript = build_transcript(segs)
    say(f"Пишу конспект ({llm.model})…")
    res = summarize(llm, transcript, meta["title"], cfg["language"], cfg["detail"],
                    int(cfg["llm"].get("chunk_chars", 18000)), load_prompts(),
                    on_progress=lambda f, m: say(m) if m.startswith(("часть", "свожу", "собираю")) else None)
    if res.truncated:
        say("Внимание: ответ упёрся в лимит контекста — уменьшите chunk_chars или увеличьте num_ctx.")
    md = build_document(linkify_timecodes(res.markdown, vid), meta, source_desc, llm.model)

    base = file_base(cfg.get("filename_template", "{title}"), meta)
    out = export_markdown(md, cfg["output_dir"], base, cfg["format"] or "md",
                          docx_template=cfg.get("docx_template", ""),
                          keep_markdown=cfg.get("keep_markdown", False))
    if save_transcript:
        (out.parent / f"{out.stem} — расшифровка.txt").write_text(transcript, encoding="utf-8")
    llm.unload()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.cli",
                                 description="Конспект YouTube-видео локальными моделями (Whisper + Ollama).")
    ap.add_argument("urls", nargs="+", help="ссылки на видео YouTube")
    ap.add_argument("-o", "--out", help="папка для сохранения")
    ap.add_argument("-f", "--format", choices=["md", "docx", "pages"], help="формат результата")
    ap.add_argument("--detail", choices=["full", "short"], help="подробный или краткий конспект")
    ap.add_argument("--lang", help="язык конспекта (ru, en, …)")
    ap.add_argument("-m", "--model", help="модель LLM (например qwen3:8b)")
    ap.add_argument("--whisper", action="store_true", help="игнорировать субтитры, распознать речь")
    ap.add_argument("--keep-md", action="store_true", help="сохранить также .md")
    ap.add_argument("--transcript", action="store_true", help="сохранить также расшифровку .txt")
    ap.add_argument("--open", action="store_true", help="открыть результат после создания")
    ap.add_argument("--config", help="путь к config.toml")
    args = ap.parse_args(argv)

    params = {"detail": args.detail, "language": args.lang, "model": args.model,
              "force_whisper": args.whisper, "out_dir": args.out}
    if args.format:
        params["format"] = args.format
    cfg = effective_config(None, params, load_toml(args.config))
    if args.keep_md:
        cfg["keep_markdown"] = True

    code = 0
    for url in args.urls:
        try:
            result = process(url, cfg, save_transcript=args.transcript)
            print(f"✓ {result}")
            if args.open:
                open_path(result)
        except PipelineError as e:
            print(f"✗ {e}", file=sys.stderr)
            code = 1
        except (Cancelled, KeyboardInterrupt):
            print("✗ Прервано.", file=sys.stderr)
            return 130
    return code


if __name__ == "__main__":
    sys.exit(main())
