"""Экспорт конспекта: .md, .docx (pandoc → python-docx), .pages (AppleScript, только macOS)."""

from __future__ import annotations

import datetime as dt
import logging
import os
import plistlib
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from . import PipelineError

log = logging.getLogger(__name__)

FORMATS = ["md", "docx", "pages"]


def available_formats() -> list[str]:
    return FORMATS if sys.platform == "darwin" else ["md", "docx"]


def safe_filename(name: str, limit: int = 120) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", name)
    name = re.sub(r"\s+", " ", name).strip(" .")
    return name[:limit].rstrip(" .") or "konspekt"


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    i = 2
    while True:
        p = path.with_name(f"{path.stem} ({i}){path.suffix}")
        if not p.exists():
            return p
        i += 1


def file_base(template: str, meta: dict, date: dt.date | None = None) -> str:
    """Имя файла по шаблону: {title}, {channel}, {date}, {id}. «/» в шаблоне — подпапка."""
    template = template or "{title}"
    values = {
        "title": meta.get("title") or meta.get("id") or "konspekt",
        "channel": meta.get("channel") or "Без канала",
        "date": f"{(date or dt.date.today()):%Y-%m-%d}",
        "id": meta.get("id") or "",
    }
    parts = []
    for piece in re.split(r"[\\/]", template):
        for k, v in values.items():
            piece = piece.replace("{" + k + "}", str(v))
        piece = safe_filename(piece)
        if piece:
            parts.append(piece)
    return "/".join(parts) or "konspekt"


def normalize_md(md: str) -> str:
    """Пустые строки перед заголовками и началом списков — иначе pandoc их не распознаёт."""
    out: list[str] = []

    def is_list(s: str) -> bool:
        return bool(re.match(r"\s*([-*+]|\d+[.)])\s+", s))

    for line in md.splitlines():
        prev = out[-1] if out else ""
        if prev.strip() and (line.lstrip().startswith("#") or (is_list(line) and not is_list(prev))):
            out.append("")
        out.append(line)
    return "\n".join(out) + "\n"


def check_writable(folder: str | Path) -> tuple[bool, str]:
    """Проверка папки сразу при выборе, а не в конце задачи."""
    if not str(folder).strip():
        return False, "Папка не указана."
    p = Path(folder).expanduser()
    try:
        p.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=p, prefix=".ytk-", delete=True):
            pass
        return True, str(p)
    except OSError as e:
        return False, f"Нет доступа к папке {p}: {e.strerror or e}"


# ───────────────────────── docx ─────────────────────────

def md_to_docx_pandoc(md_path: Path, docx_path: Path, template: str = "") -> bool:
    if not shutil.which("pandoc"):
        return False
    cmd = ["pandoc", str(md_path), "-f", "markdown", "-o", str(docx_path)]
    if template and Path(template).expanduser().is_file():
        cmd += ["--reference-doc", str(Path(template).expanduser())]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log.warning("pandoc не справился (%s), использую встроенный конвертер", r.stderr.strip()[:200])
        return False
    return True


_INLINE = re.compile(r"\*\*(.+?)\*\*|\[([^\]]+)\]\(([^)\s]+)\)|`([^`]+)`|(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)")


def _add_hyperlink(paragraph, text: str, url: str, bold: bool = False):
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    r_id = paragraph.part.relate_to(
        url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", is_external=True)
    link = OxmlElement("w:hyperlink")
    link.set(qn("r:id"), r_id)
    run = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "1A5FB4")
    rpr.append(color)
    u = OxmlElement("w:u")
    u.set(qn("w:val"), "single")
    rpr.append(u)
    if bold:
        rpr.append(OxmlElement("w:b"))
    run.append(rpr)
    t = OxmlElement("w:t")
    t.text = text
    t.set(qn("xml:space"), "preserve")
    run.append(t)
    link.append(run)
    paragraph._p.append(link)


def _add_inline(paragraph, text: str) -> None:
    pos = 0
    for m in _INLINE.finditer(text):
        if m.start() > pos:
            paragraph.add_run(text[pos:m.start()])
        if m.group(1) is not None:
            inner = m.group(1)
            lm = re.fullmatch(r"\[([^\]]+)\]\(([^)\s]+)\)", inner)
            if lm:
                _add_hyperlink(paragraph, lm.group(1), lm.group(2), bold=True)
            else:
                paragraph.add_run(inner).bold = True
        elif m.group(2) is not None:
            _add_hyperlink(paragraph, m.group(2), m.group(3))
        elif m.group(4) is not None:
            paragraph.add_run(m.group(4)).font.name = "Menlo"
        elif m.group(5) is not None:
            paragraph.add_run(m.group(5)).italic = True
        pos = m.end()
    if pos < len(text):
        paragraph.add_run(text[pos:])


def md_to_docx_builtin(md: str, docx_path: Path) -> None:
    try:
        import docx
    except ImportError as e:
        raise PipelineError("Для .docx нужен pandoc или пакет python-docx.",
                            "Установите: brew install pandoc  или  uv pip install python-docx") from e
    doc = docx.Document()
    for raw in md.splitlines():
        line = raw.rstrip()
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if not stripped or re.fullmatch(r"-{3,}|\*{3,}", stripped):
            continue
        h = re.match(r"(#{1,6})\s+(.*)", stripped)
        if h:
            p = doc.add_heading("", level=min(len(h.group(1)) - 1, 4))
            _add_inline(p, h.group(2))
            continue
        b = re.match(r"[-*+]\s+(.*)", stripped)
        n = re.match(r"\d+[.)]\s+(.*)", stripped)
        if b or n:
            style = "List Bullet" if b else "List Number"
            if indent >= 2:
                style += " 2"
            p = doc.add_paragraph(style=style)
            _add_inline(p, (b or n).group(1))
            continue
        p = doc.add_paragraph()
        _add_inline(p, stripped.rstrip("\\").rstrip())
    doc.save(str(docx_path))


PAGES_BUNDLE_IDS = ("com.apple.Pages", "com.apple.iWork.Pages")  # новый и старый идентификаторы


def _bundle_id(app: Path) -> str:
    try:
        with open(app / "Contents" / "Info.plist", "rb") as f:
            return plistlib.load(f).get("CFBundleIdentifier", "")
    except (OSError, plistlib.InvalidFileException, ValueError):
        return ""


def find_pages_app() -> Path | None:
    """Ищет Pages по идентификатору программы, а не по имени папки:
    приложение бывает «Pages.app», «Pages Creator Studio.app» и т. п."""
    if sys.platform != "darwin":
        return None
    for folder in (Path("/Applications"), Path.home() / "Applications"):
        for app in sorted(folder.glob("*Pages*.app")):
            if _bundle_id(app) in PAGES_BUNDLE_IDS:
                return app
    for bid in PAGES_BUNDLE_IDS:  # установлено в нестандартное место — спрашиваем Spotlight
        try:
            r = subprocess.run(["mdfind", f"kMDItemCFBundleIdentifier == '{bid}'"],
                               capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            continue
        for line in r.stdout.splitlines():
            if line.endswith(".app") and Path(line).exists():
                return Path(line)
    return None


def docx_to_pages(docx_path: Path, pages_path: Path) -> bool:
    if sys.platform != "darwin":
        return False
    app = find_pages_app()
    if app is None:
        log.warning("Pages не найден — оставляю .docx")
        return False
    # путь к приложению вставляем литералом: AppleScript должен знать словарь Pages при компиляции
    app_ref = str(app).replace("\\", "\\\\").replace('"', '\\"')
    script = "\n".join([
        "on run argv",
        "    set inFile to POSIX file (item 1 of argv)",
        "    set outFile to POSIX file (item 2 of argv)",
        f'    tell application "{app_ref}"',
        "        activate",
        "        set docCount to count of documents",
        "        open inFile",
        # новые версии Pages возвращают из open не документ, а объект импорта — ждём сам документ
        "        repeat 240 times",
        "            if (count of documents) > docCount then exit repeat",
        "            delay 0.5",
        "        end repeat",
        "        set d to front document",
        "        save d in outFile",
        "        close d saving no",
        "    end tell",
        "end run",
    ])
    r = subprocess.run(["osascript", "-e", script, str(docx_path), str(pages_path)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        log.warning("Не удалось сохранить в Pages: %s", r.stderr.strip()[:200])
        return False
    return True


def export_markdown(md: str, out_dir: str | Path, base: str, fmt: str, *,
                    docx_template: str = "", keep_markdown: bool = False) -> Path:
    """Сохраняет документ и возвращает путь итогового файла. Ошибки записи → PipelineError."""
    if fmt not in FORMATS:
        raise PipelineError(f"Неизвестный формат: {fmt}")
    if fmt == "pages" and sys.platform != "darwin":
        fmt = "docx"
    out = Path(out_dir).expanduser()
    target_dir = (out / base).parent
    stem = Path(base).name
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        md = normalize_md(md)
        md_path = unique_path(target_dir / f"{stem}.md")
        md_path.write_text(md, encoding="utf-8")
        if fmt == "md":
            return md_path

        docx_path = unique_path(target_dir / f"{stem}.docx")
        if not md_to_docx_pandoc(md_path, docx_path, docx_template):
            md_to_docx_builtin(md, docx_path)
        result = docx_path
        if fmt == "pages":
            pages_path = unique_path(target_dir / f"{stem}.pages")
            if docx_to_pages(docx_path, pages_path):
                docx_path.unlink(missing_ok=True)
                result = pages_path
        if not keep_markdown:
            md_path.unlink(missing_ok=True)
        return result
    except OSError as e:
        raise PipelineError(f"Не удалось сохранить файл в {out}: {e.strerror or e}",
                            "Конспект остался в библиотеке — повторите экспорт в другую папку.") from e


# ───────────────────────── открыть / показать ─────────────────────────

def open_path(path: str | Path) -> None:
    p = str(path)
    if sys.platform == "darwin":
        subprocess.Popen(["open", p])
    elif os.name == "nt":
        os.startfile(p)  # type: ignore[attr-defined]
    else:
        subprocess.Popen(["xdg-open", p])


def reveal_path(path: str | Path) -> None:
    p = Path(path)
    if sys.platform == "darwin":
        subprocess.Popen(["open", "-R", str(p)])
    elif os.name == "nt":
        subprocess.Popen(["explorer", f"/select,{p}"])
    else:
        subprocess.Popen(["xdg-open", str(p.parent)])


def pick_folder_dialog(initial: str = "") -> str | None:
    """Системный диалог выбора папки. None — пользователь отменил или диалог недоступен."""
    if sys.platform == "darwin":
        script = 'POSIX path of (choose folder with prompt "Куда сохранять конспекты?")'
        r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
        return r.stdout.strip() or None if r.returncode == 0 else None
    code = (
        "import tkinter as tk, sys\nfrom tkinter import filedialog\n"
        "r=tk.Tk(); r.withdraw(); r.attributes('-topmost', True)\n"
        "p=filedialog.askdirectory(title='Куда сохранять конспекты?', initialdir=sys.argv[1] or None)\n"
        "print(p or '')\n"
    )
    try:
        r = subprocess.run([sys.executable, "-c", code, initial or ""], capture_output=True,
                           text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout.strip() or None if r.returncode == 0 else None
