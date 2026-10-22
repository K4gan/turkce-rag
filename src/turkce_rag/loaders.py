"""Belge yükleyiciler: PDF (sayfa bazlı), Markdown (başlık bazlı), düz metin ve web sayfası."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import urlparse

from turkce_rag.models import Block, Document
from turkce_rag.normalize import fold_diacritics, normalize_text, turkish_lower

SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown", ".txt"}
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def slugify(text: str) -> str:
    text = fold_diacritics(turkish_lower(unicodedata.normalize("NFC", text)))
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-") or "belge"


def _markdown_blocks(text: str) -> tuple[str | None, list[Block]]:
    """Markdown'ı başlıklara göre bölümlere ayırır. Bölüm adı ``Üst › Alt`` biçimindedir."""
    title: str | None = None
    stack: list[tuple[int, str]] = []
    blocks: list[Block] = []
    buf: list[str] = []

    def flush() -> None:
        body = normalize_text("\n".join(buf))
        if body:
            section = " › ".join(h for _, h in stack) or None
            blocks.append(Block(text=body, section=section))
        buf.clear()

    in_code = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        m = None if in_code else _HEADING.match(line)
        if m:
            flush()
            level, heading = len(m.group(1)), m.group(2).strip()
            if level == 1 and title is None:
                title = heading
                stack = []
                continue
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading))
        else:
            buf.append(line)
    flush()
    return title, blocks


def load_markdown(path: Path, text: str | None = None) -> Document:
    raw = text if text is not None else path.read_text(encoding="utf-8")
    title, blocks = _markdown_blocks(raw)
    title = title or path.stem.replace("_", " ")
    return Document(doc_id=slugify(path.stem), title=title, source=str(path), blocks=blocks)


def load_text(path: Path) -> Document:
    if path.suffix.lower() in {".md", ".markdown"}:
        return load_markdown(path)
    body = normalize_text(path.read_text(encoding="utf-8"))
    return Document(
        doc_id=slugify(path.stem),
        title=path.stem.replace("_", " "),
        source=str(path),
        blocks=[Block(text=body)] if body else [],
    )


_SENTENCE_END = (".", "!", "?", ":", ";", ",", "…")


def _separate_pdf_headings(text: str) -> str:
    """PDF satırlarında başlıkları paragraftan ayırır.

    pypdf her görsel satırı ``\n`` ile verir; başlık ("Yıllık İzin") ilk cümleye yapışır.
    Kısa, noktalamasız ve bir paragrafın başında duran satırı başlık sayıp ardına boş satır ekler.
    """
    lines = text.split("\n")
    out: list[str] = []
    for i, line in enumerate(lines):
        out.append(line)
        stripped = line.strip()
        prev = lines[i - 1].strip() if i else ""
        at_para_start = i == 0 or not prev or prev.endswith(_SENTENCE_END)
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if (
            stripped
            and nxt
            and at_para_start
            and len(stripped) <= 60
            and len(stripped.split()) <= 8
            and not stripped.endswith(_SENTENCE_END)
            and nxt[:1].isupper()
        ):
            out.append("")
    return "\n".join(out)


def load_pdf(path: Path) -> Document:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    blocks: list[Block] = []
    for i, page in enumerate(reader.pages, start=1):
        body = normalize_text(_separate_pdf_headings(page.extract_text() or ""))
        if body:
            blocks.append(Block(text=body, page=i))
    meta_title = reader.metadata.title if reader.metadata and reader.metadata.title else None
    return Document(
        doc_id=slugify(path.stem),
        title=meta_title or path.stem.replace("_", " "),
        source=str(path),
        blocks=blocks,
    )


def html_to_document(html: str, url: str) -> Document:
    """HTML'i başlık (h1-h3) bazlı bölümlere çevirir; menü/altbilgi gibi gürültüyü atar."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
        tag.decompose()
    root = soup.find("article") or soup.find("main") or soup.body or soup
    title = (soup.title.string or "").strip() if soup.title and soup.title.string else ""
    lines: list[str] = []
    for el in root.find_all(["h1", "h2", "h3", "p", "li", "pre", "blockquote", "td"]):
        txt = el.get_text(" ", strip=True)
        if not txt:
            continue
        if el.name in {"h1", "h2", "h3"}:
            lines.append(f"\n{'#' * int(el.name[1])} {txt}\n")
        else:
            lines.append(txt + "\n")
    md_title, blocks = _markdown_blocks("\n".join(lines))
    parsed = urlparse(url)
    return Document(
        doc_id=slugify(parsed.netloc + parsed.path),
        title=title or md_title or parsed.netloc,
        source=url,
        blocks=blocks,
    )


def load_url(url: str, timeout: float = 20.0) -> Document:
    import httpx

    headers = {"User-Agent": "turkce-rag/0.1 (+https://github.com/K4gan/turkce-rag)"}
    resp = httpx.get(url, timeout=timeout, follow_redirects=True, headers=headers)
    resp.raise_for_status()
    return html_to_document(resp.text, str(resp.url))


def load_path(path: Path) -> Document:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix in SUPPORTED_SUFFIXES:
        return load_text(path)
    raise ValueError(f"Desteklenmeyen dosya türü: {path.suffix} ({path})")


def iter_sources(inputs: Iterable[str]) -> Iterable[Document]:
    """Dosya, klasör (özyinelemeli) ya da http(s) URL listesinden belgeler üretir."""
    for item in inputs:
        if item.startswith(("http://", "https://")):
            yield load_url(item)
            continue
        p = Path(item)
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix.lower() in SUPPORTED_SUFFIXES:
                    yield load_path(f)
        elif p.is_file():
            yield load_path(p)
        else:
            raise FileNotFoundError(item)
