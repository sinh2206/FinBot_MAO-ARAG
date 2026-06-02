from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable

from rag_engine.schema import Document


SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf", ".docx"}


@dataclass(slots=True)
class LoadedFile:
    path: Path
    text: str
    metadata: dict[str, str] = field(default_factory=dict)

    def to_document(self) -> Document:
        return Document(
            id=str(self.path),
            text=self.text,
            metadata={
                "source": str(self.path),
                "filename": self.path.name,
                "extension": self.path.suffix.lower(),
                **self.metadata,
            },
        )


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.parts.append(data)

    def get_text(self) -> str:
        return " ".join(self.parts)


def load_file(path: str | Path) -> LoadedFile:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    if not source.is_file():
        raise ValueError(f"Expected a file path, got directory: {source}")

    suffix = source.suffix.lower()
    if suffix == ".pdf":
        text = _load_pdf(source)
    elif suffix == ".docx":
        text = _load_docx(source)
    elif suffix in {".md", ".markdown"}:
        text = _load_markdown(source)
    elif suffix == ".txt":
        text = source.read_text(encoding="utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported file extension: {suffix}")

    return LoadedFile(path=source, text=clean_text(text))


def load_directory(
    data_dir: str | Path,
    extensions: Iterable[str] = SUPPORTED_EXTENSIONS,
    recursive: bool = True,
) -> list[LoadedFile]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(root)
    allowed = {item.lower() for item in extensions}
    pattern = "**/*" if recursive else "*"
    files = [path for path in root.glob(pattern) if path.is_file() and path.suffix.lower() in allowed]
    return [load_file(path) for path in sorted(files)]


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _load_pdf(path: Path) -> str:
    try:
        from PyPDF2 import PdfReader
    except ImportError as exc:
        raise RuntimeError("PyPDF2 is required to read PDF files") from exc

    reader = PdfReader(str(path))
    pages = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        if page_text.strip():
            pages.append(f"[page {index}]\n{page_text}")
    return "\n\n".join(pages)


def _load_markdown(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    try:
        import markdown
    except ImportError:
        return raw

    html = markdown.markdown(raw)
    parser = _HTMLTextExtractor()
    parser.feed(html)
    return parser.get_text() or raw


def _load_docx(path: Path) -> str:
    try:
        from docx import Document as DocxDocument
    except ImportError as exc:
        raise RuntimeError("python-docx is required to read DOCX files") from exc

    document = DocxDocument(str(path))
    paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
    return "\n\n".join(paragraphs)
