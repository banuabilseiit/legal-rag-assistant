"""
Очистка текста и разбиение на чанки (по статьям закона или по абзацам).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class Chunk:
    text: str
    source: str          # имя файла
    doc_type: str        # "law" | "npa" | "internal_answer"
    article: str | None  # номер статьи, если найден


def clean_text(text: str) -> str:
    """Убирает лишние пробелы, повторяющиеся переносы строк, номера страниц."""
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"Страница\s+\d+\s+из\s+\d+", "", text, flags=re.IGNORECASE)
    return text.strip()


def split_by_articles(text: str, source: str, doc_type: str) -> list[Chunk]:
    """
    Пытается резать текст по статьям ("Статья 5.", "Статья 12.1"),
    если статей не найдено — режет обычным текстовым сплиттером по абзацам.
    """
    pattern = re.compile(r"(Статья\s+\d+(?:\.\d+)?\.?)", re.IGNORECASE)
    parts = pattern.split(text)

    if len(parts) <= 1:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " "],
        )
        return [
            Chunk(text=t, source=source, doc_type=doc_type, article=None)
            for t in splitter.split_text(text)
            if t.strip()
        ]

    chunks = []
    # parts выглядит как: [текст_до_первой_статьи, "Статья 1.", текст, "Статья 2.", текст, ...]
    for i in range(1, len(parts), 2):
        article_num = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if body:
            chunks.append(Chunk(
                text=f"{article_num} {body}",
                source=source, doc_type=doc_type, article=article_num,
            ))
    return chunks


def chunks_to_jsonl(chunks: list[Chunk]) -> str:
    """Сериализует список чанков в JSONL-строку (для скачивания/сохранения)."""
    return "\n".join(json.dumps(asdict(c), ensure_ascii=False) for c in chunks)


def save_chunks(chunks: list[Chunk], out_path: Path, append: bool = True) -> None:
    """Сохраняет чанки на диск. По умолчанию дописывает к существующему файлу."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if (append and out_path.exists()) else "w"
    with out_path.open(mode, encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
