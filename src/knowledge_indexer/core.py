from __future__ import annotations

import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


class KnowledgeIndexError(Exception):
    pass


class EmptyCorpusError(KnowledgeIndexError):
    pass


TOKEN_PATTERN: re.Pattern[str] = re.compile(r"[\w‌]+", re.UNICODE)
DEFAULT_CHUNK_SIZE = 120
BM25_K1 = 1.5
BM25_B = 0.75


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    source: str | None = None


@dataclass(frozen=True)
class IndexedChunk:
    chunk_id: str
    doc_id: str
    text: str
    token_count: int

    @property
    def uid(self) -> str:
        return f"{self.doc_id}#{self.chunk_id}"


@dataclass(frozen=True)
class SearchHit:
    chunk: IndexedChunk
    score: float


@dataclass
class IndexStatistics:
    documents: int = 0
    chunks: int = 0
    vocabulary: int = 0
    total_tokens: int = 0
    built_at: float = field(default_factory=time.time)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def chunk_document(doc: Document, max_words: int = DEFAULT_CHUNK_SIZE) -> list[IndexedChunk]:
    if max_words < 10:
        raise KnowledgeIndexError("max_words must be >= 10")
    words = doc.text.split()
    if not words:
        return []
    chunks: list[IndexedChunk] = []
    for index in range(0, len(words), max_words):
        window = words[index:index + max_words]
        chunks.append(IndexedChunk(
            chunk_id=f"c{len(chunks)}",
            doc_id=doc.doc_id,
            text=" ".join(window),
            token_count=len(window),
        ))
    return chunks


class BM25Index:
    def __init__(self, documents: Sequence[Document],
                 chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
        if not documents:
            raise EmptyCorpusError("at least one document required")
        self._chunks: list[IndexedChunk] = []
        for doc in documents:
            self._chunks.extend(chunk_document(doc, chunk_size))
        self._term_freqs: list[Counter[str]] = [
            Counter(tokenize(chunk.text)) for chunk in self._chunks
        ]
        self._doc_freqs: Counter[str] = Counter()
        for freqs in self._term_freqs:
            self._doc_freqs.update(freqs.keys())
        self._avg_length = (
            sum(freqs.total() for freqs in self._term_freqs) / len(self._term_freqs)
