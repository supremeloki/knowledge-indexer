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
