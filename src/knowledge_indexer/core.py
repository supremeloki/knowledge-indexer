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

