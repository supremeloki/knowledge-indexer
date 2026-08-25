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
            if self._term_freqs else 0.0
        )
        self.statistics = IndexStatistics(
            documents=len({d.doc_id for d in documents}),
            chunks=len(self._chunks),
            vocabulary=len(self._doc_freqs),
            total_tokens=sum(freqs.total() for freqs in self._term_freqs),
        )

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def avg_length(self) -> float:
        return round(self._avg_length, 3)

    def idf(self, term: str) -> float:
        occurrences = self._doc_freqs.get(term, 0)
        if occurrences == 0:
            return 0.0
        total = len(self._term_freqs)
        return math.log((total - occurrences + 0.5) / (occurrences + 0.5) + 1.0)

    def bm25_score(self, query_tokens: Sequence[str], chunk_index: int) -> float:
        freqs = self._term_freqs[chunk_index]
        length_norm = 1.0 - BM25_B + BM25_B * (freqs.total() / (self._avg_length or 1.0))
        score = 0.0
        for term in query_tokens:
            term_frequency = freqs.get(term, 0)
            if term_frequency == 0:
                continue
            numerator = term_frequency * (BM25_K1 + 1.0)
            denominator = term_frequency + BM25_K1 * length_norm
            score += self.idf(term) * (numerator / denominator)
        return score

    def search(self, query: str, top_k: int = 5,
               filter_doc_ids: set[str] | None = None) -> list[SearchHit]:
        query_tokens = tokenize(query)
        scored: list[tuple[int, float]] = []
        allowed = filter_doc_ids or {chunk.doc_id for chunk in self._chunks}
        for index, chunk in enumerate(self._chunks):
            if chunk.doc_id not in allowed:
                continue
            score = self.bm25_score(query_tokens, index)
            if score > 0.0:
                scored.append((index, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [
            SearchHit(chunk=self._chunks[index], score=round(score, 6))
            for index, score in scored[:top_k]
        ]


class KnowledgeIndexer:
    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
        self._chunk_size = chunk_size
        self._documents: dict[str, Document] = {}
        self._index: BM25Index | None = None
        self._dirty = False

    def add(self, document: Document) -> "KnowledgeIndexer":
        self._documents[document.doc_id] = document
        self._dirty = True
        return self

    def add_many(self, documents: Iterable[Document]) -> "KnowledgeIndexer":
        for document in documents:
            self.add(document)
        return self

    def remove(self, doc_id: str) -> bool:
        removed = self._documents.pop(doc_id, None) is not None
        if removed:
            self._dirty = True
        return removed

    def build(self) -> BM25Index:
        if not self._documents:
            raise EmptyCorpusError("no documents indexed")
        self._index = BM25Index(list(self._documents.values()), self._chunk_size)
        self._dirty = False
        return self._index

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        index = self._index if (self._index and not self._dirty) else self.build()
        return index.search(query, top_k)

    @property
    def statistics(self) -> IndexStatistics | None:
        return self._index.statistics if self._index and not self._dirty else None
