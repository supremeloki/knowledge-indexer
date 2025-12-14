import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from knowledge_indexer import (
    BM25Index,
    Document,
    EmptyCorpusError,
    KnowledgeIndexError,
    KnowledgeIndexer,
    chunk_document,
    tokenize,
)


DOCS = [
    Document(doc_id="db", text=" ".join(["database index performance"] * 20)),
    Document(doc_id="nlp", text=" ".join(["persian language model tokens"] * 20)),
    Document(doc_id="ops", text=" ".join(["cache warming latency reduction"] * 20)),
]


@pytest.fixture
def indexer():
    return KnowledgeIndexer().add_many(DOCS)


def test_tokenize_lowercases_and_splits():
    assert tokenize("Hello World") == ["hello", "world"]


def test_chunk_document_splits_and_ids():
    doc = Document("d", " ".join(f"w{i}" for i in range(250)))
    chunks = chunk_document(doc, max_words=100)
    assert len(chunks) == 3
    assert [c.chunk_id for c in chunks] == ["c0", "c1", "c2"]
    assert len(chunks[0].text.split()) == 100


def test_chunk_min_size_enforced():
    with pytest.raises(KnowledgeIndexError):
        chunk_document(Document("d", "tiny"), max_words=5)


def test_bm25_index_statistics(indexer):
    index = indexer.build()
    stats = index.statistics
    assert stats.documents == 3
    assert stats.chunks >= 3
    assert stats.vocabulary >= 6
    assert stats.total_tokens > 0


def test_search_ranks_matching_topic_first(indexer):
    hits = indexer.search("persian language model")
    assert hits[0].chunk.doc_id == "nlp"


def test_search_respects_top_k(indexer):
    hits = indexer.search("database", top_k=1)
    assert len(hits) == 1


def test_filter_by_doc_ids(indexer):
    index = indexer.build()
    hits = index.search("persian model", top_k=10, filter_doc_ids={"db"})
    assert all(hit.chunk.doc_id == "db" for hit in hits)


def test_irrelevant_terms_return_empty(indexer):
    hits = indexer.search("zzz qqq nonexistent")
    assert hits == []


def test_add_after_build_dirties_index():
    indexer = KnowledgeIndexer().add_many(DOCS)
    indexer.build()
    indexer.add(Document("new", "fresh content here"))
    assert indexer.statistics is None
    indexer.search("fresh")
    assert indexer.statistics is not None


def test_remove_document(indexer):
    indexer.build()
    removed = indexer.remove("ops")
    assert removed
    hits = indexer.search("cache warming")
    assert all(h.chunk.doc_id != "ops" for h in hits)


def test_empty_corpus_rejected_at_build():
    with pytest.raises(EmptyCorpusError):
