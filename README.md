# knowledge-indexer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A BM25 knowledge indexer: document chunking, full Okapi BM25 ranking with IDF scoring, incremental add/remove with dirty rebuilds, and index statistics — the retrieval core for local knowledge bases.

## 🚀 Overview

The evolution of `rag-experiment-v1` toward production: `knowledge-indexer` chunks documents into word-window pieces, builds a proper **Okapi BM25** index (k1=1.5, b=0.75, IDF with length normalization), and serves ranked search over chunk-level granularity. Adding or removing documents marks the index dirty — the next query transparently rebuilds. Statistics expose documents, chunks, vocabulary size, and total tokens.

## ✨ Features

- **Chunking:** word-window splitting (`max_words`, default 120) with stable chunk IDs (`doc#c0`)
- **Full Okapi BM25:** term-frequency saturation + document-length normalization + smoothed IDF
- **Scoped search:** optional `filter_doc_ids` restricts candidates before ranking
- **Zero-score pruning:** non-matching chunks excluded rather than returned at 0.0
- **Dirty tracking:** mutations invalidate the index; reads rebuild automatically
- **Index statistics:** documents / chunks / vocabulary / total tokens captured at build time
- **Zero dependencies**

## 🚧 Structure

```
knowledge-indexer/
├── src/knowledge_indexer/
│   ├── __init__.py
│   └── core.py
├── tests/
│   └── test_core.py
├── README.md
└── pyproject.toml
```

## 📦 Installation

```bash
git clone https://github.com/supremeloki/knowledge-indexer.git
cd knowledge-indexer
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

## 📋 Requirements

- Python 3.11+
- No runtime dependencies

## 🏃 Quick Start

```python
from knowledge_indexer import Document, KnowledgeIndexer

indexer = (
    KnowledgeIndexer()
    .add(Document(doc_id="db", text="database index performance tuning guide"))
    .add(Document(doc_id="nlp", text="persian language model tokenization"))
)

indexer.build()
hits = indexer.search("persian model tokens", top_k=3)
print(hits[0].chunk.doc_id, hits[0].score)
print(indexer.statistics.vocabulary)
```

## 🔧 Error Handling

```text
KnowledgeIndexError
├── EmptyCorpusError      # build() with zero documents
└── chunk-size guard      # max_words < 10 rejected
```

## 🧪 Testing

```bash
pytest tests/ -v
```

## 📝 Code Quality

- Full type hints (`X | None` style), frozen chunks/hits/statistics
- Zero comments — names carry the meaning
- Score ordering and dirty-rebuild behavior explicitly tested

## 📄 License

MIT — see [LICENSE](LICENSE).

## 👤 Author

**Kooroush Masoumi** - [kooroushmasoumi@gmail.com](mailto:kooroushmasoumi@gmail.com)

---

⭐ Star this repo if you find it useful!
