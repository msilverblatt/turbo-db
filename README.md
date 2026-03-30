# turboquant-db

Lightweight embedded vector database built on [turboquant-py](https://github.com/msilverblatt/turboquant-py). Drop-in replacement for ChromaDB with 16x vector compression.

turboquant-db stores vectors using TurboQuant's near-optimal quantization (1-4 bits per coordinate) and metadata in SQLite. It provides a ChromaDB-compatible API with collections, metadata filtering, and concurrent read/write support — all in a few hundred lines of Python with no dependencies beyond turboquant-py and the standard library.

## Installation

```bash
pip install turboquant-db
```

## Quick Start

```python
import numpy as np
from turbodb import TurboDB

db = TurboDB("./my_db")
collection = db.create_collection("docs", dim=384)

collection.add(
    ids=["doc1", "doc2"],
    vectors=np.random.randn(2, 384),
    metadatas=[{"source": "wiki", "year": 2024}, {"source": "arxiv", "year": 2025}],
)

results = collection.query(vector=np.random.randn(384), k=5)
for r in results:
    print(f"{r.id}: {r.score:.3f} — {r.metadata}")
```

## API Reference

### `TurboDB(path)`

Open or create a database at the given directory path.

```python
db = TurboDB("./my_db")
```

**Methods:**

| Method | Description |
|---|---|
| `create_collection(name, dim, metric, bit_width)` | Create a new collection |
| `get_collection(name)` | Open an existing collection |
| `get_or_create_collection(name, dim, metric, bit_width)` | Get or create a collection |
| `delete_collection(name)` | Delete a collection and all its data |
| `list_collections()` | List all collection names |

---

### `Collection`

A named group of quantized vectors with metadata.

```python
collection = db.create_collection("docs", dim=384, metric="cosine", bit_width=2)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Collection name |
| `dim` | `int` | required | Vector dimensionality |
| `metric` | `str` | `"cosine"` | Distance metric: `"cosine"`, `"ip"`, or `"l2"` |
| `bit_width` | `int` | `2` | Bits per coordinate (1-4). Lower = more compression, less accuracy |

**Methods:**

#### `add(ids, vectors, metadatas, documents)`

Add vectors with string IDs, metadata dicts, and optional document text. IDs must be unique.

```python
collection.add(
    ids=["doc1", "doc2"],
    vectors=np.random.randn(2, 384),
    metadatas=[{"source": "wiki"}, {"source": "arxiv"}],
    documents=["The quick brown fox...", "A study of neural networks..."],
)
```

The `documents` parameter is optional. When provided, document text is indexed for BM25 keyword search via `hybrid_query()`.

#### `upsert(ids, vectors, metadatas, documents)`

Insert or replace vectors. If an ID already exists, its vector, metadata, and document are replaced.

```python
collection.upsert(
    ids=["doc1"],
    vectors=new_vector,
    metadatas=[{"source": "updated"}],
    documents=["Updated document text."],
)
```

#### `query(vector, k, where, format)`

Search for the top-k most similar vectors, optionally filtering by metadata.

```python
results = collection.query(vector=query_vec, k=10)
results[0].id        # "doc2"
results[0].score     # 0.934
results[0].metadata  # {"source": "arxiv"}
```

Returns a list of `QueryResult` objects sorted by descending score.

#### `hybrid_query(text, vector, k, fusion, alpha, where, format)`

Hybrid search combining BM25 keyword matching with vector similarity. Requires documents to have been passed to `add()` or `upsert()`.

```python
results = collection.hybrid_query(
    text="quick fox",
    vector=query_vec,
    k=10,
    fusion="rrf",        # "rrf", "weighted", or "dbsf"
    alpha=0.5,           # only used when fusion="weighted"
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text` | `str` | required | Query text for BM25 scoring |
| `vector` | array-like | `None` | Query vector for semantic scoring. If omitted, performs pure BM25 |
| `k` | `int` | `10` | Number of results |
| `fusion` | `str` | `"rrf"` | Fusion strategy (see below) |
| `alpha` | `float` | `0.5` | Vector weight for `"weighted"` fusion. 1.0 = pure vector, 0.0 = pure BM25 |
| `where` | `dict` | `None` | Metadata filter |
| `format` | `str` | `None` | `"chroma"` for ChromaDB format |

**Fusion strategies:**

| Strategy | Description | When to use |
|---|---|---|
| `"rrf"` | Reciprocal Rank Fusion (Cormack et al.) — combines rankings, ignores raw scores | Default. Zero tuning required, solid out-of-the-box |
| `"weighted"` | Convex combination with min-max normalization | Best accuracy when `alpha` is tuned per dataset |
| `"dbsf"` | Distribution-Based Score Fusion — normalizes via mean ± 3·stddev | Robust to score outliers without tuning |

Per Kuzi et al. (ACM TOIS 2023), `"weighted"` outperforms `"rrf"` when alpha is tuned — even ~50 labeled query pairs suffice.

#### `get(ids)`

Retrieve metadata by IDs without performing a search.

```python
items = collection.get(ids=["doc1", "doc2"])
# [{"id": "doc1", "position": 0, "metadata": {"source": "wiki"}}, ...]
```

#### `delete(ids, where)`

Delete vectors by IDs, metadata filter, or both.

```python
collection.delete(ids=["doc1"])
collection.delete(where={"source": {"$eq": "wiki"}})
```

#### `compact()`

Rewrite storage to reclaim space from deleted vectors.

```python
collection.compact()
```

#### `count()` / `name` / `dim` / `metric`

```python
collection.count()   # number of live vectors
collection.name      # "docs"
collection.dim       # 384
collection.metric    # "cosine"
```

---

### `QueryResult`

Frozen dataclass returned by `query()` and `hybrid_query()`.

| Attribute | Type | Description |
|---|---|---|
| `id` | `str` | Vector ID |
| `score` | `float` | Similarity score (higher = more similar for cosine/ip) |
| `metadata` | `dict` | Associated metadata |
| `document` | `str \| None` | Document text, if stored |

---

### ChromaDB compatibility

Pass `format="chroma"` to get results in ChromaDB's column-oriented format:

```python
results = collection.query(vector=query_vec, k=10, format="chroma")
results["ids"][0]        # ["doc2", "doc5", ...]
results["distances"][0]  # [0.934, 0.891, ...]
results["metadatas"][0]  # [{"source": "arxiv"}, ...]
```

This makes migration straightforward — change the import, update the constructor, and add `format="chroma"` to your query calls. Remove `format="chroma"` at your own pace.

## Metadata Filtering

Filter syntax matches ChromaDB and Pinecone conventions:

```python
# Comparison operators
collection.query(vector=v, k=10, where={"year": {"$eq": 2025}})
collection.query(vector=v, k=10, where={"year": {"$ne": 2024}})
collection.query(vector=v, k=10, where={"year": {"$gt": 2023}})
collection.query(vector=v, k=10, where={"year": {"$gte": 2024}})
collection.query(vector=v, k=10, where={"year": {"$lt": 2026}})
collection.query(vector=v, k=10, where={"year": {"$lte": 2025}})

# Set operators
collection.query(vector=v, k=10, where={"source": {"$in": ["wiki", "arxiv"]}})
collection.query(vector=v, k=10, where={"source": {"$nin": ["blog"]}})

# Logical combinators
collection.query(vector=v, k=10, where={
    "$and": [
        {"year": {"$gte": 2024}},
        {"source": {"$eq": "arxiv"}},
    ]
})

collection.query(vector=v, k=10, where={
    "$or": [
        {"source": {"$eq": "wiki"}},
        {"year": {"$gt": 2024}},
    ]
})
```

Multiple top-level fields are implicitly ANDed:

```python
# Equivalent to $and
collection.query(vector=v, k=10, where={"year": {"$gte": 2024}, "source": {"$eq": "arxiv"}})
```

## Distance Metrics

| Metric | Description | Score interpretation |
|---|---|---|
| `cosine` (default) | Cosine similarity | 1.0 = identical, 0.0 = orthogonal |
| `ip` | Inner product | Higher = more similar |
| `l2` | Squared L2 distance | Lower = more similar |

All metrics use TurboQuant's inner-product quantizer under the hood. Cosine normalizes vectors on add; L2 is computed from stored norms and inner products.

## Compression

turbo-db compresses vectors using TurboQuant's Lloyd-Max quantization with random orthogonal rotation:

| Bit-width | Compression ratio | Use case |
|---|---|---|
| 1 | 32x | Maximum compression, rough similarity |
| 2 (default) | 16x | Good balance of quality and size |
| 3 | 10.7x | Higher accuracy |
| 4 | 8x | Near-lossless similarity search |

At the default 2-bit setting, a collection of 1M 384-dimensional vectors uses ~9.6 MB for vector data, compared to ~1.5 GB uncompressed.

## Storage

Each database is a directory. Each collection is a subdirectory:

```
my_db/
├── docs/
│   ├── vectors/        # Quantized vectors (numpy arrays)
│   ├── metadata.db     # SQLite: IDs, metadata, positions
│   └── lock            # Write lock file
├── embeddings/
│   └── ...
└── turbodb.json        # Database config
```

Metadata is stored in SQLite with WAL mode for concurrent read/write access. Vector data uses turboquant-py's bit-packed numpy format.

## Concurrency

- **Multiple readers + one writer**: SQLite WAL mode allows concurrent reads during writes
- **Write serialization**: File locking ensures one write operation at a time per collection
- **Crash safety**: Vectors are written before metadata is committed. On restart, orphaned vectors are automatically trimmed to match SQLite state

## Migrating from ChromaDB

```python
# Before (ChromaDB)
import chromadb
client = chromadb.PersistentClient(path="./db")
collection = client.create_collection("docs")
collection.add(ids=["a"], embeddings=[[1, 2, 3]], metadatas=[{"k": "v"}])
results = collection.query(query_embeddings=[[1, 2, 3]], n_results=5)

# After (turboquant-db)
from turbodb import TurboDB
db = TurboDB("./db")
collection = db.create_collection("docs", dim=3)
collection.add(ids=["a"], vectors=[[1, 2, 3]], metadatas=[{"k": "v"}])
results = collection.query(vector=[1, 2, 3], k=5)
# Or with Chroma-compat format:
results = collection.query(vector=[1, 2, 3], k=5, format="chroma")
```

Key differences:
- `embeddings` → `vectors`
- `query_embeddings` → `vector` (single vector, not nested list)
- `n_results` → `k`
- `dim` is required on `create_collection`
- Results are `QueryResult` objects by default (use `format="chroma"` for column dicts)

## References

- **TurboQuant:** [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- **QJL:** [arXiv:2406.03482](https://arxiv.org/abs/2406.03482)
- **turboquant-py:** [github.com/msilverblatt/turboquant-py](https://github.com/msilverblatt/turboquant-py)
