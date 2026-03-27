# turbo-db

Lightweight embedded vector database built on [turboquant-py](https://github.com/msilverblatt/turboquant-py). Drop-in replacement for ChromaDB with 16x vector compression.

## Installation

```bash
pip install turbo-db
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
    metadatas=[{"source": "wiki"}, {"source": "arxiv"}],
)

results = collection.query(vector=np.random.randn(384), k=5)
for r in results:
    print(f"{r.id}: {r.score:.3f}")
```
