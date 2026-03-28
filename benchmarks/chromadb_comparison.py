"""Head-to-head benchmark: turboquant-db vs ChromaDB.

Usage:
    python benchmarks/chromadb_comparison.py [--num-vectors 20000] [--num-queries 100]

Requirements:
    pip install chromadb sentence-transformers datasets
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np


def get_texts(n: int) -> list[str]:
    """Load real text data for embedding."""
    try:
        from datasets import load_dataset

        ds = load_dataset("ag_news", split="train")
        texts = [row["text"] for row in ds.select(range(min(n, len(ds))))]
        if len(texts) >= n:
            return texts[:n]
    except Exception:
        pass

    # Fallback: synthetic but realistic-feeling texts
    rng = np.random.default_rng(42)
    topics = [
        "The stock market rallied today as investors reacted to",
        "Scientists announced a breakthrough in",
        "The government unveiled new regulations for",
        "A major earthquake struck the coast of",
        "Researchers at the university published findings on",
        "The company reported quarterly earnings that exceeded",
        "International leaders gathered to discuss",
        "A new study suggests that climate change will",
        "Technology firms are racing to develop",
        "The championship game ended with a dramatic",
    ]
    suffixes = [
        "economic policy changes",
        "quantum computing applications",
        "renewable energy sources",
        "artificial intelligence safety",
        "global supply chain disruptions",
        "healthcare technology innovation",
        "space exploration missions",
        "cybersecurity threats",
        "educational reform proposals",
        "environmental conservation efforts",
    ]
    texts = []
    for i in range(n):
        topic = topics[rng.integers(len(topics))]
        suffix = suffixes[rng.integers(len(suffixes))]
        texts.append(f"{topic} {suffix} (item {i})")
    return texts


def embed_texts(texts: list[str], batch_size: int = 256) -> np.ndarray:
    """Embed texts using sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    return np.array(embeddings, dtype=np.float64)


def get_disk_usage(path: Path) -> int:
    """Get total disk usage of a directory in bytes."""
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def format_bytes(n: int) -> str:
    """Format bytes as human-readable string."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f} GB"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f} MB"
    if n >= 1_000:
        return f"{n / 1_000:.1f} KB"
    return f"{n} B"


def benchmark_chromadb(
    embeddings: np.ndarray,
    ids: list[str],
    query_embeddings: np.ndarray,
    db_path: Path,
    k: int = 10,
) -> dict:
    """Benchmark ChromaDB."""
    import chromadb

    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.create_collection("bench", metadata={"hnsw:space": "cosine"})

    # Index
    t0 = time.perf_counter()
    batch_size = 5000
    for start in range(0, len(ids), batch_size):
        end = min(start + batch_size, len(ids))
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end].tolist(),
        )
    index_time = time.perf_counter() - t0

    disk_usage = get_disk_usage(db_path)

    # Query
    query_times = []
    all_results = []
    for i in range(len(query_embeddings)):
        t0 = time.perf_counter()
        results = collection.query(
            query_embeddings=[query_embeddings[i].tolist()],
            n_results=k,
        )
        query_times.append(time.perf_counter() - t0)
        all_results.append(results["ids"][0])

    return {
        "index_time": index_time,
        "disk_usage": disk_usage,
        "query_times": query_times,
        "results": all_results,
    }


def benchmark_turboquantdb(
    embeddings: np.ndarray,
    ids: list[str],
    query_embeddings: np.ndarray,
    db_path: Path,
    k: int = 10,
) -> dict:
    """Benchmark turboquant-db."""
    from turbodb import TurboDB

    db = TurboDB(db_path)
    dim = embeddings.shape[1]
    collection = db.create_collection("bench", dim=dim, metric="cosine", bit_width=2)

    # Index
    t0 = time.perf_counter()
    batch_size = 5000
    for start in range(0, len(ids), batch_size):
        end = min(start + batch_size, len(ids))
        collection.add(
            ids=ids[start:end],
            vectors=embeddings[start:end],
            metadatas=[{} for _ in range(end - start)],
        )
    index_time = time.perf_counter() - t0

    disk_usage = get_disk_usage(db_path)

    # Query
    query_times = []
    all_results = []
    for i in range(len(query_embeddings)):
        t0 = time.perf_counter()
        results = collection.query(vector=query_embeddings[i], k=k)
        query_times.append(time.perf_counter() - t0)
        all_results.append([r.id for r in results])

    return {
        "index_time": index_time,
        "disk_usage": disk_usage,
        "query_times": query_times,
        "results": all_results,
    }


def compute_recall(
    ground_truth: list[list[str]],
    predicted: list[list[str]],
) -> float:
    """Compute mean recall@k."""
    recalls = []
    for gt, pred in zip(ground_truth, predicted):
        gt_set = set(gt)
        pred_set = set(pred)
        if len(gt_set) == 0:
            continue
        recalls.append(len(gt_set & pred_set) / len(gt_set))
    return float(np.mean(recalls))


def main():
    parser = argparse.ArgumentParser(description="Benchmark turboquant-db vs ChromaDB")
    parser.add_argument("--num-vectors", type=int, default=20000, help="Number of vectors")
    parser.add_argument("--num-queries", type=int, default=100, help="Number of queries")
    parser.add_argument("--k", type=int, default=10, help="Top-k for search")
    args = parser.parse_args()

    n = args.num_vectors
    nq = args.num_queries
    k = args.k

    print(f"Loading {n} texts...")
    texts = get_texts(n)

    print(f"Embedding {len(texts)} texts with all-MiniLM-L6-v2...")
    all_embeddings = embed_texts(texts)
    dim = all_embeddings.shape[1]

    # Split into database and query sets
    rng = np.random.default_rng(42)
    query_indices = rng.choice(len(all_embeddings), size=nq, replace=False)
    query_embeddings = all_embeddings[query_indices]
    ids = [f"doc_{i}" for i in range(n)]

    tmpdir = Path(tempfile.mkdtemp())
    chroma_path = tmpdir / "chromadb"
    turbo_path = tmpdir / "turboquantdb"

    try:
        print("\nBenchmarking ChromaDB...")
        chroma = benchmark_chromadb(all_embeddings, ids, query_embeddings, chroma_path, k)

        print("Benchmarking turboquant-db...")
        turbo = benchmark_turboquantdb(all_embeddings, ids, query_embeddings, turbo_path, k)

        # Compute metrics
        recall = compute_recall(chroma["results"], turbo["results"])
        compression = chroma["disk_usage"] / turbo["disk_usage"] if turbo["disk_usage"] > 0 else 0

        chroma_mean_ms = np.mean(chroma["query_times"]) * 1000
        chroma_p95_ms = np.percentile(chroma["query_times"], 95) * 1000
        turbo_mean_ms = np.mean(turbo["query_times"]) * 1000
        turbo_p95_ms = np.percentile(turbo["query_times"], 95) * 1000

        # Print results
        print()
        print("=" * 58)
        print("  turboquant-db vs ChromaDB Benchmark")
        print("=" * 58)
        print()
        print(f"  Dataset: {n:,} vectors, dim={dim} (all-MiniLM-L6-v2)")
        print(f"  Queries: {nq}")
        print()

        col1 = 22
        col2 = 16
        col3 = 16

        header = f"  {'':>{col1}}  {'ChromaDB':>{col2}}  {'turboquant-db':>{col3}}"
        print(header)
        print("  " + "-" * (col1 + col2 + col3 + 4))

        print(f"  {'Vectors':>{col1}}  {n:>{col2},}  {n:>{col3},}")

        chroma_disk = format_bytes(chroma["disk_usage"])
        turbo_disk = f"{format_bytes(turbo['disk_usage'])} ({compression:.1f}x smaller)"
        print(f"  {'Disk usage':>{col1}}  {chroma_disk:>{col2}}  {turbo_disk}")

        print(f"  {'Index time':>{col1}}  {chroma['index_time']:>{col2}.1f}s  {turbo['index_time']:>{col3}.1f}s")

        chroma_qt = f"{chroma_mean_ms:.1f} ms (p95: {chroma_p95_ms:.1f})"
        turbo_qt = f"{turbo_mean_ms:.1f} ms (p95: {turbo_p95_ms:.1f})"
        print(f"  {'Avg query time':>{col1}}  {chroma_qt:>{col2}}  {turbo_qt}")

        print(f"  {'Recall@{k}':>{col1}}  {'baseline':>{col2}}  {recall:>{col3}.2f}")

        print()

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
