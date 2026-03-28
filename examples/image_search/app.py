"""Semantic image search powered by CLIP + turboquant-db.

Usage:
    python examples/image_search/app.py --images /path/to/your/photos
    Then open http://localhost:5000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from flask import Flask, render_template, request, send_from_directory
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from turbodb import TurboDB

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

app = Flask(__name__)

# Global state set in main()
_state: dict = {}


def load_clip():
    """Load CLIP model and processor."""
    model_name = "openai/clip-vit-base-patch32"
    print(f"Loading CLIP model ({model_name})...")
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor


def encode_image(image_path: Path, model, processor) -> np.ndarray:
    """Encode a single image to a CLIP embedding."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        output = model.get_image_features(**inputs)
        if hasattr(output, "pooler_output"):
            features = output.pooler_output
        elif hasattr(output, "image_embeds"):
            features = output.image_embeds
        else:
            features = output
    embedding = features.detach().cpu().numpy().flatten().astype(np.float64)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding


def encode_text(text: str, model, processor) -> np.ndarray:
    """Encode a text query to a CLIP embedding."""
    inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model.get_text_features(**inputs)
        if hasattr(output, "pooler_output"):
            features = output.pooler_output
        elif hasattr(output, "text_embeds"):
            features = output.text_embeds
        else:
            features = output
    embedding = features.detach().cpu().numpy().flatten().astype(np.float64)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding


def find_images(images_dir: Path) -> list[Path]:
    """Find all image files in a directory."""
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(images_dir.rglob(f"*{ext}"))
        images.extend(images_dir.rglob(f"*{ext.upper()}"))
    return sorted(set(images))


def index_images(images_dir: Path, index_path: Path, model, processor):
    """Index all images in the directory."""
    image_paths = find_images(images_dir)
    if not image_paths:
        print(f"No images found in {images_dir}")
        return

    db = TurboDB(index_path)
    collection = db.get_or_create_collection("images", dim=512, metric="cosine", bit_width=3)

    existing_count = collection.count()
    if existing_count == len(image_paths):
        print(f"Index up to date ({existing_count} images)")
        _state["collection"] = collection
        _state["image_count"] = existing_count
        return

    if existing_count > 0:
        print(f"Re-indexing ({existing_count} -> {len(image_paths)} images)...")
        db.delete_collection("images")
        collection = db.create_collection("images", dim=512, metric="cosine", bit_width=3)

    print(f"Indexing {len(image_paths)} images...")
    batch_ids = []
    batch_vectors = []
    batch_metas = []
    batch_size = 50

    for i, path in enumerate(image_paths):
        try:
            embedding = encode_image(path, model, processor)
            rel_path = str(path.relative_to(images_dir))
            batch_ids.append(rel_path)
            batch_vectors.append(embedding)
            batch_metas.append({"filename": path.name, "path": rel_path})

            if len(batch_ids) >= batch_size:
                collection.add(
                    ids=batch_ids,
                    vectors=np.array(batch_vectors),
                    metadatas=batch_metas,
                )
                batch_ids, batch_vectors, batch_metas = [], [], []

            if (i + 1) % 10 == 0 or i == len(image_paths) - 1:
                print(f"  [{i + 1}/{len(image_paths)}] {path.name}")

        except Exception as e:
            print(f"  Skipping {path.name}: {e}")

    if batch_ids:
        collection.add(
            ids=batch_ids,
            vectors=np.array(batch_vectors),
            metadatas=batch_metas,
        )

    print(f"Indexed {collection.count()} images")
    _state["collection"] = collection
    _state["image_count"] = collection.count()


@app.route("/", methods=["GET", "POST"])
def search():
    query = ""
    results = []
    search_time = 0
    image_count = _state.get("image_count", 0)

    if request.method == "POST":
        query = request.form.get("query", "").strip()
        if query and _state.get("collection"):
            embedding = encode_text(query, _state["model"], _state["processor"])
            t0 = time.perf_counter()
            raw_results = _state["collection"].query(vector=embedding, k=20)
            search_time = (time.perf_counter() - t0) * 1000

            results = [
                {"path": r.id, "score": r.score, "filename": r.metadata.get("filename", r.id)}
                for r in raw_results
            ]

    return render_template(
        "index.html",
        query=query,
        results=results,
        search_time=search_time,
        image_count=image_count,
    )


@app.route("/images/<path:filepath>")
def serve_image(filepath):
    return send_from_directory(_state["images_dir"], filepath)


def main():
    parser = argparse.ArgumentParser(description="Semantic image search with CLIP + turboquant-db")
    parser.add_argument("--images", required=True, help="Path to folder of images")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    args = parser.parse_args()

    images_dir = Path(args.images).resolve()
    if not images_dir.is_dir():
        print(f"Error: {images_dir} is not a directory")
        return

    index_path = images_dir / ".turbodb_index"
    _state["images_dir"] = str(images_dir)

    model, processor = load_clip()
    _state["model"] = model
    _state["processor"] = processor

    index_images(images_dir, index_path, model, processor)

    print(f"\nStarting server at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
