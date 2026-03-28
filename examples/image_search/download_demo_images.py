"""Download demo images from the Unsplash Research Dataset Lite.

Downloads the dataset TSV, selects a diverse subset of images, and downloads
them from the Unsplash CDN at a reasonable resolution.

Usage:
    python download_demo_images.py [--num-images 200] [--output ./demo_images]

The Unsplash Research Dataset Lite is used under the Unsplash Dataset License:
https://unsplash.com/data/lite
"""

from __future__ import annotations

import argparse
import csv
import json
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

DATASET_URL = (
    "https://unsplash.com/data/lite/latest"
)


def download_dataset(cache_dir: Path) -> Path:
    """Download and extract the Unsplash Lite dataset if not cached."""
    photos_path = cache_dir / "photos.csv000"
    keywords_path = cache_dir / "keywords.csv000"

    if photos_path.exists() and keywords_path.exists():
        print("Using cached dataset metadata...")
        return photos_path

    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check for local zip first
    local_zip = Path.home() / "Downloads" / "Unsplash Research Dataset Lite.zip"
    if local_zip.exists():
        print(f"Found local dataset at {local_zip}")
        with zipfile.ZipFile(local_zip) as zf:
            for name in ["photos.csv000", "keywords.csv000"]:
                if name in zf.namelist():
                    zf.extract(name, cache_dir)
        return photos_path

    print("Downloading Unsplash Lite dataset (~300MB)...")
    print("(This is a one-time download)")
    req = urllib.request.Request(DATASET_URL, headers={"User-Agent": "turboquant-db-demo/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()

    with zipfile.ZipFile(BytesIO(data)) as zf:
        for name in ["photos.csv000", "keywords.csv000"]:
            if name in zf.namelist():
                zf.extract(name, cache_dir)

    return photos_path


def load_photos(photos_path: Path) -> list[dict]:
    """Load photo metadata from the TSV."""
    photos = []
    with open(photos_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            desc = row.get("ai_description") or row.get("photo_description") or ""
            image_url = row.get("photo_image_url", "")
            if desc.strip() and image_url.strip():
                photos.append({
                    "photo_id": row["photo_id"],
                    "image_url": image_url,
                    "description": desc.strip(),
                    "location": row.get("photo_location_name", ""),
                })
    return photos


def load_keywords(keywords_path: Path) -> dict[str, list[str]]:
    """Load keywords grouped by photo_id."""
    kw_map: dict[str, list[str]] = {}
    with open(keywords_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            pid = row["photo_id"]
            kw = row.get("keyword", "")
            if kw:
                kw_map.setdefault(pid, []).append(kw)
    return kw_map


def select_diverse_subset(
    photos: list[dict],
    keywords: dict[str, list[str]],
    n: int,
) -> list[dict]:
    """Select a diverse subset using keyword-based stratification."""
    import random

    random.seed(42)

    # Score photos by keyword diversity
    target_categories = [
        "nature", "landscape", "mountain", "forest", "ocean", "beach", "sunset",
        "animal", "dog", "cat", "bird", "wildlife",
        "food", "restaurant", "coffee", "cooking",
        "city", "architecture", "building", "street", "night",
        "people", "portrait", "family", "friends",
        "sport", "fitness", "running",
        "flower", "garden", "plant",
        "car", "travel", "airplane",
        "snow", "winter", "rain", "storm",
        "art", "music", "book",
        "water", "lake", "river", "waterfall",
    ]

    # Bucket photos by category
    buckets: dict[str, list[dict]] = {cat: [] for cat in target_categories}
    uncategorized = []

    for photo in photos:
        pid = photo["photo_id"]
        photo_kws = set(kw.lower() for kw in keywords.get(pid, []))
        photo_kws.add(photo["description"].lower())

        matched = False
        for cat in target_categories:
            if cat in photo_kws or any(cat in kw for kw in photo_kws):
                buckets[cat].append(photo)
                matched = True
                break
        if not matched:
            uncategorized.append(photo)

    # Take proportionally from each bucket
    selected = []
    per_bucket = max(1, n // len(target_categories))
    for cat in target_categories:
        bucket = buckets[cat]
        random.shuffle(bucket)
        selected.extend(bucket[:per_bucket])

    # Fill remaining from uncategorized
    random.shuffle(uncategorized)
    remaining = n - len(selected)
    if remaining > 0:
        selected.extend(uncategorized[:remaining])

    random.shuffle(selected)
    return selected[:n]


def download_image(photo: dict, output_dir: Path, size: int = 640) -> dict | None:
    """Download a single image from Unsplash CDN."""
    url = f"{photo['image_url']}?w={size}&q=80"
    filename = f"{photo['photo_id']}.jpg"
    filepath = output_dir / filename

    if filepath.exists():
        return {**photo, "filename": filename}

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "turboquant-db-demo/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            filepath.write_bytes(resp.read())
        return {**photo, "filename": filename}
    except Exception as e:
        print(f"  Failed: {photo['photo_id']} — {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Download demo images from Unsplash Research Dataset Lite"
    )
    parser.add_argument("--num-images", type=int, default=2000, help="Number of images (default: 2000)")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent / "demo_images"),
        help="Output directory (default: ./demo_images)",
    )
    parser.add_argument("--size", type=int, default=640, help="Image width in pixels (default: 640)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(__file__).parent / ".dataset_cache"

    # Load dataset
    photos_path = download_dataset(cache_dir)
    keywords_path = cache_dir / "keywords.csv000"

    print("Loading metadata...")
    photos = load_photos(photos_path)
    keywords = load_keywords(keywords_path) if keywords_path.exists() else {}
    print(f"  {len(photos)} photos with descriptions")

    # Select diverse subset
    subset = select_diverse_subset(photos, keywords, args.num_images)
    print(f"  Selected {len(subset)} diverse images")

    # Download
    print(f"\nDownloading {len(subset)} images to {output_dir}...")
    results = []
    failed = 0

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(download_image, photo, output_dir, args.size): photo
            for photo in subset
        }
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result:
                results.append(result)
            else:
                failed += 1
            if i % 20 == 0 or i == len(subset):
                print(f"  [{i}/{len(subset)}] downloaded")

    # Save metadata
    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(results, indent=2))

    total_size = sum(f.stat().st_size for f in output_dir.glob("*.jpg"))
    size_mb = total_size / 1_000_000

    print(f"\nDone: {len(results)} images ({size_mb:.1f} MB)")
    if failed:
        print(f"  ({failed} failed)")
    print(f"\nRun the demo:")
    print(f"  python app.py --images {output_dir}")


if __name__ == "__main__":
    main()
