# Semantic Image Search

Search images using natural language, powered by CLIP + turboquant-db.

## Quick Start

```bash
pip install -r requirements.txt
python download_demo_images.py
python app.py --images ./demo_images
```

Then open http://localhost:5000 and start searching.

## How it works

1. `download_demo_images.py` downloads ~90 Creative Commons images from Unsplash spanning nature, animals, food, cities, people, and more
2. `app.py` encodes each image with CLIP, stores the embeddings in turboquant-db (3-bit quantization, ~5x smaller than full precision)
3. When you type a query, it's encoded with CLIP and matched against the stored embeddings

## Example queries

- "sunset at the beach"
- "dog playing in the snow"
- "city skyline at night"
- "people laughing at a party"
- "a cup of coffee on a wooden table"
- "mountains with fog"
- "colorful flowers"
- "person climbing a rock"

## Use your own images

```bash
python app.py --images /path/to/your/photos
```
