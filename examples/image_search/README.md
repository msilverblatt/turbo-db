# Semantic Image Search

Search images using natural language, powered by CLIP + turboquant-db.

## Quick Start

```bash
pip install -r requirements.txt
python download_demo_images.py          # downloads 200 diverse images from Unsplash
python app.py --images ./demo_images    # launches search UI at localhost:5000
```

Then open http://localhost:5000 and start searching.

## How it works

1. `download_demo_images.py` pulls metadata from the [Unsplash Research Dataset Lite](https://unsplash.com/data/lite) (25K photos), selects a diverse subset across categories (nature, animals, food, cities, people, sports, etc.), and downloads them at 640px
2. `app.py` encodes each image with [CLIP](https://openai.com/research/clip), stores the embeddings in turboquant-db (3-bit quantization)
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

## Options

```bash
# Download more images for better results
python download_demo_images.py --num-images 500

# Use your own photos
python app.py --images /path/to/your/photos

# Change port
python app.py --images ./demo_images --port 8080
```

## Dataset

Images are sourced from the [Unsplash Research Dataset Lite](https://unsplash.com/data/lite) under the [Unsplash Dataset License](https://unsplash.com/data/lite/license). If you have the zip already downloaded at `~/Downloads/Unsplash Research Dataset Lite.zip`, the script will use it automatically instead of re-downloading.
