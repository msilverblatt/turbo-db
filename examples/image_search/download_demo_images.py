"""Download a curated set of demo images from Unsplash.

No API key required. Images are used under the Unsplash License:
https://unsplash.com/license

Usage:
    python download_demo_images.py [--output ./demo_images]
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# (photo_id, category, description) — curated for semantic search diversity
IMAGES = [
    # Nature / Landscapes
    ("photo-1506744038136-46273834b3fb", "nature", "dramatic valley landscape with mountains and river"),
    ("photo-1470071459604-3b5ec3a7fe05", "nature", "foggy green forest with sunlight filtering through trees"),
    ("photo-1441974231531-c6227db76b6e", "nature", "dense forest path covered in green moss"),
    ("photo-1472214103451-9374bd1c798e", "nature", "calm lake reflecting snow-capped mountains at sunrise"),
    ("photo-1469474968028-56623f02e42e", "nature", "golden sunset over the ocean with waves crashing"),
    ("photo-1507525428034-b723cf961d3e", "nature", "tropical beach with white sand and turquoise water"),
    ("photo-1501785888041-af3ef285b470", "nature", "green hills and countryside landscape at golden hour"),
    ("photo-1518173946687-a544bf38a297", "nature", "snow covered mountain peak against blue sky"),
    ("photo-1476514525535-07fb3b4ae5f1", "nature", "waterfall cascading down mossy rocks in forest"),
    ("photo-1509316975850-ff9c5deb0cd9", "nature", "autumn leaves in vibrant red and orange colors"),
    ("photo-1504701954957-2010ec3bcec1", "nature", "starry night sky over desert landscape"),
    ("photo-1505765050516-f72dcac9c60e", "nature", "rolling sand dunes in sahara desert"),
    # Beaches / Ocean
    ("photo-1507525428034-b723cf961d3e", "beach", "pristine tropical beach with palm trees"),
    ("photo-1519046904884-53103b34b206", "beach", "sunset at the beach with golden light on water"),
    ("photo-1473116763249-2faaef81ccda", "beach", "aerial view of ocean waves meeting sandy shore"),
    # Animals
    ("photo-1518791841217-8f162f1e1131", "animals", "orange tabby cat lying on white surface"),
    ("photo-1574158622682-e40e69881006", "animals", "cute cat with green eyes looking at camera"),
    ("photo-1587300003388-59208cc962cb", "animals", "golden retriever dog smiling in sunlight"),
    ("photo-1548199973-03cce0bbc87b", "animals", "two dogs running through autumn leaves"),
    ("photo-1474511320723-9a56873571b7", "animals", "elephant walking across african savanna"),
    ("photo-1437622368342-7a3d73a34c8f", "animals", "sea turtle swimming in clear blue ocean"),
    ("photo-1425082661705-1834bfd09dca", "animals", "hummingbird hovering near red flower"),
    ("photo-1484406566174-9da000fda645", "animals", "horse galloping through snowy field"),
    ("photo-1535930749574-1399327ce78f", "animals", "lion resting in tall grass on savanna"),
    ("photo-1462953491269-9aff00919695", "animals", "butterfly with blue wings on purple flower"),
    # Dogs specifically
    ("photo-1517849845537-4d257902454a", "animals", "puppy playing in the snow"),
    ("photo-1558788353-f76d92f33ddc", "animals", "dog catching frisbee mid-air in park"),
    # Food / Drink
    ("photo-1504674900247-0877df9cc836", "food", "plate of pasta with tomato sauce and fresh basil"),
    ("photo-1565299624946-b28f40a0ae38", "food", "fresh pizza with melted cheese and toppings"),
    ("photo-1567620905732-2d1ec7ab7445", "food", "stack of fluffy pancakes with syrup and berries"),
    ("photo-1546069901-ba9599a7e63c", "food", "colorful sushi platter with salmon and tuna"),
    ("photo-1482049016688-2d3e1b311543", "food", "fresh avocado toast with poached egg"),
    ("photo-1495147466023-ac5c588e0c74", "food", "cup of latte with beautiful foam art"),
    ("photo-1497034825429-c343d7c6a68f", "food", "chocolate cake with rich dark frosting"),
    ("photo-1488477181946-6428a0291777", "food", "bowl of fresh fruit salad with strawberries"),
    ("photo-1506084868230-bb9d95c24759", "food", "cup of coffee on wooden table in morning light"),
    ("photo-1540189549336-e6e99c3679fe", "food", "grilled steak with herbs and vegetables"),
    # Cities / Architecture
    ("photo-1480714378408-67cf0d13bc1b", "cities", "new york city skyline at night with lights"),
    ("photo-1477959858617-67f85cf4f1df", "cities", "city street at night with neon lights and rain"),
    ("photo-1449824913935-59a10b8d2000", "cities", "aerial view of city buildings and streets"),
    ("photo-1467269204594-9661b134dd2b", "cities", "colorful row houses on european street"),
    ("photo-1502602898657-3e91760cbb34", "cities", "eiffel tower in paris at golden hour"),
    ("photo-1513635269975-59663e0ac1ad", "cities", "brooklyn bridge with manhattan skyline"),
    ("photo-1514565131-fce0801e5785", "cities", "modern glass skyscrapers reflecting sky"),
    ("photo-1519501025264-65ba15a82390", "cities", "narrow cobblestone alley in old european town"),
    ("photo-1444723121867-7a241cacace9", "cities", "london tower bridge at twilight"),
    ("photo-1518391846015-55a9cc003b25", "cities", "tokyo street with cherry blossoms and signs"),
    # People
    ("photo-1529156069898-49953e39b3ac", "people", "group of friends laughing together at party"),
    ("photo-1488161628813-04466f0cc7d4", "people", "person standing alone on mountain top"),
    ("photo-1517486808906-6ca8b3f04846", "people", "friends having fun at outdoor music festival"),
    ("photo-1522202176988-66273c2fd55f", "people", "diverse group of students studying together"),
    ("photo-1511632765486-a01980e01a18", "people", "family walking on beach at sunset"),
    ("photo-1516589178581-6cd7833ae3b2", "people", "athlete running on track at sunrise"),
    ("photo-1507003211169-0a1dd7228f2d", "people", "portrait of smiling man with beard"),
    ("photo-1494790108377-be9c29b29330", "people", "portrait of woman with curly hair smiling"),
    # Sports / Action
    ("photo-1461896836934-bd45ba8bf8bd", "sports", "surfer riding a large ocean wave"),
    ("photo-1517649763962-0c623066013b", "sports", "runners competing in marathon race"),
    ("photo-1551958219-acbc608c6377", "sports", "basketball player dunking the ball"),
    ("photo-1530549387789-4c1017266635", "sports", "skier going down powder snow mountain"),
    ("photo-1508098682722-e99c43a406b2", "sports", "rock climber scaling steep cliff face"),
    # Interior / Objects
    ("photo-1505693416388-ac5ce068fe85", "interior", "cozy living room with fireplace and bookshelves"),
    ("photo-1493663284031-b7e3aefcae8e", "interior", "minimalist workspace with laptop and plant"),
    ("photo-1507003211169-0a1dd7228f2d", "objects", "vintage camera on wooden table"),
    ("photo-1501523460185-2aa5d2a0f981", "objects", "stack of old books with reading glasses"),
    ("photo-1513542789411-b6a5d4f31634", "objects", "red bicycle leaning against brick wall"),
    # Weather / Sky
    ("photo-1534088568595-a066f410bcda", "weather", "dramatic storm clouds over open field"),
    ("photo-1501630834273-4b5604d2ee31", "weather", "rainbow arching over green countryside"),
    ("photo-1504608524841-42fe6f032b4b", "weather", "lightning bolt striking during thunderstorm"),
    # Flowers / Gardens
    ("photo-1490750967868-88aa4f44baee", "flowers", "field of sunflowers under blue sky"),
    ("photo-1455659817273-f96807779a8a", "flowers", "single red rose with water droplets"),
    ("photo-1462275646964-a0e3c11f18a6", "flowers", "cherry blossom trees in full bloom"),
    ("photo-1444021465936-c6ca81d39b84", "flowers", "lavender field stretching to the horizon"),
    # Vehicles / Transportation
    ("photo-1494976388531-d1058494cdd8", "vehicles", "classic red sports car on open road"),
    ("photo-1474487548417-781cb71495f3", "vehicles", "airplane wing view above the clouds at sunset"),
    ("photo-1532931899774-fbd12fb1d060", "vehicles", "sailboat on calm ocean at golden hour"),
    # Technology
    ("photo-1518770660439-4636190af475", "technology", "close up of computer circuit board"),
    ("photo-1531297484001-80022131f5a1", "technology", "laptop computers on desk in modern office"),
    # Abstract / Art
    ("photo-1541701494587-cb58502866ab", "abstract", "colorful abstract fluid art painting"),
    ("photo-1507908708918-778587c9e563", "abstract", "neon light trails in long exposure photograph"),
    # Winter / Snow
    ("photo-1491002052546-bf38f186af56", "winter", "snow covered cabin in the woods"),
    ("photo-1457269449834-928af64c684d", "winter", "frozen lake with mountains in background"),
    ("photo-1516912481808-3406841bd33c", "winter", "person walking through deep snow in forest"),
    # Music
    ("photo-1511671782779-c97d3d27a1d4", "music", "concert crowd with colorful stage lights"),
    ("photo-1510915361894-db8b60106cb1", "music", "acoustic guitar leaning against wall"),
    # Water
    ("photo-1433086966358-54859d0ed716", "water", "peaceful river flowing through autumn forest"),
    ("photo-1500375592092-40eb2168fd21", "water", "person swimming in crystal clear turquoise water"),
    # Night
    ("photo-1507400492013-162706c8c05e", "night", "city lights reflecting on wet street at night"),
    ("photo-1519681393784-d120267933ba", "night", "milky way galaxy visible over mountain silhouette"),
]


def download_image(photo_id: str, category: str, desc: str, output_dir: Path) -> dict | None:
    """Download a single image from Unsplash CDN."""
    url = f"https://images.unsplash.com/{photo_id}?w=640&q=80"
    safe_name = photo_id.replace("photo-", "").replace("/", "_")
    filename = f"{category}_{safe_name}.jpg"
    filepath = output_dir / filename

    if filepath.exists():
        return {"filename": filename, "category": category, "description": desc}

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "turboquant-db-demo/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            filepath.write_bytes(resp.read())
        return {"filename": filename, "category": category, "description": desc}
    except Exception as e:
        print(f"  Failed: {photo_id} — {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Download demo images for semantic search")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent / "demo_images"),
        help="Output directory (default: ./demo_images)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(IMAGES)} images to {output_dir}...")

    results = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(download_image, pid, cat, desc, output_dir): (pid, desc)
            for pid, cat, desc in IMAGES
        }
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result:
                results.append(result)
            pid, desc = futures[future]
            print(f"  [{i}/{len(IMAGES)}] {desc[:60]}")

    # Save metadata
    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(results, indent=2))

    print(f"\nDone: {len(results)}/{len(IMAGES)} images saved to {output_dir}")
    print(f"Metadata: {meta_path}")
    print(f"\nRun the demo:")
    print(f"  python app.py --images {output_dir}")


if __name__ == "__main__":
    main()
