#!/usr/bin/env python3
"""
Download real food images from public sources for test fixtures.

Sources (no authentication required):
  - Hugging Face ``food101`` dataset (test split — held out from the Swin model)
  - Wikimedia Commons (CC-licensed, permanent URLs)

Images land in ``test/fixtures/images/`` with deterministic filenames.
The companion ``test/fixtures/manifest.json`` records expected labels and
approximate nutrition ground truth.

Usage:
    python scripts/download_test_images.py              # download all
    python scripts/download_test_images.py --dry-run    # print what would happen
    python scripts/download_test_images.py --source hf  # only Hugging Face
    python scripts/download_test_images.py --source wikimedia  # only Wikimedia
"""

import argparse
import hashlib
import json
import os
import sys
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

# --- paths ----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "test" / "fixtures" / "images"
MANIFEST_PATH = ROOT / "test" / "fixtures" / "manifest.json"

# --- Wikimedia Commons images (CC-licensed, permanent hotlinks) ------------
# Each entry: (filename, url, food_label, notes)
WIKIMEDIA_IMAGES = [
    (
        "donut_wikimedia.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/e/ed/Pink-Frosted-Donut.jpg",
        "donuts",
        "Pink frosted donut on parchment — the original test image",
    ),
    (
        "pizza_wikimedia.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/a/a3/Eq_it-na_pizza-margherita_sep2005_sml.jpg",
        "pizza",
        "Margherita pizza, top-down view",
    ),
    (
        "sushi_wikimedia.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/8/8b/Sushi_platter.jpg",
        "sushi",
        "Assorted sushi platter",
    ),
    (
        "steak_wikimedia.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/3/3c/Grilled_rib_eye_steak_with_vegetables.jpg",
        "steak",
        "Grilled rib eye steak with vegetables",
    ),
    (
        "caesar_salad_wikimedia.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/2/23/Caesar_salad_%282%29.jpg",
        "caesar_salad",
        "Caesar salad close-up",
    ),
    (
        "hamburger_wikimedia.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/4/47/Hamburger_%28black_bg%29.jpg",
        "hamburger",
        "Classic hamburger on dark background",
    ),
    (
        "ice_cream_wikimedia.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/2/2e/Ice_cream_with_whipped_cream%2C_chocolate_syrup%2C_and_a_wafer.jpg",
        "ice_cream",
        "Ice cream sundae with toppings",
    ),
    (
        "french_fries_wikimedia.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/8/83/French_Fries.JPG",
        "french_fries",
        "French fries in a bowl",
    ),
    (
        "waffles_wikimedia.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/5/5b/Waffles_with_Strawberries.jpg",
        "waffles",
        "Waffles with strawberries",
    ),
    (
        "pancakes_wikimedia.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/4/40/Food_Pancakes.jpg",
        "pancakes",
        "Stack of pancakes with butter and syrup",
    ),
    (
        "chicken_wings_wikimedia.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/e/ed/Buffalo_chicken_wings.jpg",
        "chicken_wings",
        "Buffalo chicken wings",
    ),
    (
        "pad_thai_wikimedia.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/7/7c/Pad_Thai_with_chicken_and_vegetables.jpg",
        "pad_thai",
        "Pad Thai with chicken and vegetables",
    ),
    (
        "ramen_wikimedia.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/7/77/Tonkotsu_ramen_with_chashu_pork.jpg",
        "ramen",
        "Tonkotsu ramen with chashu pork",
    ),
    (
        "tacos_wikimedia.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/7/73/001_Tacos_de_carnitas%2C_carne_asada_y_al_pastor.jpg",
        "tacos",
        "Assorted Mexican tacos",
    ),
    (
        "cheesecake_wikimedia.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/8/8e/Baked_cheesecake_with_raspberries_and_blueberries.jpg",
        "cheesecake",
        "Baked cheesecake with berries",
    ),
]

# --- approximate ground-truth nutrition (per-item, from USDA FDC / common knowledge) ---
# This is NOT weighed lab data — it is "best available" for evaluating whether the
# model + catalog pipeline is directionally reasonable.  Marked clearly as approximate.
NUTRITION_GROUND_TRUTH = {
    "donuts": {
        "calories": 452,
        "mass": 100,
        "protein": 5,
        "fat": 25,
        "carbs": 51,
        "source": "USDA FDC 1841655 (glazed doughnut, per 100g); typical doughnut ~60-100g",
    },
    "pizza": {
        "calories": 266,
        "mass": 100,
        "protein": 11,
        "fat": 10,
        "carbs": 33,
        "source": "USDA FDC 17329229 (restaurant cheese pizza, per 100g)",
    },
    "sushi": {
        "calories": 145,
        "mass": 100,
        "protein": 7,
        "fat": 1,
        "carbs": 28,
        "source": "USDA FDC 2342411 (California roll, per 100g)",
    },
    "steak": {
        "calories": 271,
        "mass": 100,
        "protein": 25,
        "fat": 19,
        "carbs": 0,
        "source": "USDA FDC 168636 (rib eye steak, separable lean+fat, broiled, per 100g)",
    },
    "caesar_salad": {
        "calories": 157,
        "mass": 100,
        "protein": 5,
        "fat": 13,
        "carbs": 6,
        "source": "USDA FDC 2342484 (Caesar salad with dressing + croutons, per 100g)",
    },
    "hamburger": {
        "calories": 265,
        "mass": 100,
        "protein": 15,
        "fat": 15,
        "carbs": 20,
        "source": "USDA FDC 2342480 (cheeseburger, single patty, per 100g)",
    },
    "ice_cream": {
        "calories": 207,
        "mass": 100,
        "protein": 4,
        "fat": 11,
        "carbs": 24,
        "source": "USDA FDC 171287 (vanilla ice cream, per 100g)",
    },
    "french_fries": {
        "calories": 312,
        "mass": 100,
        "protein": 4,
        "fat": 15,
        "carbs": 41,
        "source": "USDA FDC 170698 (restaurant french fries, per 100g)",
    },
    "waffles": {
        "calories": 291,
        "mass": 100,
        "protein": 8,
        "fat": 14,
        "carbs": 33,
        "source": "USDA FDC 2342489 (plain waffle, per 100g)",
    },
    "pancakes": {
        "calories": 227,
        "mass": 100,
        "protein": 6,
        "fat": 10,
        "carbs": 28,
        "source": "USDA FDC 168758 (plain pancake, per 100g)",
    },
    "chicken_wings": {
        "calories": 247,
        "mass": 100,
        "protein": 20,
        "fat": 18,
        "carbs": 1,
        "source": "USDA FDC 2342483 (chicken wing, fried, meat+skin, per 100g)",
    },
    "pad_thai": {
        "calories": 170,
        "mass": 100,
        "protein": 5,
        "fat": 6,
        "carbs": 23,
        "source": "USDA estimated; typical restaurant pad thai ~150-200 kcal/100g",
    },
    "ramen": {
        "calories": 110,
        "mass": 100,
        "protein": 3,
        "fat": 4,
        "carbs": 16,
        "source": "USDA estimated; ramen varies widely by broth/toppings",
    },
    "tacos": {
        "calories": 226,
        "mass": 100,
        "protein": 11,
        "fat": 12,
        "carbs": 20,
        "source": "USDA FDC 2342495 (beef taco, per 100g)",
    },
    "cheesecake": {
        "calories": 321,
        "mass": 100,
        "protein": 6,
        "fat": 22,
        "carbs": 26,
        "source": "USDA FDC 2342482 (cheesecake, per 100g)",
    },
}


def fetch_image(url, timeout=60):
    """Download an image from *url* and return (bytes, content_type)."""
    req = Request(url, headers={"User-Agent": "food-tracker-test/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        ct = resp.headers.get("Content-Type", "")
        return resp.read(), ct


def download_wikimedia(dry_run=False):
    """Download Wikimedia images and return manifest entries."""
    entries = []
    for fname, url, label, notes in WIKIMEDIA_IMAGES:
        dest = IMAGE_DIR / fname
        entry = {
            "id": fname,
            "image": f"test/fixtures/images/{fname}",
            "source": "wikimedia",
            "source_url": url,
            "expected_label": label,
            "notes": notes,
        }
        # Attach approximate nutrition ground truth if we have it
        if label in NUTRITION_GROUND_TRUTH:
            entry["ground_truth"] = NUTRITION_GROUND_TRUTH[label]
        entries.append(entry)

        if dry_run:
            print(f"[dry-run] {url}  ->  {dest}")
            continue

        if dest.exists():
            print(f"[skip] {dest} (already exists)")
            continue

        print(f"[fetch] {url}")
        try:
            data, ct = fetch_image(url)
            dest.write_bytes(data)
            print(f"  -> {dest} ({len(data)} bytes, {ct})")
        except Exception as exc:
            print(f"  !! FAILED: {exc}", file=sys.stderr)
            entry["download_error"] = str(exc)
    return entries


def download_huggingface_food101(dry_run=False, max_per_class=2):
    """
    Download test-split images from the ``food101`` dataset on Hugging Face.

    The test split contains 25,250 images (250 per class) that were NOT used
    to train the Swin Food-101 model — they are a legitimate held-out set.
    We sample a diverse subset to keep the fixtures small.
    """
    entries = []
    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "[warn] 'datasets' library not installed. "
            "Install with: pip install datasets\n"
            "  Skipping Hugging Face food101 download.",
            file=sys.stderr,
        )
        return entries

    # Classes to sample — pick diverse, visually distinct foods
    sample_classes = [
        "apple_pie", "bibimbap", "bread_pudding", "chocolate_cake",
        "club_sandwich", "crab_cakes", "deviled_eggs", "edamame",
        "eggs_benedict", "falafel", "filet_mignon", "garlic_bread",
        "greek_salad", "gyoza", "lasagna", "lobster_roll_sandwich",
        "macarons", "nachos", "paella", "ravioli",
    ]

    print("[hf] Loading food101 test split from Hugging Face (first time may download ~5GB)...")
    try:
        ds = load_dataset("food101", split="test", trust_remote_code=False)
    except Exception as exc:
        print(f"[warn] Could not load food101 dataset: {exc}", file=sys.stderr)
        print("  Skipping Hugging Face download.", file=sys.stderr)
        return entries

    for cls in sample_classes:
        # Find indices for this class in the test split
        cls_images = [i for i, ex in enumerate(ds) if ex["label"] == ds.features["label"].str2int(cls)]
        if not cls_images:
            print(f"[warn] No test images found for class '{cls}'")
            continue

        # Take first N
        for idx in cls_images[:max_per_class]:
            example = ds[int(idx)]
            fname = f"food101_{cls}_{idx}.jpg"
            dest = IMAGE_DIR / fname
            entry = {
                "id": fname,
                "image": f"test/fixtures/images/{fname}",
                "source": "huggingface-food101-test",
                "expected_label": cls,
                "notes": f"Food-101 test split, index {idx}",
            }
            # Add approximate nutrition ground truth
            if cls in NUTRITION_GROUND_TRUTH:
                entry["ground_truth"] = NUTRITION_GROUND_TRUTH[cls]
            entries.append(entry)

            if dry_run:
                print(f"[dry-run] HF food101[{idx}] ({cls})  ->  {dest}")
                continue

            if dest.exists():
                print(f"[skip] {dest}")
                continue

            print(f"[fetch] HF food101 test[{idx}] class={cls}")
            try:
                img = example["image"]
                if hasattr(img, "save"):
                    img.save(str(dest))
                else:
                    dest.write_bytes(img)
                print(f"  -> {dest} ({dest.stat().st_size} bytes)")
            except Exception as exc:
                print(f"  !! FAILED: {exc}", file=sys.stderr)
                entry["download_error"] = str(exc)

    return entries


def main():
    ap = argparse.ArgumentParser(description="Download test fixture images")
    ap.add_argument("--dry-run", action="store_true", help="Print what would happen")
    ap.add_argument(
        "--source",
        choices=["all", "hf", "wikimedia"],
        default="all",
        help="Which source to download (default: all)",
    )
    args = ap.parse_args()

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    entries = []

    if args.source in ("all", "wikimedia"):
        print("=== Wikimedia Commons ===")
        entries.extend(download_wikimedia(dry_run=args.dry_run))
        print()

    if args.source in ("all", "hf"):
        print("=== Hugging Face food101 ===")
        entries.extend(download_huggingface_food101(dry_run=args.dry_run))
        print()

    if args.dry_run:
        print(f"\nWould create {len(entries)} manifest entries (dry-run).")
        return

    # Write or merge the manifest
    existing = {}
    if MANIFEST_PATH.exists():
        existing = {e["id"]: e for e in json.loads(MANIFEST_PATH.read_text())}

    merged = dict(existing)
    for e in entries:
        if e["id"] not in merged:
            merged[e["id"]] = e
        else:
            # Update fields from the new download but keep any existing ground truth
            existing_entry = merged[e["id"]]
            existing_entry.update({k: v for k, v in e.items() if k != "ground_truth"})
            if "ground_truth" in e and "ground_truth" not in existing_entry:
                existing_entry["ground_truth"] = e["ground_truth"]

    manifest_list = sorted(merged.values(), key=lambda e: e["id"])
    MANIFEST_PATH.write_text(json.dumps(manifest_list, indent=2) + "\n", encoding="utf-8")
    print(f"\nManifest: {len(manifest_list)} entries written to {MANIFEST_PATH}")

    # Summary
    n_downloaded = sum(
        1 for e in entries if not (IMAGE_DIR / Path(e["image"]).name).exists()
        and "download_error" not in e
    )
    n_errors = sum(1 for e in entries if "download_error" in e)
    print(f"Downloaded: {len(entries) - n_errors} new, {n_errors} errors, "
          f"{len(manifest_list)} total in manifest")


if __name__ == "__main__":
    main()
