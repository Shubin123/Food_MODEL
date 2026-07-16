#!/usr/bin/env python3
"""
End-to-end test for the Food Tracker model.

This runs the SAME pipeline the browser app uses, but headless:

  1. Fetch the test image (Wikimedia pink-frosted donut) from a URL, exactly
     like the app's "Photo URL" feature.
  2. Preprocess it the way models.js does (resize to the spec size, ImageNet
     normalization, channel-first [1,3,H,W] float tensor).
  3. Run model.onnx via ONNX Runtime.
  4. Postprocess like models.js: softmax -> argmax -> look up nutrition.

To avoid drift, the Food-101 LABELS and the NUTRITION table are parsed directly
out of nutrition.js, and the normalization mean/std are parsed out of models.js.
So this test exercises the real, shipped constants.

Run:
    python3 test_e2e.py
    python3 test_e2e.py --url <image-url> --expect donuts

Requires: onnxruntime, numpy, pillow  (pip install onnxruntime numpy pillow)
"""
import argparse
import json
import re
import sys
import urllib.request
from io import BytesIO

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    sys.exit("Missing dependency: pip install onnxruntime")
try:
    from PIL import Image
except ImportError:
    sys.exit("Missing dependency: pip install pillow")

HERE = __file__.rsplit("/", 1)[0] if "/" in __file__ else "."
DEFAULT_URL = "https://upload.wikimedia.org/wikipedia/commons/e/ed/Pink-Frosted-Donut.jpg"


def read(path):
    with open(path, "r") as f:
        return f.read()


def parse_labels(js):
    """Extract the LABELS array from nutrition.js."""
    m = re.search(r"const LABELS = \[(.*?)\];", js, re.S)
    if not m:
        raise RuntimeError("Could not find LABELS in nutrition.js")
    return re.findall(r"'([^']+)'", m.group(1))


def parse_nutrition(js):
    """Extract the NUTRITION table {label: [cal, mass, protein, fat, carbs]}."""
    m = re.search(r"const NUTRITION = \{(.*?)\n  \};", js, re.S)
    if not m:
        raise RuntimeError("Could not find NUTRITION in nutrition.js")
    table = {}
    for name, nums in re.findall(r"(\w+):\s*\[([^\]]+)\]", m.group(1)):
        table[name] = [float(x) for x in nums.split(",")]
    return table


def parse_norm(js):
    """Extract the swin-food101 mean/std/size from models.js."""
    block = re.search(r"'swin-food101':\s*\{(.*?)numClasses", js, re.S)
    if not block:
        raise RuntimeError("Could not find swin-food101 spec in models.js")
    b = block.group(1)
    size = int(re.search(r"size:\s*(\d+)", b).group(1))
    mean = [float(x) for x in re.search(r"mean:\s*\[([^\]]+)\]", b).group(1).split(",")]
    std = [float(x) for x in re.search(r"std:\s*\[([^\]]+)\]", b).group(1).split(",")]
    return size, mean, std


def fetch_image(url):
    req = urllib.request.Request(url, headers={"User-Agent": "food-tracker-test/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return Image.open(BytesIO(r.read())).convert("RGB")


def preprocess(img, size, mean, std):
    img = img.resize((size, size), Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 255.0            # HWC, 0..1
    arr = (arr - np.array(mean)) / np.array(std)                 # normalize
    return np.transpose(arr, (2, 0, 1))[None].astype(np.float32)  # [1,3,H,W]


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--model", default=f"{HERE}/model.onnx")
    ap.add_argument("--expect", default=None, help="fail unless this label is top-1")
    args = ap.parse_args()

    nutrition_js = read(f"{HERE}/nutrition.js")
    models_js = read(f"{HERE}/models.js")
    labels = parse_labels(nutrition_js)
    table = parse_nutrition(nutrition_js)
    size, mean, std = parse_norm(models_js)

    assert len(labels) == 101, f"expected 101 labels, got {len(labels)}"
    print(f"Loaded {len(labels)} labels, {len(table)} nutrition rows")
    print(f"Preprocess: size={size} mean={mean} std={std}")

    print(f"Fetching image: {args.url}")
    img = fetch_image(args.url)
    x = preprocess(img, size, mean, std)

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    logits = sess.run(None, {in_name: x})[0][0]
    probs = softmax(logits)
    top5 = probs.argsort()[::-1][:5]

    print("\nTop-5 predictions:")
    for i in top5:
        print(f"  {labels[i]:22s} {probs[i]*100:6.2f}%")

    idx = int(top5[0])
    label = labels[idx]
    cal, mass, protein, fat, carbs = table[label]

    print("\n" + "=" * 44)
    print(f"  Detected: {label.replace('_',' ').title()} ({probs[idx]*100:.1f}%)")
    print(f"  Calories: {int(round(cal))} kcal")
    print(f"  Mass:     {mass:g} g")
    print(f"  Protein:  {protein:g} g")
    print(f"  Fat:      {fat:g} g")
    print(f"  Carbs:    {carbs:g} g")
    print("=" * 44)

    if args.expect and label != args.expect:
        sys.exit(f"\nFAIL: expected top-1 '{args.expect}', got '{label}'")
    print("\nPASS")


if __name__ == "__main__":
    main()
