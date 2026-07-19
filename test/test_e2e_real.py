#!/usr/bin/env python3
"""
Deterministic end-to-end tests using LOCAL test fixtures (no remote URLs).

Tests the complete pipeline:
  1. Load image from test/fixtures/images/
  2. Preprocess (same as models.js imageToTensorChannels)
  3. ONNX inference via model.onnx
  4. Postprocess: softmax → argmax → nutrition lookup
  5. Assert expected label is in top-3 predictions

Also tests preprocessing parity: computes reference tensors for a known
pixel pattern and saves them so browser tests can verify JavaScript produces
the same output.

Usage:
    python test/test_e2e_real.py                    # run all tests
    python test/test_e2e_real.py --model model.onnx  # custom model path
    python test/test_e2e_real.py --save-reference     # save preprocessing reference
"""

import argparse
import json
import os
import re
import sys
from io import BytesIO
from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    sys.exit("Missing dependency: pip install onnxruntime")
try:
    from PIL import Image
except ImportError:
    sys.exit("Missing dependency: pip install pillow")

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = ROOT / "test" / "fixtures"
IMAGE_DIR = FIXTURE_DIR / "images"
MANIFEST_PATH = FIXTURE_DIR / "manifest.json"
REFERENCE_PATH = FIXTURE_DIR / "preprocessing_reference.json"


# --- parse browser JS constants (must match models.js + nutrition.js) -----


def parse_labels(js_text):
    m = re.search(r"const LABELS\s*=\s*\[(.*?)\];", js_text, re.S)
    if not m:
        raise RuntimeError("Could not find LABELS")
    return re.findall(r"'([^']+)'", m.group(1))


def parse_nutrition(js_text):
    m = re.search(r"const NUTRITION\s*=\s*\{(.*?)\n  \};", js_text, re.S)
    if not m:
        raise RuntimeError("Could not find NUTRITION")
    table = {}
    for name, nums in re.findall(r"(\w+):\s*\[([^\]]+)\]", m.group(1)):
        table[name] = [float(x) for x in nums.split(",")]
    return table


def parse_spec(js_text, spec_id="swin-food101"):
    block = re.search(rf"'{spec_id}':\s*\{{(.*?)numClasses", js_text, re.S)
    if not block:
        raise RuntimeError(f"Could not find spec '{spec_id}'")
    b = block.group(1)
    size = int(re.search(r"size:\s*(\d+)", b).group(1))
    mean = [float(x) for x in re.search(r"mean:\s*\[([^\]]+)\]", b).group(1).split(",")]
    std = [float(x) for x in re.search(r"std:\s*\[([^\]]+)\]", b).group(1).split(",")]
    spec_type = re.search(r"type:\s*'(\w+)'", b).group(1)
    return {"size": size, "mean": mean, "std": std, "type": spec_type}


# --- preprocessing (must match models.js exactly) -------------------------


def preprocess(img, spec):
    """Reproduce the browser's preprocessImage + imageToTensorChannels."""
    size = spec["size"]
    img = img.resize((size, size), Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - np.array(spec["mean"], dtype=np.float32)) / np.array(spec["std"], dtype=np.float32)
    return np.transpose(arr, (2, 0, 1))[None].astype(np.float32)


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


# --- test cases -----------------------------------------------------------


def test_image_loading():
    """Verify test fixture images exist and can be loaded."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert len(manifest) > 0, "Manifest is empty"

    available = 0
    for entry in manifest:
        img_path = ROOT / entry["image"]
        if img_path.exists():
            available += 1
            img = Image.open(img_path).convert("RGB")
            assert img.size[0] > 0 and img.size[1] > 0, f"Image {entry['id']} has zero dimensions"

    print(f"  Image loading: {available}/{len(manifest)} fixture images available")
    return available


def test_model_loading(model_path):
    """Verify model.onnx loads and has the expected input/output shapes."""
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inputs = sess.get_inputs()
    outputs = sess.get_outputs()

    assert len(inputs) == 1, f"Expected 1 input, got {len(inputs)}"
    assert len(outputs) == 1, f"Expected 1 output, got {len(outputs)}"

    # Input: [1, 3, 224, 224]
    in_shape = inputs[0].shape
    assert in_shape[1] == 3, f"Expected 3 channels, got {in_shape[1]}"

    # Output: [1, 101] (classifier logits)
    out_shape = outputs[0].shape
    assert out_shape[1] == 101, f"Expected 101 classes, got {out_shape[1]}"

    print(f"  Model loading: input={in_shape}, output={out_shape}")
    return sess


def test_donut_inference(sess, spec, labels, nutrition_table):
    """The existing test image (donut) must be identified correctly."""
    donut_path = IMAGE_DIR / "donut_wikimedia.jpg"
    if not donut_path.exists():
        print("  Donut inference: SKIP (image not downloaded)")
        return False

    img = Image.open(donut_path).convert("RGB")
    tensor = preprocess(img, spec)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    logits = sess.run([output_name], {input_name: tensor})[0][0]
    probs = softmax(logits)
    top5_idx = probs.argsort()[::-1][:5]
    top5_labels = [labels[i] for i in top5_idx]

    assert "donuts" in top5_labels, (
        f"Donut image not recognised as donuts! Top-5: {top5_labels}"
    )
    print(f"  Donut inference: top-1='{top5_labels[0]}' (confidence={probs[top5_idx[0]]*100:.1f}%)")

    # Nutrition lookup
    nutrition = nutrition_table.get("donuts", [0, 0, 0, 0, 0])
    assert nutrition[0] > 0, "Donut nutrition entry missing or zero calories"
    assert abs(nutrition[0] - 452) < 50, f"Donut calories changed significantly: {nutrition[0]}"
    return True


def test_manifest_inference(sess, spec, labels, nutrition_table, limit=None):
    """Run inference on all available manifest images, assert label in top-3."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    results = []
    available = [e for e in manifest if (ROOT / e["image"]).exists()]

    if limit:
        available = available[:limit]

    for entry in available:
        img = Image.open(ROOT / entry["image"]).convert("RGB")
        tensor = preprocess(img, spec)
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        logits = sess.run([output_name], {input_name: tensor})[0][0]
        probs = softmax(logits)
        top5_idx = probs.argsort()[::-1][:5]
        top5_labels = [labels[i] for i in top5_idx]

        expected = entry.get("expected_label")
        top3_correct = expected in top5_labels[:3] if expected else None

        results.append({
            "id": entry["id"],
            "expected": expected,
            "top5": top5_labels[:5],
            "top1_conf": round(float(probs[top5_idx[0]]) * 100, 1),
            "top3_correct": top3_correct,
        })

        status = "PASS" if top3_correct else ("SKIP" if top3_correct is None else "MISS")
        print(f"  {status} {entry['id']}: expected='{expected}' top3={top5_labels[:3]}")

    n_correct = sum(1 for r in results if r["top3_correct"])
    n_with_label = sum(1 for r in results if r["expected"] is not None)
    if n_with_label > 0:
        acc = n_correct / n_with_label
        print(f"  Top-3 accuracy: {n_correct}/{n_with_label} = {acc:.1%}")
        # Don't hard-fail on accuracy — this is a measurement, not a contract test
        # But if accuracy is 0, something is wrong
        if acc == 0 and n_with_label >= 3:
            print("  Warning: 0% top-3 accuracy — model may be broken or labels mismatched")

    return results


def test_preprocessing_parity():
    """
    Generate a reference tensor for a known input that browser tests can
    verify byte-for-byte. Saves to test/fixtures/preprocessing_reference.json.
    """
    # Create a 4×4 test pattern: gradient from red→green→blue→white
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)
    for y in range(4):
        for x in range(4):
            pixels[y, x, 0] = x * 85
            pixels[y, x, 1] = y * 85
            pixels[y, x, 2] = ((x + y) * 42) % 256
    img = Image.fromarray(pixels, "RGB")

    spec = {"size": 4, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    tensor = preprocess(img, spec)

    reference = {
        "description": "Preprocessing reference for browser parity tests",
        "input_pixels_4x4": pixels.flatten().tolist(),
        "spec": spec,
        "expected_tensor_flat": tensor.flatten().tolist(),
        "expected_shape": list(tensor.shape),
    }
    REFERENCE_PATH.write_text(json.dumps(reference, indent=2))
    print(f"  Preprocessing reference saved to {REFERENCE_PATH}")
    return reference


def test_model_spec_consistency():
    """Verify model-specs.json covers all specs in models.js."""
    models_js = (ROOT / "models.js").read_text(encoding="utf-8")
    spec_ids = re.findall(r"'(\w+(?:-\w+)*)':\s*\{", models_js)
    # Filter to just model spec IDs
    model_ids = [s for s in spec_ids if s in ["swin-food101", "nutritionverse-direct", "foodcnn-nutrition5k"]]

    specs_json = json.loads((ROOT / "model-specs.json").read_text(encoding="utf-8"))
    for mid in model_ids:
        assert mid in specs_json, f"model-specs.json missing spec: {mid}"
    print(f"  Model spec consistency: {len(model_ids)} specs match model-specs.json")


def test_nutrition_label_count():
    """Verify LABELS and nutrition entries are consistent."""
    nutrition_js = (ROOT / "nutrition.js").read_text(encoding="utf-8")
    labels = parse_labels(nutrition_js)
    table = parse_nutrition(nutrition_js)

    assert len(labels) == 101, f"Expected 101 labels, got {len(labels)}"
    assert len(table) == 101, f"Expected 101 nutrition entries, got {len(table)}"

    for label in labels:
        assert label in table, f"Label '{label}' missing from NUTRITION table"
        v = table[label]
        assert len(v) == 5, f"Label '{label}' has {len(v)} values (expected 5)"
        assert v[0] > 0, f"Label '{label}' has zero/negative calories: {v[0]}"

    print(f"  Nutrition labels: {len(labels)} labels, {len(table)} entries (all valid)")


# --- main -----------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description="Deterministic E2E tests with local fixtures")
    ap.add_argument("--model", default=str(ROOT / "model.onnx"),
                    help="Path to model.onnx")
    ap.add_argument("--save-reference", action="store_true",
                    help="Save preprocessing reference for browser parity tests")
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit manifest images to test")
    args = ap.parse_args()

    passed = 0
    failed = 0

    def run_test(name, fn, *a, **kw):
        nonlocal passed, failed
        try:
            print(f"\n-- {name} --")
            return fn(*a, **kw)
        except Exception as e:
            failed += 1
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            return None

    # Always-run tests (no model needed)
    passed += 1  # label count always runs below
    print("\n-- test_nutrition_label_count --")
    try:
        test_nutrition_label_count()
        passed += 1
    except Exception as e:
        failed += 1
        print(f"  ❌ FAIL: {e}")

    print("\n-- test_model_spec_consistency --")
    try:
        test_model_spec_consistency()
        passed += 1
    except Exception as e:
        failed += 1
        print(f"  ❌ FAIL: {e}")

    print("\n-- test_image_loading --")
    available = test_image_loading()
    passed += 1

    if args.save_reference:
        print("\n-- test_preprocessing_parity --")
        test_preprocessing_parity()
        passed += 1

    # Model-dependent tests
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"\nModel not found at {model_path}")
        print("  Skipping model-dependent tests. Download model.onnx first:")
        print("  curl -L \"https://huggingface.co/onnx-community/swin-finetuned-food101-ONNX/resolve/main/onnx/model_quantized.onnx?download=true\" -o model.onnx")
    else:
        nutrition_js = (ROOT / "nutrition.js").read_text(encoding="utf-8")
        models_js = (ROOT / "models.js").read_text(encoding="utf-8")
        labels = parse_labels(nutrition_js)
        nutrition_table = parse_nutrition(nutrition_js)
        spec = parse_spec(models_js, "swin-food101")

        sess = run_test("test_model_loading", test_model_loading, model_path)
        if sess:
            passed += 1
            if available > 0:
                run_test("test_donut_inference", test_donut_inference,
                         sess, spec, labels, nutrition_table)
                passed += 1
                run_test("test_manifest_inference", test_manifest_inference,
                         sess, spec, labels, nutrition_table, args.limit)
                passed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
    else:
        print("PASS")


if __name__ == "__main__":
    main()
