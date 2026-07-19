#!/usr/bin/env python3
"""
Benchmark the Food Tracker model against a manifest of real food images.

Runs the complete browser-equivalent pipeline for each image in the manifest:
  fetch -> preprocess (size, mean, std matching models.js) -> ONNX inference
  -> softmax -> top-k labels -> nutrition lookup -> metrics.

Outputs:
  - ``benchmark-results.json`` — per-image predictions + summary metrics
  - ``benchmark-results.csv``  — flat table for external analysis
  - ``benchmark-report.md``    — human-readable baseline report

Usage:
    python scripts/benchmark.py                          # full benchmark
    python scripts/benchmark.py --model model.onnx       # custom model path
    python scripts/benchmark.py --limit 5                # first 5 images only
    python scripts/benchmark.py --output-dir results/    # custom output dir
"""

import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    sys.exit("Missing dependency: pip install onnxruntime")
try:
    from PIL import Image
except ImportError:
    sys.exit("Missing dependency: pip install pillow")

# --- paths ----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "model.onnx"
DEFAULT_MANIFEST = ROOT / "test" / "fixtures" / "manifest.json"
DEFAULT_OUTPUT = ROOT

# --- parse constants from the browser JS (must match models.js) -----------


def parse_labels(js_text):
    """Extract LABELS array from nutrition.js."""
    m = re.search(r"const LABELS\s*=\s*\[(.*?)\];", js_text, re.S)
    if not m:
        raise RuntimeError("Could not find LABELS in nutrition.js")
    return re.findall(r"'([^']+)'", m.group(1))


def parse_nutrition(js_text):
    """Extract NUTRITION table from nutrition.js -> {label: [cal,mass,protein,fat,carbs]}."""
    m = re.search(r"const NUTRITION\s*=\s*\{(.*?)\n  \};", js_text, re.S)
    if not m:
        raise RuntimeError("Could not find NUTRITION in nutrition.js")
    table = {}
    for name, nums in re.findall(r"(\w+):\s*\[([^\]]+)\]", m.group(1)):
        table[name] = [float(x) for x in nums.split(",")]
    return table


def parse_spec(js_text, spec_id="swin-food101"):
    """Extract mean/std/size/type from models.js for a given spec."""
    pattern = rf"'{spec_id}':\s*\{{(.*?)\}}\s*}}"
    m = re.search(pattern, js_text, re.S)
    if not m:
        # Try matching until the closing of MODEL_SPECS
        block = re.search(rf"'{spec_id}':\s*\{{(.*?)numClasses", js_text, re.S)
        if not block:
            raise RuntimeError(f"Could not find spec '{spec_id}' in models.js")
        b = block.group(1)
    else:
        b = m.group(1)
    size = int(re.search(r"size:\s*(\d+)", b).group(1))
    mean = [float(x) for x in re.search(r"mean:\s*\[([^\]]+)\]", b).group(1).split(",")]
    std = [float(x) for x in re.search(r"std:\s*\[([^\]]+)\]", b).group(1).split(",")]
    spec_type = re.search(r"type:\s*'(\w+)'", b).group(1)
    return {"size": size, "mean": mean, "std": std, "type": spec_type}


# --- image loading (local file or URL) ------------------------------------


def load_image(path_or_url):
    """Load an image from a local path or URL, return PIL RGB Image."""
    if path_or_url.startswith(("http://", "https://")):
        req = Request(path_or_url, headers={"User-Agent": "food-tracker-benchmark/1.0"})
        with urlopen(req, timeout=60) as resp:
            return Image.open(BytesIO(resp.read())).convert("RGB")
    p = Path(path_or_url)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    return Image.open(p).convert("RGB")


# --- preprocessing (must match models.js imageToTensorChannels) -----------


def preprocess(img, size, mean, std):
    """Resize, normalize, return [1,3,H,W] float32 tensor matching browser pipeline."""
    img = img.resize((size, size), Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 255.0  # HWC, 0..1
    # Normalize: (pixel - mean) / std
    arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    # Transpose to [C, H, W] then add batch dim -> [1, 3, H, W]
    return np.transpose(arr, (2, 0, 1))[None].astype(np.float32)


def softmax(x):
    """Numerically stable softmax (matches models.js)."""
    e = np.exp(x - x.max())
    return e / e.sum()


def top_k(probs, k=5):
    """Indices of top-k probabilities (highest first)."""
    return probs.argsort()[::-1][:k].tolist()


# --- metrics --------------------------------------------------------------


def compute_metrics(results):
    """Compute aggregate metrics from a list of per-image result dicts."""
    n = len(results)
    if n == 0:
        return {"error": "no results"}

    # Classification accuracy
    correct_top1 = sum(1 for r in results if r.get("top1_correct"))
    correct_top3 = sum(
        1 for r in results
        if r.get("expected_label") in r.get("top5_labels", [])
    )
    confidences = [r.get("top1_confidence", 0) for r in results]
    has_gt = [r for r in results if r.get("ground_truth")]

    m = {
        "n_images": n,
        "top1_accuracy": round(correct_top1 / n, 4) if n else 0,
        "top3_accuracy": round(correct_top3 / n, 4) if n else 0,
        "top1_correct": correct_top1,
        "top3_correct": correct_top3,
        "mean_confidence": round(statistics.mean(confidences), 2) if confidences else 0,
        "inference_time_ms_mean": round(
            statistics.mean([r.get("inference_ms", 0) for r in results]), 1
        ),
        "inference_time_ms_p50": round(
            statistics.median([r.get("inference_ms", 0) for r in results]), 1
        ),
    }

    # Per-class accuracy
    by_class = {}
    for r in results:
        cls = r.get("expected_label", "unknown")
        by_class.setdefault(cls, {"total": 0, "correct": 0})
        by_class[cls]["total"] += 1
        if r.get("top1_correct"):
            by_class[cls]["correct"] += 1
    m["per_class_accuracy"] = {
        cls: round(d["correct"] / d["total"], 3)
        for cls, d in sorted(by_class.items())
    }

    # Nutrition metrics (only for images with ground truth)
    if has_gt:
        nutrient_keys = ["calories", "mass", "protein", "fat", "carbs"]
        nutrient_metrics = {}
        for key in nutrient_keys:
            preds = []
            truths = []
            for r in has_gt:
                pred_val = r.get("predicted_nutrition", {}).get(key)
                truth_val = r.get("ground_truth", {}).get(key)
                if pred_val is not None and truth_val is not None and truth_val > 0:
                    preds.append(pred_val)
                    truths.append(truth_val)
            if not preds:
                continue
            abs_errors = [abs(p - t) for p, t in zip(preds, truths)]
            apes = [
                abs(p - t) / t * 100 for p, t in zip(preds, truths)
            ]
            nutrient_metrics[key] = {
                "n": len(preds),
                "mae": round(statistics.mean(abs_errors), 2),
                "mape_pct": round(statistics.mean(apes), 1),
                "signed_bias": round(statistics.mean([p - t for p, t in zip(preds, truths)]), 2),
                "rmse": round(np.sqrt(statistics.mean([e ** 2 for e in abs_errors])), 2),
            }
        m["nutrition"] = nutrient_metrics

    # Calibration: bucket confidence vs actual accuracy
    buckets = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 90), (90, 100)]
    calibration = []
    for lo, hi in buckets:
        in_bucket = [r for r in results if lo <= r.get("top1_confidence", 0) < hi]
        if in_bucket:
            acc = sum(1 for r in in_bucket if r.get("top1_correct")) / len(in_bucket)
            calibration.append({
                "confidence_bin": f"{lo}-{hi}%",
                "n": len(in_bucket),
                "accuracy": round(acc, 3),
                "mean_confidence": round(statistics.mean([r["top1_confidence"] for r in in_bucket]), 1),
            })
    m["calibration"] = calibration

    # Expected calibration error (ECE)
    if calibration:
        ece = sum(
            abs(b["accuracy"] - b["mean_confidence"] / 100) * b["n"] / n
            for b in calibration
        )
        m["ece"] = round(ece, 4)

    return m


# --- main benchmark -------------------------------------------------------


def run_benchmark(manifest_path, model_path, limit=None, output_dir=None):
    """Run all images in manifest through the model and return (results, metrics)."""
    output_dir = Path(output_dir) if output_dir else ROOT

    # Load manifest
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if limit:
        manifest = manifest[:limit]
    print(f"Manifest: {len(manifest)} images")

    # Parse JS constants
    nutrition_js = (ROOT / "nutrition.js").read_text(encoding="utf-8")
    models_js = (ROOT / "models.js").read_text(encoding="utf-8")
    labels = parse_labels(nutrition_js)
    nutrition_table = parse_nutrition(nutrition_js)
    spec = parse_spec(models_js, "swin-food101")

    print(f"Labels: {len(labels)}, Nutrition rows: {len(nutrition_table)}")
    print(f"Spec: type={spec['type']}, size={spec['size']}, "
          f"mean={spec['mean']}, std={spec['std']}")

    # Load ONNX model
    if not Path(model_path).exists():
        sys.exit(f"Model not found: {model_path}\n"
                 f"Download it: curl -L \"https://huggingface.co/onnx-community/"
                 f"swin-finetuned-food101-ONNX/resolve/main/onnx/model_quantized.onnx"
                 f"?download=true\" -o model.onnx")
    print(f"Loading model: {model_path}")
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    print(f"Input: {input_name}, Output: {output_name}")

    # Run each image
    results = []
    for i, entry in enumerate(manifest):
        img_path = entry.get("image") or entry.get("source_url")
        expected = entry.get("expected_label")
        ground_truth = entry.get("ground_truth")

        print(f"\n[{i+1}/{len(manifest)}] {entry['id']}")
        print(f"  Expected: {expected}")

        result = {
            "id": entry["id"],
            "expected_label": expected,
            "source": entry.get("source"),
            "notes": entry.get("notes"),
        }

        try:
            img = load_image(img_path)
            tensor = preprocess(img, spec["size"], spec["mean"], spec["std"])

            t0 = time.perf_counter()
            ort_outputs = sess.run([output_name], {input_name: tensor})
            dt_ms = round((time.perf_counter() - t0) * 1000, 1)
            result["inference_ms"] = dt_ms

            logits = ort_outputs[0][0]
            probs = softmax(logits)
            top5_idx = top_k(probs, 5)

            result["top5_labels"] = [labels[idx] for idx in top5_idx]
            result["top5_confidences"] = [round(float(probs[idx]) * 100, 2) for idx in top5_idx]
            result["top1_label"] = labels[top5_idx[0]]
            result["top1_confidence"] = round(float(probs[top5_idx[0]]) * 100, 1)
            result["top1_correct"] = result["top1_label"] == expected
            result["top3_correct"] = expected in result["top5_labels"]

            # Nutrition lookup (matching postprocessClassifier in models.js)
            pred_label = result["top1_label"]
            nut = nutrition_table.get(pred_label, [0, 0, 0, 0, 0])
            result["predicted_nutrition"] = {
                "calories": round(nut[0]),
                "mass": round(nut[1], 1),
                "protein": round(nut[2], 1),
                "fat": round(nut[3], 1),
                "carbs": round(nut[4], 1),
            }
            if ground_truth:
                result["ground_truth"] = ground_truth
                # per-nutrient error
                result["nutrition_error"] = {}
                for key in ["calories", "mass", "protein", "fat", "carbs"]:
                    pred_val = result["predicted_nutrition"][key]
                    truth_val = ground_truth.get(key, 0)
                    if truth_val > 0:
                        result["nutrition_error"][key] = {
                            "predicted": pred_val,
                            "ground_truth": truth_val,
                            "abs_error": round(abs(pred_val - truth_val), 2),
                            "ape_pct": round(abs(pred_val - truth_val) / truth_val * 100, 1),
                        }

            status = "PASS" if result["top1_correct"] else (
                "TOP3" if result["top3_correct"] else "MISS"
            )
            print(f"  Predicted: {result['top1_label']} ({result['top1_confidence']}%)  "
                  f"[{status}]  {dt_ms}ms")
            if result["top5_labels"][:3] != [result["top1_label"]]:
                top3_str = ", ".join(
                    f"{l} ({c}%)"
                    for l, c in zip(result["top5_labels"][:3], result["top5_confidences"][:3])
                )
                print(f"  Top-3: {top3_str}")

        except Exception as exc:
            print(f"  ERROR: {exc}")
            result["error"] = str(exc)

        results.append(result)

    # Compute summary metrics
    metrics = compute_metrics(results)

    # Write outputs
    results_path = output_dir / "benchmark-results.json"
    results_path.write_text(
        json.dumps({"metrics": metrics, "results": results}, indent=2),
        encoding="utf-8",
    )
    print(f"\n\nResults written to {results_path}")

    # CSV
    csv_path = output_dir / "benchmark-results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "expected_label", "top1_label", "top1_confidence",
            "top1_correct", "top3_correct", "inference_ms",
            "pred_calories", "pred_mass", "pred_protein", "pred_fat", "pred_carbs",
        ])
        for r in results:
            nut = r.get("predicted_nutrition", {})
            writer.writerow([
                r["id"], r.get("expected_label"), r.get("top1_label"),
                r.get("top1_confidence"), r.get("top1_correct"), r.get("top3_correct"),
                r.get("inference_ms"),
                nut.get("calories"), nut.get("mass"), nut.get("protein"),
                nut.get("fat"), nut.get("carbs"),
            ])
    print(f"CSV written to {csv_path}")

    # Markdown report
    md_path = output_dir / "benchmark-report.md"
    write_markdown_report(metrics, results, md_path)
    print(f"Report written to {md_path}")

    # Summary to stdout
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Images:               {metrics['n_images']}")
    print(f"Top-1 accuracy:       {metrics['top1_accuracy']:.1%}  ({metrics['top1_correct']}/{metrics['n_images']})")
    print(f"Top-3 accuracy:       {metrics['top3_accuracy']:.1%}  ({metrics['top3_correct']}/{metrics['n_images']})")
    print(f"Mean confidence:      {metrics['mean_confidence']:.1f}%")
    print(f"ECE:                  {metrics.get('ece', 'N/A')}")
    print(f"Inference (mean):     {metrics['inference_time_ms_mean']:.1f} ms")
    print(f"Inference (median):   {metrics['inference_time_ms_p50']:.1f} ms")

    # Per-class
    print("\nPer-class accuracy:")
    for cls, acc in sorted(metrics.get("per_class_accuracy", {}).items(),
                           key=lambda x: x[1]):
        print(f"  {cls:25s} {acc:.1%}")

    # Nutrition errors
    nut = metrics.get("nutrition", {})
    if nut:
        print("\nNutrition estimation errors (vs USDA ground truth):")
        print(f"  {'Nutrient':<12} {'MAE':>10} {'MAPE':>10} {'Bias':>10} {'RMSE':>10} {'N':>5}")
        for key, nm in nut.items():
            print(f"  {key:<12} {nm['mae']:>10.2f} {str(nm['mape_pct'])+'%':>10} "
                  f"{nm['signed_bias']:>10.2f} {nm['rmse']:>10.2f} {nm['n']:>5}")

    return results, metrics


def write_markdown_report(metrics, results, path):
    """Generate a human-readable markdown benchmark report."""
    lines = []
    lines.append("# Food MODEL — Baseline Benchmark Report")
    lines.append(f"\n**Generated:** {time.strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"\n**Model:** Swin Food-101 classifier (ONNX)")
    lines.append(f"**Images tested:** {metrics['n_images']}")
    lines.append("")

    lines.append("## Classification Accuracy")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Top-1 accuracy | {metrics['top1_accuracy']:.1%} ({metrics['top1_correct']}/{metrics['n_images']}) |")
    lines.append(f"| Top-3 accuracy | {metrics['top3_accuracy']:.1%} ({metrics['top3_correct']}/{metrics['n_images']}) |")
    lines.append(f"| Mean confidence | {metrics['mean_confidence']:.1f}% |")
    lines.append(f"| ECE | {metrics.get('ece', 'N/A')} |")
    lines.append(f"| Mean inference time | {metrics['inference_time_ms_mean']:.1f} ms |")
    lines.append(f"| Median inference time | {metrics['inference_time_ms_p50']:.1f} ms |")
    lines.append("")

    lines.append("## Per-Class Accuracy")
    lines.append("")
    lines.append("| Class | Accuracy |")
    lines.append("|-------|----------|")
    for cls, acc in sorted(metrics.get("per_class_accuracy", {}).items(),
                           key=lambda x: x[1]):
        lines.append(f"| {cls} | {acc:.1%} |")
    lines.append("")

    nut = metrics.get("nutrition", {})
    if nut:
        lines.append("## Nutrition Estimation Errors (vs USDA FDC ground truth)")
        lines.append("")
        lines.append("> Note: Ground truth is approximate (USDA FDC per-100g values, not weighed lab data).")
        lines.append("> These metrics evaluate whether the classifier→table pipeline produces")
        lines.append("> directionally reasonable nutrition estimates for correctly identified foods.")
        lines.append("")
        lines.append("| Nutrient | MAE | MAPE | Signed Bias | RMSE | N |")
        lines.append("|----------|-----|------|-------------|------|---|")
        for key, nm in nut.items():
            lines.append(
                f"| {key} | {nm['mae']:.2f} | {nm['mape_pct']:.1f}% | "
                f"{nm['signed_bias']:+.2f} | {nm['rmse']:.2f} | {nm['n']} |"
            )
        lines.append("")

    lines.append("## Calibration")
    lines.append("")
    lines.append("| Confidence bin | N | Accuracy | Mean confidence |")
    lines.append("|----------------|---|----------|-----------------|")
    for b in metrics.get("calibration", []):
        lines.append(
            f"| {b['confidence_bin']} | {b['n']} | {b['accuracy']:.1%} | "
            f"{b['mean_confidence']:.1f}% |"
        )
    lines.append("")

    lines.append("## Per-Image Results")
    lines.append("")
    lines.append("| Image | Expected | Predicted | Conf. | Top-1 | Top-3 | Time |")
    lines.append("|-------|----------|-----------|-------|-------|-------|------|")
    for r in results:
        status = "PASS" if r.get("top1_correct") else ("TOP3" if r.get("top3_correct") else "MISS")
        lines.append(
            f"| {r['id']} | {r.get('expected_label','')} | {r.get('top1_label','ERROR')} | "
            f"{r.get('top1_confidence','-')}% | {status} | "
            f"{'Y' if r.get('top3_correct') else 'N'} | "
            f"{r.get('inference_ms','-')}ms |"
        )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# --- CLI ------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description="Benchmark the Food MODEL")
    ap.add_argument("--model", default=str(DEFAULT_MODEL),
                    help=f"Path to model.onnx (default: {DEFAULT_MODEL})")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST),
                    help=f"Path to test manifest (default: {DEFAULT_MANIFEST})")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only run the first N images")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT),
                    help=f"Directory for output files (default: {DEFAULT_OUTPUT})")
    args = ap.parse_args()

    run_benchmark(
        manifest_path=args.manifest,
        model_path=args.model,
        limit=args.limit,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
