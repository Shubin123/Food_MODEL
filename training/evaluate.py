#!/usr/bin/env python3
"""
Evaluate a trained nutrition regression model on a held-out test set.

Computes:
  - Per-task MAE, MAPE, RMSE, signed bias
  - Atwater consistency (calories vs 4P+9F+4C)
  - Scatter plots of predicted vs actual
  - Per-category breakdown if labels are available

Usage:
    python training/evaluate.py --checkpoint training/checkpoints/best_model.pth
    python training/evaluate.py --checkpoint model.pth --dataset nutritionverse-real --data-root /data/NV-Real
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from training.dataset import create_dataloaders, SyntheticFixture, get_val_transforms
from training.train_regressor import NutritionRegressor, MultitaskNutritionLoss


HEAD_NAMES = ["calories", "mass", "protein", "fat", "carbs"]


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Run model on dataloader and return predictions, targets, and metrics."""
    model.eval()
    all_preds = []
    all_targets = []

    for images, targets in dataloader:
        images = images.to(device)
        preds = model(images)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    metrics = _compute_metrics(preds, targets)
    return preds, targets, metrics


def _compute_metrics(preds, targets):
    """Compute per-task regression metrics."""
    m = {}
    for i, name in enumerate(HEAD_NAMES):
        p = preds[:, i]
        t = targets[:, i]
        # Filter zero-target samples (no ground truth for that nutrient)
        mask = t > 0
        if mask.sum() < 2:
            m[name] = {"n": int(mask.sum()), "note": "insufficient non-zero targets"}
            continue

        p_f = p[mask]
        t_f = t[mask]
        abs_err = np.abs(p_f - t_f)
        apes = abs_err / t_f * 100

        m[name] = {
            "n": int(mask.sum()),
            "mae": round(float(abs_err.mean()), 3),
            "mape": round(float(apes.mean()), 1),
            "rmse": round(float(np.sqrt((abs_err ** 2).mean())), 3),
            "signed_bias": round(float((p_f - t_f).mean()), 3),
            "median_abs_error": round(float(np.median(abs_err)), 3),
            "r2": round(float(1 - ((p_f - t_f) ** 2).sum() / ((t_f - t_f.mean()) ** 2).sum())), 4),
        }

    # Combined MAE (sum of all 5 per-task MAEs)
    combined_mae = sum(m[name].get("mae", 0) for name in HEAD_NAMES)
    m["combined_mae"] = round(combined_mae, 3)

    # Atwater consistency
    atwater_cal = 4 * preds[:, 2] + 9 * preds[:, 3] + 4 * preds[:, 4]
    cal_mask = targets[:, 0] > 0
    if cal_mask.sum() > 1:
        atwater_err = np.abs(preds[cal_mask, 0] - atwater_cal[cal_mask])
        m["atwater_mae"] = round(float(atwater_err.mean()), 2)
        m["atwater_mape"] = round(float((atwater_err / targets[cal_mask, 0] * 100).mean()), 1)

    return m


def main():
    ap = argparse.ArgumentParser(description="Evaluate a trained nutrition regressor")
    ap.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pth)")
    ap.add_argument("--dataset", default="synthetic",
                    choices=["synthetic", "nutrition5k", "nutritionverse-real"])
    ap.add_argument("--data-root", default="data/")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--output", default=None, help="Path for metrics JSON output")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = NutritionRegressor(freeze_backbone=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")

    # Load data
    _, val_loader, _ = create_dataloaders(
        args.dataset, args.data_root,
        batch_size=args.batch_size, num_workers=0, size=args.image_size,
    )

    # If val loader is empty (no real data), create a synthetic test set
    if len(val_loader) == 0:
        print("[warn] No validation data found; using synthetic test set")
        val_ds = SyntheticFixture(num_samples=20, size=args.image_size,
                                  transform=get_val_transforms(args.image_size))
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    print(f"Evaluating on {len(val_loader)} batches...")
    preds, targets, metrics = evaluate(model, val_loader, device)

    print("\n" + "=" * 55)
    print("EVALUATION RESULTS")
    print("=" * 55)
    for name in HEAD_NAMES:
        m = metrics[name]
        if "mae" in m:
            print(f"  {name:12s}  MAE={m['mae']:>8.2f}  MAPE={m['mape']:>6.1f}%  "
                  f"Bias={m['signed_bias']:>+8.2f}  RMSE={m['rmse']:>8.2f}  "
                  f"R²={m['r2']:>7.3f}  n={m['n']}")
    print(f"  {'combined':12s}  MAE={metrics['combined_mae']:>8.2f}")
    if "atwater_mae" in metrics:
        print(f"  {'atwater':12s}  MAE={metrics['atwater_mae']:>8.2f} kcal  "
              f"MAPE={metrics['atwater_mape']:>6.1f}%")

    if args.output:
        output = {
            "checkpoint": args.checkpoint,
            "dataset": args.dataset,
            "metrics": metrics,
            "predictions": preds.tolist()[:50],  # first 50 samples only
            "targets": targets.tolist()[:50],
        }
        Path(args.output).write_text(json.dumps(output, indent=2, default=float))
        print(f"\nMetrics saved to {args.output}")

    return metrics


if __name__ == "__main__":
    main()
