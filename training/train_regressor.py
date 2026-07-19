#!/usr/bin/env python3
"""
Train a baseline nutrition regression model.

Architecture: ViT-base-patch16-224 backbone (frozen or fine-tunable) +
5 regression heads predicting: calories (kcal), mass (g), protein (g),
fat (g), carbs (g).

Supports:
  - Synthetic fixture (smoke test, no data download needed)
  - Nutrition5k (requires dataset download)
  - NutritionVerse-Real (requires dataset download)

Usage:
    # Smoke test with synthetic data (no downloads needed):
    python training/train_regressor.py --dataset synthetic --epochs 2

    # Train on Nutrition5k:
    python training/train_regressor.py --dataset nutrition5k \\
        --data-root /path/to/nutrition5k --epochs 20

    # Train on NutritionVerse-Real:
    python training/train_regressor.py --dataset nutritionverse-real \\
        --data-root /path/to/NV-Real --epochs 30
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add repo root to path for sibling imports
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from training.dataset import create_dataloaders


# --- model architecture ---------------------------------------------------


class NutritionRegressor(nn.Module):
    """
    ViT-base backbone + shared MLP + 5 regression heads.

    Matches the NutritionVerse-Direct architecture from Keller et al. (2024):
      ViT-base-patch16-224 -> class token -> shared 768→512→256 FC
      -> per-task 128-hidden head -> 1 output

    Outputs (in order): calories, mass, protein, fat, carbs
    """

    HEAD_NAMES = ["calories", "mass", "protein", "fat", "carbs"]
    SHARED_DIMS = [768, 512, 256]
    HEAD_DIM = 128

    def __init__(self, freeze_backbone=True, num_tasks=5):
        super().__init__()
        import torchvision
        vit = torchvision.models.vit_b_16(weights="DEFAULT")
        vit.heads = nn.Identity()  # remove classification head -> [B, 768]
        self.backbone = vit

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.shared = nn.Sequential(
            nn.Linear(self.SHARED_DIMS[0], self.SHARED_DIMS[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(self.SHARED_DIMS[1], self.SHARED_DIMS[2]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.SHARED_DIMS[2], self.HEAD_DIM),
                nn.ReLU(inplace=True),
                nn.Linear(self.HEAD_DIM, 1),
            )
            for _ in range(num_tasks)
        ])

        # Softplus on output ensures non-negative predictions
        self.output_activation = nn.Softplus()

    def forward(self, x):
        feats = self.backbone(x)            # [B, 768]
        feats = self.shared(feats)          # [B, 256]
        outputs = [head(feats) for head in self.heads]  # list of [B, 1]
        raw = torch.cat(outputs, dim=1)     # [B, 5]
        return self.output_activation(raw)  # non-negative


# --- loss function --------------------------------------------------------


class MultitaskNutritionLoss(nn.Module):
    """
    Weighted MSE on log-transformed targets.

    Log-transform reduces the dominance of high-calorie/high-mass foods
    and better captures relative errors (important for small portions).

    Also computes Atwater consistency as a diagnostic (not used in loss):
      calories_atwater = 4*protein + 9*fat + 4*carbs
    """

    def __init__(self, task_weights=None):
        super().__init__()
        # Default weights: calories and mass are the primary signals
        self.task_weights = task_weights or torch.tensor(
            [1.0, 1.0, 0.5, 0.5, 0.5]  # cal, mass, protein, fat, carbs
        )

    def forward(self, pred, target):
        """
        Args:
            pred: [B, 5] predicted values
            target: [B, 5] ground truth values
        Returns:
            (loss, metrics_dict)
        """
        # Log-transform: predict log(1+x), target log(1+x)
        # This penalizes relative error more than absolute error
        log_pred = torch.log1p(pred)
        log_target = torch.log1p(target)

        weights = self.task_weights.to(pred.device)
        se = (log_pred - log_target) ** 2
        loss = (se * weights).mean()

        # Per-task MAE (in original space)
        with torch.no_grad():
            abs_error = (pred - target).abs()
            mae_per_task = abs_error.mean(dim=0)

            # Atwater consistency check (diagnostic)
            atwater_cal = 4 * pred[:, 2] + 9 * pred[:, 3] + 4 * pred[:, 4]
            atwater_error = (pred[:, 0] - atwater_cal).abs().mean()

        metrics = {
            "loss": loss.item(),
            "mae_calories": mae_per_task[0].item(),
            "mae_mass": mae_per_task[1].item(),
            "mae_protein": mae_per_task[2].item(),
            "mae_fat": mae_per_task[3].item(),
            "mae_carbs": mae_per_task[4].item(),
            "atwater_error": atwater_error.item(),
        }
        return loss, metrics


# --- training loop --------------------------------------------------------


def train_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_metrics = {}
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(images)
        loss, metrics = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()

        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0) + v

    n = len(loader)
    return {k: v / n for k, v in total_metrics.items()}


@torch.no_grad()
def validate_epoch(model, loader, loss_fn, device):
    model.eval()
    total_metrics = {}
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        preds = model(images)
        _, metrics = loss_fn(preds, targets)

        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0) + v

    n = len(loader)
    return {k: v / n for k, v in total_metrics.items()}


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")

    # Data
    train_loader, val_loader, target_stats = create_dataloaders(
        args.dataset, args.data_root,
        batch_size=args.batch_size, num_workers=args.num_workers,
        size=args.image_size,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(f"Target stats — mean: {target_stats['mean']}, std: {target_stats['std']}")

    # Model
    model = NutritionRegressor(freeze_backbone=not args.unfreeze_backbone)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # Loss & optimizer
    loss_fn = MultitaskNutritionLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training
    best_val_loss = float("inf")
    history = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_metrics = validate_epoch(model, val_loader, loss_fn, device)
        scheduler.step()
        dt = time.time() - t0

        combined_mae = sum(
            val_metrics.get(f"mae_{h}", 0) for h in NutritionRegressor.HEAD_NAMES
        )

        print(
            f"Epoch {epoch:3d}/{args.epochs}  {dt:.0f}s  "
            f"train_loss={train_metrics['loss']:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"combined_mae={combined_mae:.2f}"
        )
        print(
            f"  MAE — cal:{val_metrics.get('mae_calories',0):.1f}  "
            f"mass:{val_metrics.get('mae_mass',0):.1f}  "
            f"prot:{val_metrics.get('mae_protein',0):.2f}  "
            f"fat:{val_metrics.get('mae_fat',0):.2f}  "
            f"carb:{val_metrics.get('mae_carbs',0):.2f}  "
            f"atwater_err:{val_metrics.get('atwater_error',0):.1f}"
        )

        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "target_stats": target_stats,
                "args": vars(args),
            }
            torch.save(checkpoint, output_dir / "best_model.pth")
            print(f"  -> saved best model (val_loss={best_val_loss:.4f})")

    # Save final model and history
    torch.save(checkpoint, output_dir / "last_model.pth")
    with (output_dir / "training_history.json").open("w") as f:
        json.dump(history, f, indent=2, default=float)

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    print(f"Outputs: {output_dir}")
    return model, history


# --- CLI ------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description="Train a nutrition regression model")
    ap.add_argument("--dataset", default="synthetic",
                    choices=["synthetic", "nutrition5k", "nutritionverse-real"],
                    help="Dataset to train on (default: synthetic smoke test)")
    ap.add_argument("--data-root", default="data/",
                    help="Path to dataset root directory")
    ap.add_argument("--epochs", type=int, default=5,
                    help="Number of training epochs")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Batch size")
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="Learning rate")
    ap.add_argument("--weight-decay", type=float, default=1e-4,
                    help="Weight decay")
    ap.add_argument("--image-size", type=int, default=224,
                    help="Input image size")
    ap.add_argument("--num-workers", type=int, default=0,
                    help="DataLoader workers")
    ap.add_argument("--unfreeze-backbone", action="store_true",
                    help="Unfreeze ViT backbone for fine-tuning")
    ap.add_argument("--output-dir", default="training/checkpoints",
                    help="Directory for model checkpoints")
    args = ap.parse_args()

    train(args)


if __name__ == "__main__":
    main()
