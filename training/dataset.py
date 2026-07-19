"""
Dataset adapters for food nutrition regression.

Supported datasets:
  - Nutrition5k (Google Research): RGB videos + weighed ground truth
  - NutritionVerse-Real: 889 real food images with nutrition labels
  - Food-101 (for classification pre-training validation)
  - Synthetic fixture: tiny 10-sample dataset for smoke-testing the pipeline

All adapters return (image_tensor, targets_dict) where targets_dict has:
  calories (kcal), mass (g), protein (g), fat (g), carbs (g)
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


# --- image transforms matching the browser preprocessing pipeline ---------


def get_train_transforms(size=224):
    """Augmentations matching mobile food photography conditions."""
    import torchvision.transforms as T
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.ToTensor(),
        # Normalize: models.js expects (x/255 - mean) / std
        # For NV-Direct: mean=[0,0,0], std=[1,1,1] — no normalization
        # For Swin/ImageNet: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(size=224):
    """Validation transforms — resize + centre crop, no augmentation."""
    import torchvision.transforms as T
    return T.Compose([
        T.Resize(size),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# --- synthetic fixture (smoke test) ---------------------------------------


class SyntheticFixture(Dataset):
    """
    Tiny 10-sample dataset for smoke-testing the training pipeline.
    Generates random "food" images with known regression targets.
    """

    def __init__(self, num_samples=10, size=224, transform=None):
        self.num_samples = num_samples
        self.size = size
        self.transform = transform or get_val_transforms(size)

        # Generate deterministic "images" and targets
        rng = np.random.RandomState(42)
        self.images = []
        self.targets = []
        for i in range(num_samples):
            # Create a simple gradient pattern as a "food image"
            arr = np.zeros((size, size, 3), dtype=np.uint8)
            for c in range(3):
                val = (i * 25 + c * 40) % 256
                arr[:, :, c] = val + rng.randint(0, 30, (size, size))
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            self.images.append(arr)

            # Realistic nutrition targets
            self.targets.append({
                "calories": float(100 + i * 50 + rng.uniform(-20, 20)),
                "mass": float(80 + i * 30 + rng.uniform(-15, 15)),
                "protein": float(5 + i * 3 + rng.uniform(-2, 2)),
                "fat": float(3 + i * 2 + rng.uniform(-1, 1)),
                "carbs": float(10 + i * 8 + rng.uniform(-5, 5)),
            })

        # Target statistics for normalization
        all_targets = np.array([
            [t["calories"], t["mass"], t["protein"], t["fat"], t["carbs"]]
            for t in self.targets
        ])
        self.target_mean = all_targets.mean(axis=0).astype(np.float32)
        self.target_std = all_targets.std(axis=0).astype(np.float32)
        self.target_std[self.target_std < 1e-6] = 1.0

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        if self.transform:
            img = self.transform(img)
        t = self.targets[idx]
        targets = torch.tensor(
            [t["calories"], t["mass"], t["protein"], t["fat"], t["carbs"]],
            dtype=torch.float32,
        )
        return img, targets


# --- Nutrition5k adapter --------------------------------------------------


class Nutrition5kDataset(Dataset):
    """
    Nutrition5k dataset adapter.

    Expected layout::

        nutrition5k/
          imagery/
            rec_<id>/
              rgb/
                frames/*.jpg
          metadata/
            dish_metadata_cafe1.csv
            dish_metadata_cafe2.csv
            ingredient_metadata.csv

    Each dish has total weight and per-ingredient nutrition.
    We use the total per-dish aggregate from dish_metadata.
    """

    NUTRIENT_COLS = ["calories", "mass", "protein", "fat", "carbs"]

    def __init__(self, root, split="train", transform=None, frames_per_dish=4):
        self.root = Path(root)
        self.transform = transform
        self.frames_per_dish = frames_per_dish
        self.split = split

        # Load dish metadata
        self.dishes = self._load_metadata()

        # Simple train/val split by dish ID
        n = len(self.dishes)
        indices = list(range(n))
        np.random.RandomState(42).shuffle(indices)
        split_point = int(n * 0.8)
        if split == "train":
            indices = indices[:split_point]
        else:
            indices = indices[split_point:]
        self.dishes = [self.dishes[i] for i in indices]

    def _load_metadata(self):
        """Parse dish_metadata CSV files and return list of dish dicts."""
        dishes = []
        metadata_dir = self.root / "metadata"
        if not metadata_dir.exists():
            # Return empty — user needs to download the dataset
            print(f"[warn] Nutrition5k metadata not found at {metadata_dir}")
            return dishes

        import csv
        for csv_path in sorted(metadata_dir.glob("dish_metadata*.csv")):
            with csv_path.open() as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        dish = {
                            "dish_id": row.get("dish_id", ""),
                            "calories": float(row.get("total_calories", 0)),
                            "mass": float(row.get("total_mass", 0)),
                            "protein": float(row.get("total_protein", 0)),
                            "fat": float(row.get("total_fat", 0)),
                            "carbs": float(row.get("total_carbs", 0)),
                        }
                        if dish["mass"] > 0:
                            dishes.append(dish)
                    except (ValueError, KeyError):
                        continue
        return dishes

    def __len__(self):
        return len(self.dishes) * self.frames_per_dish

    def __getitem__(self, idx):
        dish_idx = idx // self.frames_per_dish
        frame_offset = idx % self.frames_per_dish
        dish = self.dishes[dish_idx]

        # Find frames for this dish
        dish_dir = self.root / "imagery" / dish["dish_id"] / "rgb" / "frames"
        if not dish_dir.exists():
            # Fallback: return a random image from another dish
            return self[np.random.randint(0, len(self))]

        frames = sorted(dish_dir.glob("*.jpg"))
        if not frames:
            return self[np.random.randint(0, len(self))]

        frame = frames[frame_offset % len(frames)]
        img = Image.open(frame).convert("RGB")
        if self.transform:
            img = self.transform(img)

        targets = torch.tensor(
            [dish[k] for k in self.NUTRIENT_COLS], dtype=torch.float32
        )
        return img, targets


# --- NutritionVerse-Real adapter ------------------------------------------


class NutritionVerseRealDataset(Dataset):
    """
    NutritionVerse-Real dataset adapter.

    NV-Real provides 889 real food images with 251 unique dishes.
    Expected layout from Kaggle (nutritionverse/nv-real)::

        NV-Real/
          images/
            <dish_id>_<angle>.jpg
          metadata.csv  (or nv_real_labels.csv)

    Each image has: calories, mass, protein, fat, carbs (per-image total).
    """

    NUTRIENT_COLS = ["calories", "mass", "protein", "fat", "carbs"]

    def __init__(self, root, split="train", transform=None):
        self.root = Path(root)
        self.transform = transform
        self.split = split
        self.samples = self._load_samples()

        n = len(self.samples)
        indices = list(range(n))
        np.random.RandomState(42).shuffle(indices)
        split_point = int(n * 0.8)
        if split == "train":
            indices = indices[:split_point]
        else:
            indices = indices[split_point:]
        self.samples = [self.samples[i] for i in indices]

    def _load_samples(self):
        """Load image paths and nutrition labels."""
        samples = []
        # Try multiple common metadata file names
        metadata_candidates = [
            "metadata.csv", "nv_real_labels.csv", "labels.csv",
            "nutrition.csv", "nv_real_metadata.csv",
        ]
        metadata_path = None
        for name in metadata_candidates:
            candidate = self.root / name
            if candidate.exists():
                metadata_path = candidate
                break

        if metadata_path is None:
            print(f"[warn] No metadata CSV found in {self.root}")
            return samples

        import csv
        with metadata_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    # Find the image file
                    img_id = row.get("image_id") or row.get("id") or row.get("filename")
                    img_path = self._find_image(img_id)
                    if img_path is None:
                        continue

                    sample = {
                        "image_path": str(img_path),
                        "calories": float(row.get("calories", 0)),
                        "mass": float(row.get("mass", 0)),
                        "protein": float(row.get("protein", 0)),
                        "fat": float(row.get("fat", 0)),
                        "carbs": float(row.get("carbs", 0)),
                    }
                    if sample["mass"] > 0 and sample["calories"] > 0:
                        samples.append(sample)
                except (ValueError, KeyError):
                    continue
        return samples

    def _find_image(self, img_id):
        """Find an image file given a partial ID/name."""
        if not img_id:
            return None
        img_dir = self.root / "images"
        if not img_dir.exists():
            return None

        # Try exact match first, then glob
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
            candidate = img_dir / (img_id + ext)
            if candidate.exists():
                return candidate
            candidate = img_dir / img_id
            if candidate.exists():
                return candidate

        # Try glob
        matches = list(img_dir.glob(f"*{img_id}*"))
        return matches[0] if matches else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample["image_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        targets = torch.tensor(
            [sample[k] for k in self.NUTRIENT_COLS], dtype=torch.float32
        )
        return img, targets


# --- helper: create data loaders ------------------------------------------


def create_dataloaders(dataset_name, root, batch_size=8, num_workers=0, size=224):
    """
    Create train/val dataloaders for a named dataset.

    Args:
        dataset_name: 'synthetic', 'nutrition5k', or 'nutritionverse-real'
        root: path to dataset directory
        batch_size: batch size
        num_workers: DataLoader workers
        size: input image size

    Returns:
        (train_loader, val_loader, target_stats)
        target_stats is a dict with 'mean' and 'std' tensors for normalization.
    """
    train_tf = get_train_transforms(size)
    val_tf = get_val_transforms(size)

    if dataset_name == "synthetic":
        train_ds = SyntheticFixture(num_samples=80, size=size, transform=train_tf)
        val_ds = SyntheticFixture(num_samples=20, size=size, transform=val_tf)
        target_stats = {
            "mean": train_ds.target_mean,
            "std": train_ds.target_std,
        }
    elif dataset_name == "nutrition5k":
        train_ds = Nutrition5kDataset(root, split="train", transform=train_tf)
        val_ds = Nutrition5kDataset(root, split="val", transform=val_tf)
        target_stats = _compute_target_stats(train_ds)
    elif dataset_name == "nutritionverse-real":
        train_ds = NutritionVerseRealDataset(root, split="train", transform=train_tf)
        val_ds = NutritionVerseRealDataset(root, split="val", transform=val_tf)
        target_stats = _compute_target_stats(train_ds)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, target_stats


def _compute_target_stats(dataset):
    """Compute mean/std of targets for normalization."""
    all_targets = []
    for i in range(len(dataset)):
        _, t = dataset[i]
        all_targets.append(t.numpy())
    all_targets = np.array(all_targets)
    return {
        "mean": all_targets.mean(axis=0).astype(np.float32),
        "std": all_targets.std(axis=0).astype(np.float32),
    }
