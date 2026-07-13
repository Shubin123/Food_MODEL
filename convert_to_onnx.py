#!/usr/bin/env python3
"""
Convert a SOTA food-nutrition PyTorch checkpoint to ONNX so it can run in the
browser with ONNX Runtime Web (no server, fully on-device).

Supported sources
-----------------
  nutritionverse-direct : NutritionVerse-Direct (Keller et al., 2024) - ViT-base
                          + 5 regression heads (calories, mass, protein, fat, carbs).
  foodcnn-nutrition5k    : FoodCNN / Nutrition5k (Thames et al., CVPR 2021) -
                          Inception-ResNet + 5 regression heads.

Weights
-------
  * NutritionVerse-Direct: request the released checkpoints (bit.ly/genai4good)
    and pass the .pth to --checkpoint.
  * FoodCNN: clone https://github.com/FoodCNN/FoodCNN and use its trained
    checkpoint, or train your own on Nutrition5k.

Usage
-----
  pip install torch torchvision
  python convert_to_onnx.py --model nutritionverse-direct \
      --checkpoint nv_direct_vit.pth --out model.onnx

Then drop model.onnx next to index.html and open it (or serve with
`python3 -m http.server`).

NOTE: the fully-connected widths below must match the released checkpoint.
If loading the state_dict fails with a size mismatch, adjust SHARED/HEAD dims
to the architecture used to train your checkpoint.
"""
import argparse

import torch
import torch.nn as nn
import torchvision


class Identity(nn.Module):
    def forward(self, x):
        return x


# ----------------------------- architectures ----------------------------- #

class NutritionVerseDirect(nn.Module):
    """ViT-base feature extractor + shared 2 FC + per-task head (5 outputs)."""

    SHARED = [768, 512, 256]   # feature_dim -> hidden1 -> hidden2
    HEAD = 128                 # per-task hidden width

    def __init__(self, num_tasks=5):
        super().__init__()
        vit = torchvision.models.vit_b_16(weights="DEFAULT")
        vit.heads = Identity()          # -> [B, 768] class token embedding
        self.backbone = vit
        self.shared = nn.Sequential(
            nn.Linear(self.SHARED[0], self.SHARED[1]),
            nn.ReLU(),
            nn.Linear(self.SHARED[1], self.SHARED[2]),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.SHARED[2], self.HEAD),
                nn.ReLU(),
                nn.Linear(self.HEAD, 1),
            )
            for _ in range(num_tasks)
        ])

    def forward(self, x):
        feats = self.shared(self.backbone(x))
        return torch.cat([h(feats) for h in self.heads], dim=1)  # [B, num_tasks]


class FoodCNNNutrition5k(nn.Module):
    """Inception-ResNet feature extractor + 4 shared FC + per-task linear (5 outputs)."""

    FEAT = 1536               # inception_resnet_v2 feature dim
    SHARED = [FEAT, 512, 256, 128]  # 4 fully connected layers

    def __init__(self, num_tasks=5):
        super().__init__()
        backbone = torchvision.models.inception_resnet_v2(weights="DEFAULT")
        backbone.aux_logits = False
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.flatten = nn.Flatten()
        layers = []
        for i in range(len(self.SHARED) - 1):
            layers += [nn.Linear(self.SHARED[i], self.SHARED[i + 1]), nn.ReLU()]
        self.shared = nn.Sequential(*layers)
        self.heads = nn.ModuleList([nn.Linear(self.SHARED[-1], 1) for _ in range(num_tasks)])

    def forward(self, x):
        feats = self.flatten(self.backbone(x))
        feats = self.shared(feats)
        return torch.cat([h(feats) for h in self.heads], dim=1)


BUILDERS = {
    "nutritionverse-direct": (NutritionVerseDirect, (3, 224, 224)),
    "foodcnn-nutrition5k": (FoodCNNNutrition5k, (3, 299, 299)),
}


def main():
    ap = argparse.ArgumentParser(description="Convert a nutrition model to ONNX")
    ap.add_argument("--model", required=True, choices=list(BUILDERS.keys()))
    ap.add_argument("--checkpoint", required=True, help="path to .pth weights")
    ap.add_argument("--out", default="model.onnx", help="output .onnx path")
    ap.add_argument("--num-tasks", type=int, default=5)
    args = ap.parse_args()

    builder, dummy_shape = BUILDERS[args.model]
    model = builder(num_tasks=args.num_tasks)
    state = torch.load(args.checkpoint, map_location="cpu")
    # accept either a raw state_dict or a {"state_dict": ...} checkpoint
    sd = state.get("state_dict", state) if isinstance(state, dict) else state
    model.load_state_dict(sd, strict=False)
    model.eval()

    dummy = torch.randn(1, *dummy_shape)
    torch.onnx.export(
        model,
        dummy,
        args.out,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported {args.model} -> {args.out}")


if __name__ == "__main__":
    main()
