#!/usr/bin/env python3
"""
Export a trained PyTorch nutrition regressor to ONNX and verify parity.

Runs the SAME input tensor through PyTorch and ONNX Runtime and asserts
the outputs match within a tight tolerance. This is the gate that ensures
the browser model (ONNX) produces identical results to the training model.

Usage:
    # Export + verify:
    python scripts/export_and_verify.py --checkpoint training/checkpoints/best_model.pth --out model.onnx

    # Verify an existing ONNX file against PyTorch:
    python scripts/export_and_verify.py --checkpoint training/checkpoints/best_model.pth --onnx model.onnx --verify-only

    # Also test against the existing convert_to_onnx.py architectures:
    python scripts/export_and_verify.py --arch nutritionverse-direct --checkpoint nv.pth --out model.onnx
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# These are imported conditionally because they require the ML stack
try:
    import onnx
    import onnxruntime as ort
    import torch
except ImportError:
    sys.exit("Missing dependencies: pip install torch onnx onnxruntime")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def export_regressor(checkpoint_path, output_path, image_size=224):
    """Export NutritionRegressor from training/ to ONNX."""
    from training.train_regressor import NutritionRegressor

    model = NutritionRegressor(freeze_backbone=True)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    dummy = torch.randn(1, 3, image_size, image_size)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported ONNX model to {output_path}")
    return model, dummy


def export_from_arch(arch, checkpoint_path, output_path):
    """Export using one of the convert_to_onnx.py architectures."""
    sys.path.insert(0, str(REPO_ROOT))
    from convert_to_onnx import BUILDERS

    builder, dummy_shape = BUILDERS[arch]
    model = builder()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = state.get("state_dict", state) if isinstance(state, dict) else state
    model.load_state_dict(sd, strict=False)
    model.eval()

    dummy = torch.randn(1, *dummy_shape)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported {arch} ONNX to {output_path}")
    return model, dummy


def verify_parity(pytorch_model, onnx_path, dummy_input, rtol=1e-4, atol=1e-6):
    """
    Run the same tensor through PyTorch and ONNX Runtime, compare outputs.

    Returns: (passed, max_diff, metrics_dict)
    """
    # PyTorch forward pass
    with torch.no_grad():
        pt_output = pytorch_model(dummy_input).cpu().numpy()

    # ONNX Runtime forward pass
    # Validate ONNX model structure first
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model verified: {onnx_model.ir_version=}, opset={onnx_model.opset_import}")

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    ort_outputs = sess.run(None, {input_name: dummy_input.numpy()})

    # Compare
    ort_output = ort_outputs[0]
    diff = np.abs(pt_output - ort_output)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    passed = max_diff < rtol

    metrics = {
        "max_absolute_difference": max_diff,
        "mean_absolute_difference": mean_diff,
        "passed": passed,
        "rtol": rtol,
        "atol": atol,
        "pt_shape": list(pt_output.shape),
        "ort_shape": list(ort_output.shape),
        "pt_output_sample": pt_output.flatten()[:10].tolist(),
        "ort_output_sample": ort_output.flatten()[:10].tolist(),
    }

    if passed:
        print(f"✅ PARITY PASS: max_diff={max_diff:.2e} < rtol={rtol}")
    else:
        print(f"❌ PARITY FAIL: max_diff={max_diff:.2e} >= rtol={rtol}")
        print(f"   Mean diff: {mean_diff:.2e}")
        # Show worst channels
        channel_diffs = diff.mean(axis=0).flatten()
        for i, d in enumerate(channel_diffs):
            print(f"   Channel {i}: mean_diff={d:.4f}")

    return passed, metrics


def main():
    ap = argparse.ArgumentParser(description="Export regressor to ONNX and verify parity")
    ap.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    ap.add_argument("--arch", default="regressor",
                    choices=["regressor", "nutritionverse-direct", "foodcnn-nutrition5k"],
                    help="Model architecture")
    ap.add_argument("--out", default="model.onnx", help="Output ONNX path")
    ap.add_argument("--onnx", default=None, help="Existing ONNX file to verify (--verify-only)")
    ap.add_argument("--verify-only", action="store_true",
                    help="Only verify an existing ONNX file, don't export")
    ap.add_argument("--image-size", type=int, default=224, help="Input image size")
    ap.add_argument("--rtol", type=float, default=1e-4, help="Relative tolerance")
    ap.add_argument("--output-json", default=None, help="Save parity metrics as JSON")
    args = ap.parse_args()

    out_path = Path(args.out)

    if args.verify_only:
        # Re-export a temporary model, then verify against existing ONNX
        if args.arch == "regressor":
            model, dummy = export_regressor(args.checkpoint, out_path.with_suffix(".tmp.onnx"),
                                            args.image_size)
        else:
            model, dummy = export_from_arch(args.arch, args.checkpoint,
                                            out_path.with_suffix(".tmp.onnx"))
        onnx_path = Path(args.onnx or args.out)
        passed, metrics = verify_parity(model, onnx_path, dummy, rtol=args.rtol)
        # Clean up temp
        tmp = out_path.with_suffix(".tmp.onnx")
        if tmp.exists():
            tmp.unlink()
    else:
        # Export and verify in one step
        if args.arch == "regressor":
            model, dummy = export_regressor(args.checkpoint, out_path, args.image_size)
        else:
            model, dummy = export_from_arch(args.arch, args.checkpoint, out_path)

        # Verify the exported model
        passed, metrics = verify_parity(model, out_path, dummy, rtol=args.rtol)

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(metrics, indent=2, default=float))

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
