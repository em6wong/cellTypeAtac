#!/usr/bin/env python
"""Cross-model specificity evaluation for per-cell-type ChromBPNet models.

Evaluates:
  1. Per-model metrics on test chromosomes (profile r, count r, JSD)
  2. Cross-model specificity: does the "correct" model predict highest signal
     for cell-type specific peaks?
  3. Cross-model correlation: how similar are predictions between models?

Usage:
    python scripts/07_evaluate_specificity.py \
        --model-dir results/chrombpnet \
        --data-dir data/training \
        --annotations data/peak_annotations.csv \
        --output-dir results/specificity
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.chrombpnet import ChromBPNet, BiasModel, ChromBPNetWithBias
from src.data.dataset import ATACDataset
from src.utils.metrics import (
    profile_pearson_r, count_pearson_r, profile_jsd,
    specificity_auc, cross_model_correlation,
)


CELL_TYPES = ["Cardiomyocyte", "Coronary_EC", "Fibroblast", "Macrophage", "Pericytes"]


def load_model(checkpoint_path: str, device: str = "cpu") -> ChromBPNetWithBias:
    """Load a trained ChromBPNetWithBias from checkpoint."""
    state = torch.load(checkpoint_path, map_location=device)
    if "state_dict" in state:
        state_dict = state["state_dict"]
    else:
        state_dict = state

    # Reconstruct model
    main_model = ChromBPNet()
    bias_model = BiasModel()
    model = ChromBPNetWithBias(main_model, bias_model)

    # Load weights
    model_state = {k.replace("model.", ""): v for k, v in state_dict.items()
                   if k.startswith("model.")}
    if model_state:
        model.load_state_dict(model_state, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model.to(device)


@torch.no_grad()
def predict_all(model, dataset, device="cpu", batch_size=64):
    """Run model on all samples in a dataset.

    Returns:
        Dict with 'profiles' (n, length), 'counts' (n,), 'target_profiles', 'target_counts'.
    """
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2,
    )

    pred_profiles = []
    pred_counts = []
    target_profiles = []
    target_counts = []

    for batch in tqdm(loader, desc="Predicting"):
        seq = batch["sequence"].to(device)
        out = model(seq)

        # Convert profile logits to probabilities for comparison
        profile_probs = torch.softmax(out["profile"], dim=-1)
        pred_count = torch.exp(out["count"])

        pred_profiles.append(profile_probs.cpu().numpy())
        pred_counts.append(pred_count.cpu().numpy())
        target_profiles.append(batch["profile"].numpy())
        target_counts.append(batch["count"].numpy())

    return {
        "pred_profiles": np.concatenate(pred_profiles),
        "pred_counts": np.concatenate(pred_counts),
        "target_profiles": np.concatenate(target_profiles),
        "target_counts": np.concatenate(target_counts),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate cell-type specificity")
    parser.add_argument("--model-dir", type=str, default="results/chrombpnet")
    parser.add_argument("--data-dir", type=str, default="data/training")
    parser.add_argument("--annotations", type=str, default="data/peak_annotations.csv")
    parser.add_argument("--output-dir", type=str, default="results/specificity")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load annotations
    annotations = pd.read_csv(args.annotations)
    print(f"Loaded {len(annotations):,} peak annotations")

    # Load and evaluate each model
    all_results = {}
    all_predictions = {}

    for ct in CELL_TYPES:
        ckpt_path = Path(args.model_dir) / ct / "best.ckpt"
        if not ckpt_path.exists():
            print(f"WARNING: Checkpoint not found for {ct}: {ckpt_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {ct}")
        print(f"{'='*60}")

        # Load model
        model = load_model(str(ckpt_path), args.device)

        # Evaluate on test set
        test_zarr = Path(args.data_dir) / ct / "test.zarr"
        if not test_zarr.exists():
            print(f"WARNING: Test data not found: {test_zarr}")
            continue

        test_ds = ATACDataset(str(test_zarr), split="test")
        print(f"Test samples: {len(test_ds):,}")

        results = predict_all(model, test_ds, args.device, args.batch_size)

        # Per-model metrics
        prof_r = profile_pearson_r(results["pred_profiles"], results["target_profiles"])
        cnt_r = count_pearson_r(results["pred_counts"], results["target_counts"])
        jsd = profile_jsd(results["pred_profiles"], results["target_profiles"])

        print(f"  Profile Pearson r: {prof_r:.4f}")
        print(f"  Count Pearson r:   {cnt_r:.4f}")
        print(f"  Profile JSD:       {jsd:.4f}")

        all_results[ct] = {
            "profile_r": prof_r,
            "count_r": cnt_r,
            "jsd": jsd,
        }
        all_predictions[ct] = results["pred_counts"]

    # Cross-model specificity analysis
    if len(all_predictions) >= 2:
        print(f"\n{'='*60}")
        print("CROSS-MODEL SPECIFICITY ANALYSIS")
        print(f"{'='*60}")

        # Specificity AUC
        spec_auc = specificity_auc(all_predictions, annotations, metric="count")
        print("\nSpecificity AUC (correct model ranks highest):")
        for ct, auc in spec_auc.items():
            print(f"  {ct}: {auc:.4f}")

        all_results["specificity_auc"] = spec_auc

        # Cross-model correlation
        corr_matrix = cross_model_correlation(all_predictions)
        ct_list = list(all_predictions.keys())
        print("\nCross-model correlation matrix:")
        print(f"{'':>15s}", end="")
        for ct in ct_list:
            print(f"{ct:>15s}", end="")
        print()
        for i, ct_i in enumerate(ct_list):
            print(f"{ct_i:>15s}", end="")
            for j in range(len(ct_list)):
                print(f"{corr_matrix[i,j]:>15.3f}", end="")
            print()

        # Mean off-diagonal correlation
        n = len(ct_list)
        off_diag = [corr_matrix[i, j] for i in range(n) for j in range(n) if i != j]
        mean_cross_r = float(np.mean(off_diag))
        print(f"\nMean cross-model correlation: {mean_cross_r:.4f}")

        all_results["cross_model_correlation"] = {
            "matrix": corr_matrix.tolist(),
            "cell_types": ct_list,
            "mean_off_diagonal": mean_cross_r,
        }

        # Success criteria
        print(f"\n{'='*60}")
        print("SUCCESS CRITERIA")
        print(f"{'='*60}")

        spec_pass = spec_auc.get("overall", 0) > 0.7
        corr_pass = mean_cross_r < 0.9

        print(f"  Specificity AUC > 0.7:      {spec_auc.get('overall', 0):.3f} {'PASS' if spec_pass else 'FAIL'}")
        print(f"  Cross-model r < 0.9:         {mean_cross_r:.3f} {'PASS' if corr_pass else 'FAIL'}")

        if spec_pass and corr_pass:
            print("\n>>> Models show cell-type specificity. Proceed to Stage 3.")
        else:
            print("\n>>> Models may not differentiate cell types sufficiently.")
            print("    Consider: more specific peaks, different loss, or data augmentation.")

    # Save results
    with open(out_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
