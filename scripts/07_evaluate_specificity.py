#!/usr/bin/env python
"""Cross-model specificity evaluation for per-cell-type ChromBPNet models.

Evaluates:
  1. Per-model metrics on test chromosomes (profile r, count r, JSD)
  2. Cross-model specificity: does the "correct" model predict highest signal
     for cell-type specific peaks?
  3. Cross-model correlation: how similar are predictions between models?

All models are evaluated on the same reference test dataset to ensure
positional alignment. Annotations are coordinate-matched to test regions.

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
import zarr
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
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
        Dict with 'pred_profiles' (n, length), 'pred_counts' (n,),
        'target_profiles', 'target_counts'.
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


def build_aligned_annotations(data_dir: Path, annotations: pd.DataFrame) -> pd.DataFrame:
    """Build annotations aligned with test prediction arrays.

    Reads coordinates from the reference test zarr and matches them to
    the full annotations by (chrom, start, end). Returns a DataFrame
    where row i corresponds to prediction position i.

    Args:
        data_dir: Path to training data directory.
        annotations: Full peak annotations DataFrame.

    Returns:
        DataFrame aligned with prediction arrays. Rows for background
        regions (not in annotations) have category='background'.
    """
    # Use first cell type's zarr as reference for coordinates
    ref_zarr_path = str(data_dir / CELL_TYPES[0] / "test.zarr")
    ref_root = zarr.open(ref_zarr_path, "r")

    # Read all coordinates from zarr
    all_chroms = np.array([
        c.decode() if isinstance(c, bytes) else c
        for c in ref_root["chrom"][:]
    ])
    all_starts = ref_root["start"][:].astype(int)
    all_ends = ref_root["end"][:].astype(int)

    # Get the dataset indices for the test split (same filtering as ATACDataset)
    ref_ds = ATACDataset(ref_zarr_path, split="test")
    test_indices = ref_ds.indices

    # Build DataFrame of test region coordinates
    test_regions = pd.DataFrame({
        "chrom": all_chroms[test_indices],
        "start": all_starts[test_indices],
        "end": all_ends[test_indices],
    })

    # Left-join with annotations on coordinates
    ann_cols = ["chrom", "start", "end", "category", "specific_celltype", "tau"]
    available_cols = [c for c in ann_cols if c in annotations.columns]
    aligned = test_regions.merge(
        annotations[available_cols],
        on=["chrom", "start", "end"],
        how="left",
    )

    # Fill NaN for regions not in annotations (background regions)
    aligned["category"] = aligned["category"].fillna("background")
    aligned["specific_celltype"] = aligned["specific_celltype"].fillna("none")

    n_matched = (aligned["category"] != "background").sum()
    n_specific = (aligned["category"] == "specific").sum()
    print(f"Test regions: {len(aligned):,}")
    print(f"  Matched to annotations: {n_matched:,}")
    print(f"  Specific peaks: {n_specific:,}")

    return aligned


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
    data_dir = Path(args.data_dir)

    # Load annotations and build aligned version for test regions
    annotations = pd.read_csv(args.annotations)
    print(f"Loaded {len(annotations):,} peak annotations")

    print("\nBuilding coordinate-aligned annotations for test regions...")
    test_annotations = build_aligned_annotations(data_dir, annotations)

    # Use reference test dataset for all models
    # All cell types share the same peaks/regions/sequences; only targets differ
    ref_zarr_path = str(data_dir / CELL_TYPES[0] / "test.zarr")
    ref_ds = ATACDataset(ref_zarr_path, split="test")
    print(f"\nReference test dataset: {len(ref_ds):,} regions")

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

        # Run model on CT's own test zarr (for per-model metrics with correct targets)
        ct_zarr_path = str(data_dir / ct / "test.zarr")
        ct_ds = ATACDataset(ct_zarr_path, split="test")
        print(f"Test samples: {len(ct_ds):,}")

        # Verify alignment: same number of regions as reference
        if len(ct_ds) != len(ref_ds):
            print(f"WARNING: {ct} test set size ({len(ct_ds)}) differs from "
                  f"reference ({len(ref_ds)}). Cross-model comparison may be invalid.")

        results = predict_all(model, ct_ds, args.device, args.batch_size)

        # Per-model metrics (model vs its own cell type's targets)
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
        # Store predictions for cross-model analysis
        all_predictions[ct] = results["pred_counts"]

    # Cross-model specificity analysis
    if len(all_predictions) >= 2:
        print(f"\n{'='*60}")
        print("CROSS-MODEL SPECIFICITY ANALYSIS")
        print(f"{'='*60}")

        # Verify all prediction arrays have the same length
        pred_lengths = {ct: len(p) for ct, p in all_predictions.items()}
        if len(set(pred_lengths.values())) > 1:
            print(f"ERROR: Prediction arrays have different lengths: {pred_lengths}")
            print("Cannot perform cross-model comparison.")
        else:
            # Specificity AUC using aligned annotations
            spec_auc = specificity_auc(all_predictions, test_annotations, metric="count")
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
