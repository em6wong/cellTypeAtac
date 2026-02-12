#!/usr/bin/env python
"""Comprehensive validation for cellTypeAtac models (Stages 2 and 3).

Adapted from multiome/scripts/validate_model.py for the ChromBPNet pipeline.

Tests:
  1. Basic inference - model loads, outputs are valid
  2. Test set correlation - profile Pearson r, count Pearson r, JSD per cell type
  3. Cell-type specificity - predictions at known cardiac marker loci
  4. Collapse detection - cross-cell-type correlation (Stage 3 only)
  5. Prediction variability - outputs differ for different inputs
  6. Specificity-only evaluation - metrics on tau-specific peaks only
  7. In-silico motif perturbation - insert/destroy TF motifs, check response
  8. Background-normalized specificity - signal ratio at specific vs all peaks
  9. Profile/scatter/boxplot plots and Stage 2 vs Stage 3 comparison

Usage:
    # Validate per-cell-type models (Stage 2)
    python scripts/09_validate_models.py --stage 2 \
        --model-dir results/chrombpnet \
        --data-dir data/training \
        --output-dir results/validation/stage2

    # Validate multi-cell model (Stage 3)
    python scripts/09_validate_models.py --stage 3 \
        --checkpoint results/multi_cell/best.ckpt \
        --data-dir data/training \
        --output-dir results/validation/stage3

    # Compare Stage 2 and Stage 3
    python scripts/09_validate_models.py --stage both \
        --model-dir results/chrombpnet \
        --checkpoint results/multi_cell/best.ckpt \
        --data-dir data/training \
        --output-dir results/validation/comparison
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy import stats
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.chrombpnet import ChromBPNet, BiasModel, ChromBPNetWithBias
from src.models.multi_cell_chrombpnet import MultiCellChromBPNet
from src.data.dataset import ATACDataset, MultiCellATACDataset


CELL_TYPES = ["Cardiomyocyte", "Coronary_EC", "Fibroblast", "Macrophage", "Pericytes"]

# Known cardiac marker genes (mm10 coordinates from GENCODE vM25)
# Used for sanity checking that models learn cell-type-specific signal
KNOWN_MARKERS = {
    "Myh6": {"chrom": "chr14", "tss": 54966927, "expected_high": "Cardiomyocyte"},
    "Myh7": {"chrom": "chr14", "tss": 54994626, "expected_high": "Cardiomyocyte"},
    "Pecam1": {"chrom": "chr11", "tss": 106750628, "expected_high": "Coronary_EC"},
    "Col1a1": {"chrom": "chr11", "tss": 94936224, "expected_high": "Fibroblast"},
    "Cd68": {"chrom": "chr11", "tss": 69663605, "expected_high": "Macrophage"},
    "Pdgfrb": {"chrom": "chr18", "tss": 61058246, "expected_high": "Pericytes"},
}


# ==============================================================================
# Model Loading
# ==============================================================================

def load_single_ct_model(
    checkpoint_path: str,
    device: str = "cpu",
) -> ChromBPNetWithBias:
    """Load a per-cell-type ChromBPNetWithBias from checkpoint."""
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "state_dict" in state:
        state_dict = state["state_dict"]
    else:
        state_dict = state

    main_model = ChromBPNet()
    bias_model = BiasModel()
    model = ChromBPNetWithBias(main_model, bias_model)

    model_state = {k.replace("model.", ""): v for k, v in state_dict.items()
                   if k.startswith("model.")}
    if model_state:
        model.load_state_dict(model_state, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model.to(device)


def load_multi_cell_model(
    checkpoint_path: str,
    device: str = "cpu",
) -> MultiCellChromBPNet:
    """Load a multi-cell ChromBPNet from checkpoint."""
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "state_dict" in state:
        state_dict = state["state_dict"]
    else:
        state_dict = state

    model = MultiCellChromBPNet(num_cell_types=len(CELL_TYPES))

    model_state = {k.replace("model.", ""): v for k, v in state_dict.items()
                   if k.startswith("model.")}
    if model_state:
        model.load_state_dict(model_state, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model.to(device)


# ==============================================================================
# Prediction Helpers
# ==============================================================================

@torch.no_grad()
def predict_single_ct(
    model: ChromBPNetWithBias,
    dataset: ATACDataset,
    device: str,
    max_samples: int = 0,
    batch_size: int = 64,
) -> Dict[str, np.ndarray]:
    """Run per-cell-type model on dataset."""
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2,
    )

    pred_profiles, pred_counts = [], []
    target_profiles, target_counts = [], []

    n = 0
    for batch in tqdm(loader, desc="Predicting"):
        seq = batch["sequence"].to(device)
        out = model(seq)

        profile_probs = torch.softmax(out["profile"], dim=-1)
        pred_count = torch.expm1(out["count"])

        pred_profiles.append(profile_probs.cpu().numpy())
        pred_counts.append(pred_count.cpu().numpy())
        target_profiles.append(batch["profile"].numpy())
        target_counts.append(batch["count"].numpy())

        n += seq.size(0)
        if max_samples > 0 and n >= max_samples:
            break

    return {
        "pred_profiles": np.concatenate(pred_profiles),
        "pred_counts": np.concatenate(pred_counts),
        "target_profiles": np.concatenate(target_profiles),
        "target_counts": np.concatenate(target_counts),
    }


@torch.no_grad()
def predict_multi_cell(
    model: MultiCellChromBPNet,
    dataset: MultiCellATACDataset,
    device: str,
    max_samples: int = 0,
    batch_size: int = 64,
) -> Dict[str, np.ndarray]:
    """Run multi-cell model on dataset. Returns per-cell-type predictions."""
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2,
    )

    pred_profiles = {ct: [] for ct in CELL_TYPES}
    pred_counts = {ct: [] for ct in CELL_TYPES}
    target_profiles = {ct: [] for ct in CELL_TYPES}
    target_counts = {ct: [] for ct in CELL_TYPES}

    n = 0
    for batch in tqdm(loader, desc="Predicting (multi-cell)"):
        seq = batch["sequence"].to(device)
        out = model(seq)  # profile: (B, n_ct, 1000), count: (B, n_ct)

        for ct_idx, ct in enumerate(CELL_TYPES):
            ct_profile = torch.softmax(out["profile"][:, ct_idx, :], dim=-1)
            ct_count = torch.expm1(out["count"][:, ct_idx])

            pred_profiles[ct].append(ct_profile.cpu().numpy())
            pred_counts[ct].append(ct_count.cpu().numpy())
            target_profiles[ct].append(batch["profile"][:, ct_idx, :].numpy())
            target_counts[ct].append(batch["count"][:, ct_idx].numpy())

        n += seq.size(0)
        if max_samples > 0 and n >= max_samples:
            break

    return {
        ct: {
            "pred_profiles": np.concatenate(pred_profiles[ct]),
            "pred_counts": np.concatenate(pred_counts[ct]),
            "target_profiles": np.concatenate(target_profiles[ct]),
            "target_counts": np.concatenate(target_counts[ct]),
        }
        for ct in CELL_TYPES
    }


# ==============================================================================
# Metrics
# ==============================================================================

def compute_per_peak_profile_r(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-peak Pearson r between predicted and target profiles.

    Returns array of per-peak correlations.
    """
    correlations = []
    for i in range(len(pred)):
        if target[i].std() < 1e-8 and pred[i].std() < 1e-8:
            correlations.append(1.0)
        elif target[i].std() < 1e-8 or pred[i].std() < 1e-8:
            correlations.append(0.0)
        else:
            r, _ = stats.pearsonr(pred[i], target[i])
            correlations.append(r if not np.isnan(r) else 0.0)
    return np.array(correlations)


def compute_jsd(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-peak JSD between predicted and target profiles."""
    jsds = []
    for i in range(len(pred)):
        p = target[i] + eps
        p = p / p.sum()
        q = pred[i] + eps
        q = q / q.sum()
        jsds.append(float(jensenshannon(p, q) ** 2))
    return np.array(jsds)


# ==============================================================================
# Test 1: Basic Inference
# ==============================================================================

def test_basic_inference(
    model: torch.nn.Module,
    device: str,
    is_multi_cell: bool = False,
) -> bool:
    """Test that model loads and produces valid output."""
    print("\n" + "=" * 60)
    print("TEST 1: Basic Inference")
    print("=" * 60)

    try:
        # Random sequence (batch=1, 4, 2114)
        seq = torch.randn(1, 4, 2114).to(device)
        # Make it look like one-hot (softmax along channel dim)
        seq = torch.softmax(seq, dim=1)

        if is_multi_cell:
            out = model(seq)
            print(f"  Input: {seq.shape}")
            print(f"  Profile output: {out['profile'].shape}")
            print(f"  Count output: {out['count'].shape}")

            assert out["profile"].shape == (1, len(CELL_TYPES), 1000), \
                f"Expected profile (1, {len(CELL_TYPES)}, 1000), got {out['profile'].shape}"
            assert out["count"].shape == (1, len(CELL_TYPES)), \
                f"Expected count (1, {len(CELL_TYPES)}), got {out['count'].shape}"
        else:
            out = model(seq)
            print(f"  Input: {seq.shape}")
            print(f"  Profile output: {out['profile'].shape}")
            print(f"  Count output: {out['count'].shape}")

            assert out["profile"].shape[1] == 1000, \
                f"Expected profile length 1000, got {out['profile'].shape[1]}"

        # Check for NaN/Inf
        for key in ["profile", "count"]:
            if torch.isnan(out[key]).any():
                print(f"  FAILED: NaN in {key}")
                return False
            if torch.isinf(out[key]).any():
                print(f"  FAILED: Inf in {key}")
                return False

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")
        print("  PASSED")
        return True

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ==============================================================================
# Test 2: Test Set Correlation
# ==============================================================================

def test_correlation_stage2(
    model_dir: str,
    data_dir: str,
    device: str,
    max_samples: int,
    batch_size: int,
) -> Dict[str, Dict[str, float]]:
    """Test correlation for per-cell-type models on test set."""
    print("\n" + "=" * 60)
    print("TEST 2: Test Set Correlation (Stage 2 - Per-Cell-Type)")
    print("=" * 60)

    all_results = {}
    all_profile_corrs = {}

    for ct in CELL_TYPES:
        ckpt_path = Path(model_dir) / ct / "best.ckpt"
        if not ckpt_path.exists():
            print(f"  WARNING: Checkpoint not found for {ct}: {ckpt_path}")
            continue

        print(f"\n  {ct}:")
        model = load_single_ct_model(str(ckpt_path), device)

        zarr_path = str(Path(data_dir) / ct / "test.zarr")
        ds = ATACDataset(zarr_path, split="test")
        print(f"    Test samples: {len(ds):,}")

        results = predict_single_ct(model, ds, device, max_samples, batch_size)

        # Per-peak profile Pearson r
        profile_r = compute_per_peak_profile_r(
            results["pred_profiles"], results["target_profiles"]
        )
        # Count Pearson r
        if results["pred_counts"].std() > 1e-8 and results["target_counts"].std() > 1e-8:
            count_r, _ = stats.pearsonr(results["pred_counts"], results["target_counts"])
        else:
            count_r = 0.0
        # Profile JSD
        jsd = compute_jsd(results["pred_profiles"], results["target_profiles"])

        mean_profile_r = float(profile_r.mean())
        mean_jsd = float(jsd.mean())

        print(f"    Profile Pearson r: {mean_profile_r:.4f} (median: {np.median(profile_r):.4f})")
        print(f"    Count Pearson r:   {count_r:.4f}")
        print(f"    Profile JSD:       {mean_jsd:.4f}")

        status = "GOOD" if mean_profile_r > 0.5 else "OK" if mean_profile_r > 0.3 else "LOW"
        print(f"    Status: [{status}]")

        all_results[ct] = {
            "profile_r_mean": mean_profile_r,
            "profile_r_median": float(np.median(profile_r)),
            "count_r": float(count_r),
            "jsd_mean": mean_jsd,
        }
        all_profile_corrs[ct] = profile_r

    # Summary
    if all_results:
        mean_across_ct = np.mean([v["profile_r_mean"] for v in all_results.values()])
        print(f"\n  Mean profile r across cell types: {mean_across_ct:.4f}")

    return all_results, all_profile_corrs


def test_correlation_stage3(
    checkpoint: str,
    data_dir: str,
    device: str,
    max_samples: int,
    batch_size: int,
) -> Dict[str, Dict[str, float]]:
    """Test correlation for multi-cell model on test set."""
    print("\n" + "=" * 60)
    print("TEST 2: Test Set Correlation (Stage 3 - Multi-Cell)")
    print("=" * 60)

    model = load_multi_cell_model(checkpoint, device)

    # Build multi-cell dataset
    zarr_paths = [str(Path(data_dir) / ct / "test.zarr") for ct in CELL_TYPES]
    for p in zarr_paths:
        if not Path(p).exists():
            print(f"  WARNING: Test zarr not found: {p}")
            return {}, {}

    ds = MultiCellATACDataset(zarr_paths, split="test")
    print(f"  Test samples: {len(ds):,}")

    results = predict_multi_cell(model, ds, device, max_samples, batch_size)

    all_results = {}
    all_profile_corrs = {}

    for ct in CELL_TYPES:
        ct_results = results[ct]

        profile_r = compute_per_peak_profile_r(
            ct_results["pred_profiles"], ct_results["target_profiles"]
        )
        if ct_results["pred_counts"].std() > 1e-8 and ct_results["target_counts"].std() > 1e-8:
            count_r, _ = stats.pearsonr(ct_results["pred_counts"], ct_results["target_counts"])
        else:
            count_r = 0.0
        jsd = compute_jsd(ct_results["pred_profiles"], ct_results["target_profiles"])

        mean_profile_r = float(profile_r.mean())
        mean_jsd = float(jsd.mean())

        status = "GOOD" if mean_profile_r > 0.5 else "OK" if mean_profile_r > 0.3 else "LOW"
        print(f"  {ct}: profile_r={mean_profile_r:.4f}, count_r={count_r:.4f}, jsd={mean_jsd:.4f} [{status}]")

        all_results[ct] = {
            "profile_r_mean": mean_profile_r,
            "profile_r_median": float(np.median(profile_r)),
            "count_r": float(count_r),
            "jsd_mean": mean_jsd,
        }
        all_profile_corrs[ct] = profile_r

    if all_results:
        mean_across_ct = np.mean([v["profile_r_mean"] for v in all_results.values()])
        print(f"\n  Mean profile r across cell types: {mean_across_ct:.4f}")

    return all_results, all_profile_corrs


# ==============================================================================
# Test 3: Cell-Type Specificity at Known Marker Loci
# ==============================================================================

def test_marker_specificity_stage2(
    model_dir: str,
    genome_path: str,
    device: str,
) -> Dict[str, Dict]:
    """Check if per-cell-type models produce highest signal at their marker genes."""
    print("\n" + "=" * 60)
    print("TEST 3: Known Marker Specificity (Stage 2)")
    print("=" * 60)
    print("  Each per-CT model should predict higher count at its own marker genes.")

    import pyfaidx
    genome = pyfaidx.Fasta(genome_path)

    # Load all models
    models = {}
    for ct in CELL_TYPES:
        ckpt = Path(model_dir) / ct / "best.ckpt"
        if ckpt.exists():
            models[ct] = load_single_ct_model(str(ckpt), device)

    if len(models) < 2:
        print("  SKIPPED: Need at least 2 models for comparison")
        return {}

    results = {}

    for gene_name, info in KNOWN_MARKERS.items():
        chrom = info["chrom"]
        tss = info["tss"]
        expected = info["expected_high"]

        # Extract 2114bp centered on TSS
        half = 2114 // 2
        start = tss - half
        end = tss + half
        chrom_len = len(genome[chrom])
        if start < 0 or end > chrom_len:
            print(f"  Skipping {gene_name}: too close to chromosome edge")
            continue

        seq_str = str(genome[chrom][start:end]).upper()
        seq = _one_hot_encode(seq_str).unsqueeze(0).to(device)

        print(f"\n  {gene_name} ({chrom}:{tss:,}), expected: {expected}")

        # Get predicted counts from each model
        ct_counts = {}
        for ct, model in models.items():
            with torch.no_grad():
                out = model(seq)
            pred_count = torch.expm1(out["count"]).item()
            ct_counts[ct] = pred_count

        # Display
        max_ct = max(ct_counts, key=ct_counts.get)
        for ct in CELL_TYPES:
            if ct not in ct_counts:
                continue
            marker = "*" if ct == expected else " "
            top = " <-- TOP" if ct == max_ct else ""
            print(f"    {marker} {ct:>15s}: {ct_counts[ct]:.2f}{top}")

        is_correct = max_ct == expected
        status = "CORRECT" if is_correct else "WRONG"
        print(f"    Result: [{status}]")

        results[gene_name] = {
            "expected": expected,
            "predicted_top": max_ct,
            "correct": is_correct,
            "counts": ct_counts,
        }

    # Summary
    n_correct = sum(1 for v in results.values() if v["correct"])
    n_total = len(results)
    print(f"\n  Marker accuracy: {n_correct}/{n_total}")

    return results


def test_marker_specificity_stage3(
    checkpoint: str,
    genome_path: str,
    device: str,
) -> Dict[str, Dict]:
    """Check if multi-cell model predicts highest count for correct cell type."""
    print("\n" + "=" * 60)
    print("TEST 3: Known Marker Specificity (Stage 3)")
    print("=" * 60)
    print("  Multi-cell model should predict highest count for correct cell type at marker loci.")

    import pyfaidx
    genome = pyfaidx.Fasta(genome_path)
    model = load_multi_cell_model(checkpoint, device)

    results = {}

    for gene_name, info in KNOWN_MARKERS.items():
        chrom = info["chrom"]
        tss = info["tss"]
        expected = info["expected_high"]

        half = 2114 // 2
        start = tss - half
        end = tss + half
        chrom_len = len(genome[chrom])
        if start < 0 or end > chrom_len:
            print(f"  Skipping {gene_name}: too close to chromosome edge")
            continue

        seq_str = str(genome[chrom][start:end]).upper()
        seq = _one_hot_encode(seq_str).unsqueeze(0).to(device)

        print(f"\n  {gene_name} ({chrom}:{tss:,}), expected: {expected}")

        with torch.no_grad():
            out = model(seq)  # profile: (1, n_ct, 1000), count: (1, n_ct)

        pred_counts = torch.expm1(out["count"]).squeeze(0).cpu().numpy()  # (n_ct,)

        max_idx = np.argmax(pred_counts)
        max_ct = CELL_TYPES[max_idx]

        for ct_idx, ct in enumerate(CELL_TYPES):
            marker = "*" if ct == expected else " "
            top = " <-- TOP" if ct_idx == max_idx else ""
            print(f"    {marker} {ct:>15s}: {pred_counts[ct_idx]:.2f}{top}")

        is_correct = max_ct == expected
        status = "CORRECT" if is_correct else "WRONG"
        print(f"    Result: [{status}]")

        results[gene_name] = {
            "expected": expected,
            "predicted_top": max_ct,
            "correct": is_correct,
            "counts": {ct: float(pred_counts[i]) for i, ct in enumerate(CELL_TYPES)},
        }

    n_correct = sum(1 for v in results.values() if v["correct"])
    n_total = len(results)
    print(f"\n  Marker accuracy: {n_correct}/{n_total}")

    return results


# ==============================================================================
# Test 4: Collapse Detection (Stage 3 only)
# ==============================================================================

def test_collapse_detection(
    checkpoint: str,
    data_dir: str,
    device: str,
    max_samples: int = 200,
    batch_size: int = 64,
) -> Dict[str, float]:
    """Check if multi-cell model produces differentiated predictions across cell types.

    Computes:
    1. Cross-cell-type correlation matrix of predicted counts
    2. Profile similarity across cell types at same loci
    3. Signal CV across cell types
    """
    print("\n" + "=" * 60)
    print("TEST 4: Collapse Detection (Stage 3)")
    print("=" * 60)

    model = load_multi_cell_model(checkpoint, device)

    # Build multi-cell dataset
    zarr_paths = [str(Path(data_dir) / ct / "test.zarr") for ct in CELL_TYPES]
    ds = MultiCellATACDataset(zarr_paths, split="test")

    results = predict_multi_cell(model, ds, device, max_samples, batch_size)

    # 1. Cross-cell-type count correlation
    print("\n  Cross-cell-type count correlation matrix:")
    print("  (Values close to 1.0 = collapsed predictions)")

    count_arrays = {ct: results[ct]["pred_counts"] for ct in CELL_TYPES}
    n_ct = len(CELL_TYPES)
    corr_matrix = np.ones((n_ct, n_ct))

    for i in range(n_ct):
        for j in range(i + 1, n_ct):
            r, _ = stats.pearsonr(count_arrays[CELL_TYPES[i]], count_arrays[CELL_TYPES[j]])
            corr_matrix[i, j] = r
            corr_matrix[j, i] = r

    # Print matrix
    header = "              " + "  ".join([f"{ct[:8]:>10s}" for ct in CELL_TYPES])
    print(f"  {header}")
    for i, ct in enumerate(CELL_TYPES):
        row = "  ".join([f"{corr_matrix[i, j]:10.3f}" for j in range(n_ct)])
        print(f"  {ct[:12]:12s}: {row}")

    off_diag = corr_matrix[np.triu_indices(n_ct, k=1)]
    mean_cross_corr = float(off_diag.mean())
    min_cross_corr = float(off_diag.min())

    print(f"\n  Mean cross-CT correlation: {mean_cross_corr:.3f}")
    print(f"  Min cross-CT correlation:  {min_cross_corr:.3f}")

    if mean_cross_corr > 0.95:
        print("  WARNING: Predictions highly similar - model likely collapsed!")
    elif mean_cross_corr > 0.85:
        print("  CAUTION: Predictions moderately similar across cell types")
    else:
        print("  GOOD: Predictions show cell-type differentiation")

    # 2. Signal CV across cell types
    mean_counts = np.array([count_arrays[ct].mean() for ct in CELL_TYPES])
    signal_cv = float(np.std(mean_counts) / (np.mean(mean_counts) + 1e-8))

    print(f"\n  Mean predicted count per cell type:")
    for i, ct in enumerate(CELL_TYPES):
        bar = "#" * int(mean_counts[i] * 20 / (mean_counts.max() + 1e-8))
        print(f"    {ct:>15s}: {mean_counts[i]:8.2f} {bar}")
    print(f"  Signal CV: {signal_cv:.3f}")

    if signal_cv < 0.05:
        print("  WARNING: Very similar signal levels - possible collapse")

    # 3. Profile similarity
    print("\n  Mean cross-CT profile correlation (at same peaks):")
    n_check = min(100, len(ds))
    profile_corrs = []
    for sample_idx in range(n_check):
        profiles = []
        for ct in CELL_TYPES:
            profiles.append(results[ct]["pred_profiles"][sample_idx])

        for i in range(n_ct):
            for j in range(i + 1, n_ct):
                r, _ = stats.pearsonr(profiles[i], profiles[j])
                profile_corrs.append(r if not np.isnan(r) else 0.0)

    mean_profile_cross = float(np.mean(profile_corrs))
    print(f"  Mean cross-CT profile r: {mean_profile_cross:.3f}")

    if mean_profile_cross > 0.95:
        print("  WARNING: Profile shapes nearly identical across cell types!")

    return {
        "mean_cross_ct_count_corr": mean_cross_corr,
        "min_cross_ct_count_corr": min_cross_corr,
        "signal_cv": signal_cv,
        "mean_cross_ct_profile_corr": mean_profile_cross,
        "corr_matrix": corr_matrix.tolist(),
    }


# ==============================================================================
# Test 5: Prediction Variability
# ==============================================================================

def test_prediction_variability(
    model: torch.nn.Module,
    device: str,
    is_multi_cell: bool = False,
) -> bool:
    """Check that predictions vary across different random sequences."""
    print("\n" + "=" * 60)
    print("TEST 5: Prediction Variability")
    print("=" * 60)

    predictions = []
    for _ in range(5):
        seq = torch.randn(1, 4, 2114).to(device)
        seq = torch.softmax(seq, dim=1)

        with torch.no_grad():
            out = model(seq)

        pred = out["count"].squeeze().cpu().numpy()
        predictions.append(pred)

    predictions = np.array(predictions)

    if predictions.ndim == 1:
        std_across = float(np.std(predictions))
    else:
        std_across = float(np.std(predictions, axis=0).mean())

    mean_pred = float(np.mean(np.abs(predictions)))
    cv = std_across / (mean_pred + 1e-8)

    print(f"  Count std across sequences: {std_across:.4f}")
    print(f"  Count CV: {cv:.4f}")

    if cv < 0.01:
        print("  WARNING: Predictions nearly identical for different inputs!")
        return False
    else:
        print("  PASSED (predictions vary across inputs)")
        return True


# ==============================================================================
# Test 6: Specificity-Only Evaluation
# ==============================================================================

def test_specific_peaks_only(
    model_dir: str,
    data_dir: str,
    annotations_path: str,
    device: str,
    max_samples: int = 500,
    batch_size: int = 64,
) -> Dict[str, Dict]:
    """Evaluate models ONLY on cell-type-specific peaks (tau > threshold).

    This addresses the CM bias problem: CM has many peaks, so overall correlation
    can be high without learning CM-specific grammar. By restricting to peaks
    where tau labels one cell type as "specific", we test whether the correct
    model truly outperforms others on those peaks.

    For each CT-specific peak, compares the "correct" model's count prediction
    vs other models' predictions, normalized by each model's mean prediction
    (removing the "CM predicts high everywhere" confound).
    """
    print("\n" + "=" * 60)
    print("TEST 6: Specificity-Only Evaluation")
    print("=" * 60)
    print("  Tests only on tau-specific peaks, with background-normalized counts.")
    print("  This controls for cell types with higher overall signal (e.g., CM).")

    import pandas as pd
    import zarr

    if not Path(annotations_path).exists():
        print(f"  SKIPPED: Annotations not found at {annotations_path}")
        return {}

    annotations = pd.read_csv(annotations_path)
    specific = annotations[annotations["category"] == "specific"]
    print(f"  Total annotations: {len(annotations):,}")
    print(f"  Specific peaks: {len(specific):,}")

    # Load models and compute mean predicted count per model (background level)
    models = {}
    model_mean_counts = {}

    for ct in CELL_TYPES:
        ckpt = Path(model_dir) / ct / "best.ckpt"
        zarr_path = str(Path(data_dir) / ct / "test.zarr")
        if not ckpt.exists() or not Path(zarr_path).exists():
            continue

        model = load_single_ct_model(str(ckpt), device)
        models[ct] = model

        # Get mean predicted count across ALL test peaks (background level)
        ds = ATACDataset(zarr_path, split="test")
        results = predict_single_ct(model, ds, device, max_samples=max_samples, batch_size=batch_size)
        model_mean_counts[ct] = float(results["pred_counts"].mean())
        print(f"  {ct} mean predicted count: {model_mean_counts[ct]:.2f}")

    if len(models) < 2:
        print("  SKIPPED: Need at least 2 models")
        return {}

    # Build coordinate index for test regions
    ref_ct = CELL_TYPES[0]
    ref_zarr = str(Path(data_dir) / ref_ct / "test.zarr")
    ref_root = zarr.open(ref_zarr, mode="r")

    chroms = np.array([c.decode() if isinstance(c, bytes) else c for c in ref_root["chrom"][:]])
    starts = ref_root["start"][:] if "start" in ref_root else None
    ends = ref_root["end"][:] if "end" in ref_root else None

    if starts is None:
        print("  SKIPPED: Test zarr missing start/end coordinates")
        return {}

    # Build test-region DataFrame for filtering
    from src.data.dataset import CHROM_SPLITS
    test_mask = np.array([c in CHROM_SPLITS["test"] for c in chroms])
    test_indices = np.where(test_mask)[0]

    test_regions = pd.DataFrame({
        "chrom": chroms[test_indices],
        "start": starts[test_indices].astype(int),
        "end": ends[test_indices].astype(int),
        "test_idx": test_indices,
    })

    # Match specific peaks to test regions
    aligned = test_regions.merge(
        specific[["chrom", "start", "end", "specific_celltype"]],
        on=["chrom", "start", "end"],
        how="inner",
    )

    print(f"  Specific peaks in test set: {len(aligned):,}")

    if len(aligned) < 5:
        print("  SKIPPED: Too few specific peaks in test set")
        return {}

    # For each specific peak, get normalized counts from each model
    results = {ct: {"correct_enrichment": [], "n_peaks": 0} for ct in CELL_TYPES}

    for ct in CELL_TYPES:
        if ct not in models:
            continue

        ct_peaks = aligned[aligned["specific_celltype"] == ct]
        if len(ct_peaks) == 0:
            continue

        results[ct]["n_peaks"] = len(ct_peaks)

        for _, peak in ct_peaks.iterrows():
            test_idx = peak["test_idx"]

            # Get each model's prediction at this peak
            seq = torch.from_numpy(ref_root["sequences"][test_idx]).unsqueeze(0).to(device)

            normalized_preds = {}
            for model_ct, model in models.items():
                with torch.no_grad():
                    out = model(seq)
                raw_count = torch.expm1(out["count"]).item()
                # Normalize by model's mean count (removes "high everywhere" confound)
                normalized_preds[model_ct] = raw_count / (model_mean_counts[model_ct] + 1e-8)

            # The correct model should have highest normalized prediction
            correct_norm = normalized_preds.get(ct, 0)
            other_norms = [v for k, v in normalized_preds.items() if k != ct]
            mean_other = np.mean(other_norms) if other_norms else 1e-8

            enrichment = correct_norm / (mean_other + 1e-8)
            results[ct]["correct_enrichment"].append(enrichment)

    # Summary
    print("\n  Background-normalized specificity (correct model enrichment over others):")
    print("  (>1.0 means the correct model outperforms, controlling for overall signal level)")

    overall_enrichments = []
    for ct in CELL_TYPES:
        if results[ct]["n_peaks"] == 0:
            print(f"    {ct:>15s}: no specific peaks")
            continue

        enrichments = results[ct]["correct_enrichment"]
        mean_enr = float(np.mean(enrichments))
        median_enr = float(np.median(enrichments))
        frac_correct = float(np.mean([e > 1.0 for e in enrichments]))

        status = "GOOD" if mean_enr > 1.5 else "OK" if mean_enr > 1.0 else "FAIL"
        print(f"    {ct:>15s}: mean={mean_enr:.2f}x, median={median_enr:.2f}x, "
              f"frac_correct={frac_correct:.0%} (n={results[ct]['n_peaks']}) [{status}]")

        overall_enrichments.extend(enrichments)
        results[ct]["mean_enrichment"] = mean_enr
        results[ct]["median_enrichment"] = median_enr
        results[ct]["frac_correct"] = frac_correct

    if overall_enrichments:
        overall_mean = float(np.mean(overall_enrichments))
        overall_frac = float(np.mean([e > 1.0 for e in overall_enrichments]))
        print(f"\n  Overall: mean enrichment={overall_mean:.2f}x, fraction correct={overall_frac:.0%}")
        results["overall"] = {"mean_enrichment": overall_mean, "frac_correct": overall_frac}

    return results


# ==============================================================================
# Test 7: In-Silico Motif Perturbation
# ==============================================================================

# Known TF motifs for each cell type (consensus PWM cores)
CELL_TYPE_MOTIFS = {
    "Cardiomyocyte": [
        {"name": "GATA4", "seq": "AGATAAGG", "description": "GATA family (cardiac TF)"},
        {"name": "NKX2-5", "seq": "CACTTAA", "description": "NK homeodomain (cardiac)"},
        {"name": "MEF2C", "seq": "CTAAAAATAG", "description": "MADS box (cardiac muscle)"},
    ],
    "Coronary_EC": [
        {"name": "ETS1", "seq": "ACGGAAGT", "description": "ETS family (endothelial)"},
        {"name": "ERG", "seq": "ACGGAAG", "description": "ETS family (vascular)"},
    ],
    "Fibroblast": [
        {"name": "AP-1", "seq": "TGAGTCA", "description": "AP-1/JUN (fibroblast)"},
        {"name": "TEAD", "seq": "CATTCCA", "description": "TEAD/Hippo (mesenchymal)"},
    ],
    "Macrophage": [
        {"name": "PU.1", "seq": "AAAGAGGAAGTG", "description": "ETS/SPI1 (myeloid)"},
        {"name": "CEBP", "seq": "TTGCGCAA", "description": "C/EBP (macrophage)"},
    ],
    "Pericytes": [
        {"name": "FOXF1", "seq": "TGTTTAC", "description": "FOX family (pericyte/mesenchymal)"},
    ],
}


@torch.no_grad()
def test_motif_perturbation(
    model_dir: str,
    genome_path: str,
    device: str,
) -> Dict[str, Dict]:
    """In-silico motif injection test.

    For each cell type's known TF motif:
    1. Take a neutral background sequence (from a non-peak region)
    2. Insert the motif at the center
    3. Check if the correct model's prediction increases more than other models'

    This is the strongest test of whether the model has learned motif grammar
    rather than just background signal levels.
    """
    print("\n" + "=" * 60)
    print("TEST 7: In-Silico Motif Perturbation")
    print("=" * 60)
    print("  Insert known TF motifs into neutral sequence.")
    print("  Correct model should show larger increase than others.")

    import pyfaidx

    if not Path(genome_path).exists():
        print(f"  SKIPPED: Genome not found at {genome_path}")
        return {}

    genome = pyfaidx.Fasta(genome_path)

    # Load all models
    models = {}
    for ct in CELL_TYPES:
        ckpt = Path(model_dir) / ct / "best.ckpt"
        if ckpt.exists():
            models[ct] = load_single_ct_model(str(ckpt), device)

    if len(models) < 2:
        print("  SKIPPED: Need at least 2 models")
        return {}

    # Get a background sequence (intergenic region on chr10)
    bg_center = 60000000
    half = 2114 // 2
    bg_seq_str = str(genome["chr10"][bg_center - half:bg_center + half]).upper()
    bg_seq = _one_hot_encode(bg_seq_str)

    # Get baseline predictions from all models on background
    bg_tensor = bg_seq.unsqueeze(0).to(device)
    baseline_counts = {}
    for ct, model in models.items():
        out = model(bg_tensor)
        baseline_counts[ct] = torch.expm1(out["count"]).item()

    results = {}

    for target_ct, motifs in CELL_TYPE_MOTIFS.items():
        if target_ct not in models:
            continue

        print(f"\n  {target_ct} motifs:")

        ct_results = []

        for motif_info in motifs:
            motif_name = motif_info["name"]
            motif_seq = motif_info["seq"]

            # Insert motif at center of background sequence
            center = len(bg_seq_str) // 2
            perturbed_str = (bg_seq_str[:center] + motif_seq +
                             bg_seq_str[center + len(motif_seq):])
            perturbed = _one_hot_encode(perturbed_str).unsqueeze(0).to(device)

            # Get predictions with motif inserted
            perturbed_counts = {}
            for ct, model in models.items():
                out = model(perturbed)
                perturbed_counts[ct] = torch.expm1(out["count"]).item()

            # Compute fold-change for each model
            fold_changes = {}
            for ct in models:
                fc = perturbed_counts[ct] / (baseline_counts[ct] + 1e-8)
                fold_changes[ct] = fc

            # The target cell type's model should have highest fold-change
            target_fc = fold_changes.get(target_ct, 1.0)
            other_fcs = [v for k, v in fold_changes.items() if k != target_ct]
            mean_other_fc = np.mean(other_fcs) if other_fcs else 1.0

            specificity_ratio = target_fc / (mean_other_fc + 1e-8)

            best_ct = max(fold_changes, key=fold_changes.get)
            is_correct = best_ct == target_ct
            status = "CORRECT" if is_correct else "WRONG"

            print(f"    {motif_name} ({motif_seq}):")
            for ct in CELL_TYPES:
                if ct not in fold_changes:
                    continue
                marker = "*" if ct == target_ct else " "
                top = " <-- HIGHEST FC" if ct == best_ct else ""
                print(f"      {marker} {ct:>15s}: {fold_changes[ct]:.3f}x{top}")
            print(f"      Specificity ratio: {specificity_ratio:.2f}x [{status}]")

            ct_results.append({
                "motif": motif_name,
                "target_fc": float(target_fc),
                "mean_other_fc": float(mean_other_fc),
                "specificity_ratio": float(specificity_ratio),
                "correct": is_correct,
            })

        results[target_ct] = ct_results

    # Summary
    all_correct = []
    all_ratios = []
    for ct, motif_results in results.items():
        for r in motif_results:
            all_correct.append(r["correct"])
            all_ratios.append(r["specificity_ratio"])

    if all_correct:
        accuracy = np.mean(all_correct)
        mean_ratio = np.mean(all_ratios)
        print(f"\n  Motif perturbation accuracy: {sum(all_correct)}/{len(all_correct)} ({accuracy:.0%})")
        print(f"  Mean specificity ratio: {mean_ratio:.2f}x")

        if accuracy >= 0.7 and mean_ratio > 1.5:
            print("  GOOD: Models respond correctly to cell-type-specific motifs")
        elif accuracy >= 0.5:
            print("  MODERATE: Some motif specificity learned")
        else:
            print("  WARNING: Models may not have learned cell-type-specific motif grammar")

        results["summary"] = {
            "accuracy": float(accuracy),
            "mean_specificity_ratio": float(mean_ratio),
        }

    return results


# ==============================================================================
# Test 8: Multi-Cell Motif Perturbation (Stage 3)
# ==============================================================================

@torch.no_grad()
def test_motif_perturbation_stage3(
    checkpoint: str,
    genome_path: str,
    device: str,
) -> Dict[str, Dict]:
    """In-silico motif injection for multi-cell model.

    Same concept as Test 7, but tests whether the multi-cell model's
    cell-type-conditioned output responds correctly to motif insertion.
    """
    print("\n" + "=" * 60)
    print("TEST 8: In-Silico Motif Perturbation (Stage 3)")
    print("=" * 60)

    import pyfaidx

    if not Path(genome_path).exists():
        print(f"  SKIPPED: Genome not found at {genome_path}")
        return {}

    model = load_multi_cell_model(checkpoint, device)
    genome = pyfaidx.Fasta(genome_path)

    # Background sequence
    bg_center = 60000000
    half = 2114 // 2
    bg_seq_str = str(genome["chr10"][bg_center - half:bg_center + half]).upper()
    bg_tensor = _one_hot_encode(bg_seq_str).unsqueeze(0).to(device)

    out = model(bg_tensor)
    baseline_counts = torch.expm1(out["count"]).squeeze(0).cpu().numpy()  # (n_ct,)

    results = {}

    for target_ct, motifs in CELL_TYPE_MOTIFS.items():
        target_idx = CELL_TYPES.index(target_ct)
        print(f"\n  {target_ct} motifs:")

        ct_results = []

        for motif_info in motifs:
            motif_name = motif_info["name"]
            motif_seq = motif_info["seq"]

            center = len(bg_seq_str) // 2
            perturbed_str = (bg_seq_str[:center] + motif_seq +
                             bg_seq_str[center + len(motif_seq):])
            perturbed = _one_hot_encode(perturbed_str).unsqueeze(0).to(device)

            out = model(perturbed)
            perturbed_counts = torch.expm1(out["count"]).squeeze(0).cpu().numpy()

            fold_changes = perturbed_counts / (baseline_counts + 1e-8)

            best_idx = np.argmax(fold_changes)
            is_correct = best_idx == target_idx
            status = "CORRECT" if is_correct else "WRONG"

            target_fc = fold_changes[target_idx]
            other_fcs = [fold_changes[i] for i in range(len(CELL_TYPES)) if i != target_idx]
            specificity = target_fc / (np.mean(other_fcs) + 1e-8)

            print(f"    {motif_name} ({motif_seq}):")
            for i, ct in enumerate(CELL_TYPES):
                marker = "*" if i == target_idx else " "
                top = " <-- HIGHEST FC" if i == best_idx else ""
                print(f"      {marker} {ct:>15s}: {fold_changes[i]:.3f}x{top}")
            print(f"      Specificity ratio: {specificity:.2f}x [{status}]")

            ct_results.append({
                "motif": motif_name,
                "target_fc": float(target_fc),
                "specificity_ratio": float(specificity),
                "correct": is_correct,
            })

        results[target_ct] = ct_results

    # Summary
    all_correct = [r["correct"] for motifs in results.values() for r in motifs]
    all_ratios = [r["specificity_ratio"] for motifs in results.values() for r in motifs]

    if all_correct:
        accuracy = np.mean(all_correct)
        mean_ratio = np.mean(all_ratios)
        print(f"\n  Motif perturbation accuracy: {sum(all_correct)}/{len(all_correct)} ({accuracy:.0%})")
        print(f"  Mean specificity ratio: {mean_ratio:.2f}x")
        results["summary"] = {"accuracy": float(accuracy), "mean_specificity_ratio": float(mean_ratio)}

    return results


# ==============================================================================
# Plotting
# ==============================================================================

def generate_correlation_boxplots(
    profile_corrs: Dict[str, np.ndarray],
    output_dir: str,
    stage_label: str,
):
    """Generate correlation boxplots by cell type."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    data = [profile_corrs[ct] for ct in CELL_TYPES if ct in profile_corrs]
    labels = [ct for ct in CELL_TYPES if ct in profile_corrs]

    bp = ax.boxplot(data, labels=labels, patch_artist=True)

    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_ylabel("Profile Pearson r")
    ax.set_title(f"Test Set Profile Correlation ({stage_label})")
    ax.axhline(0.5, color="green", linestyle="--", alpha=0.5, label="Good (0.5)")
    ax.axhline(0.3, color="orange", linestyle="--", alpha=0.5, label="OK (0.3)")
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    filename = f"{output_dir}/correlation_boxplot_{stage_label.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


def generate_count_scatter(
    all_results: Dict[str, Dict[str, np.ndarray]],
    output_dir: str,
    stage_label: str,
):
    """Scatter plot of predicted vs actual counts per cell type."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_ct = len(all_results)
    ncols = min(3, n_ct)
    nrows = (n_ct + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    if n_ct == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, (ct, res) in enumerate(all_results.items()):
        ax = axes[i]
        pred = res["pred_counts"]
        target = res["target_counts"]

        ax.scatter(target, pred, alpha=0.3, s=8)

        if len(pred) > 2 and pred.std() > 1e-8 and target.std() > 1e-8:
            r, _ = stats.pearsonr(target, pred)
            z = np.polyfit(target, pred, 1)
            p_line = np.poly1d(z)
            x_range = np.linspace(target.min(), target.max(), 100)
            ax.plot(x_range, p_line(x_range), "r--", alpha=0.7)
            ax.set_title(f"{ct}\nr={r:.3f}")
        else:
            ax.set_title(ct)

        ax.set_xlabel("Actual count")
        ax.set_ylabel("Predicted count")

    # Hide extra axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Predicted vs Actual Counts ({stage_label})")
    plt.tight_layout()

    filename = f"{output_dir}/count_scatter_{stage_label.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


def generate_profile_plots(
    all_results: Dict[str, Dict[str, np.ndarray]],
    output_dir: str,
    stage_label: str,
    n_peaks: int = 3,
):
    """Profile plots: predicted vs actual for top peaks per cell type."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for ct, res in all_results.items():
        pred_profiles = res["pred_profiles"]
        target_profiles = res["target_profiles"]
        pred_counts = res["pred_counts"]

        # Pick top peaks by predicted count (most interesting)
        top_idx = np.argsort(pred_counts)[-n_peaks:][::-1]

        fig, axes = plt.subplots(n_peaks, 1, figsize=(12, 3 * n_peaks), sharex=True)
        if n_peaks == 1:
            axes = [axes]

        for i, idx in enumerate(top_idx):
            ax = axes[i]
            x = np.arange(len(target_profiles[idx]))

            # Normalize target profile to probability for comparison
            t = target_profiles[idx]
            t_prob = t / (t.sum() + 1e-8)

            ax.fill_between(x, t_prob, alpha=0.4, color="orange", label="Actual")
            ax.fill_between(x, pred_profiles[idx], alpha=0.4, color="blue", label="Predicted")
            ax.plot(x, t_prob, color="orange", alpha=0.7, linewidth=0.5)
            ax.plot(x, pred_profiles[idx], color="blue", alpha=0.7, linewidth=0.5)

            r_val = compute_per_peak_profile_r(
                pred_profiles[idx:idx+1], t_prob[np.newaxis, :]
            )[0]
            ax.set_ylabel(f"Peak {idx}\nr={r_val:.3f}")

            if i == 0:
                ax.legend(loc="upper right")

        axes[-1].set_xlabel("Position (bp)")
        fig.suptitle(f"{ct} - Profile Comparison ({stage_label})", fontweight="bold")
        plt.tight_layout()

        filename = f"{output_dir}/profile_{ct}_{stage_label.lower().replace(' ', '_')}.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {filename}")


def generate_stage_comparison_plot(
    stage2_results: Dict[str, Dict[str, float]],
    stage3_results: Dict[str, Dict[str, float]],
    output_dir: str,
):
    """Bar chart comparing Stage 2 vs Stage 3 metrics per cell type."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cts = [ct for ct in CELL_TYPES if ct in stage2_results and ct in stage3_results]
    if not cts:
        print("  No overlapping cell types for comparison")
        return

    x = np.arange(len(cts))
    width = 0.35

    # Profile r comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax_idx, (metric, label) in enumerate([
        ("profile_r_mean", "Profile Pearson r"),
        ("count_r", "Count Pearson r"),
        ("jsd_mean", "Profile JSD"),
    ]):
        ax = axes[ax_idx]
        s2_vals = [stage2_results[ct][metric] for ct in cts]
        s3_vals = [stage3_results[ct][metric] for ct in cts]

        ax.bar(x - width/2, s2_vals, width, label="Stage 2 (per-CT)", color="steelblue")
        ax.bar(x + width/2, s3_vals, width, label="Stage 3 (multi-cell)", color="coral")
        ax.set_xlabel("Cell Type")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_xticks(x)
        ax.set_xticklabels([ct[:6] for ct in cts], rotation=30, ha="right")
        ax.legend()

    plt.tight_layout()
    filename = f"{output_dir}/stage_comparison.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


def generate_collapse_heatmap(
    corr_matrix: np.ndarray,
    output_dir: str,
):
    """Heatmap of cross-cell-type correlation matrix."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(corr_matrix, cmap="RdYlBu_r", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(CELL_TYPES)))
    ax.set_yticks(np.arange(len(CELL_TYPES)))
    ax.set_xticklabels([ct[:8] for ct in CELL_TYPES], rotation=45, ha="right")
    ax.set_yticklabels([ct[:8] for ct in CELL_TYPES])

    # Annotate values
    for i in range(len(CELL_TYPES)):
        for j in range(len(CELL_TYPES)):
            ax.text(j, i, f"{corr_matrix[i, j]:.2f}", ha="center", va="center",
                    color="white" if corr_matrix[i, j] > 0.7 else "black", fontsize=10)

    ax.set_title("Cross-Cell-Type Count Correlation (Stage 3)\nLower off-diagonal = better differentiation")
    plt.tight_layout()

    filename = f"{output_dir}/collapse_heatmap.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


# ==============================================================================
# Helpers
# ==============================================================================

def _one_hot_encode(seq: str) -> torch.Tensor:
    """One-hot encode DNA sequence (4, length)."""
    base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    one_hot = torch.zeros(4, len(seq))
    for i, base in enumerate(seq):
        if base in base_to_idx:
            one_hot[base_to_idx[base], i] = 1.0
    return one_hot


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validate cellTypeAtac models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--stage", choices=["2", "3", "both"], default="both",
                        help="Which stage(s) to validate (default: both)")
    parser.add_argument("--model-dir", type=str, default="results/chrombpnet",
                        help="Stage 2 per-cell-type model directory")
    parser.add_argument("--checkpoint", type=str, default="results/multi_cell/best.ckpt",
                        help="Stage 3 multi-cell model checkpoint")
    parser.add_argument("--data-dir", type=str, default="data/training",
                        help="Training data directory with per-CT zarrs")
    parser.add_argument("--genome", type=str, default="data/genome/mm10.fa",
                        help="Path to reference genome FASTA")
    parser.add_argument("--annotations", type=str, default="data/peak_annotations.csv",
                        help="Peak annotations from step 02")
    parser.add_argument("--output-dir", type=str, default="results/validation",
                        help="Directory for output plots and metrics")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--max-samples", type=int, default=500,
                        help="Max test samples for correlation analysis")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for inference")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("cellTypeAtac Model Validation")
    print("=" * 60)
    print(f"Stage: {args.stage}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output_dir}")

    all_metrics = {}

    # ---- Stage 2 Validation ----
    if args.stage in ("2", "both"):
        print("\n\n" + "#" * 60)
        print("# STAGE 2: Per-Cell-Type Models")
        print("#" * 60)

        # Test 1: Basic inference (use first available model)
        for ct in CELL_TYPES:
            ckpt = Path(args.model_dir) / ct / "best.ckpt"
            if ckpt.exists():
                model = load_single_ct_model(str(ckpt), args.device)
                basic_ok = test_basic_inference(model, args.device, is_multi_cell=False)
                all_metrics["stage2_basic_inference"] = basic_ok
                break

        # Test 2: Correlation
        s2_corr_results, s2_profile_corrs = test_correlation_stage2(
            args.model_dir, args.data_dir, args.device,
            args.max_samples, args.batch_size,
        )
        all_metrics["stage2_correlation"] = s2_corr_results

        # Test 3: Marker specificity
        if Path(args.genome).exists():
            s2_markers = test_marker_specificity_stage2(
                args.model_dir, args.genome, args.device,
            )
            all_metrics["stage2_markers"] = {
                k: {kk: vv for kk, vv in v.items() if kk != "counts"}
                for k, v in s2_markers.items()
            }
        else:
            print(f"\n  Skipping marker test: genome not found at {args.genome}")

        # Test 5: Variability
        for ct in CELL_TYPES:
            ckpt = Path(args.model_dir) / ct / "best.ckpt"
            if ckpt.exists():
                model = load_single_ct_model(str(ckpt), args.device)
                var_ok = test_prediction_variability(model, args.device, is_multi_cell=False)
                all_metrics["stage2_variability"] = var_ok
                break

        # Test 6: Specificity-only evaluation (tau-specific peaks)
        if Path(args.annotations).exists():
            s2_specific = test_specific_peaks_only(
                args.model_dir, args.data_dir, args.annotations,
                args.device, args.max_samples, args.batch_size,
            )
            all_metrics["stage2_specific_peaks"] = {
                k: {kk: vv for kk, vv in v.items() if kk != "correct_enrichment"}
                for k, v in s2_specific.items() if isinstance(v, dict)
            }

        # Test 7: In-silico motif perturbation
        if Path(args.genome).exists():
            s2_motifs = test_motif_perturbation(
                args.model_dir, args.genome, args.device,
            )
            all_metrics["stage2_motif_perturbation"] = s2_motifs

        # Plots
        print("\n" + "=" * 60)
        print("GENERATING STAGE 2 PLOTS")
        print("=" * 60)
        try:
            if s2_profile_corrs:
                generate_correlation_boxplots(s2_profile_corrs, str(out_dir), "Stage 2")

            # Collect scatter data
            scatter_data = {}
            for ct in CELL_TYPES:
                ckpt = Path(args.model_dir) / ct / "best.ckpt"
                zarr_path = str(Path(args.data_dir) / ct / "test.zarr")
                if ckpt.exists() and Path(zarr_path).exists():
                    model = load_single_ct_model(str(ckpt), args.device)
                    ds = ATACDataset(zarr_path, split="test")
                    res = predict_single_ct(model, ds, args.device, max_samples=200, batch_size=args.batch_size)
                    scatter_data[ct] = res

            if scatter_data:
                generate_count_scatter(scatter_data, str(out_dir), "Stage 2")
                generate_profile_plots(scatter_data, str(out_dir), "Stage 2")
        except ImportError:
            print("  Skipping plots (matplotlib not available)")

    # ---- Stage 3 Validation ----
    if args.stage in ("3", "both"):
        print("\n\n" + "#" * 60)
        print("# STAGE 3: Multi-Cell Model")
        print("#" * 60)

        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            print(f"  WARNING: Stage 3 checkpoint not found: {args.checkpoint}")
            print("  Skipping Stage 3 validation.")
        else:
            # Test 1: Basic inference
            model = load_multi_cell_model(args.checkpoint, args.device)
            basic_ok = test_basic_inference(model, args.device, is_multi_cell=True)
            all_metrics["stage3_basic_inference"] = basic_ok

            # Test 2: Correlation
            s3_corr_results, s3_profile_corrs = test_correlation_stage3(
                args.checkpoint, args.data_dir, args.device,
                args.max_samples, args.batch_size,
            )
            all_metrics["stage3_correlation"] = s3_corr_results

            # Test 3: Marker specificity
            if Path(args.genome).exists():
                s3_markers = test_marker_specificity_stage3(
                    args.checkpoint, args.genome, args.device,
                )
                all_metrics["stage3_markers"] = {
                    k: {kk: vv for kk, vv in v.items() if kk != "counts"}
                    for k, v in s3_markers.items()
                }

            # Test 4: Collapse detection
            collapse_results = test_collapse_detection(
                args.checkpoint, args.data_dir, args.device,
                max_samples=args.max_samples, batch_size=args.batch_size,
            )
            all_metrics["stage3_collapse"] = {
                k: v for k, v in collapse_results.items()
                if k != "corr_matrix"
            }

            # Test 5: Variability
            var_ok = test_prediction_variability(model, args.device, is_multi_cell=True)
            all_metrics["stage3_variability"] = var_ok

            # Test 8: Multi-cell motif perturbation
            if Path(args.genome).exists():
                s3_motifs = test_motif_perturbation_stage3(
                    args.checkpoint, args.genome, args.device,
                )
                all_metrics["stage3_motif_perturbation"] = s3_motifs

            # Plots
            print("\n" + "=" * 60)
            print("GENERATING STAGE 3 PLOTS")
            print("=" * 60)
            try:
                if s3_profile_corrs:
                    generate_correlation_boxplots(s3_profile_corrs, str(out_dir), "Stage 3")

                # Scatter & profile plots
                zarr_paths = [str(Path(args.data_dir) / ct / "test.zarr") for ct in CELL_TYPES]
                if all(Path(p).exists() for p in zarr_paths):
                    ds = MultiCellATACDataset(zarr_paths, split="test")
                    mc_results = predict_multi_cell(
                        model, ds, args.device, max_samples=200, batch_size=args.batch_size,
                    )
                    generate_count_scatter(mc_results, str(out_dir), "Stage 3")
                    generate_profile_plots(mc_results, str(out_dir), "Stage 3")

                # Collapse heatmap
                if "corr_matrix" in collapse_results:
                    generate_collapse_heatmap(
                        np.array(collapse_results["corr_matrix"]), str(out_dir)
                    )
            except ImportError:
                print("  Skipping plots (matplotlib not available)")

    # ---- Stage Comparison ----
    if args.stage == "both":
        s2_corr = all_metrics.get("stage2_correlation", {})
        s3_corr = all_metrics.get("stage3_correlation", {})

        if s2_corr and s3_corr:
            print("\n\n" + "#" * 60)
            print("# STAGE COMPARISON: Per-Cell-Type vs Multi-Cell")
            print("#" * 60)

            print(f"\n  {'Cell Type':>15s}  {'S2 prof_r':>10s}  {'S3 prof_r':>10s}  {'S2 cnt_r':>10s}  {'S3 cnt_r':>10s}  {'S2 JSD':>8s}  {'S3 JSD':>8s}")
            print("  " + "-" * 80)

            for ct in CELL_TYPES:
                if ct in s2_corr and ct in s3_corr:
                    s2 = s2_corr[ct]
                    s3 = s3_corr[ct]
                    print(f"  {ct:>15s}  {s2['profile_r_mean']:10.4f}  {s3['profile_r_mean']:10.4f}"
                          f"  {s2['count_r']:10.4f}  {s3['count_r']:10.4f}"
                          f"  {s2['jsd_mean']:8.4f}  {s3['jsd_mean']:8.4f}")

            s2_mean = np.mean([s2_corr[ct]["profile_r_mean"] for ct in CELL_TYPES if ct in s2_corr])
            s3_mean = np.mean([s3_corr[ct]["profile_r_mean"] for ct in CELL_TYPES if ct in s3_corr])
            print("  " + "-" * 80)
            print(f"  {'Mean':>15s}  {s2_mean:10.4f}  {s3_mean:10.4f}")

            if s3_mean > s2_mean:
                print("\n  Stage 3 outperforms Stage 2 on average profile correlation.")
            elif s2_mean > s3_mean:
                print("\n  Stage 2 outperforms Stage 3 on average profile correlation.")
                print("  (This is expected for per-CT models vs multi-cell model)")
            else:
                print("\n  Stage 2 and Stage 3 perform similarly.")

            try:
                generate_stage_comparison_plot(s2_corr, s3_corr, str(out_dir))
            except ImportError:
                pass

    # ---- Summary ----
    print("\n\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    # Print key metrics
    for key, val in all_metrics.items():
        if isinstance(val, bool):
            status = "PASS" if val else "FAIL"
            print(f"  {key}: {status}")
        elif isinstance(val, dict) and key.endswith("_correlation"):
            stage = "Stage 2" if "stage2" in key else "Stage 3"
            mean_r = np.mean([v["profile_r_mean"] for v in val.values() if isinstance(v, dict)])
            print(f"  {stage} mean profile r: {mean_r:.4f}")
        elif isinstance(val, dict) and key.endswith("_markers"):
            n_correct = sum(1 for v in val.values() if isinstance(v, dict) and v.get("correct"))
            n_total = sum(1 for v in val.values() if isinstance(v, dict))
            print(f"  {key}: {n_correct}/{n_total} correct")
        elif isinstance(val, dict) and key.endswith("_collapse"):
            print(f"  Stage 3 cross-CT corr: {val.get('mean_cross_ct_count_corr', 'N/A'):.3f}")
            print(f"  Stage 3 signal CV: {val.get('signal_cv', 'N/A'):.3f}")

    # Save metrics
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\n  Metrics saved to: {metrics_path}")
    print(f"  Plots saved to: {out_dir}")

    # Print summary line for log parsing (used by compare_validations.py)
    if "stage2_correlation" in all_metrics:
        s2_mean = np.mean([v["profile_r_mean"] for v in all_metrics["stage2_correlation"].values()])
        print(f"\nStage 2 ATAC correlation: {s2_mean:.4f}")
    if "stage3_correlation" in all_metrics:
        s3_mean = np.mean([v["profile_r_mean"] for v in all_metrics["stage3_correlation"].values()])
        print(f"Stage 3 ATAC correlation: {s3_mean:.4f}")
    if "stage3_collapse" in all_metrics:
        print(f"Mean cross-CT correlation: {all_metrics['stage3_collapse'].get('mean_cross_ct_count_corr', 'N/A')}")
        print(f"Signal CV across CTs: {all_metrics['stage3_collapse'].get('signal_cv', 'N/A')}")
    if "stage2_markers" in all_metrics:
        n_c = sum(1 for v in all_metrics["stage2_markers"].values() if isinstance(v, dict) and v.get("correct"))
        n_t = sum(1 for v in all_metrics["stage2_markers"].values() if isinstance(v, dict))
        print(f"Stage 2 marker accuracy: {n_c}/{n_t}")
    if "stage3_markers" in all_metrics:
        n_c = sum(1 for v in all_metrics["stage3_markers"].values() if isinstance(v, dict) and v.get("correct"))
        n_t = sum(1 for v in all_metrics["stage3_markers"].values() if isinstance(v, dict))
        print(f"Stage 3 marker accuracy: {n_c}/{n_t}")
    if "stage2_specific_peaks" in all_metrics:
        overall = all_metrics["stage2_specific_peaks"].get("overall", {})
        if overall:
            print(f"Stage 2 specific-peak enrichment: {overall.get('mean_enrichment', 'N/A'):.2f}x (frac correct: {overall.get('frac_correct', 'N/A'):.0%})")
    if "stage2_motif_perturbation" in all_metrics:
        summary = all_metrics["stage2_motif_perturbation"].get("summary", {})
        if summary:
            print(f"Stage 2 motif perturbation: accuracy={summary.get('accuracy', 'N/A'):.0%}, specificity={summary.get('mean_specificity_ratio', 'N/A'):.2f}x")
    if "stage3_motif_perturbation" in all_metrics:
        summary = all_metrics["stage3_motif_perturbation"].get("summary", {})
        if summary:
            print(f"Stage 3 motif perturbation: accuracy={summary.get('accuracy', 'N/A'):.0%}, specificity={summary.get('mean_specificity_ratio', 'N/A'):.2f}x")


if __name__ == "__main__":
    main()
