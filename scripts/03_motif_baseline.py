#!/usr/bin/env python
"""TF motif baseline for cell-type specificity prediction.

Scans peak sequences for known TF motifs (JASPAR) and trains classifiers
to predict cell-type specificity from motif scores alone.

This establishes whether DNA sequence is predictive of cell-type specificity
before investing in DNN training.

Usage:
    python scripts/03_motif_baseline.py \
        --annotations data/peak_annotations.csv \
        --genome data/genome/mm10.fa \
        --output-dir results/motif_baseline
"""

import argparse
import json
import multiprocessing as mp
from functools import partial
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings("ignore")

import pyfaidx
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from Bio import motifs
    from Bio.Seq import Seq
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False

try:
    from pyjaspar import jaspardb
    HAS_PYJASPAR = True
except ImportError:
    HAS_PYJASPAR = False


CELL_TYPES = ["Cardiomyocyte", "Coronary_EC", "Fibroblast", "Macrophage", "Pericytes"]
WINDOW_SIZE = 2000  # 2kb centered on peak


def load_jaspar_motifs(tax_group: str = "vertebrates") -> list:
    """Load JASPAR CORE motifs.

    Returns:
        List of (name, pwm_matrix) tuples.
    """
    if not HAS_PYJASPAR:
        raise ImportError("pyjaspar required. Install: pip install pyjaspar")

    jdb = jaspardb()
    jdb.fetch_motif_by_id("MA0001.1")  # Warm up

    motif_list = jdb.fetch_motifs(
        collection="CORE",
        tax_group=[tax_group],
    )

    result = []
    for m in motif_list:
        try:
            name = f"{m.name}_{m.matrix_id}"
            pwm = m.pwm
            # Convert to numpy: rows = A, C, G, T; cols = positions
            mat = np.array([pwm.get(b, [0]*len(pwm["A"])) for b in "ACGT"])
            result.append((name, mat))
        except Exception:
            continue

    print(f"Loaded {len(result)} JASPAR motifs")
    return result


def precompute_log_odds(motif_list, pseudocount=0.01):
    """Pre-compute log-odds matrices for all motifs (once, not per-sequence).

    Returns:
        List of (name, log_odds) tuples where log_odds is (4, motif_len) float32.
    """
    result = []
    for name, pwm in motif_list:
        pwm_norm = pwm + pseudocount
        pwm_norm = pwm_norm / pwm_norm.sum(axis=0, keepdims=True)
        log_odds = np.log2(pwm_norm / 0.25).astype(np.float32)
        result.append((name, log_odds))
    return result


def score_sequence_motifs(
    sequence: str,
    motif_log_odds: list,
) -> np.ndarray:
    """Score a DNA sequence against all motifs using vectorized log-likelihood ratio.

    Uses numpy sliding_window_view to scan all positions at once instead of
    Python for-loops, giving ~50-100x speedup.

    Args:
        sequence: DNA sequence string.
        motif_log_odds: List of (name, log_odds_matrix) from precompute_log_odds.

    Returns:
        Array of max scores, shape (n_motifs,).
    """
    seq = sequence.upper()
    n_motifs = len(motif_log_odds)
    scores = np.zeros(n_motifs, dtype=np.float32)

    # Encode sequence to indices: A=0, C=1, G=2, T=3, other=-1
    base_map = np.full(128, -1, dtype=np.int8)
    for i, b in enumerate("ACGT"):
        base_map[ord(b)] = i
    seq_idx = base_map[np.frombuffer(seq.encode(), dtype=np.uint8)]

    # Reverse complement: A(0)<->T(3), C(1)<->G(2)
    rc_table = np.array([3, 2, 1, 0, -1], dtype=np.int8)
    seq_idx_safe = np.where(seq_idx >= 0, seq_idx, 4).astype(np.int8)
    rc_idx = rc_table[seq_idx_safe[::-1]]

    for m_idx, (name, log_odds) in enumerate(motif_log_odds):
        motif_len = log_odds.shape[1]
        if motif_len > len(seq):
            continue

        max_score = 0.0  # clip to >= 0
        pos_range = np.arange(motif_len)

        for strand_idx in (seq_idx, rc_idx):
            # Sliding windows: (n_positions, motif_len) — zero-copy view
            windows = np.lib.stride_tricks.sliding_window_view(strand_idx, motif_len)
            # Filter valid windows (no N's / ambiguous bases)
            valid_mask = np.all(windows >= 0, axis=1)
            if not valid_mask.any():
                continue
            valid_windows = windows[valid_mask]
            # Vectorized score: sum log_odds[base, position] across motif length
            window_scores = np.sum(log_odds[valid_windows, pos_range], axis=1)
            strand_max = float(np.max(window_scores))
            if strand_max > max_score:
                max_score = strand_max

        scores[m_idx] = max_score

    return scores


# Module-level globals for worker processes (avoids pickling motifs per call)
_worker_genome = None
_worker_motifs = None
_worker_window_size = None


def _init_worker(genome_path, motif_log_odds, window_size):
    """Initialize worker process: open genome and store motifs once."""
    global _worker_genome, _worker_motifs, _worker_window_size
    _worker_genome = pyfaidx.Fasta(genome_path)
    _worker_motifs = motif_log_odds
    _worker_window_size = window_size


def _score_peak_worker(args):
    """Worker function for parallel motif scoring.

    Args:
        args: Tuple of (chrom, center).

    Returns:
        Array of motif scores for one peak.
    """
    chrom, center = args
    start = max(0, center - _worker_window_size // 2)
    end = start + _worker_window_size
    try:
        seq = str(_worker_genome[chrom][start:end])
    except (KeyError, ValueError):
        return np.zeros(len(_worker_motifs), dtype=np.float32)
    return score_sequence_motifs(seq, _worker_motifs)


def build_feature_matrix(
    annotations: pd.DataFrame,
    genome_path: str,
    motif_log_odds: list,
    window_size: int = WINDOW_SIZE,
    n_workers: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build motif score feature matrix for specific peaks.

    Args:
        annotations: Peak annotations with category and specific_celltype.
        genome_path: Path to reference genome FASTA.
        motif_log_odds: Pre-computed log-odds from precompute_log_odds().
        window_size: Sequence window centered on peak.
        n_workers: Number of parallel workers (default: 1).

    Returns:
        X: (n_peaks, n_motifs) feature matrix.
        y: (n_peaks,) cell type labels (encoded).
        peak_ids: Array of peak IDs.
        chroms: Array of chromosome names (for grouped CV).
    """
    specific = annotations[annotations["category"] == "specific"].reset_index(drop=True)
    n_peaks = len(specific)
    n_motifs = len(motif_log_odds)

    print(f"Building feature matrix: {n_peaks} specific peaks × {n_motifs} motifs")
    print(f"  Using {n_workers} workers")

    y = np.array([CELL_TYPES.index(ct) for ct in specific["specific_celltype"]])
    peak_ids = specific["peak_id"].values
    chroms = specific["chrom"].values

    # Work items: only peak-specific data (motifs shared via worker init)
    work_items = []
    for i in range(n_peaks):
        row = specific.iloc[i]
        center = (row["start"] + row["end"]) // 2
        work_items.append((row["chrom"], center))

    if n_workers > 1:
        with mp.Pool(
            n_workers,
            initializer=_init_worker,
            initargs=(genome_path, motif_log_odds, window_size),
        ) as pool:
            results = []
            for i, result in enumerate(pool.imap(
                _score_peak_worker, work_items, chunksize=200,
            )):
                results.append(result)
                if (i + 1) % 5000 == 0:
                    print(f"  Scored {i+1}/{n_peaks} peaks...")
            X = np.array(results, dtype=np.float32)
    else:
        genome = pyfaidx.Fasta(genome_path)
        X = np.zeros((n_peaks, n_motifs), dtype=np.float32)
        for i in range(n_peaks):
            row = specific.iloc[i]
            chrom = row["chrom"]
            center = (row["start"] + row["end"]) // 2
            start = max(0, center - window_size // 2)
            end = start + window_size
            try:
                seq = str(genome[chrom][start:end])
            except (KeyError, ValueError):
                continue
            X[i] = score_sequence_motifs(seq, motif_log_odds)
            if (i + 1) % 1000 == 0:
                print(f"  Scored {i+1}/{n_peaks} peaks...")

    return X, y, peak_ids, chroms


def train_binary_classifiers(
    X: np.ndarray,
    y: np.ndarray,
    chroms: np.ndarray,
    n_folds: int = 5,
) -> Dict:
    """Train per-cell-type binary classifiers with chromosome-grouped CV.

    Splits are by chromosome to prevent leakage from nearby peaks on
    the same chromosome appearing in both train and validation.

    Args:
        X: Feature matrix (n_peaks, n_motifs).
        y: Cell type labels.
        chroms: Chromosome names per peak (for grouped CV).
        n_folds: Number of CV folds.

    Returns:
        Dict with per-cell-type AUC and top motifs.
    """
    if not HAS_XGBOOST:
        raise ImportError("xgboost required. Install: conda install xgboost")

    results = {}
    cv = GroupKFold(n_splits=n_folds)

    for ct_idx, ct in enumerate(CELL_TYPES):
        print(f"\n--- {ct} vs. rest ---")
        y_binary = (y == ct_idx).astype(int)

        n_pos = y_binary.sum()
        n_neg = len(y_binary) - n_pos
        print(f"  Positive: {n_pos}, Negative: {n_neg}")

        if n_pos < 10:
            print(f"  Skipping: too few positive examples")
            results[ct] = {"auc": 0.0, "top_motifs": []}
            continue

        aucs = []
        feature_importances = np.zeros(X.shape[1])

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y_binary, groups=chroms)):
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=n_neg / max(n_pos, 1),
                use_label_encoder=False,
                eval_metric="logloss",
                verbosity=0,
            )
            model.fit(X[train_idx], y_binary[train_idx])
            probs = model.predict_proba(X[val_idx])[:, 1]

            try:
                auc = roc_auc_score(y_binary[val_idx], probs)
            except ValueError:
                auc = 0.5

            aucs.append(auc)
            feature_importances += model.feature_importances_

        mean_auc = np.mean(aucs)
        feature_importances /= n_folds

        results[ct] = {
            "auc": mean_auc,
            "auc_std": np.std(aucs),
            "n_positive": int(n_pos),
        }
        print(f"  AUC: {mean_auc:.3f} ± {np.std(aucs):.3f}")

    return results


def train_multiclass_classifier(
    X: np.ndarray,
    y: np.ndarray,
    motif_names: list,
    chroms: np.ndarray,
    n_folds: int = 5,
) -> Dict:
    """Train multi-class classifier: which cell type is this peak specific to?

    Uses chromosome-grouped CV to prevent leakage.

    Returns:
        Dict with accuracy, per-class metrics, and top motifs.
    """
    if not HAS_XGBOOST:
        raise ImportError("xgboost required")

    cv = GroupKFold(n_splits=n_folds)
    accuracies = []
    feature_importances = np.zeros(X.shape[1])

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups=chroms)):
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            objective="multi:softprob",
            num_class=len(CELL_TYPES),
            use_label_encoder=False,
            eval_metric="mlogloss",
            verbosity=0,
        )
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[val_idx])
        acc = accuracy_score(y[val_idx], preds)
        accuracies.append(acc)
        feature_importances += model.feature_importances_

    feature_importances /= n_folds

    # Top motifs
    top_idx = np.argsort(feature_importances)[::-1][:20]
    top_motifs = [(motif_names[i], float(feature_importances[i])) for i in top_idx]

    return {
        "accuracy": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "top_motifs": top_motifs,
    }


def main():
    parser = argparse.ArgumentParser(description="Motif baseline for cell-type specificity")
    parser.add_argument("--annotations", type=str, default="data/peak_annotations.csv")
    parser.add_argument("--genome", type=str, default="data/genome/mm10.fa")
    parser.add_argument("--output-dir", type=str, default="results/motif_baseline")
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers for motif scanning")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading annotations...")
    annotations = pd.read_csv(args.annotations)
    n_specific = (annotations["category"] == "specific").sum()
    print(f"  Total peaks: {len(annotations):,}")
    print(f"  Specific peaks: {n_specific:,}")

    if n_specific < 50:
        print("ERROR: Too few specific peaks for motif baseline.")
        print("Check peak annotations or adjust tau/FDR thresholds.")
        return

    print("\nLoading JASPAR motifs...")
    motif_list = load_jaspar_motifs()
    motif_names = [name for name, _ in motif_list]

    # Pre-compute log-odds matrices (once, shared across all peaks)
    print("Pre-computing log-odds matrices...")
    motif_log_odds = precompute_log_odds(motif_list)

    # Build feature matrix
    n_workers = args.workers
    print(f"\nBuilding feature matrix (workers={n_workers})...")
    X, y, peak_ids, chroms = build_feature_matrix(
        annotations, args.genome, motif_log_odds, args.window_size, n_workers=n_workers,
    )

    # Save feature matrix
    np.savez(
        out_dir / "features.npz",
        X=X, y=y, peak_ids=peak_ids, motif_names=np.array(motif_names),
    )

    # Train binary classifiers
    print("\n" + "="*60)
    print("Per-cell-type binary classifiers (specific vs. rest)")
    print("="*60)
    binary_results = train_binary_classifiers(X, y, chroms)

    # Train multi-class classifier
    print("\n" + "="*60)
    print("Multi-class classifier (which cell type?)")
    print("="*60)
    multi_results = train_multiclass_classifier(X, y, motif_names, chroms)
    print(f"Accuracy: {multi_results['accuracy']:.3f} ± {multi_results['accuracy_std']:.3f}")
    print(f"\nTop 10 motifs:")
    for name, imp in multi_results["top_motifs"][:10]:
        print(f"  {name}: {imp:.4f}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    success_count = 0
    for ct in CELL_TYPES:
        auc = binary_results[ct]["auc"]
        status = "PASS" if auc > 0.65 else "FAIL"
        if auc > 0.65:
            success_count += 1
        print(f"  {ct}: AUC = {auc:.3f} [{status}]")

    print(f"\nMulti-class accuracy: {multi_results['accuracy']:.3f}")
    print(f"\nSuccess: {success_count}/5 cell types with AUC > 0.65")

    if success_count >= 3:
        print("\n>>> PROCEED to DNN training (sequence is predictive)")
    else:
        print("\n>>> CAUTION: Sequence may not be strongly predictive of specificity")
        print("    Consider: (1) more data, (2) different specificity thresholds,")
        print("    (3) alternative problem framing")

    # Save results
    all_results = {
        "binary_classifiers": binary_results,
        "multiclass": multi_results,
        "n_specific_peaks": int(n_specific),
        "n_motifs": len(motif_list),
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
