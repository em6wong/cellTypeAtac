"""Evaluation metrics for ATAC-seq models.

Includes:
  - Profile correlation (Pearson r per peak)
  - Count correlation
  - Jensen-Shannon divergence
  - Specificity AUC (cross-model analysis)
"""

import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Tuple


def profile_pearson_r(
    pred: np.ndarray,
    target: np.ndarray,
) -> float:
    """Pearson correlation between predicted and target profiles.

    Args:
        pred: Predicted profile (length,) or (n_peaks, length).
        target: Target profile (same shape).

    Returns:
        Mean Pearson r across peaks.
    """
    if pred.ndim == 1:
        pred = pred[np.newaxis, :]
        target = target[np.newaxis, :]

    correlations = []
    for i in range(len(pred)):
        if target[i].std() < 1e-8 and pred[i].std() < 1e-8:
            correlations.append(1.0)
        elif target[i].std() < 1e-8 or pred[i].std() < 1e-8:
            correlations.append(0.0)
        else:
            r, _ = stats.pearsonr(pred[i], target[i])
            correlations.append(r)

    return float(np.mean(correlations))


def count_pearson_r(
    pred_counts: np.ndarray,
    target_counts: np.ndarray,
) -> float:
    """Pearson correlation between predicted and target total counts.

    Args:
        pred_counts: Predicted counts (n_peaks,).
        target_counts: Target counts (n_peaks,).

    Returns:
        Pearson r.
    """
    if pred_counts.std() < 1e-8 or target_counts.std() < 1e-8:
        return 0.0
    r, _ = stats.pearsonr(pred_counts, target_counts)
    return float(r)


def profile_jsd(
    pred: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """Jensen-Shannon divergence between predicted and target profiles.

    Args:
        pred: Predicted profile (length,) or (n_peaks, length).
        target: Target profile (same shape).
        eps: Numerical stability constant.

    Returns:
        Mean JSD across peaks.
    """
    if pred.ndim == 1:
        pred = pred[np.newaxis, :]
        target = target[np.newaxis, :]

    jsds = []
    for i in range(len(pred)):
        # Normalize to probability distributions
        p = target[i] + eps
        p = p / p.sum()
        q = pred[i] + eps
        q = q / q.sum()

        jsds.append(float(jensenshannon(p, q) ** 2))  # JSD is sqrt of JS divergence

    return float(np.mean(jsds))


def specificity_auc(
    predictions: Dict[str, np.ndarray],
    aligned_annotations: "pd.DataFrame",
    metric: str = "count",
) -> Dict[str, float]:
    """Cross-model specificity AUC.

    For each cell-type specific peak, checks whether the "correct" model
    predicts higher signal than other models.

    IMPORTANT: aligned_annotations must be pre-aligned with prediction arrays.
    Row i of aligned_annotations must correspond to predictions[ct][i] for all
    cell types. The caller is responsible for building this alignment (e.g. by
    matching zarr coordinates to annotation coordinates).

    Args:
        predictions: Dict mapping cell_type -> predicted values.
            For 'count' metric: (n_regions,) arrays.
            For 'profile' metric: (n_regions, length) arrays.
        aligned_annotations: DataFrame with 'specific_celltype' and 'category'
            columns, pre-aligned so row i matches predictions position i.
            Rows without annotations (background regions) should have
            category != 'specific'.
        metric: 'count' or 'profile_sum'.

    Returns:
        Dict with per-cell-type AUC and overall AUC.
    """
    cell_types = list(predictions.keys())
    n_preds = len(next(iter(predictions.values())))
    annotations = aligned_annotations.reset_index(drop=True)

    assert len(annotations) == n_preds, (
        f"Annotations ({len(annotations)}) must match prediction length ({n_preds}). "
        f"Pass pre-aligned annotations, not the full annotations file."
    )

    specific = annotations[annotations["category"] == "specific"]
    results = {}

    for ct in cell_types:
        ct_peaks = specific[specific["specific_celltype"] == ct]
        if len(ct_peaks) < 10:
            results[ct] = 0.5
            continue

        indices = ct_peaks.index.values

        correct_scores = []
        other_scores = []

        for idx in indices:
            if metric == "count":
                ct_score = float(predictions[ct][idx])
                other = [float(predictions[c][idx]) for c in cell_types if c != ct]
            else:  # profile_sum
                ct_score = float(predictions[ct][idx].sum())
                other = [float(predictions[c][idx].sum()) for c in cell_types if c != ct]

            correct_scores.append(ct_score)
            other_scores.extend(other)

        if len(correct_scores) == 0:
            results[ct] = 0.5
            continue

        y_true = np.concatenate([np.ones(len(correct_scores)),
                                  np.zeros(len(other_scores))])
        y_score = np.concatenate([correct_scores, other_scores])

        try:
            auc = roc_auc_score(y_true, y_score)
        except ValueError:
            auc = 0.5

        results[ct] = float(auc)

    all_aucs = list(results.values())
    results["overall"] = float(np.mean(all_aucs))

    return results


def cross_model_correlation(
    predictions: Dict[str, np.ndarray],
) -> np.ndarray:
    """Compute pairwise Pearson correlation between model predictions.

    Low correlation = good differentiation between cell types.

    Args:
        predictions: Dict mapping cell_type -> (n_peaks,) predicted counts.

    Returns:
        (n_cell_types, n_cell_types) correlation matrix.
    """
    cell_types = list(predictions.keys())
    n_ct = len(cell_types)
    corr_matrix = np.ones((n_ct, n_ct))

    for i in range(n_ct):
        for j in range(i + 1, n_ct):
            r, _ = stats.pearsonr(predictions[cell_types[i]], predictions[cell_types[j]])
            corr_matrix[i, j] = r
            corr_matrix[j, i] = r

    return corr_matrix
