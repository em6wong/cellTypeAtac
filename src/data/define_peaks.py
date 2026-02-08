"""Define cell-type specific peaks using differential accessibility analysis.

Reads the per-mouse logCPM matrix (4 replicates × 5 cell types) and computes:
  1. Tau specificity index per peak (0 = ubiquitous, 1 = exclusive)
  2. Kruskal-Wallis test across cell types
  3. Pairwise comparisons: each cell type vs. rest
  4. Peak classification: specific / shared / intermediate
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple


CELL_TYPES = ["Cardiomyocyte", "Coronary_EC", "Fibroblast", "Macrophage", "Pericytes"]

# Column name mapping: CSV uses dots, we use underscores
_COLNAME_MAP = {
    "Coronary.EC": "Coronary_EC",
}


def _parse_mouse_columns(columns: list) -> dict:
    """Parse per-mouse CSV column names into {cell_type: [col_indices]}.

    Column format: 'YS1.4.F_Cardiomyocyte', 'YS1.4.M_Coronary.EC', etc.
    """
    ct_cols = {ct: [] for ct in CELL_TYPES}

    for col in columns:
        if col == "":
            continue
        # Normalize column name: CSV uses dots (Coronary.EC), we use underscores
        col_check = col.replace("Coronary.EC", "Coronary_EC")

        for ct in CELL_TYPES:
            if col_check.endswith(f"_{ct}"):
                ct_cols[ct].append(col)
                break

    return ct_cols


def compute_tau(expression_matrix: np.ndarray) -> np.ndarray:
    """Compute tau specificity index for each row (peak).

    Tau = sum(1 - x_hat_i) / (n - 1), where x_hat_i = x_i / max(x)

    Args:
        expression_matrix: (n_peaks, n_cell_types) mean logCPM values.

    Returns:
        Array of tau values, shape (n_peaks,).
    """
    # Handle all-zero rows
    row_max = expression_matrix.max(axis=1, keepdims=True)
    row_max = np.where(row_max == 0, 1, row_max)

    x_hat = expression_matrix / row_max
    n = expression_matrix.shape[1]
    tau = (1 - x_hat).sum(axis=1) / (n - 1)

    return tau


def differential_analysis(
    mouse_csv_path: str,
    fdr_threshold: float = 0.05,
    tau_specific: float = 0.6,
    tau_shared: float = 0.3,
) -> pd.DataFrame:
    """Run differential accessibility analysis on per-mouse logCPM data.

    Args:
        mouse_csv_path: Path to YoungSed_DownSample_Peak_logCPM_CellType_Mouse.csv
        fdr_threshold: FDR cutoff for significance.
        tau_specific: Tau threshold above which a peak is considered specific.
        tau_shared: Tau threshold below which a peak is considered shared.

    Returns:
        DataFrame with columns: peak_id, chrom, start, end, tau,
        specific_celltype, category, kw_pvalue, plus per-CT fold changes.
    """
    df = pd.read_csv(mouse_csv_path, index_col=0)
    ct_cols = _parse_mouse_columns(list(df.columns))

    n_peaks = len(df)
    results = []

    # Pre-compute cell-type mean expression for tau
    ct_means = np.zeros((n_peaks, len(CELL_TYPES)))
    ct_data = {}  # cell_type -> (n_peaks, n_replicates) array

    for j, ct in enumerate(CELL_TYPES):
        cols = ct_cols[ct]
        vals = df[cols].values  # (n_peaks, n_replicates)
        ct_data[ct] = vals
        ct_means[:, j] = vals.mean(axis=1)

    # Tau specificity
    tau = compute_tau(ct_means)

    # Kruskal-Wallis test per peak
    kw_pvalues = np.ones(n_peaks)
    for i in range(n_peaks):
        groups = [ct_data[ct][i, :] for ct in CELL_TYPES]
        # Need variance in at least one group
        if any(g.std() > 0 for g in groups):
            try:
                stat, pval = stats.kruskal(*groups)
                kw_pvalues[i] = pval
            except ValueError:
                pass

    # FDR correction (Benjamini-Hochberg)
    kw_fdr = _benjamini_hochberg(kw_pvalues)

    # Pairwise: each cell type vs. rest (Mann-Whitney U)
    pairwise_pvalues = np.ones((n_peaks, len(CELL_TYPES)))
    pairwise_fc = np.zeros((n_peaks, len(CELL_TYPES)))

    for j, ct in enumerate(CELL_TYPES):
        ct_vals = ct_data[ct]  # (n_peaks, n_replicates)
        other_cts = [c for c in CELL_TYPES if c != ct]
        other_vals = np.concatenate([ct_data[c] for c in other_cts], axis=1)

        for i in range(n_peaks):
            x = ct_vals[i, :]
            y = other_vals[i, :]
            fc = x.mean() - y.mean()  # log-space fold change
            pairwise_fc[i, j] = fc

            if x.std() > 0 or y.std() > 0:
                try:
                    _, pval = stats.mannwhitneyu(x, y, alternative="two-sided")
                    pairwise_pvalues[i, j] = pval
                except ValueError:
                    pass

    # FDR per cell type
    pairwise_fdr = np.zeros_like(pairwise_pvalues)
    for j in range(len(CELL_TYPES)):
        pairwise_fdr[:, j] = _benjamini_hochberg(pairwise_pvalues[:, j])

    # Build results DataFrame
    peak_ids = df.index.values
    records = []

    for i in range(n_peaks):
        pid = peak_ids[i]
        parts = str(pid).split("-")
        chrom = parts[0]
        start = int(parts[1])
        end = int(parts[2])

        # Find most specific cell type
        best_ct_idx = int(np.argmax(pairwise_fc[i, :]))
        best_ct = CELL_TYPES[best_ct_idx]
        best_fc = pairwise_fc[i, best_ct_idx]
        best_fdr = pairwise_fdr[i, best_ct_idx]

        # Classify
        if tau[i] > tau_specific and best_fdr < fdr_threshold and best_fc > 0:
            category = "specific"
            specific_ct = best_ct
        elif tau[i] < tau_shared:
            category = "shared"
            specific_ct = "none"
        else:
            category = "intermediate"
            specific_ct = "none"

        record = {
            "peak_id": pid,
            "chrom": chrom,
            "start": start,
            "end": end,
            "tau": tau[i],
            "specific_celltype": specific_ct,
            "category": category,
            "kw_pvalue": kw_pvalues[i],
            "kw_fdr": kw_fdr[i],
        }

        # Add per-CT fold changes
        for j, ct in enumerate(CELL_TYPES):
            record[f"fc_{ct}"] = pairwise_fc[i, j]
            record[f"fdr_{ct}"] = pairwise_fdr[i, j]

        records.append(record)

    return pd.DataFrame(records)


def _benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    n = len(pvalues)
    sorted_idx = np.argsort(pvalues)
    sorted_pvals = pvalues[sorted_idx]

    # BH formula: adjusted_p[i] = min(p[i] * n / rank, 1.0)
    ranks = np.arange(1, n + 1)
    adjusted = sorted_pvals * n / ranks

    # Enforce monotonicity (running minimum from the right)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)

    # Map back to original order
    result = np.empty(n)
    result[sorted_idx] = adjusted
    return result


def summarize_annotations(annotations: pd.DataFrame) -> str:
    """Print a summary of peak annotations."""
    lines = []
    lines.append(f"Total peaks: {len(annotations):,}")
    lines.append(f"\nCategory distribution:")
    for cat, count in annotations["category"].value_counts().items():
        lines.append(f"  {cat}: {count:,} ({100*count/len(annotations):.1f}%)")

    lines.append(f"\nTau distribution:")
    lines.append(f"  Mean: {annotations['tau'].mean():.3f}")
    lines.append(f"  Median: {annotations['tau'].median():.3f}")
    lines.append(f"  Std: {annotations['tau'].std():.3f}")

    specific = annotations[annotations["category"] == "specific"]
    if len(specific) > 0:
        lines.append(f"\nSpecific peaks per cell type:")
        for ct, count in specific["specific_celltype"].value_counts().items():
            lines.append(f"  {ct}: {count:,}")

    return "\n".join(lines)
