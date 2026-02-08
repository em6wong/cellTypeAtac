#!/usr/bin/env python3
"""Compare validation results across different model runs.

Parses metrics.json files from validation output directories and produces
a comparison table. Adapted from multiome/scripts/compare_validations.py.

Usage:
    python scripts/compare_validations.py                             # Compare all found
    python scripts/compare_validations.py results/validation/stage2 results/validation/stage3
    python scripts/compare_validations.py --csv                       # CSV output
    python scripts/compare_validations.py --sort profile_r            # Sort by metric
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional


def load_metrics(metrics_path: str) -> Dict:
    """Load metrics from a validation metrics.json file."""
    if not os.path.exists(metrics_path):
        return {}
    with open(metrics_path) as f:
        return json.load(f)


def extract_summary(metrics: Dict, label: str) -> Dict[str, str]:
    """Extract key summary metrics from a metrics dict."""
    summary = {
        "label": label,
        "s2_profile_r": "N/A",
        "s3_profile_r": "N/A",
        "s2_count_r": "N/A",
        "s3_count_r": "N/A",
        "s2_markers": "N/A",
        "s3_markers": "N/A",
        "cross_ct": "N/A",
        "signal_cv": "N/A",
        "s2_basic": "N/A",
        "s3_basic": "N/A",
    }

    # Stage 2 correlation
    s2_corr = metrics.get("stage2_correlation", {})
    if s2_corr:
        profile_rs = [v["profile_r_mean"] for v in s2_corr.values() if isinstance(v, dict)]
        count_rs = [v["count_r"] for v in s2_corr.values() if isinstance(v, dict)]
        if profile_rs:
            import numpy as np
            summary["s2_profile_r"] = f"{np.mean(profile_rs):.4f}"
        if count_rs:
            summary["s2_count_r"] = f"{np.mean(count_rs):.4f}"

    # Stage 3 correlation
    s3_corr = metrics.get("stage3_correlation", {})
    if s3_corr:
        profile_rs = [v["profile_r_mean"] for v in s3_corr.values() if isinstance(v, dict)]
        count_rs = [v["count_r"] for v in s3_corr.values() if isinstance(v, dict)]
        if profile_rs:
            import numpy as np
            summary["s3_profile_r"] = f"{np.mean(profile_rs):.4f}"
        if count_rs:
            summary["s3_count_r"] = f"{np.mean(count_rs):.4f}"

    # Marker accuracy
    s2_markers = metrics.get("stage2_markers", {})
    if s2_markers:
        n_correct = sum(1 for v in s2_markers.values() if isinstance(v, dict) and v.get("correct"))
        n_total = sum(1 for v in s2_markers.values() if isinstance(v, dict))
        summary["s2_markers"] = f"{n_correct}/{n_total}"

    s3_markers = metrics.get("stage3_markers", {})
    if s3_markers:
        n_correct = sum(1 for v in s3_markers.values() if isinstance(v, dict) and v.get("correct"))
        n_total = sum(1 for v in s3_markers.values() if isinstance(v, dict))
        summary["s3_markers"] = f"{n_correct}/{n_total}"

    # Collapse metrics
    collapse = metrics.get("stage3_collapse", {})
    if collapse:
        cross_ct = collapse.get("mean_cross_ct_count_corr")
        if cross_ct is not None:
            summary["cross_ct"] = f"{cross_ct:.3f}"
        sig_cv = collapse.get("signal_cv")
        if sig_cv is not None:
            summary["signal_cv"] = f"{sig_cv:.3f}"

    # Basic inference
    if "stage2_basic_inference" in metrics:
        summary["s2_basic"] = "PASS" if metrics["stage2_basic_inference"] else "FAIL"
    if "stage3_basic_inference" in metrics:
        summary["s3_basic"] = "PASS" if metrics["stage3_basic_inference"] else "FAIL"

    return summary


def extract_from_log(log_path: str) -> Dict[str, str]:
    """Extract metrics from validation log file (fallback if metrics.json missing)."""
    summary = {
        "s2_profile_r": "N/A",
        "s3_profile_r": "N/A",
        "cross_ct": "N/A",
        "signal_cv": "N/A",
        "s2_markers": "N/A",
        "s3_markers": "N/A",
    }

    if not os.path.exists(log_path):
        return summary

    with open(log_path) as f:
        content = f.read()

    patterns = {
        "s2_profile_r": r"Stage 2 ATAC correlation:\s+([\d.-]+)",
        "s3_profile_r": r"Stage 3 ATAC correlation:\s+([\d.-]+)",
        "cross_ct": r"Mean cross-CT correlation:\s+([\d.]+)",
        "signal_cv": r"Signal CV across CTs:\s+([\d.]+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            summary[key] = match.group(1)

    s2_marker_match = re.search(r"Stage 2 marker accuracy:\s+(\d+/\d+)", content)
    if s2_marker_match:
        summary["s2_markers"] = s2_marker_match.group(1)

    s3_marker_match = re.search(r"Stage 3 marker accuracy:\s+(\d+/\d+)", content)
    if s3_marker_match:
        summary["s3_markers"] = s3_marker_match.group(1)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Compare validation results")
    parser.add_argument("dirs", nargs="*", help="Validation output directories to compare")
    parser.add_argument("--csv", action="store_true", help="Output as CSV")
    parser.add_argument("--sort", choices=["label", "s2_profile_r", "s3_profile_r", "cross_ct"],
                        default="label", help="Sort by metric")
    args = parser.parse_args()

    # Find validation directories
    if args.dirs:
        val_dirs = args.dirs
    else:
        # Auto-discover
        results_dir = Path("results/validation")
        if results_dir.exists():
            val_dirs = sorted([
                str(d) for d in results_dir.iterdir()
                if d.is_dir() and (d / "metrics.json").exists()
            ])
        else:
            val_dirs = []

        # Also check top-level
        if (results_dir / "metrics.json").exists():
            val_dirs.insert(0, str(results_dir))

    if not val_dirs:
        print("No validation results found.")
        print("Run: python scripts/09_validate_models.py first")
        sys.exit(1)

    # Collect results
    results = []
    for d in val_dirs:
        metrics_path = os.path.join(d, "metrics.json")
        label = os.path.basename(d)

        if os.path.exists(metrics_path):
            metrics = load_metrics(metrics_path)
            summary = extract_summary(metrics, label)
        else:
            # Try log-based extraction
            log_path = os.path.join(d, "validation.log")
            summary = extract_from_log(log_path)
            summary["label"] = label

        results.append(summary)

    # Sort
    def sort_key(r):
        if args.sort == "s2_profile_r":
            try:
                return -float(r["s2_profile_r"])
            except ValueError:
                return 999
        elif args.sort == "s3_profile_r":
            try:
                return -float(r["s3_profile_r"])
            except ValueError:
                return 999
        elif args.sort == "cross_ct":
            try:
                return float(r["cross_ct"])
            except ValueError:
                return 999
        return r["label"]

    results.sort(key=sort_key)

    # CSV output
    if args.csv:
        print("Label,S2_Profile_r,S3_Profile_r,S2_Count_r,S3_Count_r,S2_Markers,S3_Markers,CrossCT,SignalCV")
        for r in results:
            print(f"{r['label']},{r['s2_profile_r']},{r['s3_profile_r']},"
                  f"{r.get('s2_count_r', 'N/A')},{r.get('s3_count_r', 'N/A')},"
                  f"{r['s2_markers']},{r['s3_markers']},"
                  f"{r['cross_ct']},{r['signal_cv']}")
        return

    # Pretty table
    print()
    print("=" * 110)
    print("                         VALIDATION RESULTS COMPARISON")
    print("=" * 110)
    print(f"{'Label':<20} | {'S2 prof_r':>9} | {'S3 prof_r':>9} | {'S2 cnt_r':>9} | {'S3 cnt_r':>9} | "
          f"{'S2 mark':>7} | {'S3 mark':>7} | {'CrossCT':>7} | {'SigCV':>7}")
    print("-" * 110)

    for r in results:
        print(f"{r['label']:<20} | {r['s2_profile_r']:>9} | {r['s3_profile_r']:>9} | "
              f"{r.get('s2_count_r', 'N/A'):>9} | {r.get('s3_count_r', 'N/A'):>9} | "
              f"{r['s2_markers']:>7} | {r['s3_markers']:>7} | "
              f"{r['cross_ct']:>7} | {r['signal_cv']:>7}")

    print("=" * 110)
    print()
    print("Legend:")
    print("  S2/S3 prof_r : Mean profile Pearson r on test set (higher=better)")
    print("  S2/S3 cnt_r  : Mean count Pearson r on test set (higher=better)")
    print("  S2/S3 mark   : Marker gene accuracy (correct/total)")
    print("  CrossCT      : Mean cross-cell-type count correlation (lower=better differentiation)")
    print("  SigCV        : Count coefficient of variation across cell types (higher=more variation)")
    print()


if __name__ == "__main__":
    main()
