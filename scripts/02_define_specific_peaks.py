#!/usr/bin/env python
"""Define cell-type specific peaks using differential accessibility.

Usage:
    python scripts/02_define_specific_peaks.py \
        --mouse-csv data/YoungSed_DownSample_Peak_logCPM_CellType_Mouse.csv \
        --output data/peak_annotations.csv
"""

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.define_peaks import differential_analysis, summarize_annotations


def main():
    parser = argparse.ArgumentParser(description="Define cell-type specific peaks")
    parser.add_argument(
        "--mouse-csv", type=str,
        default="data/YoungSed_DownSample_Peak_logCPM_CellType_Mouse.csv",
    )
    parser.add_argument("--fdr-threshold", type=float, default=0.05)
    parser.add_argument("--output", type=str, default="data/peak_annotations.csv")
    args = parser.parse_args()

    print("Running differential accessibility analysis...")
    print(f"  Input: {args.mouse_csv}")
    print(f"  FDR threshold: {args.fdr_threshold}")

    annotations = differential_analysis(args.mouse_csv, fdr_threshold=args.fdr_threshold)

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    annotations.to_csv(args.output, index=False)
    print(f"\nSaved annotations to {args.output}")

    # Summary
    print(f"\n{'='*60}")
    print(summarize_annotations(annotations))
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
