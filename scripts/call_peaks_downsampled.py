#!/usr/bin/env python3
"""Call peaks on downsampled fragment files using MACS2.

Converts 10X fragment BED files to Tn5 cut-site tagAlign format, runs MACS2
per cell type, then merges into a unified peak set.

This addresses the bias where the original peak set was called on all cells
(before downsampling), giving over-representation of peaks from abundant
cell types (especially CM).

Usage:
    python scripts/call_peaks_downsampled.py
    python scripts/call_peaks_downsampled.py --fragments-dir data --output-dir data/peaks_downsampled
    python scripts/call_peaks_downsampled.py --qvalue 0.01  # Stricter threshold
"""

import argparse
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


CELL_TYPES = [
    "Cardiomyocyte",
    "Coronary_EC",
    "Fibroblast",
    "Macrophage",
    "Pericytes",
]


def fragments_to_cutsites(fragments_path: str, output_path: str):
    """Convert 10X fragment BED to Tn5-corrected cut-site tagAlign.

    Each fragment produces two cut sites:
      - Left:  start + 4 (forward strand Tn5 correction)
      - Right: end - 5   (reverse strand Tn5 correction)

    Output is BED6 tagAlign format for MACS2:
      chr, start, end, name, score, strand

    Args:
        fragments_path: Path to 10X fragment BED (chr, start, end, barcode, count).
        output_path: Path for output tagAlign BED.
    """
    print(f"  Converting {fragments_path} to cut sites...")
    n_frags = 0

    with open(fragments_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            parts = line.rstrip("\n").split("\t")
            chrom = parts[0]

            # Skip non-standard chromosomes
            if not (chrom.startswith("chr") and (chrom[3:].isdigit() or chrom in ("chrX", "chrY"))):
                continue

            start = int(parts[1])
            end = int(parts[2])

            # Tn5-corrected left cut site (+4, forward strand)
            left = start + 4
            fout.write(f"{chrom}\t{left}\t{left + 1}\tN\t0\t+\n")

            # Tn5-corrected right cut site (-5, reverse strand)
            right = end - 5
            if right > 0:
                fout.write(f"{chrom}\t{right}\t{right + 1}\tN\t0\t-\n")

            n_frags += 1

    print(f"  Wrote {n_frags * 2:,} cut sites from {n_frags:,} fragments")
    return n_frags


def run_macs2(
    cutsites_path: str,
    output_dir: str,
    name: str,
    genome_size: str = "mm",
    qvalue: float = 0.05,
    shift: int = -75,
    extsize: int = 150,
):
    """Run MACS2 callpeak on cut-site BED file.

    Uses ArchR-style parameters for snATAC-seq:
      --nomodel --shift -75 --extsize 150 --keep-dup all

    Args:
        cutsites_path: Path to Tn5 cut-site tagAlign BED.
        output_dir: Output directory for MACS2 results.
        name: Sample name prefix for output files.
        genome_size: MACS2 genome size (default 'mm' for mouse).
        qvalue: Q-value threshold (default 0.05).
        shift: Read shift for ATAC-seq (default -75).
        extsize: Extension size for ATAC-seq (default 150).
    """
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "macs2", "callpeak",
        "-t", cutsites_path,
        "-f", "BED",
        "-g", genome_size,
        "-n", name,
        "--outdir", output_dir,
        "--nomodel",
        "--shift", str(shift),
        "--extsize", str(extsize),
        "--keep-dup", "all",
        "--call-summits",
        "-q", str(qvalue),
    ]

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  MACS2 stderr:\n{result.stderr}")
        raise RuntimeError(f"MACS2 failed for {name}")

    # Count peaks
    narrowpeak = os.path.join(output_dir, f"{name}_peaks.narrowPeak")
    if os.path.exists(narrowpeak):
        n_peaks = sum(1 for _ in open(narrowpeak))
        print(f"  {name}: {n_peaks:,} peaks called")
    else:
        print(f"  WARNING: No narrowPeak file produced for {name}")


def merge_peaks(
    peak_files: list,
    output_path: str,
    chrom_sizes_path: str,
    merge_distance: int = 200,
    min_width: int = 200,
    max_width: int = 5000,
):
    """Merge peaks from multiple cell types into a unified peak set.

    Pure Python implementation (no bedtools dependency). Reads narrowPeak
    files, sorts by position, merges overlapping/nearby intervals, and
    filters by width.

    Args:
        peak_files: List of narrowPeak file paths.
        output_path: Path for merged peak BED output.
        chrom_sizes_path: Path to chromosome sizes file.
        merge_distance: Max distance to merge overlapping peaks (default 200bp).
        min_width: Minimum peak width to keep (default 200bp).
        max_width: Maximum peak width to keep (default 5000bp).
    """
    # Collect all peaks
    all_peaks = []
    for pf in peak_files:
        if not os.path.exists(pf):
            print(f"  WARNING: {pf} not found, skipping")
            continue
        with open(pf) as fin:
            for line in fin:
                parts = line.strip().split("\t")
                chrom, start, end = parts[0], int(parts[1]), int(parts[2])
                all_peaks.append((chrom, start, end))

    print(f"  Total peaks before merge: {len(all_peaks):,}")

    # Sort by chrom, start
    all_peaks.sort(key=lambda x: (x[0], x[1]))

    # Merge overlapping/nearby intervals
    merged = []
    for chrom, start, end in all_peaks:
        if merged and merged[-1][0] == chrom and start <= merged[-1][2] + merge_distance:
            # Extend existing interval
            merged[-1] = (chrom, merged[-1][1], max(merged[-1][2], end))
        else:
            merged.append((chrom, start, end))

    # Filter by width and write
    n_kept = 0
    with open(output_path, "w") as fout:
        for chrom, start, end in merged:
            width = end - start
            if min_width <= width <= max_width:
                fout.write(f"{chrom}\t{start}\t{end}\tpeak_{n_kept}\t0\t.\n")
                n_kept += 1

    print(f"  Merged: {len(merged):,} → {n_kept:,} peaks (after width filter {min_width}-{max_width}bp)")
    return n_kept


def _count_overlaps(query_df, subject_df):
    """Count query intervals that overlap at least one subject interval.

    Uses numpy binary search for speed — O(n log m) instead of O(n*m).
    Both DataFrames must have chrom, start, end columns.
    """
    n_overlap = 0
    for chrom in query_df["chrom"].unique():
        q = query_df[query_df["chrom"] == chrom]
        s = subject_df[subject_df["chrom"] == chrom]
        if len(s) == 0:
            continue

        s_starts = s["start"].values
        s_ends = s["end"].values
        order = np.argsort(s_starts)
        s_starts = s_starts[order]
        s_ends = s_ends[order]

        for _, row in q.iterrows():
            # Binary search: find subject intervals that could overlap
            # A subject overlaps query if s_start < q_end AND s_end > q_start
            idx = np.searchsorted(s_starts, row["end"], side="left")
            # Check candidates from idx backwards
            for i in range(idx - 1, -1, -1):
                if s_ends[i] <= row["start"]:
                    break
                if s_starts[i] < row["end"] and s_ends[i] > row["start"]:
                    n_overlap += 1
                    break
    return n_overlap


def compare_peak_sets(original_bed: str, new_bed: str):
    """Print comparison between original and new peak sets."""
    orig = pd.read_csv(original_bed, sep="\t", header=None, usecols=[0, 1, 2],
                       names=["chrom", "start", "end"])
    new = pd.read_csv(new_bed, sep="\t", header=None, usecols=[0, 1, 2],
                      names=["chrom", "start", "end"])

    orig["width"] = orig["end"] - orig["start"]
    new["width"] = new["end"] - new["start"]

    print("\n  Peak set comparison:")
    print(f"  {'':20s} {'Original':>10s}  {'Downsampled':>12s}")
    print(f"  {'Total peaks':20s} {len(orig):>10,d}  {len(new):>12,d}")
    print(f"  {'Mean width (bp)':20s} {orig['width'].mean():>10.0f}  {new['width'].mean():>12.0f}")
    print(f"  {'Median width (bp)':20s} {orig['width'].median():>10.0f}  {new['width'].median():>12.0f}")
    print(f"  {'Total bp covered':20s} {orig['width'].sum():>10,d}  {new['width'].sum():>12,d}")

    print("  Computing overlap...")
    n_new_in_orig = _count_overlaps(new, orig)
    n_orig_in_new = _count_overlaps(orig, new)

    print(f"  {'New in original':20s} {n_new_in_orig:>10,d}  ({100*n_new_in_orig/max(1,len(new)):.1f}%)")
    print(f"  {'Original in new':20s} {n_orig_in_new:>10,d}  ({100*n_orig_in_new/max(1,len(orig)):.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Call peaks on downsampled fragments using MACS2",
    )
    parser.add_argument("--fragments-dir", default="data",
                        help="Directory containing *_downsample.bed files")
    parser.add_argument("--output-dir", default="data/peaks_downsampled",
                        help="Output directory for MACS2 results")
    parser.add_argument("--chrom-sizes", default="data/genome/mm10.chrom.sizes",
                        help="Chromosome sizes file")
    parser.add_argument("--genome-size", default="mm",
                        help="MACS2 genome size parameter (default: mm)")
    parser.add_argument("--qvalue", type=float, default=0.05,
                        help="MACS2 q-value threshold (default: 0.05)")
    parser.add_argument("--merge-distance", type=int, default=200,
                        help="Distance to merge nearby peaks (default: 200bp)")
    parser.add_argument("--original-peaks", default="data/YoungSed_DownSample_Peak.bed",
                        help="Original peak BED for comparison")
    parser.add_argument("--cell-types", nargs="*", default=None,
                        help="Subset of cell types to process (default: all)")
    args = parser.parse_args()

    cell_types = args.cell_types or CELL_TYPES
    fragments_dir = Path(args.fragments_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cutsites_dir = output_dir / "cutsites"
    cutsites_dir.mkdir(exist_ok=True)
    macs2_dir = output_dir / "macs2"
    macs2_dir.mkdir(exist_ok=True)

    # Step 1: Convert fragments to cut sites per cell type
    print("=" * 60)
    print("Step 1: Convert fragments to Tn5 cut sites")
    print("=" * 60)

    cutsite_files = {}
    for ct in cell_types:
        frag_path = fragments_dir / f"{ct}_downsample.bed"
        if not frag_path.exists():
            print(f"  WARNING: {frag_path} not found, skipping {ct}")
            continue

        cs_path = cutsites_dir / f"{ct}_cutsites.bed"
        if cs_path.exists() and cs_path.stat().st_size > 0:
            print(f"  {ct}: using existing {cs_path}")
        else:
            fragments_to_cutsites(str(frag_path), str(cs_path))
        cutsite_files[ct] = str(cs_path)

    # Step 2: Run MACS2 per cell type
    print()
    print("=" * 60)
    print("Step 2: MACS2 peak calling per cell type")
    print("=" * 60)

    peak_files = []
    for ct, cs_path in cutsite_files.items():
        ct_macs2_dir = str(macs2_dir / ct)
        print(f"\n  --- {ct} ---")
        run_macs2(
            cs_path,
            ct_macs2_dir,
            name=ct,
            genome_size=args.genome_size,
            qvalue=args.qvalue,
        )
        narrowpeak = os.path.join(ct_macs2_dir, f"{ct}_peaks.narrowPeak")
        if os.path.exists(narrowpeak):
            peak_files.append(narrowpeak)

    if not peak_files:
        print("ERROR: No peaks called for any cell type")
        return

    # Step 3: Merge peaks across cell types
    print()
    print("=" * 60)
    print("Step 3: Merge peaks across cell types")
    print("=" * 60)

    merged_bed = str(output_dir / "merged_peaks.bed")
    n_merged = merge_peaks(
        peak_files,
        merged_bed,
        args.chrom_sizes,
        merge_distance=args.merge_distance,
    )

    # Step 4: Per-cell-type peak counts
    print()
    print("=" * 60)
    print("Step 4: Per-cell-type peak summary")
    print("=" * 60)

    for ct in cell_types:
        narrowpeak = str(macs2_dir / ct / f"{ct}_peaks.narrowPeak")
        if os.path.exists(narrowpeak):
            n = sum(1 for _ in open(narrowpeak))
            print(f"  {ct:20s}: {n:>8,d} peaks")

    print(f"  {'Merged (union)':20s}: {n_merged:>8,d} peaks")

    # Step 5: Compare with original peak set
    print()
    print("=" * 60)
    print("Step 5: Compare with original peak set")
    print("=" * 60)

    if os.path.exists(args.original_peaks):
        compare_peak_sets(args.original_peaks, merged_bed)
    else:
        print(f"  Original peaks not found at {args.original_peaks}")

    print()
    print("=" * 60)
    print(f"Output: {merged_bed}")
    print(f"Per-cell-type results: {macs2_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
