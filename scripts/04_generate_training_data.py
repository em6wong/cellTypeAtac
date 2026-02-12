#!/usr/bin/env python
"""Generate zarr training datasets for each cell type.

Usage:
    python scripts/04_generate_training_data.py \
        --peaks data/peaks_downsampled/merged_peaks.bed \
        --bigwig-dir data/bigwig \
        --genome data/genome/mm10.fa \
        --chrom-sizes data/genome/mm10.chrom.sizes \
        --output-dir data/training
"""

import argparse
import multiprocessing as mp
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.generate_dataset import generate_dataset


CELL_TYPES = ["Cardiomyocyte", "Coronary_EC", "Fibroblast", "Macrophage", "Pericytes"]


def _generate_one(task):
    """Worker: generate one zarr dataset."""
    label, kwargs = task
    print(f"\n{'='*60}")
    print(f"[START] {label}")
    print(f"{'='*60}")
    generate_dataset(**kwargs)
    print(f"[DONE]  {label}")


def main():
    parser = argparse.ArgumentParser(description="Generate training zarr datasets")
    parser.add_argument("--peaks", type=str, default="data/peaks_downsampled/merged_peaks.bed")
    parser.add_argument("--bigwig-dir", type=str, default="data/bigwig")
    parser.add_argument("--genome", type=str, default="data/genome/mm10.fa")
    parser.add_argument("--chrom-sizes", type=str, default="data/genome/mm10.chrom.sizes")
    parser.add_argument("--output-dir", type=str, default="data/training")
    parser.add_argument("--input-length", type=int, default=2114)
    parser.add_argument("--output-length", type=int, default=1000)
    parser.add_argument("--max-jitter", type=int, default=0,
                        help="Random jitter in bp for training augmentation (default: 0)")
    parser.add_argument("--gc-match", action="store_true",
                        help="Match background GC content to peak distribution")
    parser.add_argument("--cell-types", type=str, default=None,
                        help="Comma-separated cell types (default: all)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (default: 1 = sequential)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cell_types = args.cell_types.split(",") if args.cell_types else CELL_TYPES

    # Build task list
    tasks = []
    for ct in cell_types:
        bw_path = Path(args.bigwig_dir) / f"{ct}.bw"
        if not bw_path.exists():
            print(f"WARNING: bigWig not found for {ct}: {bw_path}")
            continue

        for split in ["train", "val", "test"]:
            zarr_path = out_dir / ct / f"{split}.zarr"
            zarr_path.parent.mkdir(parents=True, exist_ok=True)

            # Only use jitter for training split
            jitter = args.max_jitter if split == "train" else 0

            tasks.append((
                f"{ct}/{split}",
                dict(
                    peaks_bed=args.peaks,
                    bigwig_path=str(bw_path),
                    genome_path=args.genome,
                    chrom_sizes_path=args.chrom_sizes,
                    output_path=str(zarr_path),
                    input_length=args.input_length,
                    output_length=args.output_length,
                    max_jitter=jitter,
                    include_background=(split == "train"),
                    gc_match=args.gc_match if split == "train" else False,
                    split=split,
                ),
            ))

    # Also generate merged dataset for bias model
    merged_bw = Path(args.bigwig_dir) / "merged.bw"
    if merged_bw.exists():
        for split in ["train", "val"]:
            zarr_path = out_dir / "merged" / f"{split}.zarr"
            zarr_path.parent.mkdir(parents=True, exist_ok=True)

            jitter = args.max_jitter if split == "train" else 0

            tasks.append((
                f"merged/{split}",
                dict(
                    peaks_bed=args.peaks,
                    bigwig_path=str(merged_bw),
                    genome_path=args.genome,
                    chrom_sizes_path=args.chrom_sizes,
                    output_path=str(zarr_path),
                    input_length=args.input_length,
                    output_length=args.output_length,
                    max_jitter=jitter,
                    include_background=True,
                    gc_match=args.gc_match if split == "train" else False,
                    split=split,
                ),
            ))

    print(f"Total tasks: {len(tasks)}, Workers: {args.workers}")

    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            pool.map(_generate_one, tasks)
    else:
        for task in tasks:
            _generate_one(task)

    print("\nAll training datasets generated successfully.")


if __name__ == "__main__":
    main()
