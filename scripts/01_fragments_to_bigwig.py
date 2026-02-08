#!/usr/bin/env python
"""Generate per-cell-type Tn5-corrected bigWig files from fragment BEDs.

Usage:
    python scripts/01_fragments_to_bigwig.py \
        --chrom-sizes data/genome/mm10.chrom.sizes \
        --fragments-dir data \
        --output-dir data/bigwig
"""

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.fragments_to_bigwig import create_celltype_bigwig, create_merged_bigwig


CELL_TYPES = ["Cardiomyocyte", "Coronary_EC", "Fibroblast", "Macrophage", "Pericytes"]


def main():
    parser = argparse.ArgumentParser(description="Fragment BED → bigWig")
    parser.add_argument("--chrom-sizes", type=str, required=True)
    parser.add_argument("--fragments-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="data/bigwig")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fragment_paths = []

    for ct in CELL_TYPES:
        frag_file = Path(args.fragments_dir) / f"{ct}_downsample.bed"
        if not frag_file.exists():
            # Try with dot notation (Coronary.EC)
            alt = ct.replace("_", ".")
            frag_file = Path(args.fragments_dir) / f"{alt}_downsample.bed"

        if not frag_file.exists():
            print(f"WARNING: Fragment file not found for {ct}: {frag_file}")
            continue

        out_path = out_dir / f"{ct}.bw"
        print(f"\n{'='*60}")
        print(f"Cell type: {ct}")
        print(f"  Input:  {frag_file}")
        print(f"  Output: {out_path}")
        print(f"{'='*60}")

        create_celltype_bigwig(str(frag_file), str(out_path), args.chrom_sizes)
        fragment_paths.append(str(frag_file))

    # Create merged bigWig
    if fragment_paths:
        merged_path = out_dir / "merged.bw"
        print(f"\n{'='*60}")
        print(f"Creating merged bigWig from all {len(fragment_paths)} cell types")
        print(f"  Output: {merged_path}")
        print(f"{'='*60}")
        create_merged_bigwig(fragment_paths, str(merged_path), args.chrom_sizes)

    print("\nAll bigWig files generated successfully.")


if __name__ == "__main__":
    main()
