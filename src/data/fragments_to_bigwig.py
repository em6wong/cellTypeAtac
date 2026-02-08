"""Fragment BED files → Tn5-corrected bigWig coverage tracks.

Adapted from ../multiome/src/data/tn5_correction.py for the cell-type
specific ATAC pipeline. Reads 10X-format fragment BED files and produces
base-resolution cut-site coverage bigWig files.

Tn5 correction: The Tn5 transposase binds as a dimer and cuts with a 9bp
stagger. To get true cut sites:
  - Left end (forward strand): +4 bp
  - Right end (reverse strand): -5 bp
"""

import numpy as np
import pandas as pd
import pyBigWig
from pathlib import Path
from typing import Optional
from tqdm import tqdm


def load_chrom_sizes(chrom_sizes_path: str) -> dict:
    """Load chromosome sizes from a tab-separated file."""
    chrom_sizes = {}
    with open(chrom_sizes_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                chrom_sizes[parts[0]] = int(parts[1])
    return chrom_sizes


def fragments_to_cutsites(
    fragments_path: str,
    chrom_sizes: dict,
    chunk_size: int = 1_000_000,
) -> dict:
    """
    Convert fragment BED to Tn5-corrected cut-site coverage arrays.

    Args:
        fragments_path: Path to fragment BED (chr, start, end, barcode, count).
        chrom_sizes: Dict mapping chromosome -> size.
        chunk_size: Number of rows to process at a time.

    Returns:
        Dict mapping chromosome -> numpy array of cut-site counts.
    """
    coverage = {}

    # Standard chromosomes only
    valid_chroms = {c for c in chrom_sizes if c.startswith("chr") and c[3:].isdigit()}
    valid_chroms.add("chrX")
    valid_chroms.add("chrY")

    for chrom in valid_chroms:
        if chrom in chrom_sizes:
            coverage[chrom] = np.zeros(chrom_sizes[chrom], dtype=np.float32)

    total_frags = 0
    for chunk in pd.read_csv(
        fragments_path,
        sep="\t",
        header=None,
        names=["chrom", "start", "end", "barcode", "count"],
        chunksize=chunk_size,
        dtype={"chrom": str, "start": int, "end": int, "barcode": str, "count": int},
    ):
        for chrom in chunk["chrom"].unique():
            if chrom not in coverage:
                continue

            mask = chunk["chrom"] == chrom
            starts = chunk.loc[mask, "start"].values
            ends = chunk.loc[mask, "end"].values
            cov = coverage[chrom]
            chrom_len = len(cov)

            # Tn5-corrected left cut sites (+4)
            left_sites = starts + 4
            left_valid = (left_sites >= 0) & (left_sites < chrom_len)
            np.add.at(cov, left_sites[left_valid], 1)

            # Tn5-corrected right cut sites (-5)
            right_sites = ends - 5
            right_valid = (right_sites >= 0) & (right_sites < chrom_len)
            np.add.at(cov, right_sites[right_valid], 1)

        total_frags += len(chunk)

    print(f"  Processed {total_frags:,} fragments")
    return coverage


def write_bigwig(coverage: dict, output_path: str, chrom_sizes: dict):
    """Write coverage arrays to a bigWig file.

    Args:
        coverage: Dict mapping chromosome -> numpy array of values.
        output_path: Path for the output bigWig.
        chrom_sizes: Dict mapping chromosome -> size.
    """
    bw = pyBigWig.open(output_path, "w")

    # Header: only chromosomes present in coverage, sorted
    header = sorted(
        [(c, chrom_sizes[c]) for c in coverage if c in chrom_sizes],
        key=lambda x: x[0],
    )
    bw.addHeader(header)

    for chrom, _ in header:
        cov = coverage[chrom]
        nonzero_idx = np.nonzero(cov)[0]
        if len(nonzero_idx) == 0:
            continue
        bw.addEntries(
            [chrom] * len(nonzero_idx),
            nonzero_idx.tolist(),
            ends=(nonzero_idx + 1).tolist(),
            values=cov[nonzero_idx].tolist(),
        )

    bw.close()


def create_celltype_bigwig(
    fragments_path: str,
    output_path: str,
    chrom_sizes_path: str,
):
    """Create a Tn5-corrected bigWig from a cell-type fragment BED file.

    Args:
        fragments_path: Path to fragment BED file.
        output_path: Path for the output bigWig.
        chrom_sizes_path: Path to chromosome sizes file.
    """
    chrom_sizes = load_chrom_sizes(chrom_sizes_path)
    print(f"Processing {fragments_path}...")
    coverage = fragments_to_cutsites(fragments_path, chrom_sizes)
    print(f"Writing {output_path}...")
    write_bigwig(coverage, output_path, chrom_sizes)
    print("Done.")


def create_merged_bigwig(
    fragment_paths: list,
    output_path: str,
    chrom_sizes_path: str,
):
    """Create a merged bigWig from multiple fragment BED files.

    Args:
        fragment_paths: List of paths to fragment BED files.
        output_path: Path for the output merged bigWig.
        chrom_sizes_path: Path to chromosome sizes file.
    """
    chrom_sizes = load_chrom_sizes(chrom_sizes_path)
    merged_coverage = {}

    for fpath in fragment_paths:
        print(f"Processing {fpath}...")
        coverage = fragments_to_cutsites(fpath, chrom_sizes)

        for chrom, cov in coverage.items():
            if chrom not in merged_coverage:
                merged_coverage[chrom] = cov.copy()
            else:
                merged_coverage[chrom] += cov

    print(f"Writing merged bigWig to {output_path}...")
    write_bigwig(merged_coverage, output_path, chrom_sizes)
    print("Done.")
