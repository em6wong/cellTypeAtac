"""Generate training zarr datasets from bigWig files and reference genome.

For each cell type, creates regions (peak + background) with:
  - One-hot encoded DNA sequence (4, input_length)
  - Base-resolution profile from bigWig (output_length,)
  - Total count scalar
  - Peak mask indicating peak region within the window

Adapted from ../multiome/src/data/generate_dataset.py.
"""

import numpy as np
import pandas as pd
import zarr
import pyfaidx
import pyBigWig
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional


CHROM_SPLITS = {
    "test": ["chr8", "chr9"],
    "val": ["chr1"],
    "train": [
        "chr2", "chr3", "chr4", "chr5", "chr6", "chr7",
        "chr10", "chr11", "chr12", "chr13", "chr14", "chr15",
        "chr16", "chr17", "chr18", "chr19",
    ],
}

ONE_HOT = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1],
    "N": [0, 0, 0, 0],
}


def one_hot_encode(sequence: str) -> np.ndarray:
    """One-hot encode a DNA sequence.

    Args:
        sequence: DNA string (ACGTN).

    Returns:
        Array of shape (4, len(sequence)).
    """
    encoded = np.zeros((4, len(sequence)), dtype=np.float32)
    for i, base in enumerate(sequence.upper()):
        if base in ONE_HOT:
            encoded[:, i] = ONE_HOT[base]
    return encoded


def load_regions(bed_path: str) -> pd.DataFrame:
    """Load peak regions from BED file."""
    return pd.read_csv(
        bed_path,
        sep="\t",
        header=None,
        usecols=[0, 1, 2],
        names=["chrom", "start", "end"],
    )


def extract_profile(
    bw: pyBigWig.pyBigWig,
    chrom: str,
    start: int,
    end: int,
) -> np.ndarray:
    """Extract base-resolution profile from bigWig.

    Args:
        bw: Open pyBigWig file handle.
        chrom: Chromosome name.
        start: Start position.
        end: End position.

    Returns:
        Array of shape (end - start,).
    """
    try:
        values = bw.values(chrom, max(0, start), end)
        values = np.nan_to_num(values, nan=0.0)
        return values.astype(np.float32)
    except Exception:
        return np.zeros(end - start, dtype=np.float32)


def _compute_gc_content(genome, chrom, start, end):
    """Compute GC fraction for a genomic region."""
    try:
        seq = str(genome[chrom][start:end]).upper()
        gc = sum(1 for b in seq if b in "GC")
        return gc / max(len(seq), 1)
    except (KeyError, ValueError):
        return 0.5


def sample_background_regions(
    peaks: pd.DataFrame,
    chrom_sizes: dict,
    n_samples: int,
    region_width: int = 2114,
    seed: int = 42,
    genome=None,
    gc_match: bool = False,
    gc_tolerance: float = 0.02,
) -> pd.DataFrame:
    """Sample background regions avoiding peaks, optionally GC-matched.

    When gc_match=True and genome is provided, samples regions whose GC
    content matches the distribution of peak regions (within gc_tolerance).
    This prevents the bias model from confounding GC content with Tn5
    sequence preference.

    Args:
        peaks: DataFrame with chrom, start, end.
        chrom_sizes: Dict mapping chromosome -> size.
        n_samples: Number of background regions to sample.
        region_width: Width of each region.
        seed: Random seed.
        genome: pyfaidx.Fasta genome for GC matching (optional).
        gc_match: Whether to match GC content of peak regions.
        gc_tolerance: Max GC fraction difference for matching.

    Returns:
        DataFrame with chrom, start, end for background regions.
    """
    rng = np.random.RandomState(seed)

    # Build blacklist from peaks (with padding)
    blacklist = set()
    for _, row in peaks.iterrows():
        chrom = row["chrom"]
        for pos in range(max(0, row["start"] - region_width), row["end"] + region_width, 1000):
            blacklist.add((chrom, pos // 1000))

    # Compute GC distribution of peaks for matching
    peak_gc_bins = None
    if gc_match and genome is not None:
        print("  Computing peak GC content for matching...")
        peak_gcs = []
        for _, row in peaks.iterrows():
            gc = _compute_gc_content(genome, row["chrom"], row["start"], row["end"])
            peak_gcs.append(gc)
        # Bin GC values into 0.02-wide bins for matching
        peak_gc_bins = np.round(np.array(peak_gcs) / gc_tolerance) * gc_tolerance
        # Count how many peaks per GC bin
        unique_bins, bin_counts = np.unique(peak_gc_bins, return_counts=True)
        # Target: sample proportional to peak GC distribution
        gc_target = dict(zip(unique_bins, (bin_counts / bin_counts.sum() * n_samples).astype(int)))
        gc_collected = {b: 0 for b in unique_bins}

    # Valid chromosomes
    valid_chroms = sorted([c for c in chrom_sizes if c.startswith("chr") and c[3:].isdigit()])
    chrom_weights = np.array([chrom_sizes[c] for c in valid_chroms], dtype=float)
    chrom_weights /= chrom_weights.sum()

    regions = []
    attempts = 0
    max_attempts = n_samples * 50 if gc_match else n_samples * 20

    while len(regions) < n_samples and attempts < max_attempts:
        chrom_idx = rng.choice(len(valid_chroms), p=chrom_weights)
        chrom = valid_chroms[chrom_idx]
        csize = chrom_sizes[chrom]

        start = rng.randint(0, max(1, csize - region_width))
        center_bin = (start + region_width // 2) // 1000

        if (chrom, center_bin) not in blacklist:
            if gc_match and genome is not None:
                gc = _compute_gc_content(genome, chrom, start, start + region_width)
                gc_bin = round(gc / gc_tolerance) * gc_tolerance
                if gc_bin in gc_target and gc_collected[gc_bin] < gc_target[gc_bin]:
                    regions.append({"chrom": chrom, "start": start, "end": start + region_width})
                    gc_collected[gc_bin] += 1
            else:
                regions.append({"chrom": chrom, "start": start, "end": start + region_width})

        attempts += 1

    if gc_match:
        print(f"  GC-matched: sampled {len(regions)}/{n_samples} regions")

    return pd.DataFrame(regions)


def _zarr_store(root, name: str, data: np.ndarray):
    """Create a zarr array from numpy data (zarr v2/v3 compatible)."""
    try:
        # zarr v3: create_array with shape, then assign
        arr = root.create_array(name, shape=data.shape, dtype=data.dtype)
        arr[:] = data
    except (TypeError, AttributeError):
        # zarr v2: create_dataset with data
        root.create_dataset(name, data=data)


def _zarr_zeros(root, name: str, shape, dtype, chunks):
    """Create a zero-filled zarr array (zarr v2/v3 compatible)."""
    try:
        # zarr v3: create_array with fill_value
        return root.create_array(name, shape=shape, dtype=dtype,
                                 chunks=chunks, fill_value=0)
    except (TypeError, AttributeError):
        # zarr v2: zeros convenience method
        return root.zeros(name, shape=shape, dtype=dtype, chunks=chunks)


def generate_dataset(
    peaks_bed: str,
    bigwig_path: str,
    genome_path: str,
    chrom_sizes_path: str,
    output_path: str,
    input_length: int = 2114,
    output_length: int = 1000,
    max_jitter: int = 0,
    include_background: bool = True,
    gc_match: bool = False,
    split: Optional[str] = None,
):
    """Generate a zarr training dataset for one cell type.

    Args:
        peaks_bed: Path to peaks BED file.
        bigwig_path: Path to cell-type bigWig file.
        genome_path: Path to reference genome FASTA.
        chrom_sizes_path: Path to chromosome sizes file.
        output_path: Path for output zarr store.
        input_length: DNA sequence input length (default 2114).
        output_length: Profile output length (default 1000).
        max_jitter: Random jitter in bp for training augmentation (default 0).
            When > 0, stores wider regions so ATACDataset can random-crop.
        include_background: Whether to include background regions.
        gc_match: Match background GC content to peak distribution.
        split: If set, only include regions from this split ('train', 'val', 'test').
    """
    # When jitter is enabled, store wider sequences and profiles
    stored_input_length = input_length + 2 * max_jitter
    stored_output_length = output_length + 2 * max_jitter

    # Load resources
    peaks = load_regions(peaks_bed)
    genome = pyfaidx.Fasta(genome_path)
    bw = pyBigWig.open(bigwig_path)

    chrom_sizes = {}
    with open(chrom_sizes_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            chrom_sizes[parts[0]] = int(parts[1])

    # Filter by split
    if split is not None:
        split_chroms = set(CHROM_SPLITS[split])
        peaks = peaks[peaks["chrom"].isin(split_chroms)].reset_index(drop=True)

    print(f"Peak regions: {len(peaks):,}")
    if max_jitter > 0:
        print(f"Jitter: {max_jitter}bp (stored: {stored_input_length}bp seq, "
              f"{stored_output_length}bp profile)")

    # Sample background regions
    if include_background:
        bg = sample_background_regions(
            peaks, chrom_sizes, n_samples=len(peaks),
            region_width=stored_input_length,
            genome=genome if gc_match else None,
            gc_match=gc_match,
        )
        if split is not None:
            split_chroms = set(CHROM_SPLITS[split])
            bg = bg[bg["chrom"].isin(split_chroms)].reset_index(drop=True)
        print(f"Background regions: {len(bg):,}")
    else:
        bg = pd.DataFrame(columns=["chrom", "start", "end"])

    # Combine all regions
    peaks["is_peak"] = True
    bg["is_peak"] = False
    all_regions = pd.concat([peaks, bg], ignore_index=True)
    n_regions = len(all_regions)
    print(f"Total regions: {n_regions:,}")

    # Create zarr store (compatible with zarr v2 and v3)
    root = zarr.open(output_path, mode="w")

    # Metadata arrays: create with explicit shape, then write data
    chrom_data = np.array(all_regions["chrom"].values, dtype="S10")
    start_data = all_regions["start"].values.astype(np.int64)
    end_data = all_regions["end"].values.astype(np.int64)
    is_peak_data = all_regions["is_peak"].values

    _zarr_store(root, "chrom", chrom_data)
    _zarr_store(root, "start", start_data)
    _zarr_store(root, "end", end_data)
    _zarr_store(root, "is_peak", is_peak_data)

    # Large arrays: create zero-filled, populate incrementally
    # Store wider regions when jitter is enabled
    seqs = _zarr_zeros(root, "sequences",
                       shape=(n_regions, 4, stored_input_length),
                       dtype=np.float32,
                       chunks=(100, 4, stored_input_length))
    profiles = _zarr_zeros(root, "profiles",
                           shape=(n_regions, stored_output_length),
                           dtype=np.float32,
                           chunks=(100, stored_output_length))
    counts = _zarr_zeros(root, "counts", shape=(n_regions,),
                         dtype=np.float32, chunks=(1000,))

    # Extract data for each region
    for idx in tqdm(range(n_regions), desc="Generating dataset"):
        row = all_regions.iloc[idx]
        chrom = row["chrom"]
        start = int(row["start"])
        end = int(row["end"])

        # Center the region
        center = (start + end) // 2
        seq_start = center - stored_input_length // 2
        seq_end = seq_start + stored_input_length
        profile_start = center - stored_output_length // 2
        profile_end = profile_start + stored_output_length

        # Clamp to chromosome bounds
        csize = chrom_sizes.get(chrom, seq_end + 1)
        seq_start = max(0, seq_start)
        seq_end = min(csize, seq_end)
        profile_start = max(0, profile_start)
        profile_end = min(csize, profile_end)

        # Extract sequence
        try:
            seq_str = str(genome[chrom][seq_start:seq_end])
            # Pad if near chromosome boundary
            if len(seq_str) < stored_input_length:
                seq_str = seq_str + "N" * (stored_input_length - len(seq_str))
            seqs[idx] = one_hot_encode(seq_str)
        except (KeyError, ValueError):
            pass

        # Extract profile
        profile = extract_profile(bw, chrom, profile_start, profile_end)
        if len(profile) < stored_output_length:
            profile = np.pad(profile, (0, stored_output_length - len(profile)))
        profiles[idx] = profile[:stored_output_length]
        counts[idx] = profile[:stored_output_length].sum()

    bw.close()

    print(f"Dataset saved to {output_path}")
    print(f"  Sequences: {seqs.shape}")
    print(f"  Profiles: {profiles.shape}")
    print(f"  Counts: {counts.shape}")
