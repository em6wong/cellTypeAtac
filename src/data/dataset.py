"""PyTorch Dataset classes for ATAC-seq training data.

Provides:
  - ATACDataset: loads from zarr, returns (sequence, profile, count) tuples
  - Supports reverse complement augmentation
  - Supports chromosome-based train/val/test splits
"""

import numpy as np
import zarr
import torch
from torch.utils.data import Dataset
from typing import Optional, List


CHROM_SPLITS = {
    "test": {"chr8", "chr9"},
    "val": {"chr1"},
    "train": {
        "chr2", "chr3", "chr4", "chr5", "chr6", "chr7",
        "chr10", "chr11", "chr12", "chr13", "chr14", "chr15",
        "chr16", "chr17", "chr18", "chr19",
    },
}


def reverse_complement(one_hot: np.ndarray) -> np.ndarray:
    """Reverse complement a one-hot encoded sequence.

    Args:
        one_hot: (4, length) one-hot encoded DNA.

    Returns:
        Reverse-complemented (4, length) array.
    """
    # Reverse along length axis, flip A<->T and C<->G
    return one_hot[::-1, ::-1].copy()


class ATACDataset(Dataset):
    """PyTorch Dataset for ATAC-seq zarr data.

    Each sample provides:
        sequence: (4, input_length) one-hot DNA
        profile: (output_length,) base-resolution target signal
        count: scalar total count
        is_peak: bool whether this region overlaps a peak

    Args:
        zarr_path: Path to zarr store.
        split: 'train', 'val', or 'test' for chromosome filtering.
        augment_rc: Apply reverse complement augmentation (50% prob).
    """

    def __init__(
        self,
        zarr_path: str,
        split: Optional[str] = None,
        augment_rc: bool = False,
    ):
        self.root = zarr.open(zarr_path, mode="r")
        self.augment_rc = augment_rc

        # Load metadata
        chroms = np.array([c.decode() if isinstance(c, bytes) else c
                          for c in self.root["chrom"][:]])

        # Filter by split
        if split is not None and split in CHROM_SPLITS:
            valid_chroms = CHROM_SPLITS[split]
            self.indices = np.where([c in valid_chroms for c in chroms])[0]
        else:
            self.indices = np.arange(len(chroms))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        real_idx = self.indices[idx]

        sequence = self.root["sequences"][real_idx]  # (4, input_length)
        profile = self.root["profiles"][real_idx]     # (output_length,)
        count = self.root["counts"][real_idx]          # scalar
        is_peak = bool(self.root["is_peak"][real_idx])

        # Reverse complement augmentation
        if self.augment_rc and np.random.random() < 0.5:
            sequence = reverse_complement(sequence)
            profile = profile[::-1].copy()

        return {
            "sequence": torch.from_numpy(sequence),
            "profile": torch.from_numpy(profile),
            "count": torch.tensor(count, dtype=torch.float32),
            "is_peak": torch.tensor(is_peak, dtype=torch.bool),
        }


class BiasDataset(Dataset):
    """Dataset for bias model training on background (non-peak) regions.

    Only returns samples where is_peak is False.

    Args:
        zarr_path: Path to zarr store.
        split: 'train', 'val', or 'test'.
    """

    def __init__(self, zarr_path: str, split: Optional[str] = None):
        self.root = zarr.open(zarr_path, mode="r")

        chroms = np.array([c.decode() if isinstance(c, bytes) else c
                          for c in self.root["chrom"][:]])
        is_peak = self.root["is_peak"][:]

        # Filter: non-peak regions in the right split
        mask = ~is_peak
        if split is not None and split in CHROM_SPLITS:
            valid_chroms = CHROM_SPLITS[split]
            chrom_mask = np.array([c in valid_chroms for c in chroms])
            mask = mask & chrom_mask

        self.indices = np.where(mask)[0]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        real_idx = self.indices[idx]

        sequence = self.root["sequences"][real_idx]
        profile = self.root["profiles"][real_idx]
        count = self.root["counts"][real_idx]

        return {
            "sequence": torch.from_numpy(sequence),
            "profile": torch.from_numpy(profile),
            "count": torch.tensor(count, dtype=torch.float32),
        }
