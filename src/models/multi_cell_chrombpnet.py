"""Multi-cell-type ChromBPNet model (Stage 3).

Shared encoder with separate per-cell-type output heads.
Each cell type gets its own ProfileHead and CountHead, allowing
independent feature selection from the shared encoder.
Task-specific normalization (GroupNorm) before each head gives
each cell type a different view of the shared features.

Architecture:
    Input: 2114bp one-hot DNA (4, 2114)
    → Stem: Conv1d(4→512, k=21, valid) + ReLU
    → Dilated Conv Stack: 8 layers (d=2..256), same as single-cell ChromBPNet
    → Per-cell-type GroupNorm(1, 512)
    → Per-cell-type ProfileHead: Conv1d(512→1, k=75, valid)
    → Per-cell-type CountHead: AdaptiveAvgPool1d(1) → Linear(512→1)
"""

import torch
import torch.nn as nn
from typing import Dict

from .layers import ConvBlock, DilatedConvStack, ProfileHead, CountHead


class MultiCellChromBPNet(nn.Module):
    """Multi-cell-type ChromBPNet with shared encoder and separate heads.

    Each cell type gets:
    - Its own GroupNorm (task-specific normalization)
    - Its own ProfileHead (512→1, k=75)
    - Its own CountHead (512→1)

    This is strictly more expressive than a single multi-channel head,
    and task-specific normalization allows each cell type to amplify
    different features from the shared encoder.

    Args:
        num_cell_types: Number of cell types (default 5).
        input_length: DNA sequence length (default 2114).
        output_length: Profile output length (default 1000).
        stem_channels: Number of channels (default 512).
        stem_kernel_size: Stem kernel size (default 21).
        num_dilated_layers: Number of dilated layers (default 8).
        dilated_kernel_size: Dilated conv kernel size (default 3).
        profile_kernel_size: Profile head kernel size (default 75).
        dropout: Dropout rate (default 0.0).
    """

    def __init__(
        self,
        num_cell_types: int = 5,
        input_length: int = 2114,
        output_length: int = 1000,
        stem_channels: int = 512,
        stem_kernel_size: int = 21,
        num_dilated_layers: int = 8,
        dilated_kernel_size: int = 3,
        profile_kernel_size: int = 75,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_cell_types = num_cell_types
        self.input_length = input_length
        self.output_length = output_length

        # Shared encoder (same as single-cell ChromBPNet)
        self.stem = ConvBlock(4, stem_channels, stem_kernel_size, dropout=dropout)
        self.dilated_convs = DilatedConvStack(
            channels=stem_channels,
            kernel_size=dilated_kernel_size,
            num_layers=num_dilated_layers,
            dropout=dropout,
        )

        # Per-cell-type normalization + heads
        self.task_norms = nn.ModuleList([
            nn.GroupNorm(1, stem_channels) for _ in range(num_cell_types)
        ])
        self.profile_heads = nn.ModuleList([
            ProfileHead(stem_channels, num_outputs=1,
                        output_length=output_length, kernel_size=profile_kernel_size)
            for _ in range(num_cell_types)
        ])
        self.count_heads = nn.ModuleList([
            CountHead(stem_channels, num_outputs=1)
            for _ in range(num_cell_types)
        ])

    def forward(
        self,
        sequence: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass producing all cell type outputs.

        Args:
            sequence: One-hot DNA (batch, 4, input_length).

        Returns:
            Dict with:
                'profile': (batch, n_cell_types, output_length)
                'count': (batch, n_cell_types)
        """
        x = self.stem(sequence)
        x = self.dilated_convs(x)

        profiles = []
        counts = []
        for ct_idx in range(self.num_cell_types):
            x_ct = self.task_norms[ct_idx](x)
            # Squeeze single output channel: (batch, 1, length) → (batch, length)
            profiles.append(self.profile_heads[ct_idx](x_ct).squeeze(1))
            # Squeeze single output: (batch, 1) → (batch,)
            counts.append(self.count_heads[ct_idx](x_ct).squeeze(-1))

        profile = torch.stack(profiles, dim=1)  # (batch, n_ct, output_length)
        count = torch.stack(counts, dim=1)       # (batch, n_ct)
        return {"profile": profile, "count": count}

    def forward_single_celltype(
        self,
        sequence: torch.Tensor,
        cell_type_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass returning a single cell type's output.

        Args:
            sequence: One-hot DNA (batch, 4, input_length).
            cell_type_idx: Index of the cell type to return.

        Returns:
            Dict with 'profile' (batch, output_length) and 'count' (batch,).
        """
        out = self.forward(sequence)
        return {
            "profile": out["profile"][:, cell_type_idx, :],
            "count": out["count"][:, cell_type_idx],
        }

    def forward_all_celltypes(
        self,
        sequence: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Alias for forward() — returns all cell types."""
        return self.forward(sequence)
