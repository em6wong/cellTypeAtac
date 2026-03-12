"""Multi-cell-type ChromBPNet model (Stage 3).

Shared encoder with multi-output heads (Enformer/Scooby-style).
No cell-type conditioning in the encoder — the encoder learns general
DNA sequence features, and each head channel specializes for one cell type.

Supports variants via constructor flags:
  - separate_heads: independent ProfileHead/CountHead per cell type
  - task_norms: per-cell-type GroupNorm before heads

Architecture (default):
    Input: 2114bp one-hot DNA (4, 2114)
    → Stem: Conv1d(4→512, k=21, valid) + ReLU
    → Dilated Conv Stack: 8 layers (d=2..256), same as single-cell ChromBPNet
    → Profile Head: Conv1d(512→n_cell_types, k=75, valid)
    → Count Head: AdaptiveAvgPool1d(1) → Linear(512→n_cell_types)
"""

import torch
import torch.nn as nn
from typing import Dict

from .layers import ConvBlock, DilatedConvStack, ProfileHead, CountHead


class MultiCellChromBPNet(nn.Module):
    """Multi-cell-type ChromBPNet with shared encoder and output heads.

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
        separate_heads: If True, use independent heads per cell type.
        task_norms: If True, add per-cell-type GroupNorm before heads.
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
        separate_heads: bool = False,
        task_norms: bool = False,
    ):
        super().__init__()
        self.num_cell_types = num_cell_types
        self.input_length = input_length
        self.output_length = output_length
        self.use_separate_heads = separate_heads
        self.use_task_norms = task_norms

        # Shared encoder (same as single-cell ChromBPNet)
        self.stem = ConvBlock(4, stem_channels, stem_kernel_size, dropout=dropout)
        self.dilated_convs = DilatedConvStack(
            channels=stem_channels,
            kernel_size=dilated_kernel_size,
            num_layers=num_dilated_layers,
            dropout=dropout,
        )

        # Task-specific normalization (optional)
        if task_norms:
            self.task_norms = nn.ModuleList([
                nn.GroupNorm(1, stem_channels) for _ in range(num_cell_types)
            ])

        # Output heads
        if separate_heads:
            self.profile_heads = nn.ModuleList([
                ProfileHead(stem_channels, num_outputs=1,
                            output_length=output_length, kernel_size=profile_kernel_size)
                for _ in range(num_cell_types)
            ])
            self.count_heads = nn.ModuleList([
                CountHead(stem_channels, num_outputs=1)
                for _ in range(num_cell_types)
            ])
        else:
            self.profile_head = ProfileHead(
                stem_channels, num_outputs=num_cell_types,
                output_length=output_length, kernel_size=profile_kernel_size,
            )
            self.count_head = CountHead(stem_channels, num_outputs=num_cell_types)

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

        if self.use_separate_heads:
            profiles = []
            counts = []
            for ct_idx in range(self.num_cell_types):
                x_ct = self.task_norms[ct_idx](x) if self.use_task_norms else x
                profiles.append(self.profile_heads[ct_idx](x_ct).squeeze(1))
                counts.append(self.count_heads[ct_idx](x_ct).squeeze(-1))
            profile = torch.stack(profiles, dim=1)
            count = torch.stack(counts, dim=1)
        else:
            if self.use_task_norms:
                # With shared heads + task norms: run head once per norm
                profiles = []
                counts = []
                for ct_idx in range(self.num_cell_types):
                    x_ct = self.task_norms[ct_idx](x)
                    # Use only this cell type's channel from the shared head
                    p = self.profile_head(x_ct)[:, ct_idx, :]
                    c = self.count_head(x_ct)[:, ct_idx]
                    profiles.append(p)
                    counts.append(c)
                profile = torch.stack(profiles, dim=1)
                count = torch.stack(counts, dim=1)
            else:
                profile = self.profile_head(x)
                count = self.count_head(x)

        return {"profile": profile, "count": count}

    def forward_single_celltype(
        self,
        sequence: torch.Tensor,
        cell_type_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass returning a single cell type's output."""
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
