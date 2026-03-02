"""Multi-cell-type ChromBPNet model (Stage 3).

Shared encoder with multi-output heads (Enformer/Scooby-style).
No cell-type conditioning in the encoder — the encoder learns general
DNA sequence features, and each head channel specializes for one cell type.

Architecture:
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
    """Multi-cell-type ChromBPNet with shared encoder and multi-output heads.

    Same architecture as single-cell ChromBPNet but with n_cell_types
    output channels instead of 1. All cell types share the encoder;
    each head channel specializes for one cell type.

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

        # Multi-output heads
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
        profile = self.profile_head(x)  # (batch, n_cell_types, output_length)
        count = self.count_head(x)       # (batch, n_cell_types)
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

    @classmethod
    def from_pretrained_single(
        cls,
        checkpoint_path: str,
        num_cell_types: int = 5,
        **kwargs,
    ) -> "MultiCellChromBPNet":
        """Initialize encoder from a pre-trained single-cell ChromBPNet.

        Transfers stem and dilated conv weights. Heads are randomly
        initialized (use init_heads_from_models() to transfer heads).

        Args:
            checkpoint_path: Path to single-cell-type model checkpoint.
            num_cell_types: Number of cell types.
            **kwargs: Additional model arguments.

        Returns:
            MultiCellChromBPNet with transferred encoder weights.
        """
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # Strip Lightning module prefix
        cleaned = {}
        for k, v in state_dict.items():
            if k.startswith("model.main_model."):
                cleaned[k.replace("model.main_model.", "")] = v
            elif k.startswith("model."):
                cleaned[k.replace("model.", "")] = v
            else:
                cleaned[k] = v
        state_dict = cleaned

        model = cls(num_cell_types=num_cell_types, **kwargs)

        # Transfer stem
        stem_keys = {k: v for k, v in state_dict.items() if k.startswith("stem.")}
        if stem_keys:
            model.stem.load_state_dict(
                {k.replace("stem.", "", 1): v for k, v in stem_keys.items()},
                strict=False,
            )
            print(f"  Transferred {len(stem_keys)} stem parameters")

        # Transfer dilated conv weights
        n_transferred = 0
        for i, layer in enumerate(model.dilated_convs.layers):
            src_prefix = f"dilated_convs.layers.{i}."
            matching = {
                k.replace(src_prefix, ""): v
                for k, v in state_dict.items()
                if k.startswith(src_prefix)
            }
            if matching:
                layer.load_state_dict(matching, strict=False)
                n_transferred += len(matching)
        print(f"  Transferred {n_transferred} dilated conv parameters")

        return model
