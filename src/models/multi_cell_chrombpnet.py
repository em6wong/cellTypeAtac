"""Multi-cell-type ChromBPNet model (Stage 3).

Uses FiLM conditioning with cell-type embeddings to predict profiles
for all cell types from a shared encoder. Designed with anti-collapse
measures informed by the multiome project's lessons.

Architecture:
    Input: 2114bp one-hot DNA (4, 2114) + cell_type_id
    → Stem: Conv1d(4→512, k=21, valid) + ReLU
    → FiLM-conditioned Dilated Conv Stack: 8 layers (d=2..256) with cell-type FiLM
    → Profile Head: Conv1d(512→n_cell_types, k=75, valid)
    → Count Head: AdaptiveAvgPool1d(1) → Linear(512→n_cell_types)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .layers import (
    ConvBlock, FiLMDilatedConvStack, ProfileHead, CountHead, CellTypeEmbedding,
)


class MultiCellChromBPNet(nn.Module):
    """Multi-cell-type ChromBPNet with FiLM conditioning.

    Anti-collapse measures:
      1. FiLM conditioning at every dilated conv layer
      2. Support for single-cell-type training per forward pass (Scooby-style)
      3. Differential loss penalty on collapsed predictions
      4. Can initialize from pre-trained per-cell-type model

    Args:
        num_cell_types: Number of cell types (default 5).
        embedding_dim: Cell type embedding dimension (default 128).
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
        embedding_dim: int = 128,
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

        # Cell type embeddings
        self.cell_type_embedding = CellTypeEmbedding(num_cell_types, embedding_dim)

        # Shared stem
        self.stem = ConvBlock(4, stem_channels, stem_kernel_size, dropout=dropout)

        # FiLM-conditioned dilated conv stack
        self.dilated_convs = FiLMDilatedConvStack(
            channels=stem_channels,
            embedding_dim=embedding_dim,
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
        cell_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            sequence: One-hot DNA (batch, 4, input_length).
            cell_type_ids: Optional cell type indices (batch,).
                If provided, uses FiLM conditioning for that cell type
                and returns predictions for all cell types.
                If None, returns predictions for all cell types using
                a mean embedding (useful for evaluation).

        Returns:
            Dict with:
                'profile': (batch, n_cell_types, output_length)
                'count': (batch, n_cell_types)
        """
        batch_size = sequence.size(0)

        # Get cell-type embedding
        if cell_type_ids is not None:
            embedding = self.cell_type_embedding(cell_type_ids)  # (batch, emb_dim)
        else:
            # Use mean embedding for evaluation
            all_ids = torch.arange(self.num_cell_types, device=sequence.device)
            all_emb = self.cell_type_embedding(all_ids)  # (n_ct, emb_dim)
            embedding = all_emb.mean(dim=0).unsqueeze(0).expand(batch_size, -1)

        # Shared stem
        x = self.stem(sequence)

        # FiLM-conditioned dilated convs
        x = self.dilated_convs(x, embedding)

        # Prediction heads
        profile = self.profile_head(x)  # (batch, n_cell_types, output_length)
        count = self.count_head(x)       # (batch, n_cell_types)

        return {"profile": profile, "count": count}

    def forward_single_celltype(
        self,
        sequence: torch.Tensor,
        cell_type_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for a single cell type (Scooby-style training).

        During training, process one random cell type per batch to prevent
        the model from learning collapsed (identical) predictions.

        Args:
            sequence: One-hot DNA (batch, 4, input_length).
            cell_type_idx: Index of the cell type to predict.

        Returns:
            Dict with 'profile' (batch, output_length) and 'count' (batch,).
        """
        cell_type_ids = torch.full(
            (sequence.size(0),), cell_type_idx,
            dtype=torch.long, device=sequence.device,
        )
        out = self.forward(sequence, cell_type_ids)

        return {
            "profile": out["profile"][:, cell_type_idx, :],
            "count": out["count"][:, cell_type_idx],
        }

    @classmethod
    def from_pretrained_single(
        cls,
        checkpoint_path: str,
        num_cell_types: int = 5,
        embedding_dim: int = 128,
        **kwargs,
    ) -> "MultiCellChromBPNet":
        """Initialize from a pre-trained per-cell-type ChromBPNet.

        Transfers the shared conv stack weights from the best single-cell-type
        model. Profile and count heads are re-initialized.

        Args:
            checkpoint_path: Path to single-cell-type model checkpoint.
            num_cell_types: Number of cell types for the multi-cell model.
            embedding_dim: Cell type embedding dimension.
            **kwargs: Additional model arguments.

        Returns:
            MultiCellChromBPNet with transferred weights.
        """
        # Load single-cell-type checkpoint
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # Strip Lightning module prefix: checkpoint keys are like
        # "model.main_model.stem.*" from ChromBPNetModule -> ChromBPNetWithBias -> ChromBPNet
        cleaned = {}
        for k, v in state_dict.items():
            # Try stripping "model.main_model." (full ChromBPNetWithBias path)
            if k.startswith("model.main_model."):
                cleaned[k.replace("model.main_model.", "")] = v
            # Fallback: strip just "model." (direct ChromBPNet)
            elif k.startswith("model."):
                cleaned[k.replace("model.", "")] = v
            else:
                cleaned[k] = v
        state_dict = cleaned

        # Create multi-cell model
        model = cls(num_cell_types=num_cell_types, embedding_dim=embedding_dim, **kwargs)

        # Transfer stem weights
        stem_keys = {k: v for k, v in state_dict.items() if k.startswith("stem.")}
        if stem_keys:
            model.stem.load_state_dict(
                {k.replace("stem.", "", 1): v for k, v in stem_keys.items()},
                strict=False,
            )
            print(f"  Transferred {len(stem_keys)} stem parameters")
        else:
            print("  WARNING: No stem weights found in checkpoint")

        # Transfer dilated conv weights (conv layers only, not FiLM)
        n_transferred = 0
        for i, conv_layer in enumerate(model.dilated_convs.conv_layers):
            src_prefix = f"dilated_convs.layers.{i}."
            matching = {
                k.replace(src_prefix, ""): v
                for k, v in state_dict.items()
                if k.startswith(src_prefix)
            }
            if matching:
                conv_layer.load_state_dict(matching, strict=False)
                n_transferred += len(matching)
        print(f"  Transferred {n_transferred} dilated conv parameters")

        return model
