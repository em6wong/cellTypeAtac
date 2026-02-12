"""Per-cell-type ChromBPNet model.

Architecture (following official Kundaje lab ChromBPNet):
    Input: 2114bp one-hot DNA sequence (4, 2114)
    → Stem: Conv1d(4→512, k=21, valid) + ReLU → 2094bp
    → Dilated Conv Stack: 8 layers, k=3, d=2..256, valid, cropped residual → 1074bp
    → Profile Head: Conv1d(512→1, k=75, valid) → 1000bp
    → Count Head: AdaptiveAvgPool1d(1) → Linear(512→1)

Bias model: Same architecture but smaller (128 channels, 4 layers).
Frozen during main model training. Combination:
    profile: additive in logit space (bias_logits + main_logits)
    count: logsumexp (additive in count space)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .layers import ConvBlock, DilatedConvStack, ProfileHead, CountHead


class ChromBPNet(nn.Module):
    """Per-cell-type ChromBPNet for base-resolution ATAC-seq prediction.

    Args:
        input_length: DNA sequence length (default 2114).
        output_length: Profile output length (default 1000).
        stem_channels: Number of channels after stem conv (default 512).
        stem_kernel_size: Stem convolution kernel size (default 21).
        num_dilated_layers: Number of dilated conv layers (default 8).
        dilated_kernel_size: Kernel size for dilated convs (default 3).
        profile_kernel_size: Kernel size for profile head (default 75).
        dropout: Dropout rate (default 0.0, official uses no dropout).
        use_batch_norm: Use batch norm (default False, official uses none).
    """

    def __init__(
        self,
        input_length: int = 2114,
        output_length: int = 1000,
        stem_channels: int = 512,
        stem_kernel_size: int = 21,
        num_dilated_layers: int = 8,
        dilated_kernel_size: int = 3,
        profile_kernel_size: int = 75,
        dropout: float = 0.0,
        use_batch_norm: bool = False,
    ):
        super().__init__()
        self.input_length = input_length
        self.output_length = output_length

        # Stem convolution
        self.stem = ConvBlock(
            4, stem_channels, stem_kernel_size,
            dropout=dropout, use_batch_norm=use_batch_norm,
        )

        # Dilated conv stack
        self.dilated_convs = DilatedConvStack(
            channels=stem_channels,
            kernel_size=dilated_kernel_size,
            num_layers=num_dilated_layers,
            dropout=dropout,
        )

        # Prediction heads
        self.profile_head = ProfileHead(
            stem_channels, num_outputs=1,
            output_length=output_length,
            kernel_size=profile_kernel_size,
        )
        self.count_head = CountHead(stem_channels, num_outputs=1)

    def forward(self, sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            sequence: One-hot encoded DNA (batch, 4, input_length).

        Returns:
            Dict with 'profile' (batch, output_length) and 'count' (batch,).
        """
        x = self.stem(sequence)
        x = self.dilated_convs(x)

        profile = self.profile_head(x).squeeze(1)  # (batch, output_length)
        count = self.count_head(x).squeeze(-1)      # (batch,)

        return {"profile": profile, "count": count}


class BiasModel(nn.Module):
    """Smaller ChromBPNet for learning Tn5 sequence bias.

    Trained on non-peak background regions. Uses fewer channels
    and layers than the main model. Official uses 128 channels.

    Args:
        input_length: DNA sequence length (default 2114).
        output_length: Profile output length (default 1000).
        channels: Number of channels (default 128, matching official).
        num_dilated_layers: Number of dilated conv layers (default 4).
        profile_kernel_size: Kernel size for profile head (default 75).
        dropout: Dropout rate (default 0.0).
    """

    def __init__(
        self,
        input_length: int = 2114,
        output_length: int = 1000,
        channels: int = 128,
        num_dilated_layers: int = 4,
        profile_kernel_size: int = 75,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_length = input_length
        self.output_length = output_length

        self.stem = ConvBlock(4, channels, kernel_size=21, dropout=dropout)
        self.dilated_convs = DilatedConvStack(
            channels=channels, kernel_size=3,
            num_layers=num_dilated_layers, dropout=dropout,
        )
        self.profile_head = ProfileHead(
            channels, num_outputs=1,
            output_length=output_length,
            kernel_size=profile_kernel_size,
        )
        self.count_head = CountHead(channels, num_outputs=1)

    def forward(self, sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(sequence)
        x = self.dilated_convs(x)

        profile = self.profile_head(x).squeeze(1)
        count = self.count_head(x).squeeze(-1)

        return {"profile": profile, "count": count}


class ChromBPNetWithBias(nn.Module):
    """ChromBPNet with frozen bias model correction.

    The bias model captures Tn5 sequence preferences. The main model
    learns the residual (true biological signal):
        total_prediction = bias_prediction + main_prediction

    During training, bias model is frozen.

    Args:
        main_model: The main ChromBPNet model.
        bias_model: Pre-trained bias model (will be frozen).
    """

    def __init__(self, main_model: ChromBPNet, bias_model: BiasModel):
        super().__init__()
        self.main_model = main_model
        self.bias_model = bias_model

        # Freeze bias model
        for param in self.bias_model.parameters():
            param.requires_grad = False
        self.bias_model.eval()

    def forward(self, sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with bias correction.

        Profile: additive in logit space (bias_logits + main_logits).
        Count: logsumexp = log(exp(bias) + exp(main)), i.e. additive in
               count space, NOT multiplicative (which plain addition would be).
        """
        # Bias prediction (no grad)
        with torch.no_grad():
            bias_out = self.bias_model(sequence)

        # Main model prediction
        main_out = self.main_model(sequence)

        # Profile: additive in logit space (same as official)
        total_profile = bias_out["profile"] + main_out["profile"]

        # Count: logsumexp (additive in count space, matching official)
        # log(exp(bias_logcount) + exp(main_logcount))
        total_count = torch.logsumexp(
            torch.stack([bias_out["count"], main_out["count"]], dim=-1),
            dim=-1,
        )

        return {
            "profile": total_profile,
            "count": total_count,
            "bias_profile": bias_out["profile"],
            "bias_count": bias_out["count"],
            "main_profile": main_out["profile"],
            "main_count": main_out["count"],
        }

    def train(self, mode=True):
        """Override to keep bias model in eval mode."""
        super().train(mode)
        self.bias_model.eval()
        return self
