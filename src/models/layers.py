"""Neural network layers for ChromBPNet-style models.

Adapted from ../multiome/src/models/layers.py. Simplified for per-cell-type
ATAC prediction (Stage 2) and multi-cell-type conditioning (Stage 3).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolutional block with optional batch norm and dropout.

    Official ChromBPNet uses valid padding (no padding), NO batch norm,
    and NO dropout. Feature maps shrink at each layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        dropout: float = 0.0,
        activation: str = "relu",
        use_batch_norm: bool = False,
    ):
        super().__init__()
        # Valid padding (no padding) matching official ChromBPNet.
        # Output shrinks by (kernel_size - 1) * dilation.
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=0, dilation=dilation,
        )
        self.norm = nn.BatchNorm1d(out_channels) if use_batch_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class DilatedConvStack(nn.Module):
    """Stack of dilated convolutions with residual connections.

    Uses valid padding with cropped residuals, matching official ChromBPNet.
    Dilations start at 2^1=2 (not 2^0=1).
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        num_layers: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(1, num_layers + 1):
            dilation = 2 ** i
            self.layers.append(
                ConvBlock(channels, channels, kernel_size,
                          dilation=dilation, dropout=dropout)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            conv_out = layer(x)
            # Crop residual to match valid-conv output length
            crop = (x.size(2) - conv_out.size(2)) // 2
            x = x[:, :, crop:crop + conv_out.size(2)] + conv_out
        return x


class ProfileHead(nn.Module):
    """Head for base-resolution profile prediction.

    Uses a large kernel (75) with valid padding, matching official ChromBPNet.
    With correct input dimensions (2114bp, 8 dilated layers starting at d=2),
    the valid convolutions naturally produce exactly 1000bp output.
    """

    def __init__(
        self,
        in_channels: int,
        num_outputs: int = 1,
        output_length: int = 1000,
        kernel_size: int = 75,
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, num_outputs, kernel_size=kernel_size, padding=0)
        self.output_length = output_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)  # (batch, num_outputs, length)

        # Center crop if needed (safety — should be exactly output_length
        # with correct architecture dimensions)
        if x.size(2) > self.output_length:
            start = (x.size(2) - self.output_length) // 2
            x = x[:, :, start:start + self.output_length]

        return x


class CountHead(nn.Module):
    """Head for total count prediction (log-scale)."""

    def __init__(self, in_channels: int, num_outputs: int = 1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_channels, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x).squeeze(-1)  # (batch, channels)
        x = self.fc(x)                # (batch, num_outputs)
        return x


class CellTypeEmbedding(nn.Module):
    """Learnable embeddings for cell type conditioning."""

    def __init__(self, num_cell_types: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_cell_types, embedding_dim)

    def forward(self, cell_type_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(cell_type_ids)


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation for cell-type conditioning.

    Modulates conv features: output = gamma * x + beta,
    where gamma and beta are derived from the cell-type embedding.
    """

    def __init__(self, channels: int, embedding_dim: int):
        super().__init__()
        self.gamma = nn.Linear(embedding_dim, channels)
        self.beta = nn.Linear(embedding_dim, channels)

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, length)
        # embedding: (batch, embedding_dim)
        gamma = self.gamma(embedding).unsqueeze(-1)  # (batch, channels, 1)
        beta = self.beta(embedding).unsqueeze(-1)
        return gamma * x + beta


class FiLMDilatedConvStack(nn.Module):
    """Dilated conv stack with FiLM conditioning at each layer.

    Used in Stage 3 multi-cell-type model. Uses valid padding with
    cropped residuals, matching official ChromBPNet dilations (2^1..2^n).
    """

    def __init__(
        self,
        channels: int,
        embedding_dim: int,
        kernel_size: int = 3,
        num_layers: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        self.film_layers = nn.ModuleList()

        for i in range(1, num_layers + 1):
            dilation = 2 ** i
            self.conv_layers.append(
                ConvBlock(channels, channels, kernel_size,
                          dilation=dilation, dropout=dropout)
            )
            self.film_layers.append(FiLMLayer(channels, embedding_dim))

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        for conv, film in zip(self.conv_layers, self.film_layers):
            conv_out = conv(x)
            conv_out = film(conv_out, embedding)
            # Crop residual to match valid-conv output length
            crop = (x.size(2) - conv_out.size(2)) // 2
            x = x[:, :, crop:crop + conv_out.size(2)] + conv_out
        return x
