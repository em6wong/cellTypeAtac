"""Neural network layers for ChromBPNet-style models.

Adapted from ../multiome/src/models/layers.py. Simplified for per-cell-type
ATAC prediction (Stage 2) and multi-cell-type conditioning (Stage 3).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolutional block with batch norm and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation,
        )
        self.norm = nn.BatchNorm1d(out_channels)
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
    """Stack of dilated convolutions with residual connections."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        num_layers: int = 9,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            dilation = 2 ** i
            self.layers.append(
                ConvBlock(channels, channels, kernel_size,
                          dilation=dilation, dropout=dropout)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + layer(x)
        return x


class ProfileHead(nn.Module):
    """Head for base-resolution profile prediction.

    Uses a large kernel (75) following ChromBPNet design.
    """

    def __init__(
        self,
        in_channels: int,
        num_outputs: int = 1,
        output_length: int = 1000,
        kernel_size: int = 75,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, num_outputs, kernel_size=kernel_size, padding=padding)
        self.output_length = output_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)  # (batch, num_outputs, length)

        # Center crop to output_length
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

    Used in Stage 3 multi-cell-type model.
    """

    def __init__(
        self,
        channels: int,
        embedding_dim: int,
        kernel_size: int = 3,
        num_layers: int = 9,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        self.film_layers = nn.ModuleList()

        for i in range(num_layers):
            dilation = 2 ** i
            self.conv_layers.append(
                ConvBlock(channels, channels, kernel_size,
                          dilation=dilation, dropout=dropout)
            )
            self.film_layers.append(FiLMLayer(channels, embedding_dim))

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        for conv, film in zip(self.conv_layers, self.film_layers):
            residual = x
            x = conv(x)
            x = film(x, embedding)
            x = x + residual
        return x
