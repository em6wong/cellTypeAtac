"""PyTorch Lightning trainer modules for ChromBPNet models.

Provides:
  - BiasModelModule: trains the Tn5 bias model
  - ChromBPNetModule: trains per-cell-type models with bias correction
  - MultiCellModule: trains multi-cell-type model (Stage 3)
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Dict, Optional
import numpy as np

from ..models.chrombpnet import ChromBPNet, BiasModel, ChromBPNetWithBias
from ..models.multi_cell_chrombpnet import MultiCellChromBPNet
from ..training.loss import ChromBPNetLoss
from ..data.dataset import ATACDataset, BiasDataset


def compute_dynamic_count_weight(dataset, n_samples: int = 5000) -> float:
    """Compute count loss weight as median(counts) / 10.

    This matches the official ChromBPNet implementation which dynamically
    sets the count loss weight based on the data distribution.

    Args:
        dataset: Dataset with 'count' field in samples.
        n_samples: Number of samples to estimate median from.

    Returns:
        Count weight value.
    """
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    counts = []
    for i in indices:
        sample = dataset[int(i)]
        counts.append(float(sample["count"]))
    median_count = float(np.median(counts))
    weight = max(median_count / 10.0, 0.01)  # floor to prevent near-zero weight
    return weight


class BiasModelModule(pl.LightningModule):
    """Lightning module for bias model training.

    Trains on background (non-peak) regions to learn Tn5 sequence preference.

    Args:
        model: BiasModel instance.
        learning_rate: Learning rate (default 0.001).
        weight_decay: Weight decay (default 0.0).
        profile_weight: Profile loss weight (default 1.0).
        count_weight: Count loss weight (default 0.5).
    """

    def __init__(
        self,
        model: BiasModel,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        profile_weight: float = 1.0,
        count_weight: float = 0.5,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_fn = ChromBPNetLoss(profile_weight, count_weight)
        self.save_hyperparameters(ignore=["model"])

    def forward(self, sequence):
        return self.model(sequence)

    def _shared_step(self, batch, prefix):
        out = self.model(batch["sequence"])
        losses = self.loss_fn(
            out["profile"], out["count"],
            batch["profile"], batch["count"],
        )
        for k, v in losses.items():
            self.log(f"{prefix}/{k}", v, prog_bar=(k == "loss"))
        return losses["loss"]

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )


class ChromBPNetModule(pl.LightningModule):
    """Lightning module for per-cell-type ChromBPNet training.

    Wraps ChromBPNetWithBias (main model + frozen bias model).

    Args:
        model: ChromBPNetWithBias instance.
        learning_rate: Learning rate (default 0.001, matching official).
        profile_weight: Profile loss weight (default 1.0).
        count_weight: Count loss weight (default 0.5).
    """

    def __init__(
        self,
        model: ChromBPNetWithBias,
        learning_rate: float = 0.001,
        profile_weight: float = 1.0,
        count_weight: float = 0.5,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss_fn = ChromBPNetLoss(profile_weight, count_weight)
        self.save_hyperparameters(ignore=["model"])

    def forward(self, sequence):
        return self.model(sequence)

    def _shared_step(self, batch, prefix):
        out = self.model(batch["sequence"])
        losses = self.loss_fn(
            out["profile"], out["count"],
            batch["profile"], batch["count"],
        )
        for k, v in losses.items():
            self.log(f"{prefix}/{k}", v, prog_bar=(k == "loss"))
        return losses["loss"]

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        # Official ChromBPNet uses Adam with constant LR (no scheduler)
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
        )


class MultiCellModule(pl.LightningModule):
    """Lightning module for multi-cell-type ChromBPNet (Stage 3).

    Single forward pass produces all cell type outputs. Uses:
    - Uncertainty weighting (Kendall et al. 2018): learns per-cell-type
      loss weights automatically via homoscedastic uncertainty
    - Orthogonality regularization: pushes profile head weights to be
      different across cell types

    Args:
        model: MultiCellChromBPNet instance.
        num_cell_types: Number of cell types.
        learning_rate: Learning rate (default 0.001, matching official).
        profile_weight: Profile loss weight (default 1.0).
        count_weight: Count loss weight (default 0.5).
        ortho_weight: Weight for orthogonality regularization (default 0.01).
    """

    def __init__(
        self,
        model: MultiCellChromBPNet,
        num_cell_types: int = 5,
        learning_rate: float = 0.001,
        profile_weight: float = 1.0,
        count_weight: float = 0.5,
        ortho_weight: float = 0.01,
    ):
        super().__init__()
        self.model = model
        self.num_cell_types = num_cell_types
        self.learning_rate = learning_rate
        self.ortho_weight = ortho_weight

        self.loss_fn = ChromBPNetLoss(profile_weight, count_weight)

        # Learnable per-cell-type log-variance for uncertainty weighting
        self.log_vars = nn.Parameter(torch.zeros(num_cell_types))

        self.save_hyperparameters(ignore=["model"])

    def forward(self, sequence):
        return self.model(sequence)

    def _ortho_loss(self):
        """Orthogonality regularization on profile head conv weights.

        Penalizes similarity between different cell types' profile filters,
        forcing each head to learn distinct feature combinations.
        """
        weights = []
        for head in self.model.profile_heads:
            weights.append(head.conv.weight.view(-1))  # flatten each head's weights
        W = torch.stack(weights)  # (n_ct, n_params)
        # Normalize rows for cosine-similarity-based orthogonality
        W_norm = W / (W.norm(dim=1, keepdim=True) + 1e-8)
        gram = W_norm @ W_norm.T  # (n_ct, n_ct)
        identity = torch.eye(self.num_cell_types, device=gram.device)
        return ((gram - identity) ** 2).sum()

    def _shared_step(self, batch, prefix):
        out = self.model(batch["sequence"])

        # Per-cell-type losses with uncertainty weighting
        total_loss = torch.tensor(0.0, device=batch["sequence"].device)
        for ct_idx in range(self.num_cell_types):
            losses = self.loss_fn(
                out["profile"][:, ct_idx, :],
                out["count"][:, ct_idx],
                batch["profile"][:, ct_idx],
                batch["count"][:, ct_idx],
            )
            # Uncertainty weighting: precision * loss + log_var
            precision = torch.exp(-self.log_vars[ct_idx])
            total_loss = total_loss + precision * losses["loss"] + self.log_vars[ct_idx]
        total_loss = total_loss / self.num_cell_types

        # Orthogonality regularization (only during training)
        if prefix == "train":
            ortho = self._ortho_loss()
            total_loss = total_loss + self.ortho_weight * ortho
            self.log("train/ortho_loss", ortho, prog_bar=False)

        self.log(f"{prefix}/loss", total_loss, prog_bar=True)
        return total_loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )
