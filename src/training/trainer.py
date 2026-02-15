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
from ..training.loss import ChromBPNetLoss, MultiCellChromBPNetLoss
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

    Supports two training modes:
      1. Single cell type per batch (Scooby-style): randomly selects one
         cell type per forward pass to prevent collapse.
      2. All cell types: standard multi-output training.

    Args:
        model: MultiCellChromBPNet instance.
        num_cell_types: Number of cell types.
        learning_rate: Learning rate (default 1e-4).
        weight_decay: Weight decay (default 1e-5).
        warmup_steps: Linear warmup steps (default 1000).
        profile_weight: Profile loss weight (default 1.0).
        count_weight: Count loss weight (default 0.5).
        diff_weight: Differential penalty weight (default 0.1).
        single_ct_training: Use single cell type per batch (default True).
    """

    def __init__(
        self,
        model: MultiCellChromBPNet,
        num_cell_types: int = 5,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        warmup_steps: int = 1000,
        profile_weight: float = 1.0,
        count_weight: float = 0.5,
        diff_weight: float = 0.1,
        single_ct_training: bool = True,
    ):
        super().__init__()
        self.model = model
        self.num_cell_types = num_cell_types
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.single_ct_training = single_ct_training

        self.single_loss_fn = ChromBPNetLoss(profile_weight, count_weight)
        self.multi_loss_fn = MultiCellChromBPNetLoss(
            profile_weight, count_weight, diff_weight,
        )
        self.save_hyperparameters(ignore=["model"])

    def forward(self, sequence, cell_type_ids=None):
        return self.model(sequence, cell_type_ids)

    def training_step(self, batch, batch_idx):
        if self.single_ct_training:
            # Random cell type per batch (Scooby-style)
            ct_idx = torch.randint(0, self.num_cell_types, (1,)).item()
            out = self.model.forward_single_celltype(batch["sequence"], ct_idx)
            losses = self.single_loss_fn(
                out["profile"], out["count"],
                batch["profile"][:, ct_idx],
                batch["count"][:, ct_idx],
            )
        else:
            out = self.model(batch["sequence"])
            losses = self.multi_loss_fn(
                out["profile"], out["count"],
                batch["profile"], batch["count"],
            )

        for k, v in losses.items():
            self.log(f"train/{k}", v, prog_bar=(k == "loss"))
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        # Validate using same strategy as training
        if self.single_ct_training:
            ct_idx = torch.randint(0, self.num_cell_types, (1,)).item()
            out = self.model.forward_single_celltype(batch["sequence"], ct_idx)
            losses = self.single_loss_fn(
                out["profile"], out["count"],
                batch["profile"][:, ct_idx],
                batch["count"][:, ct_idx],
            )
        else:
            out = self.model(batch["sequence"])
            losses = self.multi_loss_fn(
                out["profile"], out["count"],
                batch["profile"], batch["count"],
            )

        for k, v in losses.items():
            self.log(f"val/{k}", v, prog_bar=(k == "loss"))
        return losses["loss"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            progress = (step - self.warmup_steps) / max(1, self.trainer.estimated_stepping_batches - self.warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * min(progress, 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
