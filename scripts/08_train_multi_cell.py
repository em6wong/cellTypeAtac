#!/usr/bin/env python
"""Train multi-cell-type ChromBPNet model (Stage 3).

Initializes from the best per-cell-type model and uses FiLM conditioning
with cell-type embeddings. Anti-collapse training with Scooby-style
single cell type per forward pass.

Usage:
    python scripts/08_train_multi_cell.py \
        --config configs/chrombpnet.yaml \
        --init-from results/chrombpnet/Cardiomyocyte/best.ckpt \
        --output-dir results/multi_cell
"""

import argparse
from pathlib import Path
import yaml
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from torch.utils.data import DataLoader, ConcatDataset

from src.models.multi_cell_chrombpnet import MultiCellChromBPNet
from src.training.trainer import MultiCellModule
from src.data.dataset import MultiCellATACDataset


CELL_TYPES = ["Cardiomyocyte", "Coronary_EC", "Fibroblast", "Macrophage", "Pericytes"]


def main():
    parser = argparse.ArgumentParser(description="Train multi-cell-type ChromBPNet")
    parser.add_argument("--config", type=str, default="configs/chrombpnet.yaml")
    parser.add_argument("--init-from", type=str, default=None,
                        help="Initialize from per-cell-type checkpoint")
    parser.add_argument("--data-dir", type=str, default="data/training")
    parser.add_argument("--output-dir", type=str, default="results/multi_cell")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--single-ct-training", action="store_true", default=True,
                        help="Train one cell type per batch (Scooby-style)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = config["model"]
    train_cfg = config["training"]

    # Create model
    if args.init_from and Path(args.init_from).exists():
        print(f"Initializing from pre-trained model: {args.init_from}")
        model = MultiCellChromBPNet.from_pretrained_single(
            args.init_from,
            num_cell_types=len(CELL_TYPES),
            embedding_dim=128,
            input_length=model_cfg["input_length"],
            output_length=model_cfg["output_length"],
            stem_channels=model_cfg["stem_channels"],
            stem_kernel_size=model_cfg["stem_kernel_size"],
            num_dilated_layers=model_cfg["num_dilated_layers"],
            dilated_kernel_size=model_cfg["dilated_kernel_size"],
            profile_kernel_size=model_cfg["profile_kernel_size"],
            dropout=model_cfg["dropout"],
        )
    else:
        print("Training from scratch (no initialization)")
        model = MultiCellChromBPNet(
            num_cell_types=len(CELL_TYPES),
            embedding_dim=128,
            input_length=model_cfg["input_length"],
            output_length=model_cfg["output_length"],
            stem_channels=model_cfg["stem_channels"],
            stem_kernel_size=model_cfg["stem_kernel_size"],
            num_dilated_layers=model_cfg["num_dilated_layers"],
            dilated_kernel_size=model_cfg["dilated_kernel_size"],
            profile_kernel_size=model_cfg["profile_kernel_size"],
            dropout=model_cfg["dropout"],
        )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Datasets: load profiles from ALL cell types for the same regions
    data_dir = Path(args.data_dir)

    # Collect zarr paths for all cell types (must all exist)
    train_zarrs = []
    val_zarrs = []
    for ct in CELL_TYPES:
        ct_train = data_dir / ct / "train.zarr"
        ct_val = data_dir / ct / "val.zarr"
        if not ct_train.exists():
            print(f"ERROR: Missing training data for {ct}: {ct_train}")
            return
        train_zarrs.append(str(ct_train))
        if ct_val.exists():
            val_zarrs.append(str(ct_val))

    if len(val_zarrs) != len(CELL_TYPES):
        print("WARNING: Not all cell types have validation data, disabling validation")
        val_zarrs = []

    train_ds = MultiCellATACDataset(
        train_zarrs, split="train",
        augment_rc=train_cfg.get("reverse_complement_augment", True),
    )
    val_ds = MultiCellATACDataset(val_zarrs, split="val") if val_zarrs else None

    print(f"Cell types: {len(CELL_TYPES)}")
    print(f"Train samples: {len(train_ds):,}")
    if val_ds:
        print(f"Val samples: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"],
        shuffle=True, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg["batch_size"],
        shuffle=False, num_workers=4, pin_memory=True,
    ) if val_ds else None

    # Lightning module
    module = MultiCellModule(
        model=model,
        num_cell_types=len(CELL_TYPES),
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        warmup_steps=train_cfg["warmup_steps"],
        profile_weight=train_cfg["profile_weight"],
        count_weight=train_cfg["count_weight"],
        diff_weight=0.1,
        single_ct_training=args.single_ct_training,
    )

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=out_dir, filename="best",
        monitor="val/loss" if val_loader else "train/loss",
        mode="min", save_top_k=1,
    )
    early_stop_cb = EarlyStopping(
        monitor="val/loss" if val_loader else "train/loss",
        patience=train_cfg["early_stopping_patience"],
        mode="min",
    )
    lr_cb = LearningRateMonitor(logging_interval="step")

    # Logger
    logger = None
    wandb_cfg = config.get("wandb", {})
    if wandb_cfg.get("project"):
        try:
            from pytorch_lightning.loggers import WandbLogger
            logger = WandbLogger(
                project=wandb_cfg["project"],
                entity=wandb_cfg.get("entity"),
                name="multi_cell_chrombpnet",
                save_dir=str(out_dir),
            )
        except ImportError:
            pass

    # Trainer
    trainer = pl.Trainer(
        max_epochs=train_cfg["max_epochs"],
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else 1,
        precision=train_cfg.get("precision", "16-mixed"),
        callbacks=[checkpoint_cb, early_stop_cb, lr_cb],
        default_root_dir=str(out_dir),
        logger=logger,
        log_every_n_steps=50,
    )

    print("\nStarting multi-cell-type model training...")
    trainer.fit(module, train_loader, val_loader)

    print(f"\nBest model: {checkpoint_cb.best_model_path}")
    print(f"Best val loss: {checkpoint_cb.best_model_score:.4f}")


if __name__ == "__main__":
    main()
