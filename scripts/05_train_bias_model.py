#!/usr/bin/env python
"""Train the Tn5 sequence bias model on background regions.

Usage:
    python scripts/05_train_bias_model.py \
        --config configs/bias_model.yaml \
        --data-dir data/training/merged \
        --output-dir results/bias_model
"""

import argparse
from pathlib import Path
import yaml
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader

from src.models.chrombpnet import BiasModel
from src.training.trainer import BiasModelModule
from src.data.dataset import BiasDataset


def main():
    parser = argparse.ArgumentParser(description="Train bias model")
    parser.add_argument("--config", type=str, default="configs/bias_model.yaml")
    parser.add_argument("--data-dir", type=str, default="data/training/merged")
    parser.add_argument("--output-dir", type=str, default="results/bias_model")
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create model
    model_cfg = config["model"]
    model = BiasModel(
        input_length=model_cfg["input_length"],
        output_length=model_cfg["output_length"],
        channels=model_cfg["channels"],
        num_dilated_layers=model_cfg["num_dilated_layers"],
        profile_kernel_size=model_cfg["profile_kernel_size"],
        dropout=model_cfg["dropout"],
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Bias model parameters: {n_params:,}")

    # Create datasets
    train_cfg = config["training"]
    train_ds = BiasDataset(str(Path(args.data_dir) / "train.zarr"), split="train")
    val_ds = BiasDataset(str(Path(args.data_dir) / "val.zarr"), split="val")

    print(f"Train samples: {len(train_ds):,}")
    print(f"Val samples: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"],
        shuffle=True, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg["batch_size"],
        shuffle=False, num_workers=4, pin_memory=True,
    )

    # Lightning module
    module = BiasModelModule(
        model=model,
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
        profile_weight=train_cfg["profile_weight"],
        count_weight=train_cfg["count_weight"],
    )

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=out_dir,
        filename="best",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
    )
    early_stop_cb = EarlyStopping(
        monitor="val/loss",
        patience=train_cfg.get("early_stopping_patience", 5),
        mode="min",
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=train_cfg["max_epochs"],
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else 1,
        precision=train_cfg.get("precision", "16-mixed"),
        callbacks=[checkpoint_cb, early_stop_cb],
        default_root_dir=out_dir,
        log_every_n_steps=50,
    )

    # Train
    print("\nStarting bias model training...")
    trainer.fit(module, train_loader, val_loader)

    print(f"\nBest model saved to {checkpoint_cb.best_model_path}")
    print(f"Best val loss: {checkpoint_cb.best_model_score:.4f}")


if __name__ == "__main__":
    main()
