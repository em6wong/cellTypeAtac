#!/usr/bin/env python
"""Train per-cell-type ChromBPNet models with bias correction.

Trains 5 independent models (one per cell type). Each model uses the
frozen bias model for Tn5 correction.

Usage:
    python scripts/06_train_chrombpnet.py \
        --config configs/chrombpnet.yaml \
        --cell-type Cardiomyocyte \
        --output-dir results/chrombpnet

    # Or train all cell types:
    python scripts/06_train_chrombpnet.py --config configs/chrombpnet.yaml --all
"""

import argparse
from pathlib import Path
import yaml
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from torch.utils.data import DataLoader

from src.models.chrombpnet import ChromBPNet, BiasModel, ChromBPNetWithBias
from src.training.trainer import ChromBPNetModule, compute_dynamic_count_weight
from src.data.dataset import ATACDataset


CELL_TYPES = ["Cardiomyocyte", "Coronary_EC", "Fibroblast", "Macrophage", "Pericytes"]


def train_cell_type(cell_type: str, config: dict, output_dir: Path, gpus: int):
    """Train ChromBPNet for one cell type."""
    print(f"\n{'='*60}")
    print(f"Training ChromBPNet for: {cell_type}")
    print(f"{'='*60}")

    model_cfg = config["model"]
    train_cfg = config["training"]

    ct_dir = output_dir / cell_type
    ct_dir.mkdir(parents=True, exist_ok=True)

    # Remove old checkpoints to prevent Lightning version suffix issue
    for old_ckpt in ct_dir.glob("best*.ckpt"):
        print(f"Removing old checkpoint: {old_ckpt}")
        old_ckpt.unlink()

    # Create main model
    main_model = ChromBPNet(
        input_length=model_cfg["input_length"],
        output_length=model_cfg["output_length"],
        stem_channels=model_cfg["stem_channels"],
        stem_kernel_size=model_cfg["stem_kernel_size"],
        num_dilated_layers=model_cfg["num_dilated_layers"],
        dilated_kernel_size=model_cfg["dilated_kernel_size"],
        profile_kernel_size=model_cfg["profile_kernel_size"],
        dropout=model_cfg["dropout"],
    )

    # Load pre-trained bias model
    bias_cfg = config["bias_model"]
    bias_model = BiasModel(
        input_length=model_cfg["input_length"],
        output_length=model_cfg["output_length"],
    )

    bias_ckpt = bias_cfg["checkpoint"]
    if Path(bias_ckpt).exists():
        print(f"Loading bias model from {bias_ckpt}")
        state = torch.load(bias_ckpt, map_location="cpu", weights_only=False)
        if "state_dict" in state:
            # Lightning checkpoint
            bias_state = {k.replace("model.", ""): v
                          for k, v in state["state_dict"].items()
                          if k.startswith("model.")}
            bias_model.load_state_dict(bias_state)
        else:
            bias_model.load_state_dict(state)
    else:
        print(f"WARNING: Bias model checkpoint not found: {bias_ckpt}")
        print("Training without bias correction.")

    # Combine models
    model = ChromBPNetWithBias(main_model, bias_model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # Datasets
    data_dir = Path(config["data"]["zarr_dir"]) / cell_type
    max_jitter = train_cfg.get("max_jitter", 0)
    train_ds = ATACDataset(
        str(data_dir / "train.zarr"), split="train",
        augment_rc=train_cfg.get("reverse_complement_augment", True),
        max_jitter=max_jitter,
    )
    val_ds = ATACDataset(str(data_dir / "val.zarr"), split="val")

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

    # Compute count loss weight
    count_weight = train_cfg["count_weight"]
    if count_weight == "auto":
        count_weight = compute_dynamic_count_weight(train_ds)
        print(f"Dynamic count weight: {count_weight:.2f} (median(counts)/10)")
    else:
        count_weight = float(count_weight)

    # Lightning module
    module = ChromBPNetModule(
        model=model,
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        warmup_steps=train_cfg["warmup_steps"],
        profile_weight=train_cfg["profile_weight"],
        count_weight=count_weight,
    )

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=ct_dir, filename="best",
        monitor="val/loss", mode="min", save_top_k=1,
    )
    early_stop_cb = EarlyStopping(
        monitor="val/loss",
        patience=train_cfg["early_stopping_patience"],
        mode="min",
    )
    lr_cb = LearningRateMonitor(logging_interval="step")

    # WandB logger
    logger = None
    wandb_cfg = config.get("wandb", {})
    if wandb_cfg.get("project"):
        try:
            from pytorch_lightning.loggers import WandbLogger
            logger = WandbLogger(
                project=wandb_cfg["project"],
                entity=wandb_cfg.get("entity"),
                name=f"chrombpnet_{cell_type}",
                save_dir=str(ct_dir),
            )
        except ImportError:
            pass

    # Trainer
    trainer = pl.Trainer(
        max_epochs=train_cfg["max_epochs"],
        accelerator="gpu" if gpus > 0 else "cpu",
        devices=gpus if gpus > 0 else 1,
        precision=train_cfg.get("precision", "16-mixed"),
        callbacks=[checkpoint_cb, early_stop_cb, lr_cb],
        default_root_dir=str(ct_dir),
        logger=logger,
        log_every_n_steps=50,
    )

    trainer.fit(module, train_loader, val_loader)

    print(f"\nBest model: {checkpoint_cb.best_model_path}")
    print(f"Best val loss: {checkpoint_cb.best_model_score:.4f}")

    return checkpoint_cb.best_model_path


def main():
    parser = argparse.ArgumentParser(description="Train per-cell-type ChromBPNet")
    parser.add_argument("--config", type=str, default="configs/chrombpnet.yaml")
    parser.add_argument("--cell-type", type=str, default=None,
                        help="Single cell type to train (default: use --all)")
    parser.add_argument("--all", action="store_true", help="Train all cell types")
    parser.add_argument("--output-dir", type=str, default="results/chrombpnet")
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output_dir)

    if args.all:
        cell_types = CELL_TYPES
    elif args.cell_type:
        cell_types = [args.cell_type]
    else:
        parser.error("Specify --cell-type or --all")

    results = {}
    for ct in cell_types:
        best_path = train_cell_type(ct, config, output_dir, args.gpus)
        results[ct] = best_path

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    for ct, path in results.items():
        print(f"  {ct}: {path}")


if __name__ == "__main__":
    main()
