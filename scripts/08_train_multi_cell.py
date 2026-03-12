#!/usr/bin/env python
"""Train multi-cell-type ChromBPNet model (Stage 3).

Shared encoder with output heads. Supports variant flags for ablation:
  --separate-heads: independent heads per cell type
  --task-norms: per-cell-type GroupNorm before heads
  --uncertainty-weighting: learned per-cell-type loss weights
  --ortho-weight: orthogonality regularization on head weights

Usage:
    python scripts/08_train_multi_cell.py \
        --config configs/chrombpnet.yaml \
        --model-dir results/chrombpnet \
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
from torch.utils.data import DataLoader

from src.models.multi_cell_chrombpnet import MultiCellChromBPNet
from src.training.trainer import MultiCellModule
from src.data.dataset import MultiCellATACDataset

import numpy as np


CELL_TYPES = ["Cardiomyocyte", "Coronary_EC", "Fibroblast", "Macrophage", "Pericytes"]


def main():
    parser = argparse.ArgumentParser(description="Train multi-cell-type ChromBPNet")
    parser.add_argument("--config", type=str, default="configs/chrombpnet.yaml")
    parser.add_argument("--model-dir", type=str, default="results/chrombpnet",
                        help="Directory with per-cell-type models for head init")
    parser.add_argument("--data-dir", type=str, default="data/training")
    parser.add_argument("--output-dir", type=str, default="results/multi_cell")
    parser.add_argument("--gpus", type=int, default=1)
    # Variant flags
    parser.add_argument("--separate-heads", action="store_true",
                        help="Use independent heads per cell type")
    parser.add_argument("--task-norms", action="store_true",
                        help="Add per-cell-type GroupNorm before heads")
    parser.add_argument("--uncertainty-weighting", action="store_true",
                        help="Learn per-cell-type loss weights (Kendall 2018)")
    parser.add_argument("--ortho-weight", type=float, default=0.0,
                        help="Orthogonality regularization weight (0 = off)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = config["model"]
    train_cfg = config["training"]

    # Print variant config
    variants = []
    if args.separate_heads:
        variants.append("separate_heads")
    if args.task_norms:
        variants.append("task_norms")
    if args.uncertainty_weighting:
        variants.append("uncertainty_weighting")
    if args.ortho_weight > 0:
        variants.append(f"ortho_weight={args.ortho_weight}")
    print(f"Variant: {', '.join(variants) if variants else 'base (no variants)'}")

    # Create model
    print("Training from scratch (heads initialized from per-cell-type models)")
    model = MultiCellChromBPNet(
        num_cell_types=len(CELL_TYPES),
        input_length=model_cfg["input_length"],
        output_length=model_cfg["output_length"],
        stem_channels=model_cfg["stem_channels"],
        stem_kernel_size=model_cfg["stem_kernel_size"],
        num_dilated_layers=model_cfg["num_dilated_layers"],
        dilated_kernel_size=model_cfg["dilated_kernel_size"],
        profile_kernel_size=model_cfg["profile_kernel_size"],
        dropout=model_cfg["dropout"],
        separate_heads=args.separate_heads,
        task_norms=args.task_norms,
    )

    # Initialize heads from all per-cell-type models
    model_dir = Path(args.model_dir)
    n_heads_loaded = 0
    for ct_idx, ct in enumerate(CELL_TYPES):
        ckpt_path = model_dir / ct / "best.ckpt"
        if not ckpt_path.exists():
            print(f"  WARNING: No checkpoint for {ct}, head {ct_idx} keeps random init")
            continue
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "state_dict" in sd:
            sd = sd["state_dict"]
        if args.separate_heads:
            for k, v in sd.items():
                if "main_model.profile_head.conv.weight" in k:
                    model.profile_heads[ct_idx].conv.weight.data.copy_(v)
                elif "main_model.profile_head.conv.bias" in k:
                    model.profile_heads[ct_idx].conv.bias.data.copy_(v)
                elif "main_model.count_head.fc.weight" in k:
                    model.count_heads[ct_idx].fc.weight.data.copy_(v)
                elif "main_model.count_head.fc.bias" in k:
                    model.count_heads[ct_idx].fc.bias.data.copy_(v)
        else:
            for k, v in sd.items():
                if "main_model.profile_head.conv.weight" in k:
                    model.profile_head.conv.weight.data[ct_idx] = v.squeeze(0)
                elif "main_model.profile_head.conv.bias" in k:
                    model.profile_head.conv.bias.data[ct_idx] = v.item()
                elif "main_model.count_head.fc.weight" in k:
                    model.count_head.fc.weight.data[ct_idx] = v.squeeze(0)
                elif "main_model.count_head.fc.bias" in k:
                    model.count_head.fc.bias.data[ct_idx] = v.item()
        n_heads_loaded += 1
        print(f"  Transferred heads from {ct}")
    print(f"  Initialized {n_heads_loaded}/{len(CELL_TYPES)} head channels")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} (all trainable)")

    # Datasets
    data_dir = Path(args.data_dir)
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

    # Compute count loss weight
    count_weight = train_cfg["count_weight"]
    if count_weight == "auto":
        n_samples = min(5000, len(train_ds))
        indices = np.random.choice(len(train_ds), n_samples, replace=False)
        counts = []
        for i in indices:
            sample = train_ds[int(i)]
            counts.append(float(sample["count"].float().mean()))
        count_weight = max(float(np.median(counts)) / 10.0, 0.01)
        print(f"Dynamic count weight: {count_weight:.2f} (median(mean_counts)/10)")
    else:
        count_weight = float(count_weight)

    multi_cell_lr = train_cfg["learning_rate"]
    print(f"Multi-cell learning rate: {multi_cell_lr}")

    module = MultiCellModule(
        model=model,
        num_cell_types=len(CELL_TYPES),
        learning_rate=multi_cell_lr,
        profile_weight=train_cfg["profile_weight"],
        count_weight=count_weight,
        uncertainty_weighting=args.uncertainty_weighting,
        ortho_weight=args.ortho_weight,
    )

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=out_dir, filename="best",
        monitor="val/loss" if val_loader else "train/loss",
        mode="min", save_top_k=1,
    )
    early_stop_cb = EarlyStopping(
        monitor="val/loss" if val_loader else "train/loss",
        patience=10,
        mode="min",
    )
    lr_cb = LearningRateMonitor(logging_interval="step")

    # Logger
    logger = None
    wandb_cfg = config.get("wandb", {})
    if wandb_cfg.get("project"):
        try:
            from pytorch_lightning.loggers import WandbLogger
            variant_name = "_".join(variants) if variants else "base"
            logger = WandbLogger(
                project=wandb_cfg["project"],
                entity=wandb_cfg.get("entity"),
                name=f"multi_cell_{variant_name}",
                save_dir=str(out_dir),
            )
        except ImportError:
            pass

    trainer = pl.Trainer(
        max_epochs=train_cfg["max_epochs"],
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else 1,
        precision="bf16-mixed",
        gradient_clip_val=1.0,
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
