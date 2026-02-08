#!/usr/bin/env bash
# Run this ONCE on Katana to set up the project.
# NOT a PBS job — run interactively on a login node.
set -euo pipefail

# === EDIT THESE ===
PROJECT_DIR="/srv/scratch/fabilab/emily/cellTypeAtac"
GENOME_DIR="/srv/scratch/wonglab/useful/genomes"
# ==================

echo "=== Setting up cellTypeAtac on Katana ==="

# Symlink genome files so scripts find them at data/genome/
mkdir -p "$PROJECT_DIR/data/genome"
ln -sf "$GENOME_DIR/mm10.fa"          "$PROJECT_DIR/data/genome/mm10.fa"
ln -sf "$GENOME_DIR/mm10.fa.fai"      "$PROJECT_DIR/data/genome/mm10.fa.fai"
ln -sf "$GENOME_DIR/mm10.chrom.sizes" "$PROJECT_DIR/data/genome/mm10.chrom.sizes"

echo "Symlinked genome files:"
ls -la "$PROJECT_DIR/data/genome/"

# Create output directories
mkdir -p "$PROJECT_DIR/data/bigwig"
mkdir -p "$PROJECT_DIR/data/training"
mkdir -p "$PROJECT_DIR/results/bias_model"
mkdir -p "$PROJECT_DIR/results/chrombpnet"
mkdir -p "$PROJECT_DIR/results/multi_cell"
mkdir -p "$PROJECT_DIR/results/motif_baseline"
mkdir -p "$PROJECT_DIR/results/specificity"
mkdir -p "$PROJECT_DIR/logs"

# Create conda environment (if not exists)
if ! conda info --envs | grep -q celltypeatac; then
    echo "Creating conda environment..."
    conda env create -f "$PROJECT_DIR/environment.yaml"
else
    echo "Conda env 'celltypeatac' already exists"
fi

echo ""
echo "=== Setup complete ==="
echo "Activate with: conda activate celltypeatac"
echo "Project dir:   $PROJECT_DIR"
