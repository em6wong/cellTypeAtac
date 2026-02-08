#!/bin/bash
# Source this at the start of every job/session
# Usage: source scripts/setup_env.sh

# System modules
module load cuda/11.8.0
module load cudnn/8.6.0.163-11.8
module load git/2.38.1
module load bedtools2/2.30.0
module load samtools/1.20
module load htslib/1.20

# Conda (user miniconda)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate multiome

# Project paths
export PROJECT_DIR="/srv/scratch/fabilab/emily/cellTypeAtac"
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

# Shared lab resources
export GENOME_FA="/srv/scratch/wonglab/useful/genomes/mm10.fa"
export CHROM_SIZES="/srv/scratch/wonglab/useful/genomes/mm10.chrom.sizes"

echo "Environment loaded: cellTypeAtac"
