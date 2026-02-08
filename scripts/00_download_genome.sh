#!/usr/bin/env bash
# Download mm10 reference genome and chromosome sizes
set -euo pipefail

GENOME_DIR="data/genome"
mkdir -p "$GENOME_DIR"

echo "=== Downloading mm10 reference genome ==="

# Download mm10.fa from UCSC
if [ ! -f "$GENOME_DIR/mm10.fa" ]; then
    echo "Downloading mm10.fa.gz..."
    wget -q "https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz" \
        -O "$GENOME_DIR/mm10.fa.gz"
    echo "Decompressing..."
    gunzip "$GENOME_DIR/mm10.fa.gz"
    echo "Done: $GENOME_DIR/mm10.fa"
else
    echo "mm10.fa already exists, skipping download"
fi

# Index with samtools
if [ ! -f "$GENOME_DIR/mm10.fa.fai" ]; then
    echo "Indexing with samtools..."
    samtools faidx "$GENOME_DIR/mm10.fa"
    echo "Done: $GENOME_DIR/mm10.fa.fai"
else
    echo "mm10.fa.fai already exists, skipping indexing"
fi

# Download chrom.sizes
if [ ! -f "$GENOME_DIR/mm10.chrom.sizes" ]; then
    echo "Downloading mm10.chrom.sizes..."
    wget -q "https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.chrom.sizes" \
        -O "$GENOME_DIR/mm10.chrom.sizes"
    echo "Done: $GENOME_DIR/mm10.chrom.sizes"
else
    echo "mm10.chrom.sizes already exists, skipping download"
fi

echo ""
echo "=== Genome files ready ==="
echo "  FASTA: $GENOME_DIR/mm10.fa"
echo "  Index: $GENOME_DIR/mm10.fa.fai"
echo "  Sizes: $GENOME_DIR/mm10.chrom.sizes"
