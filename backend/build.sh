#!/usr/bin/env bash
# ==============================================================================
# Render Build Script for MAITRI Backend
# ==============================================================================
set -e

echo ">>> [1/3] Upgrading pip & installing CPU-optimized PyTorch..."
pip install --upgrade pip
# Install CPU-only PyTorch wheels (saves ~2GB download, prevents memory exhaustion & build timeout)
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo ">>> [2/3] Installing MAITRI Dependencies..."
pip install --no-cache-dir -r requirements.txt

echo ">>> [3/3] Checking Model Checkpoints..."
# Download model checkpoints if missing (with non-blocking failure guard so build succeeds cleanly)
python download_models.py || true

echo ">>> MAITRI Backend Build Succeeded!"
