#!/bin/bash
# =============================================================================
# scripts/setup_runpod.sh
# =============================================================================
# One-shot environment setup for a fresh RunPod GPU pod.
# Run this ONCE after cloning the repo.
#
# Usage:
#     bash scripts/setup_runpod.sh
#
# What it does:
#   1. Installs system dependencies (git-lfs for model downloads)
#   2. Installs Python training stack (torch, trl, transformers, wandb)
#   3. Installs this package in editable mode
#   4. Runs smoke test to verify installation
#   5. (Optional) downloads the base model weights
#
# Expected environment: RunPod PyTorch template, CUDA 12.1+, Python 3.10+
# =============================================================================

set -e   # exit on first error
echo ""
echo "======================================================"
echo "  CDPO Financial — RunPod Setup"
echo "======================================================"
echo ""

# ── 1. System deps ────────────────────────────────────────────────────────────
echo "[1/5] Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq git-lfs htop tmux
git lfs install

# ── 2. Python packages ────────────────────────────────────────────────────────
echo ""
echo "[2/5] Installing Python training stack..."
echo "      This takes 5-10 minutes on first run."
echo ""

pip install --upgrade pip --quiet

# Install PyTorch first (usually already installed in RunPod template)
# If you see CUDA errors, check your RunPod template's CUDA version
# and install the matching torch: https://pytorch.org/get-started/locally/
python -c "import torch; print(f'  PyTorch {torch.__version__} already installed')" \
    2>/dev/null || pip install torch --quiet

# Install the rest
pip install \
    transformers>=4.40 \
    datasets>=2.18 \
    accelerate>=0.28 \
    peft>=0.10 \
    "trl>=0.8.6" \
    wandb>=0.16 \
    tqdm pyyaml jsonlines sentencepiece protobuf \
    --quiet

echo "  Training stack installed."

# ── 3. Install this package ───────────────────────────────────────────────────
echo ""
echo "[3/5] Installing cdpo-financial package..."
pip install -e . --quiet
echo "  Package installed."

# ── 4. Smoke test ─────────────────────────────────────────────────────────────
echo ""
echo "[4/5] Running smoke test..."
python scripts/smoke_test.py
echo ""

# ── 5. Check GPU ──────────────────────────────────────────────────────────────
echo "[5/5] GPU check:"
python -c "
import torch
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'  VRAM: {vram:.1f} GB')
    if vram < 20:
        print('  WARNING: < 20 GB VRAM. Use --batch-size 1 for 7B model.')
    elif vram < 40:
        print('  OK: 20-40 GB VRAM. --batch-size 2-4 recommended for 7B model.')
    else:
        print('  GREAT: 40+ GB VRAM. Default batch sizes will work.')
else:
    print('  No CUDA GPU detected. Training will be very slow on CPU.')
"

# ── W&B login reminder ────────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo ""
echo "  1. Log in to W&B (for experiment tracking):"
echo "       wandb login"
echo ""
echo "  2. Generate training data:"
echo "       python scripts/generate_pilot.py"
echo "       # review calibration, then:"
echo "       python scripts/generate_full.py"
echo ""
echo "  3. Add client descriptions (GPT-4o or Claude API):"
echo "       See README.md → 'Generate client descriptions'"
echo ""
echo "  4. Run a quick smoke training (5 steps, no GPU needed):"
echo "       bash scripts/run_experiment.sh cdpo portfolio --max-steps 5 --no-wandb"
echo ""
echo "  5. Run real experiments:"
echo "       bash scripts/run_experiment.sh cdpo portfolio"
echo "       bash scripts/run_experiment.sh grpo portfolio"
echo "       bash scripts/run_experiment.sh gdpo portfolio"
echo "======================================================"
echo ""
