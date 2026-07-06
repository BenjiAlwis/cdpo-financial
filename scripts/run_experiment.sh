#!/bin/bash
# =============================================================================
# scripts/run_experiment.sh
# =============================================================================
# Launch one training run: one method × one task.
# Wraps training/train.py with sensible defaults and logging.
#
# Usage:
#     bash scripts/run_experiment.sh <method> <task> [extra args]
#
# Examples:
#     bash scripts/run_experiment.sh cdpo portfolio
#     bash scripts/run_experiment.sh grpo portfolio --seed 1
#     bash scripts/run_experiment.sh cdpo all --max-steps 200
#     bash scripts/run_experiment.sh cdpo portfolio --max-steps 5 --no-wandb
#
# Arguments:
#     method   : grpo | gdpo | cdpo
#     task     : portfolio | retirement | loan | all
#     All other args are passed directly to training/train.py
# =============================================================================

set -e

METHOD=${1:-cdpo}
TASK=${2:-portfolio}
shift 2 2>/dev/null || true   # consume first two args; remaining go to train.py

# Validate method
if [[ ! "$METHOD" =~ ^(grpo|gdpo|cdpo)$ ]]; then
    echo "ERROR: method must be grpo, gdpo, or cdpo. Got: $METHOD"
    exit 1
fi

# Validate task
if [[ ! "$TASK" =~ ^(portfolio|retirement|loan|all)$ ]]; then
    echo "ERROR: task must be portfolio, retirement, loan, or all. Got: $TASK"
    exit 1
fi

# Check data exists
if [ ! -f "data/full/${TASK}_train.jsonl" ] && [ "$TASK" != "all" ]; then
    echo "ERROR: Training data not found: data/full/${TASK}_train.jsonl"
    echo "Run: python scripts/generate_full.py"
    exit 1
fi
if [ "$TASK" = "all" ]; then
    for t in portfolio retirement loan; do
        if [ ! -f "data/full/${t}_train.jsonl" ]; then
            echo "ERROR: Missing data/full/${t}_train.jsonl"
            echo "Run: python scripts/generate_full.py"
            exit 1
        fi
    done
fi

# Handle --no-wandb flag (convert to --wandb-project disabled)
EXTRA_ARGS="$@"
if echo "$EXTRA_ARGS" | grep -q "\-\-no-wandb"; then
    EXTRA_ARGS=$(echo "$EXTRA_ARGS" | sed 's/--no-wandb//')
    EXTRA_ARGS="$EXTRA_ARGS --wandb-project disabled"
fi

echo ""
echo "======================================================"
echo "  CDPO Financial — Training Run"
echo "  Method : $METHOD"
echo "  Task   : $TASK"
echo "  Extra  : $EXTRA_ARGS"
echo "======================================================"
echo ""

# Log GPU memory before starting
python -c "
import torch
if torch.cuda.is_available():
    gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'  GPU VRAM: {gb:.1f} GB')
" 2>/dev/null || true

# Run training
python training/train.py \
    --method "$METHOD" \
    --task   "$TASK"   \
    $EXTRA_ARGS

echo ""
echo "  Training complete: $METHOD × $TASK"
echo ""
