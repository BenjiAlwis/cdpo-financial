#!/usr/bin/env bash
# =============================================================================
# scripts/run_matrix.sh
# =============================================================================
# Launch the CDPO experiment matrix sequentially on a single pod.
#
# Default: the 9-run PORTFOLIO matrix (3 methods x 3 seeds) — the comparison the
# paper hinges on. Parameterized so you can widen to more tasks later.
#
# Design for robustness (we fought these all through Day 5):
#   - Sequential on one GPU: one place for things to go wrong, not N.
#   - Resumable: each run writes a .done marker; re-running SKIPS finished runs,
#     so an SSH drop / pod restart costs you at most the current run.
#   - tmux-friendly: meant to run detached. Logs per-run to logs/<run>.log.
#   - Config matches the Day-5 proven-stable point on a 44GB L40S:
#     G=4, batch=1, max_new_tokens=384, gradient checkpointing ON.
#   - W&B used if WANDB_API_KEY is set; otherwise CSV-only (curves still saved).
#
# Usage
# -----
#   # one-time, on the pod:
#   tmux new -s matrix
#   cd /workspace/cdpo-financial
#   # (optional) export WANDB_API_KEY=...     # else CSV-only
#   bash scripts/run_matrix.sh
#   # detach: Ctrl-b then d ;  reattach: tmux attach -t matrix
#
#   # widen later:
#   TASKS="portfolio retirement loan" bash scripts/run_matrix.sh
#   SEEDS="42 43 44 45 46" bash scripts/run_matrix.sh
#
# Override any of these via environment variables:
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
TASKS="${TASKS:-portfolio}"
METHODS="${METHODS:-grpo gdpo cdpo}"
SEEDS="${SEEDS:-42 43 44}"
STEPS="${STEPS:-400}"
G="${G:-4}"
BATCH="${BATCH:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-384}"
DATA_DIR="${DATA_DIR:-data/full}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
WANDB_PROJECT="${WANDB_PROJECT:-cdpo-financial}"
COST_PER_HOUR="${COST_PER_HOUR:-1.00}"
SEC_PER_STEP="${SEC_PER_STEP:-254}"   # measured Day-5 default; for the estimate only
# =============================================================================

set -u
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p logs "${OUTPUT_DIR}"

# W&B: on if a key is present, else disabled (train.py handles 'disabled').
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  WANDB_PROJECT="disabled"
  echo ">> No WANDB_API_KEY found — logging to CSV only (curves still saved)."
else
  echo ">> W&B enabled: project=${WANDB_PROJECT}"
fi

# ---- enumerate runs ----
RUNS=()
for task in ${TASKS}; do
  for method in ${METHODS}; do
    for seed in ${SEEDS}; do
      RUNS+=("${method}|${task}|${seed}")
    done
  done
done
N=${#RUNS[@]}

# ---- cost estimate up front ----
total_sec=$(python3 -c "print(${N} * ${STEPS} * ${SEC_PER_STEP})")
total_h=$(python3 -c "print(f'{${total_sec}/3600:.1f}')")
total_cost=$(python3 -c "print(f'{${total_sec}/3600*${COST_PER_HOUR}:.0f}')")
echo "============================================================"
echo "  CDPO MATRIX — ${N} runs"
echo "  tasks=[${TASKS}] methods=[${METHODS}] seeds=[${SEEDS}]"
echo "  per run: ${STEPS} steps @ ~${SEC_PER_STEP}s/step"
echo "  estimated total: ${total_h} h  ≈  \$${total_cost} @ \$${COST_PER_HOUR}/hr"
echo "  (estimate only; real time depends on measured sec/step)"
echo "============================================================"
echo ""

# ---- run loop (resumable) ----
i=0
start_all=$(date +%s)
for spec in "${RUNS[@]}"; do
  i=$((i+1))
  IFS='|' read -r method task seed <<< "${spec}"
  run_id="${method}_${task}_seed${seed}"
  done_marker="${OUTPUT_DIR}/${run_id}/.done"
  log_file="logs/${run_id}.log"

  if [[ -f "${done_marker}" ]]; then
    echo "[${i}/${N}] SKIP ${run_id} (already done)"
    continue
  fi

  echo "[${i}/${N}] START ${run_id}  ($(date '+%H:%M:%S'))  -> ${log_file}"
  run_start=$(date +%s)

  python training/train.py \
      --method "${method}" \
      --task "${task}" \
      --seed "${seed}" \
      --model "${MODEL}" \
      --max-steps "${STEPS}" \
      --num-generations "${G}" \
      --batch-size "${BATCH}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --data-dir "${DATA_DIR}" \
      --output-dir "${OUTPUT_DIR}" \
      --wandb-project "${WANDB_PROJECT}" \
      --run-name "${run_id}" \
      > "${log_file}" 2>&1

  rc=$?
  run_end=$(date +%s)
  dur=$(( (run_end - run_start) / 60 ))

  if [[ ${rc} -eq 0 ]]; then
    mkdir -p "${OUTPUT_DIR}/${run_id}"
    touch "${done_marker}"
    echo "[${i}/${N}] DONE  ${run_id}  (${dur} min)"
  else
    echo "[${i}/${N}] FAIL  ${run_id}  (rc=${rc}, ${dur} min) — see ${log_file}"
    echo "         continuing to next run; re-run this script to retry failures."
  fi
done

end_all=$(date +%s)
echo ""
echo "============================================================"
echo "  MATRIX COMPLETE — $(( (end_all - start_all) / 3600 ))h elapsed"
echo "  done markers: ${OUTPUT_DIR}/*/.done"
echo "  per-step CSVs: ${OUTPUT_DIR}/*/trajectory_*.csv"
echo ""
echo "  Build the hero CCR figure:"
echo "    python plot_ccr_curves.py --logdir ${OUTPUT_DIR}/ --task portfolio \\"
echo "        --metric ccr --smooth 3"
echo "============================================================"
