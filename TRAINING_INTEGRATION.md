# CDPO — Training Integration (Day 3)

This document explains how the decoupled trainer is wired in, and the smoke gate
you must pass before spending money on a 7B GPU.

## What changed in Day 3

Day 2 built the corrected trainer (`CDPODecoupledGRPOTrainer`) in isolation and
proved it. Day 3 wires it into the actual training entry point so a real run
exercises the fix instead of the bug.

The data flow is now:

```
prompts ──► model.generate ──► completions
                                   │
                  AdvantageBridge.build_batch   (recovers r_hard / r_prox / r_soft)
                                   │
                  advantage_computer.compute    (CDPO / GDPO / GRPO)  ──►  Â  (B,G)
                                   │
       CDPODecoupledGRPOTrainer injects Â DIRECTLY into the policy loss
                                   │
                          (NO TRL re-normalization)
```

The TRL reward channel is **bypassed**. A trivial `zero_reward_fn` is still
passed because TRL requires a reward function to exist, but its output never
reaches the loss. This is the whole point: in the old code the advantage was
smuggled through the reward channel and TRL re-normalized it with vanilla GRPO,
collapsing all three methods into ~the same algorithm.

### Files touched
- `training/trainers.py` — `CDPODecoupledGRPOTrainer` (Day 2; overrides `compute_loss`).
- `training/advantage_bridge.py` — pure-numpy `AdvantageBridge` (Day 2).
- `training/reward_fns.py` — added `make_advantage_computer` and `make_zero_reward_fn`;
  the old `make_reward_fn` is kept but **deprecated** (advantage-as-reward = the bug).
- `training/train.py` — instantiates the decoupled trainer, builds the bridge +
  advantage computer + trajectory logger, adds LoRA (default on), and rejects
  `--task all` for a single run (K/M differ per task).
- `setup.py`, `requirements-train.txt`, `requirements-frozen.txt` — TRL pinned to
  `==0.14.0` (the version the `compute_loss` override was written against).

## Version pin — important

`GRPOTrainer` does **not exist** before TRL 0.14.0, and its `compute_loss` body
changes between versions. The override in `trainers.py` was copied from 0.14.0.
If you change the TRL version, diff that version's `GRPOTrainer.compute_loss`
against the copied body in `trainers.py` and re-verify the advantage swap point.
A runtime guard warns if the installed TRL ≠ 0.14.0.

```
pip install -e ".[train]"     # pulls trl==0.14.0, transformers==4.44.2, peft, etc.
```

## The Day-3 smoke gate (do this before any 7B run)

```
# tiny model, a few steps, all three methods — proves the override works
python scripts/smoke_integration.py --model Qwen/Qwen2.5-0.5B-Instruct --steps 6
```

What it asserts:
1. The trainer instantiates and `.train()` runs without crashing; loss is finite.
2. The advantage computer is actually called (the override routes through it).
3. The **step-0 advantage vectors differ across methods**. The critical check is
   GRPO vs CDPO: if `max|Δ| ≤ 1e-3` they are identical, meaning the decoupling
   did NOT take effect and the re-normalization bug is still present.

It also writes `outputs/smoke_integration/step0_advantage_histograms.png` so you
can eyeball the divergence. **If this script prints `FAIL`, do not launch a 7B
run** — the wiring is broken and you would burn GPU on a null result.

## Running a real experiment

Single run (one method × one task), LoRA by default:

```
python training/train.py --method cdpo --task portfolio \
    --model Qwen/Qwen2.5-7B-Instruct --max-steps 400 --seed 42
```

All three methods, three seeds, portfolio (the primary comparison):

```
for m in grpo gdpo cdpo; do
  for s in 42 43 44; do
    python training/train.py --method $m --task portfolio --seed $s
  done
done
```

Per-step CCR/SPS/CQ are logged to W&B and to
`outputs/<run>/trajectory_<method>_<task>_seed<seed>.csv`. Build the figure with:

```
python plot_ccr_curves.py --logdir outputs/ --task portfolio --metric ccr --smooth 3
```

## LoRA vs full fine-tuning

LoRA is the default (`--use-lora`, r=16, α=32, all attention+MLP projections).
All three methods use identical LoRA config, so any difference in results is
attributable to the advantage method, not the adapter. For a full-FT confirmation
run, pass `--no-lora` (needs much more VRAM).

## Known limitations of the override
- Implements the **regular** HF generation path only. `use_vllm=True` raises
  `NotImplementedError` — port the vLLM branch from your TRL's `compute_loss` if
  you need it (the advantage swap point is unchanged).
- `--task all` is rejected for training (run one task at a time).

## Day 4 — parser hardening against real model output

The Day-3 gate used synthetic reward scoring. Day 4 confronts the parser with
real 0.5B output, which is messier than the hand-written fixtures.

### Diagnostic (run first)
```
python scripts/parser_diagnostic.py --model Qwen/Qwen2.5-0.5B-Instruct \
    --tasks portfolio retirement loan --n-prompts 8 --num-generations 4
```
It generates real completions, runs them through the real parser, and prints a
per-task breakdown of parse success vs failure CATEGORY (no_tags / fenced_json /
trailing_comma / unterminated_tag / schema_missing_field / ...). Raw outputs go
to `outputs/parser_diagnostic/<task>_raw.jsonl` and representative failure
samples to `outputs/parser_diagnostic/failure_samples.txt`. Read those before
trusting any parse rate.

### Hardening already applied
`extract_json_block` now tries, in order: strict tags → fenced JSON inside tags
→ unterminated tag (ran out of tokens) → bare `{...}` object. It also repairs
trailing commas and single-quoted JSON. **Crucially, it recovers SYNTAX, never
SCHEMA**: a JSON object missing required fields still fails and still zeros the
rewards. See `tests/test_parser_hardening.py` (14 tests covering both recovery
and "garbage still fails").

### Parse-failure policy
A parse failure returns an all-zero `RewardBundle` with `parse_failed=True`. The
flag distinguishes an *unparseable* rollout from a *valid-but-fully-failing* one
(both have hard=zeros). The advantage bridge counts `parse_failed` for the
`parse_failure_rate` logged each step — watch it during training; a high rate
means fix the prompt or generation length, not the parser.

### Loop
Run the diagnostic → read `failure_samples.txt` → if a failure category the
hardening doesn't cover dominates, extend `extract_json_block` (or the prompt) →
re-run the diagnostic to confirm the rate improved. Repeat until parse failures
are dominated by genuinely-bad plans, not format noise.

## Day 4 — empirical findings & dataset generation

### Parser-hardening results (0.5B stress test)
Running `parser_diagnostic.py` on Qwen2.5-0.5B with the rewritten prompt +
number-comma repair lifted the parse rate from **23% → 68%** overall (portfolio
88%). The remaining failures are model-capability limits (the 0.5B invents
schemas like `credit_score_rating`) and genuine task errors (weights not summing
to 1 — which correctly fire the hard constraint), not format noise. A 7B-Instruct
will parse far better with the same prompt.

**Prefill was measured and rejected.** `--prefill` (starting the assistant turn
inside `<financial_plan>{`) *lowered* the rate to 33%: it denies the model
reasoning space and trades recoverable format errors for unrecoverable schema
errors. It remains an opt-in flag (default OFF) but is not recommended.

### Dataset generation workflow (Day-4 step 3 / Day-5 prep)
```
# 1. scenarios + tier calibration (no LLM, deterministic)
python scripts/generate_pilot.py --output data/pilot          # 60/task, sanity
python scripts/generate_full.py  --output data/full           # 550/task

# 2. attach client descriptions (templated, no API, reproducible)
python scripts/generate_descriptions.py --data-dir data/full

# -> produces data/full/<task>_train_with_desc.jsonl etc.
```
Descriptions are templated deterministically from each instance's
`narrative_hints` — free, reproducible, no external API. The scenario NUMBERS
(constraints, returns) are rendered from metadata separately, so the description
is framing; templating is sufficient to train on. To get richer/more diverse
descriptions later, regenerate with an LLM via `finplanenv.GENERATION_PROMPTS`
and overwrite the `client_description` field (same format). Point the training
ScenarioStore at the `*_with_desc.jsonl` files.

## Day 5 — single short 7B run (go/no-go)

Prereq: `data/full/<task>_train_with_desc.jsonl` exists (generate_full.py +
generate_descriptions.py). train.py now AUTO-prefers the `_with_desc.jsonl`
files and falls back to plain `_train.jsonl` with a warning.

```
python scripts/day5_short_run.py \
    --model Qwen/Qwen2.5-7B-Instruct --task portfolio \
    --steps 30 --num-generations 8 --use-lora --cost-per-hour 0.69
```
It runs the real decoupled-trainer path for ~30 steps, measures median sec/step
and peak VRAM, and prints a cost projection for one 400-step run and for the full
matrix (default 3 methods × 1 task × 3 seeds). Compare the projected total to
your RunPod credit BEFORE launching the matrix; cut --matrix-seeds/-tasks/-steps
if needed. Portfolio-only × 3 seeds is the minimum publishable comparison.

Once the projection is acceptable, the real runs use train.py directly:
```
for m in grpo gdpo cdpo; do for s in 42 43 44; do
  python training/train.py --method $m --task portfolio --seed $s \
      --model Qwen/Qwen2.5-7B-Instruct --max-steps 400
done; done
```
Then build the CCR figure with plot_ccr_curves.py.

## vLLM generation (the throughput fix)

Regular HF generation dominates step time (~16 min/step measured for 7B at G=4
on an L40S). vLLM cuts generation 5-20x. The trainer already contains TRL 0.14.0's
native vLLM branch; enable it with a flag.

```
pip install vllm
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/day5_short_run.py --model Qwen/Qwen2.5-7B-Instruct --task portfolio \
    --steps 30 --num-generations 8 --batch-size 1 \
    --use-vllm --vllm-device cuda:0 --vllm-gpu-mem-util 0.30
```

### Single-GPU memory math (important)
On a 1-GPU pod, vLLM shares the card with training. TRL's default `vllm_device`
is `auto` -> `cuda:1`, which does NOT exist on a 1-GPU pod and raises. We default
`--vllm-device cuda:0` so vLLM shares the training GPU.

Budget on a 44GB card with a 7B:
- training model (bf16 LoRA): ~15 GB
- vLLM inference copy + KV cache: governed by `--vllm-gpu-mem-util` (0.30 ≈ 13 GB)
- training activations / backward: the rest

If vLLM init OOMs, lower `--vllm-gpu-mem-util` (e.g. 0.25). If generation is still
slow or vLLM complains it can't fit the KV cache, raise it (e.g. 0.40) and/or keep
gradient checkpointing on to free training memory. With vLLM doing generation, the
backward pass is the main training-side cost; checkpointing can stay on.

A dedicated 2nd GPU is cleaner (set `--vllm-device cuda:1` and launch training with
`accelerate --num_processes 1`), but single-GPU sharing works for the matrix.

### After a fast baseline
Re-run the projection with the vLLM sec/step. Expect step time to drop from
~4-16 min to well under a minute, turning the 9-run portfolio matrix from ~$250
into roughly $30-60. Then launch the real runs via train.py --use-vllm.

## Running the experiment matrix (scripts/run_matrix.sh)

The launcher runs the matrix sequentially on one pod, resumably. Default is the
9-run PORTFOLIO matrix (3 methods x 3 seeds) — the paper's core comparison — at
the Day-5 proven-stable config (G=4, batch=1, max_new_tokens=384, gradient
checkpointing on, no vLLM).

```
tmux new -s matrix
cd /workspace/cdpo-financial
# optional: export WANDB_API_KEY=...   (else CSV-only; curves still saved)
bash scripts/run_matrix.sh
# detach: Ctrl-b then d ;  reattach later: tmux attach -t matrix
```

Robustness:
- **Resumable** — each finished run writes outputs/<run>/.done; re-running skips
  done runs, so an SSH drop or pod restart costs at most the current run.
- **Failure-continue** — a failed run is logged and skipped (no .done marker);
  re-run the script to retry only failures.
- **Per-run logs** at logs/<run>.log; live-watch with `tail -f logs/<run>.log`.

Widen the matrix later (only after the portfolio result is confirmed):
```
TASKS="portfolio retirement loan" bash scripts/run_matrix.sh   # 27 runs
SEEDS="42 43 44 45 46" bash scripts/run_matrix.sh
```

After the matrix, build the hero figure (reads the per-step CSVs, no W&B needed):
```
python plot_ccr_curves.py --logdir outputs/ --task portfolio --metric ccr --smooth 3
```

Cost: 9 runs x 400 steps x ~254 s/step ≈ 250 GPU-h ≈ ~$250 on an L40S ($1/hr).
