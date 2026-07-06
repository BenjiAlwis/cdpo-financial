"""
evaluation/eval.py
==================
Evaluate a trained model checkpoint on the held-out audit split.

Generates one plan per prompt (greedy decoding), parses it, computes
all reward signals, and reports CCR / SPS / CQ per task and per tier.

Usage
-----
    # Evaluate a single checkpoint
    python evaluation/eval.py \
        --checkpoint outputs/cdpo_portfolio_seed42_0101_1234/final \
        --method cdpo \
        --tasks portfolio

    # Compare all three methods after training
    python evaluation/eval.py \
        --checkpoint outputs/grpo_portfolio_seed42/final \
        --method grpo \
        --tasks portfolio retirement loan

    # Save results to JSON for later comparison
    python evaluation/eval.py \
        --checkpoint outputs/cdpo_all_seed42/final \
        --method cdpo \
        --tasks all \
        --output results/cdpo_eval.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Evaluate CDPO/GRPO/GDPO checkpoint")
parser.add_argument("--checkpoint", required=True,
                    help="Path to model checkpoint directory")
parser.add_argument("--method",     required=True,
                    choices=["grpo", "gdpo", "cdpo"])
parser.add_argument("--tasks",      nargs="+",
                    default=["portfolio", "retirement", "loan"],
                    help="Tasks to evaluate")
parser.add_argument("--data-dir",   default="data/full")
parser.add_argument("--desc-dir",   default="data/full")
parser.add_argument("--batch-size", type=int, default=8,
                    help="Inference batch size")
parser.add_argument("--max-new-tokens", type=int, default=1024)
parser.add_argument("--output",     default=None,
                    help="Save results to this JSON file")
parser.add_argument("--split",      default="audit",
                    choices=["audit", "train"],
                    help="Which data split to evaluate on")
args = parser.parse_args()

TASKS = (["portfolio", "retirement", "loan"]
         if "all" in args.tasks else args.tasks)

# ── Imports ───────────────────────────────────────────────────────────────────
logger.info("Loading evaluation stack...")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from finplanenv.parser    import compute_rewards_from_output
from finplanenv.dataset   import DatasetConfig, DatasetGenerator

from training.scenario_store  import ScenarioStore
from training.dataset_builder import build_eval_dataset
from evaluation.metrics       import compute_metrics, print_comparison_table

# ── Load model ────────────────────────────────────────────────────────────────
logger.info("Loading checkpoint: %s", args.checkpoint)
tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    args.checkpoint,
    torch_dtype = torch.bfloat16,
    device_map  = "auto",
)
model.eval()
logger.info("Model loaded.")

# ── Build eval dataset and scenario store ─────────────────────────────────────
data_dir = Path(args.data_dir)
desc_dir = Path(args.desc_dir)

metadata_paths = {t: data_dir / f"{t}_train.jsonl" for t in TASKS}
descriptions_paths = {t: desc_dir / f"{t}_descriptions.json" for t in TASKS}

eval_dataset = build_eval_dataset(
    metadata_paths     = metadata_paths,
    descriptions_paths = descriptions_paths,
    tokenizer          = tokenizer,
    split              = args.split,
)
logger.info("Eval dataset: %d prompts", len(eval_dataset))

ds_config = DatasetConfig()
store     = ScenarioStore(ds_config)
for task in TASKS:
    path = data_dir / f"{task}_{args.split}.jsonl"
    if not path.exists():
        path = data_dir / f"{task}_train.jsonl"
    gen  = DatasetGenerator(ds_config)
    meta = gen.load_metadata(path)
    store.build(meta)

# ── Inference ─────────────────────────────────────────────────────────────────
logger.info("Running inference...")

all_results: dict[str, list[dict]] = {task: [] for task in TASKS}

gen_config = GenerationConfig(
    max_new_tokens  = args.max_new_tokens,
    do_sample       = False,   # greedy for eval
    temperature     = 1.0,
    pad_token_id    = tokenizer.pad_token_id,
    eos_token_id    = tokenizer.eos_token_id,
)

# Batch the prompts
prompts    = eval_dataset["prompt"]
inst_ids   = eval_dataset["instance_id"]
tasks_col  = eval_dataset["task"]
diffs_col  = eval_dataset["difficulty"]

BS = args.batch_size
for start in range(0, len(prompts), BS):
    batch_prompts = prompts[start:start+BS]
    batch_ids     = inst_ids[start:start+BS]
    batch_tasks   = tasks_col[start:start+BS]
    batch_diffs   = diffs_col[start:start+BS]

    if start % 50 == 0:
        logger.info("  Inference: %d / %d", start, len(prompts))

    # Tokenise
    inputs = tokenizer(
        batch_prompts,
        return_tensors = "pt",
        padding        = True,
        truncation     = True,
        max_length     = 2048,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config = gen_config,
        )

    # Decode only the newly generated tokens
    input_len   = inputs["input_ids"].shape[1]
    completions = tokenizer.batch_decode(
        outputs[:, input_len:],
        skip_special_tokens = True,
    )

    # Score each completion
    for completion, prompt, iid, task, diff in zip(
        completions, batch_prompts, batch_ids, batch_tasks, batch_diffs
    ):
        entry = store.lookup(prompt)
        if entry is None:
            all_results[task].append({
                "instance_id": iid,
                "difficulty":  diff,
                "hard":        [0] * 4,
                "soft":        [0.0] * 3,
                "parsed":      False,
            })
            continue

        bundle = compute_rewards_from_output(
            completion,
            entry.task,
            entry.scenario,
            entry.constraints,
        )
        parsed = not (bundle.hard.sum() == 0 and bundle.soft.sum() == 0)

        all_results[task].append({
            "instance_id": iid,
            "difficulty":  diff,
            "hard":        bundle.hard.tolist(),
            "soft":        bundle.soft.tolist(),
            "parsed":      parsed,
            "completion":  completion[:500],  # truncate for storage
        })

logger.info("Inference complete.")

# ── Compute metrics ───────────────────────────────────────────────────────────
task_metrics = {}
for task in TASKS:
    m = compute_metrics(
        results = all_results[task],
        task    = task,
        method  = args.method,
        split   = args.split,
    )
    task_metrics[task] = m
    logger.info(
        "%s | CCR=%.3f  SPS=%.3f  CQ=%.3f  parse_fail=%.1f%%",
        task.upper(), m.ccr, m.sps, m.cq, 100*m.parse_failure_rate,
    )
    if m.tier_ccr:
        for tier, ccr in m.tier_ccr.items():
            logger.info("    %s tier: CCR=%.3f", tier, ccr)

# ── Print comparison table ────────────────────────────────────────────────────
print_comparison_table({args.method: list(task_metrics.values())})

# ── Save results ──────────────────────────────────────────────────────────────
if args.output:
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_data = {
        "method":     args.method,
        "checkpoint": args.checkpoint,
        "split":      args.split,
        "metrics":    {t: m.to_dict() for t, m in task_metrics.items()},
        "raw_results": all_results,
    }
    with open(out, "w") as f:
        json.dump(save_data, f, indent=2)
    logger.info("Results saved → %s", out)
