"""
training/train.py
=================
Main training entry point for GRPO, GDPO, and CDPO.

Usage
-----
    # Train CDPO on portfolio task
    python training/train.py --method cdpo --task portfolio

    # Train GRPO baseline
    python training/train.py --method grpo --task portfolio

    # Train all three methods sequentially (for comparison)
    for method in grpo gdpo cdpo; do
        python training/train.py --method $method --task portfolio
    done

    # Full experiment: all methods × all tasks
    python training/train.py --method cdpo --task all

Arguments
---------
See argparse section below. All key hyperparameters can be overridden
from the command line; defaults match the paper's Week 2 configuration.

Output
------
Checkpoints saved to: outputs/<method>_<task>_<run_id>/
W&B run name:         <method>_<task>_seed<seed>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# ── Logging setup (before any imports that might log) ─────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    datefmt= "%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train CDPO / GRPO / GDPO")

# Method and task
parser.add_argument("--method", required=True,
                    choices=["grpo", "gdpo", "cdpo"],
                    help="Advantage computation method")
parser.add_argument("--task",   required=True,
                    choices=["portfolio", "retirement", "loan", "all"],
                    help="Financial planning task (or 'all' for all three)")

# Model
parser.add_argument("--model",  default="Qwen/Qwen2.5-7B-Instruct",
                    help="HuggingFace model ID or local path")

# Training config
parser.add_argument("--num-generations", type=int, default=8,
                    help="G: rollouts per prompt (group size)")
parser.add_argument("--max-steps",       type=int, default=400,
                    help="Total training steps")
parser.add_argument("--batch-size",      type=int, default=4,
                    help="Prompts per batch (effective batch = batch_size × num_generations)")
parser.add_argument("--lr",              type=float, default=1e-6,
                    help="Learning rate")
parser.add_argument("--max-new-tokens",  type=int, default=1024,
                    help="Max tokens to generate per rollout")
parser.add_argument("--seed",            type=int, default=42)

# CDPO-specific
parser.add_argument("--alpha-schedule",  default="adaptive",
                    choices=["adaptive", "fixed", "annealing"],
                    help="CDPO mixing strategy (ignored for grpo/gdpo)")
parser.add_argument("--beta-minus",      type=float, default=2.0,
                    help="CDPO asymmetric hard penalty (β−)")

# Parameter-efficient finetuning (default ON: makes 3×3×3 runs affordable)
parser.add_argument("--use-lora", action="store_true", default=True,
                    help="Train with LoRA adapters (default). All methods share "
                         "identical LoRA config, so the comparison stays fair.")
parser.add_argument("--no-lora", dest="use_lora", action="store_false",
                    help="Full fine-tuning instead of LoRA.")
parser.add_argument("--lora-r",     type=int, default=16)
parser.add_argument("--lora-alpha", type=int, default=32)

# vLLM generation (the throughput fix). On a single-GPU pod vLLM shares cuda:0.
parser.add_argument("--use-vllm", action="store_true",
                    help="Use vLLM for generation (5-20x faster). pip install vllm.")
parser.add_argument("--vllm-device", default="cuda:0",
                    help="vLLM device; keep cuda:0 on a 1-GPU pod (TRL 'auto' picks "
                         "cuda:1 and fails).")
parser.add_argument("--vllm-gpu-mem-util", type=float, default=0.30,
                    help="Fraction of GPU memory vLLM reserves (single-GPU share).")

# Data paths
parser.add_argument("--data-dir",   default="data/full",
                    help="Directory containing <task>_train.jsonl files")
parser.add_argument("--desc-dir",   default="data/full",
                    help="Directory containing <task>_descriptions.json files")
parser.add_argument("--output-dir", default="outputs",
                    help="Root directory for checkpoints and logs")

# W&B
parser.add_argument("--wandb-project", default="cdpo-financial",
                    help="W&B project name (set to 'disabled' to skip)")
parser.add_argument("--run-name", default=None,
                    help="W&B run name (auto-generated if not set)")

args = parser.parse_args()

# ── Derived config ────────────────────────────────────────────────────────────
TASKS = ["portfolio", "retirement", "loan"] if args.task == "all" else [args.task]

# A single GRPO training run must be single-task: each task has a different
# number of hard constraints K (portfolio=4, retirement=2, loan=3), and the
# advantage computer / bridge are built for one fixed (K, M). Mixing tasks in one
# run would feed mismatched-width reward vectors through a single computer.
if args.task == "all":
    logger.error(
        "--task all is not supported for a single training run because K/M "
        "differ per task. Launch one run per task, e.g.:\n"
        "    for t in portfolio retirement loan; do "
        "python training/train.py --method %s --task $t; done",
        args.method,
    )
    sys.exit(1)

# Number of hard constraints and soft prefs per task
TASK_DIMS = {
    "portfolio":  {"K": 4, "M": 3},
    "retirement": {"K": 2, "M": 3},
    "loan":       {"K": 3, "M": 3},
}

run_id   = args.run_name or (
    f"{args.method}_{args.task}_seed{args.seed}_"
    f"{datetime.now().strftime('%m%d_%H%M')}"
)
out_dir  = Path(args.output_dir) / run_id
out_dir.mkdir(parents=True, exist_ok=True)

logger.info("Run: %s", run_id)
logger.info("Output dir: %s", out_dir)

# ── Imports (heavy; after args are validated) ─────────────────────────────────
logger.info("Importing training stack...")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig

from finplanenv.cdpo      import CDPOConfig
from finplanenv.baselines import BaselineConfig
from finplanenv.dataset   import DatasetConfig

from training.scenario_store    import ScenarioStore
from training.dataset_builder   import build_dataset
from training.reward_fns        import make_advantage_computer, make_zero_reward_fn
from training.advantage_bridge  import AdvantageBridge
from training.trainers          import CDPODecoupledGRPOTrainer
from training.trajectory_logger import TrajectoryLogger
from finplanenv.parser          import compute_rewards_from_output

logger.info("Imports done. CUDA available: %s", torch.cuda.is_available())
if torch.cuda.is_available():
    logger.info("GPU: %s  VRAM: %.1f GB",
                torch.cuda.get_device_name(0),
                torch.cuda.get_device_properties(0).total_memory / 1e9)

# ── W&B ───────────────────────────────────────────────────────────────────────
if args.wandb_project != "disabled":
    try:
        import wandb
        wandb.init(
            project = args.wandb_project,
            name    = run_id,
            config  = vars(args),
        )
        logger.info("W&B initialised: %s/%s", args.wandb_project, run_id)
    except ImportError:
        logger.warning("wandb not installed — skipping experiment tracking")

# ── Load tokenizer and model ──────────────────────────────────────────────────
logger.info("Loading tokenizer: %s", args.model)
tokenizer = AutoTokenizer.from_pretrained(args.model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ── Build dataset ─────────────────────────────────────────────────────────────
data_dir = Path(args.data_dir)
desc_dir = Path(args.desc_dir)

def _resolve_train_path(task: str) -> Path:
    """Prefer the with-descriptions file if generate_descriptions.py was run."""
    with_desc = data_dir / f"{task}_train_with_desc.jsonl"
    plain     = data_dir / f"{task}_train.jsonl"
    if with_desc.exists():
        return with_desc
    return plain

metadata_paths = {task: _resolve_train_path(task) for task in TASKS}
# Separate descriptions JSON is only a fallback; inline client_description in the
# *_with_desc.jsonl takes precedence inside build_dataset.
descriptions_paths = {
    task: desc_dir / f"{task}_descriptions.json"
    for task in TASKS
}

# Check data exists
for task, path in metadata_paths.items():
    if not path.exists():
        logger.error(
            "Training data not found: %s\n"
            "Run scripts/generate_full.py, then "
            "scripts/generate_descriptions.py.", path
        )
        sys.exit(1)
    if path.name.endswith("_train.jsonl"):
        logger.warning(
            "Using %s without inline descriptions. For richer prompts run "
            "scripts/generate_descriptions.py --data-dir %s first.",
            path.name, data_dir,
        )

logger.info("Building dataset for tasks: %s", TASKS)
train_dataset = build_dataset(
    metadata_paths     = metadata_paths,
    descriptions_paths = descriptions_paths,
    tokenizer          = tokenizer,
)
logger.info("Dataset size: %d prompts", len(train_dataset))

# ── Build ScenarioStore ───────────────────────────────────────────────────────
ds_config = DatasetConfig()
store     = ScenarioStore(ds_config)

for task, path in metadata_paths.items():
    from finplanenv.dataset import DatasetGenerator
    gen  = DatasetGenerator(ds_config)
    meta = gen.load_metadata(path)
    store.build(meta)

logger.info("ScenarioStore: %d scenarios loaded", len(store))

# ── Build CDPO/GRPO config ────────────────────────────────────────────────────
# Use the dimensions of the first task (or portfolio if training all tasks)
primary_task = TASKS[0]
K = TASK_DIMS[primary_task]["K"]
M = TASK_DIMS[primary_task]["M"]
G = args.num_generations

logger.info("Method: %s  Task: %s  K=%d  M=%d  G=%d",
            args.method, primary_task, K, M, G)

cdpo_config = CDPOConfig(
    G               = G,
    beta_minus      = args.beta_minus,
    alpha_schedule  = args.alpha_schedule,
)
grpo_config = BaselineConfig(G=G)

# Advantage computer for the chosen method — its output goes STRAIGHT into the
# loss via CDPODecoupledGRPOTrainer (no reward-channel re-normalization).
advantage_computer = make_advantage_computer(
    method      = args.method,
    K           = K,
    M           = M,
    G           = G,
    cdpo_config = cdpo_config,
    grpo_config = grpo_config,
)

# Bridge: recovers per-signal rewards (r_hard/r_prox/r_soft) from completions.
advantage_bridge = AdvantageBridge(
    store              = store,
    reward_from_output = compute_rewards_from_output,
    K                  = K,
    M                  = M,
)

# Per-step trajectory logging for the CCR-vs-steps figure.
trajectory_logger = TrajectoryLogger(
    method  = args.method,
    task    = primary_task,
    seed    = args.seed,
    out_dir = str(out_dir),
)

# TRL still needs *a* reward function present; it is bypassed (output unused).
zero_reward_fn = make_zero_reward_fn(G)

# ── TRL GRPOConfig ────────────────────────────────────────────────────────────
trl_config = GRPOConfig(
    # Generation
    num_generations      = G,
    max_completion_length  = args.max_new_tokens,
    temperature          = 1.0,
    # NOTE: trl 0.14.0 GRPOConfig has no top_p field; sampling is temperature-only.

    # Training
    learning_rate        = args.lr,
    per_device_train_batch_size = max(args.batch_size, args.num_generations),
    gradient_accumulation_steps = 1,
    gradient_checkpointing = True,   # required for 7B backprop on <=48GB cards
    gradient_checkpointing_kwargs = {"use_reentrant": False},
    max_steps            = args.max_steps,
    seed                 = args.seed,

    # Output
    output_dir           = str(out_dir),
    logging_steps        = 1,
    save_steps           = 50,
    save_total_limit     = 3,

    # W&B
    report_to            = "wandb" if args.wandb_project != "disabled" else "none",
    run_name             = run_id,

    # Misc
    remove_unused_columns = False,
    dataloader_num_workers = 0,
    use_vllm = args.use_vllm,
    vllm_device = args.vllm_device,
    vllm_gpu_memory_utilization = args.vllm_gpu_mem_util,
)

# ── Load model ────────────────────────────────────────────────────────────────
logger.info("Loading model: %s", args.model)
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype  = torch.bfloat16,
    device_map   = "auto",
)
logger.info("Model loaded. Parameters: %.1fB",
            sum(p.numel() for p in model.parameters()) / 1e9)

# ── LoRA (default) ────────────────────────────────────────────────────────────
# All three methods use IDENTICAL LoRA config, so any difference in results is
# attributable to the advantage method, not the adapter. Flip with --no-lora.
#
# IMPORTANT: we pass the LoraConfig to the trainer via `peft_config` and let TRL
# apply it. In TRL 0.14.0, GRPOTrainer only skips building a full deepcopy
# reference model when it receives peft_config directly (ref_model=None, adapter
# disabled to get the reference policy). Pre-applying LoRA with get_peft_model
# makes TRL deepcopy the full base model -> OOM on a 24GB card with a 7B.
peft_config = None
if args.use_lora:
    from peft import LoraConfig
    peft_config = LoraConfig(
        r              = args.lora_r,
        lora_alpha     = args.lora_alpha,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_dropout   = 0.05,
        bias           = "none",
        task_type      = "CAUSAL_LM",
    )
    logger.info("LoRA enabled (r=%d, alpha=%d), applied by TRL (no ref-model copy).",
                args.lora_r, args.lora_alpha)
    # Gradient checkpointing + LoRA + frozen base requires this, else backward
    # fails with 'element 0 of tensors does not require grad'.
    model.enable_input_require_grads()

# ── Train ─────────────────────────────────────────────────────────────────────
trainer = CDPODecoupledGRPOTrainer(
    model              = model,
    processing_class   = tokenizer,
    reward_funcs       = [zero_reward_fn],   # bypassed; present to satisfy TRL
    args               = trl_config,
    train_dataset      = train_dataset,
    peft_config        = peft_config,
    advantage_computer = advantage_computer,
    advantage_bridge   = advantage_bridge,
    trajectory_logger  = trajectory_logger,
)

logger.info("Starting training — %d steps", args.max_steps)
trainer.train()

# ── Save ─────────────────────────────────────────────────────────────────────
final_path = out_dir / "final"
trainer.save_model(str(final_path))
tokenizer.save_pretrained(str(final_path))
logger.info("Model saved → %s", final_path)

# Save run config
with open(out_dir / "run_config.json", "w") as f:
    json.dump(vars(args), f, indent=2)

logger.info("Training complete. Run: %s", run_id)
