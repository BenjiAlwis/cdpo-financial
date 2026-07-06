"""
training/trainers.py
====================
CDPODecoupledGRPOTrainer — injects CDPO/GDPO/GRPO advantages into TRL's NATIVE,
memory-correct GRPO training loop, changing exactly one block.

Why this version
----------------
An earlier version copied TRL's compute_loss but rewrote internals (log-prob
loop, generation), diverging from TRL's tested memory behaviour and causing a
cascade of scale-only OOMs. This version copies TRL 0.14.0's compute_loss
VERBATIM and swaps ONLY the reward->advantage block:

    # native TRL (the bit we must change):
    #   rewards = rewards_per_func.sum(dim=1)
    #   advantages = (rewards - grouped_mean) / (grouped_std + 1e-4)
    # our swap:
    #   advantages = <CDPO/GDPO/GRPO decomposed advantage, used as-is>

Everything else — vLLM/regular generation, the memory-peak log-prob loop, KL,
masking, loss, metrics — is identical to TRL's implementation. This keeps TRL's
memory behaviour and makes vLLM available via args.use_vllm, while guaranteeing
the decomposed advantage reaches the loss un-renormalized.

Memory note: 7B training needs gradient checkpointing on <=48GB cards. When
checkpointing is enabled, the model must also have enable_input_require_grads()
called (TRL 0.14.0 does not do this); the train scripts handle that.

VERSION: trl==0.14.0. If you change TRL, diff GRPOTrainer.compute_loss against
this body and re-apply the one-block swap.
"""

from __future__ import annotations

import warnings

import numpy as np

try:
    import torch
    import trl
    from trl import GRPOTrainer
    from trl.data_utils import (
        apply_chat_template,
        is_conversational,
        maybe_apply_chat_template,
    )
    from trl.models import unwrap_model_for_generation
    from transformers import PreTrainedModel
    from accelerate.utils import broadcast_object_list, gather_object
    try:
        from trl.trainer.utils import pad
    except Exception:
        pad = None
    _TRL_OK = True
except Exception as _e:  # pragma: no cover
    _TRL_OK = False
    _IMPORT_ERR = _e

from finplanenv.cdpo import BatchRewards
from training.advantage_bridge import AdvantageBridge

_TESTED_TRL = "0.14.0"


if _TRL_OK:

    class CDPODecoupledGRPOTrainer(GRPOTrainer):
        """GRPOTrainer whose advantages come from a CDPO/GDPO/GRPO computer,
        injected into TRL's native compute_loss via a one-block swap."""

        def __init__(
            self,
            *args,
            advantage_computer,
            advantage_bridge: AdvantageBridge,
            trajectory_logger=None,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self.advantage_computer = advantage_computer
            self.advantage_bridge = advantage_bridge
            self.trajectory_logger = trajectory_logger
            self._cdpo_step = 0
            if trl.__version__ != _TESTED_TRL:
                warnings.warn(
                    f"CDPODecoupledGRPOTrainer written against trl=={_TESTED_TRL}; "
                    f"trl=={trl.__version__} installed. Diff compute_loss and "
                    "re-apply the advantage swap.",
                    RuntimeWarning,
                )

        def _cdpo_advantages(self, prompts, completions, device):
            """Recover per-signal rewards from completions, run the decomposed
            computer, return a (B*G,) advantage tensor. Replaces TRL's
            rewards.sum -> group z-score."""
            batch, diag = self.advantage_bridge.build_batch(
                prompts, completions, self.num_generations
            )
            adv_BG, _m = self.advantage_computer.compute(batch, step=self._cdpo_step)
            adv_flat = np.asarray(adv_BG, dtype=np.float32).reshape(-1)
            if self.trajectory_logger is not None:
                self.trajectory_logger.log_step(
                    step=self._cdpo_step,
                    r_hard=batch.r_hard, r_soft=batch.r_soft, A_hat=adv_BG,
                    parse_failures=diag["parse_failures"], total=diag["total"],
                )
            self._cdpo_step += 1
            return torch.tensor(adv_flat, device=device)

        def compute_loss(self, model, inputs, return_outputs=False,
                         num_items_in_batch=None):
            if return_outputs:
                raise ValueError("The GRPOTrainer does not support returning outputs")

            device = self.accelerator.device
            prompts = [x["prompt"] for x in inputs]
            prompts_text = [
                maybe_apply_chat_template(example, self.processing_class)["prompt"]
                for example in inputs
            ]
            prompt_inputs = self.processing_class(
                prompts_text, return_tensors="pt", padding=True,
                padding_side="left", add_special_tokens=False,
            )
            # GRPOTrainer._prepare_inputs is a no-op in trl 0.14.0, so inputs are
            # NOT moved to the accelerator device automatically. Move them
            # explicitly, otherwise generate() gets CPU input_ids vs CUDA model.
            prompt_inputs = {k: v.to(device) for k, v in prompt_inputs.items()}

            if self.max_prompt_length is not None:
                prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length:]
                prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length:]

            # ---- generation: vLLM or regular (native) ----
            if self.args.use_vllm:
                if self.state.global_step != self._last_loaded_step:
                    with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                        state_dict = unwrapped_model.state_dict()
                    if self.accelerator.is_main_process:
                        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                        llm_model.load_weights(state_dict.items())
                    self._last_loaded_step = self.state.global_step
                all_prompts_text = gather_object(prompts_text)
                if self.accelerator.is_main_process:
                    outputs = self.llm.generate(
                        all_prompts_text, sampling_params=self.sampling_params, use_tqdm=False
                    )
                    completion_ids = [out.token_ids for comp in outputs for out in comp.outputs]
                else:
                    completion_ids = [None] * len(all_prompts_text) * self.num_generations
                completion_ids = broadcast_object_list(completion_ids, from_process=0)
                process_slice = slice(
                    self.accelerator.process_index * len(prompts) * self.num_generations,
                    (self.accelerator.process_index + 1) * len(prompts) * self.num_generations,
                )
                completion_ids = completion_ids[process_slice]
                completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
                completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
                prompt_inputs_repeated = torch.repeat_interleave(
                    prompt_inputs["input_ids"], self.num_generations, dim=0
                )
                prompt_completion_ids = torch.cat([prompt_inputs_repeated, completion_ids], dim=1)
            else:
                with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                    prompt_completion_ids = unwrapped_model.generate(
                        **prompt_inputs, generation_config=self.generation_config
                    )

            prompt_length = prompt_inputs["input_ids"].size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]

            # ---- per-token log-probs (native memory-peak loop) ----
            def get_per_token_logps(model, input_ids, num_logits_to_keep):
                logits = model(input_ids, num_logits_to_keep=num_logits_to_keep + 1).logits
                logits = logits[:, :-1, :]
                per_token_logps = []
                for logits_row, input_ids_row in zip(logits, input_ids[:, -num_logits_to_keep:]):
                    log_probs = logits_row.log_softmax(dim=-1)
                    token_log_prob = torch.gather(
                        log_probs, dim=1, index=input_ids_row.unsqueeze(1)
                    ).squeeze(1)
                    per_token_logps.append(token_log_prob)
                return torch.stack(per_token_logps)

            num_logits_to_keep = completion_ids.size(1)
            per_token_logps = get_per_token_logps(model, prompt_completion_ids, num_logits_to_keep)

            with torch.inference_mode():
                if self.ref_model is not None:
                    ref_per_token_logps = get_per_token_logps(
                        self.ref_model, prompt_completion_ids, num_logits_to_keep
                    )
                else:
                    with self.accelerator.unwrap_model(model).disable_adapter():
                        ref_per_token_logps = get_per_token_logps(
                            model, prompt_completion_ids, num_logits_to_keep
                        )

            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps) - 1
            )

            is_eos = completion_ids == self.processing_class.eos_token_id
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

            completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
            if is_conversational(inputs[0]):
                completions = [[{"role": "assistant", "content": c}] for c in completions]

            prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]

            # ================= >>> CDPO ADVANTAGE SWAP <<< =================
            # Native TRL: rewards = rewards_per_func.sum(1);
            #             advantages = (rewards - grouped_mean)/(grouped_std+1e-4)
            # Replaced with the decomposed advantage, used directly.
            advantages = self._cdpo_advantages(prompts, completions, device)
            # =============== >>> END CDPO ADVANTAGE SWAP <<< ===============

            per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
            per_token_loss = -(per_token_loss - self.beta * per_token_kl)
            loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

            completion_length = self.accelerator.gather_for_metrics(
                completion_mask.sum(1)
            ).float().mean().item()
            self._metrics["completion_length"].append(completion_length)
            mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
            return loss

else:  # pragma: no cover

    class CDPODecoupledGRPOTrainer:  # type: ignore
        def __init__(self, *a, **k):
            raise ImportError(
                "trl/transformers not importable; install the 'train' extra. "
                f"Original error: {_IMPORT_ERR}"
            )
