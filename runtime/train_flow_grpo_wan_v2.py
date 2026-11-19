#!/usr/bin/env python
"""Flow-GRPO (clipped GRPO on the SDE-sampled flow-matching chain) with a LoRA adapter on Wan2.1-T2V-1.3B, prototype
area "video generation" (MULTIMODAL-SPEC row 3; prototype, unpromotable). Attempt 2 (amendment).

Attempt 2 (decided 2026-11-15, README "Attempt 2 (amendment)"): this file is train_flow_grpo_wan.py with only
the profiles and the default prompt directory changed; train_flow_grpo_wan.py stays byte-identical so that the first
attempt's launch records (which pin its SHA-256, EXT-83) still re-check. Differences, nothing else:
  - profiles `smoke2` (2 epochs, cut 2) and `full2` (100 epochs, cuts every 25): 8 prompts x 4 videos = 32 samples per
    epoch, still 2 optimizer steps per epoch (num_batches_per_epoch 2, gradient_accumulation_steps 1: 16 samples per
    step), training micro-batch 2 and sampling batch 4 as before; the profiles `smoke` and `full` do not exist here;
  - `--prompts` defaults to prompts-short/ (short-target shard, make_short_split.py).
The SHARED SAMPLER block is byte-identical to train_flow_grpo_wan.py's and eval_videos.py's.

A minimal port of yifan123/flow_grpo at commit 879042cf5707f8b90daa98d147d7deac2317c5da (MIT License, Copyright (c)
2025 Jie Liu) onto current diffusers + peft, configuration `general_ocr_wan2_1`, built from area 5's port
(the SD3.5 OCR recipe's `runtime/train_flow_grpo.py`: run layout, seeds, logs, cuts) and copied, not imported.
Ported pieces, cited by file:line of that commit:
  - the Wan SDE step with log-probability: flow_grpo/diffusers_patch/wan_pipeline_with_logprob.py:10-84;
  - the sampling loop with classifier-free guidance by two transformer calls: wan_pipeline_with_logprob.py:228-364;
  - prompt encoding (umT5, 512 tokens, cut at the mask, zero-padded): flow_grpo/diffusers_patch/wan_prompt_embedding.py
    :4-34;
  - `compute_log_prob`: scripts/train_wan2_1.py:190-230; the GRPO update with the KL term against the adapter-disabled
    transformer: train_wan2_1.py:878-1001;
  - advantages per prompt group with the group's std: flow_grpo/stat_tracking.py:5-36 (`global_std` False,
    config/grpo.py:64), cleared every epoch (train_wan2_1.py:829); the zero-advantage filter, its all-zero `+1e-6`
    rule and its padding to a multiple of the batch count: train_wan2_1.py:846-873;
  - the reward `video_ocr`: flow_grpo/rewards.py:143-158 and flow_grpo/ocr.py:65-123 (PaddleOCR on every 4th frame);
  - LoRA: r 32, alpha 64, gaussian init, flow_grpo's Wan target list: train_wan2_1.py:451-466;
  - hyperparameters: config/base.py:5-127 and config/grpo.py:29-81 (`general_ocr_wan2_1`).
flow_grpo's wan_pipeline_with_logprob.py is derived from huggingface/diffusers
`src/diffusers/pipelines/wan/pipeline_wan.py` (commit cb8e61ed2f792db8bcb9606c12e9ae7e400e4b49: "Copyright 2025 The Wan
Team and The HuggingFace Team. All rights reserved.", licensed under the Apache License, Version 2.0); the Wan SDE step
and the sampling loop ported from it carry that attribution (THIRD_PARTY_NOTICES.md).
The deployment sampler, the prompt encoder, the sampling loop and the reward are the evaluator's code: the shared
sampler block below is byte-identical in eval_videos.py (its SHA-256 is every cut's `sampler_code_sha256`).

Deviations from flow_grpo and from area 5's outputs: README.md "Trainer". In short: 17 frames instead of 33 (decision);
umT5 encodes every prompt of the run once at start and is freed before the transformer is loaded; transformer stored
in bf16 under bf16 autocast; training micro-batches of `train_batch` videos with flow_grpo's gradient; prompts in the
shard's file order; per-sample recorded noise seeds; EMA off (decision); no skipped first two epochs, no wandb, no
in-run evaluation; cuts every `save_every` epochs, written atomically with `cut.json` last.

Outputs in --out (the run's only read-write mount), as area 5:
  env-freeze.txt     `name==version` lines of the runtime, written before torch is imported (MM-33)
  env-runtime.json   interpreter, image CUDA version, lock and script SHA-256 (EXT-83), sampler code SHA-256
  run_args.json      every setting, seeds and their derivation, base model binding, data binding
  samples.jsonl      observed consumption log: one line per generated video (per-frame OCR), after the epoch's rewards
  epoch_plan.jsonl   per epoch, before its first optimizer step: kept and filtered (zero-advantage) sample ids
  train_steps.jsonl  one line per optimizer step: the sample ids whose gradients it applied, written before the step
  train_log.jsonl    per optimizer step: policy loss, KL, approx_kl, clipfrac, grad norm
  memory.jsonl       per epoch: seconds, peak device memory (sampling, training), peak RSS, reward statistics
  attempts.jsonl     one line per start and finish of this command (a rerun after a preemption restarts)
  checkpoint-<E>/    the cut after epoch E (1-based); `cut.json` written last

Usage (inside the container; README.md "Trainer" has the launch):
  python train_flow_grpo_wan_v2.py --profile smoke2|full2 --out /out [--seed S] [--epochs N] [--save-every K]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_NAME = "wan2.1-t2v-1.3b"
DEFAULT_BASE = HERE / "models" / BASE_NAME
DEFAULT_PROMPTS = HERE / "prompts-short"                      # attempt 2: short-target shard
MODELS_JSON = HERE / "models" / "MODELS.json"                 # written by fetch_models.py
# The SD3.5 OCR recipe's runtime (OCR engine A, its weights, the lock and the venv): the override
# GATEDTRAIN_RECIPE_FLOW_GRPO_WAN_VIDEO_OCR_RUNTIME, else the sibling checkout ../gatedtrain-recipe-flow-grpo-sd35-ocr/runtime.
OCR_RUNTIME = Path(os.environ.get("GATEDTRAIN_RECIPE_FLOW_GRPO_WAN_VIDEO_OCR_RUNTIME")
                   or HERE.parent.parent / "gatedtrain-recipe-flow-grpo-sd35-ocr" / "runtime")
OCR_MODELS = OCR_RUNTIME / "ocr-models"
OCR_MODELS_JSON = OCR_RUNTIME / "MODELS.json"
FRAME_INTERVAL_A = 4                                            # flow_grpo ocr.py:77 (used by the shared block)

# scripts/train_wan2_1.py:451-466 (bare names, matched by peft as suffixes: attn1 and attn2 of every block; the
# `add_*` names match nothing in the T2V model, `added_kv_proj_dim` null)
LORA = {"r": 32, "lora_alpha": 64, "init_lora_weights": "gaussian",
        "target_modules": ["add_k_proj", "add_q_proj", "add_v_proj", "to_add_out", "to_k", "to_out.0", "to_q", "to_v"]}

COMMON = {
    # training sampler (config/grpo.py:35-44; frames: README "Decisions", flow_grpo 33)
    "num_steps": 20,
    "guidance_scale": 4.5,
    "height": 240,
    "width": 416,
    "num_frames": 17,
    "max_sequence_length": 512,     # train_wan2_1.py:574, :656
    # deployment sampler = the evaluator's (eval_videos.py SAMPLER_DEFAULT): Euler ODE, 20 steps, CFG 4.5
    "deploy_num_steps": 20,
    # training (config/base.py:67-111, config/grpo.py:45-65)
    "num_batches_per_epoch": 2,     # config/grpo.py:45; gradient_accumulation_steps = 2 // 2 = 1 (:49)
    "learning_rate": 1.0e-4,
    "adam_beta1": 0.9, "adam_beta2": 0.999, "adam_weight_decay": 1.0e-4, "adam_epsilon": 1.0e-8,
    "max_grad_norm": 1.0,
    "adv_clip_max": 5.0,
    "clip_range": 1.0e-3,
    "timestep_fraction": 0.99,      # int(20 * 0.99) = 19 trained steps, the last one is not trained
    "beta": 0.004,                  # KL coefficient against the adapter-disabled transformer
    "train_cfg": True,              # config/base.py:95 (always on here)
    "num_inner_epochs": 1,
    "global_std": False,            # config/grpo.py:64: the group's own std
    "ema": False,                   # decision (area 5 and area 3 video): off; flow_grpo True (decay 0.9, every 8)
    # reward (config/grpo.py:74-76)
    "reward": "video_ocr",
    "prompt_order": "file",         # epoch e uses shard rows (e - 1) * P .. e * P - 1 in file order
    # memory knobs
    "precision": "bf16",            # transformer and umT5 bf16 (flow_grpo mixed_precision bf16), VAE fp32, LoRA fp32
    "gradient_checkpointing": True,
    "allow_tf32": True,
    "ocr_cpu_threads": 8,
}
# Attempt 2 (decided 2026-11-15): 100 epochs of 8 prompts x 4 videos. Attempt 1 measured 186-223 s per 16-video
# epoch, so a 32-video epoch is about 6.5-7.5 min and 100 epochs about 11-12.5 GPU-h; cuts every 25 epochs.
FULL_EPOCHS = 100
PROFILES = {
    # epochs, prompts per epoch P, videos per prompt G (key name kept from area 5), sampling batch (videos per
    # transformer batch), train micro-batch, cut every N epochs. P x G = 32 videos per epoch, 2 optimizer steps of 16
    # (num_batches_per_epoch 2, config/grpo.py:43-49).
    "smoke2": {"epochs": 2, "prompts_per_epoch": 8, "images_per_prompt": 4, "sample_batch": 4, "train_batch": 2,
               "save_every": 2},
    "full2": {"epochs": FULL_EPOCHS, "prompts_per_epoch": 8, "images_per_prompt": 4, "sample_batch": 4,
              "train_batch": 2, "save_every": 25},
}
DTYPES = ("bf16", "fp32")
CUT_MARKER = "cut.json"
LORA_WEIGHTS = "pytorch_lora_weights.safetensors"
CUT_FILES = (LORA_WEIGHTS, "adapter_config.json", "sampler.json", "base_ref.json", "trainer_state.json")
DEPLOYABLE = (LORA_WEIGHTS, "adapter_config.json", "sampler.json", "base_ref.json")
ALGORITHM_REFERENCE = "yifan123/flow_grpo 879042cf5707f8b90daa98d147d7deac2317c5da"
SAMPLE_ID_DOMAIN = "flow-grpo-wan"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def canonical(doc) -> bytes:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def write_json(path: Path, doc) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(doc, indent=2, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def rss_now_mb() -> float | None:
    """Current resident set size (Linux /proc), MB; rss_mb() is the peak."""
    try:
        pages = int(Path("/proc/self/statm").read_text().split()[1])
    except (OSError, IndexError, ValueError):
        return None
    return round(pages * os.sysconf("SC_PAGE_SIZE") / 2**20, 1)


# ---------------------------------------------------------------------------
# BEGIN SHARED SAMPLER (byte-identical in train_flow_grpo_wan.py and eval_videos.py; its SHA-256 is
# `sampler_code_sha256`)
# ---------------------------------------------------------------------------

def derive_seed(*parts) -> int:
    """63-bit seed from the SHA-256 of the parts joined by ':' (recorded derivation, MM-29)."""
    return int.from_bytes(hashlib.sha256(":".join(str(p) for p in parts).encode("utf-8")).digest()[:8], "big") >> 1


def rss_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def encode_prompts(tokenizer, text_encoder, prompts, max_sequence_length, batch_size=8):
    """flow_grpo wan_prompt_embedding.py:4-33 (`_get_t5_prompt_embeds`): tokenize to max_length with the attention
    mask, no prompt cleaning (unlike diffusers' WanPipeline, pipeline_wan.py `prompt_clean`), run the encoder with the
    mask, cast to the encoder's dtype, cut every row at its mask length. Returns {prompt: (L_i, D) tensor on the CPU},
    trimmed; `pad_embeds` restores the zero padding to max_sequence_length bit for bit."""
    import torch

    device = next(text_encoder.parameters()).device
    dtype = text_encoder.dtype                                                      # encode_prompt :78-79
    out = {}
    uniq = list(dict.fromkeys(prompts))
    with torch.no_grad():
        for b0 in range(0, len(uniq), batch_size):
            batch = uniq[b0:b0 + batch_size]
            inputs = tokenizer(batch, padding="max_length", max_length=max_sequence_length, truncation=True,
                               add_special_tokens=True, return_attention_mask=True, return_tensors="pt")  # :17-25
            ids, mask = inputs.input_ids, inputs.attention_mask
            seq_lens = mask.gt(0).sum(dim=1).long()                                 # :27
            embeds = text_encoder(ids.to(device), mask.to(device)).last_hidden_state.to(dtype=dtype)  # :29-30
            for k, p in enumerate(batch):
                out[p] = embeds[k, : int(seq_lens[k])].to("cpu").clone()            # :31
    return out


def pad_embeds(trimmed, max_sequence_length):
    """wan_prompt_embedding.py:32-34: zero rows up to max_sequence_length, stacked to (B, max_sequence_length, D)."""
    import torch

    return torch.stack([torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in trimmed])


def encode_all_prompts(base_dir, prompts, device, dtype, max_sequence_length, batch_size=8):
    """Load the umT5-XXL encoder, encode every prompt once, free the encoder (it never shares the device with the
    transformer: about 11 GB in bf16). As flow_grpo train_wan2_1.py:440, every module is cast to `dtype`
    (`.to(dtype)` also casts the `wo` layers that transformers keeps in float32 at load)."""
    import gc

    import torch
    from transformers import AutoTokenizer, UMT5EncoderModel

    base_dir = str(base_dir)
    kw = {"torch_dtype": dtype, "local_files_only": True}
    if str(device).startswith("cuda"):
        kw["device_map"] = str(device)       # load straight to the GPU: keeps the host RSS of the load low
    tokenizer = AutoTokenizer.from_pretrained(base_dir, subfolder="tokenizer", local_files_only=True)
    encoder = UMT5EncoderModel.from_pretrained(base_dir, subfolder="text_encoder", **kw).to(device=device,
                                                                                           dtype=dtype)
    encoder.requires_grad_(False)
    encoder.eval()
    out = encode_prompts(tokenizer, encoder, prompts, max_sequence_length, batch_size)
    del encoder
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return out


def load_transformer_vae(base_dir, device, dtype):
    """Wan transformer (frozen, `dtype`; diffusers keeps its `_keep_in_fp32_modules` in float32), VAE in float32
    (flow_grpo train_wan2_1.py:439), and the UniPC scheduler, used as the sigma table only (as flow_grpo)."""
    import torch
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanTransformer3DModel

    base_dir = str(base_dir)
    transformer = WanTransformer3DModel.from_pretrained(base_dir, subfolder="transformer", torch_dtype=dtype,
                                                        local_files_only=True).to(device)
    vae = AutoencoderKLWan.from_pretrained(base_dir, subfolder="vae", torch_dtype=torch.float32,
                                           local_files_only=True).to(device)
    scheduler = UniPCMultistepScheduler.from_pretrained(base_dir, subfolder="scheduler", local_files_only=True)
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    return transformer, vae, scheduler


def schedule(scheduler, num_steps):
    """wan_pipeline_with_logprob.py:229-230 and :41-42: `set_timesteps` of the UniPC scheduler (flow sigmas with
    flow_shift; int64 timesteps, float32 sigmas with the terminal 0), and the sigma index of each timestep by the
    scheduler's own `index_for_timestep`. Returns timesteps (N,) int64, sigmas (N + 1,) float32, step indices (list)."""
    scheduler.set_timesteps(num_steps, device="cpu")
    timesteps = scheduler.timesteps.clone()
    sigmas = scheduler.sigmas.clone().float()
    step_index = [int(scheduler.index_for_timestep(t)) for t in timesteps]
    return timesteps, sigmas, step_index


def wan_sde_step_with_logprob(sigmas, step_index, model_output, sample, deterministic, prev_sample=None, noise=None):
    """flow_grpo wan_pipeline_with_logprob.py:10-84. `step_index` (B,) long. float32 throughout (:35-39);
    std_dev_t = sigma_min + (sigma_max - sigma_min) * sigma with sigma_max = sigmas[1], sigma_min = sigmas[-1]
    (:47-51). Deterministic (the deployment sampler, :70-71): the Euler ODE step sample + dt * v, log_prob None.
    Returns (prev_sample, log_prob, prev_sample_mean, std_dev_t * sqrt(-dt)) (:84); log_prob is the mean over all but
    the batch dimension (:80)."""
    import torch

    model_output = model_output.float()
    sample = sample.float()
    shape = (-1,) + (1,) * (sample.ndim - 1)
    sigmas = sigmas.to(sample.device, torch.float32)
    sigma = sigmas[step_index.to(sample.device)].view(shape)                                  # :45
    sigma_prev = sigmas[step_index.to(sample.device) + 1].view(shape)                         # :46
    sigma_max = float(sigmas[1])                                                               # :47
    sigma_min = float(sigmas[-1])                                                              # :48
    dt = sigma_prev - sigma                                                                    # :49
    std_dev_t = sigma_min + (sigma_max - sigma_min) * sigma                                    # :51
    prev_sample_mean = (sample * (1 + std_dev_t ** 2 / (2 * sigma) * dt)
                        + model_output * (1 + std_dev_t ** 2 * (1 - sigma) / (2 * sigma)) * dt)  # :52
    if deterministic:
        return sample + dt * model_output, None, prev_sample_mean, std_dev_t * torch.sqrt(-1 * dt)  # :70-71
    if prev_sample is None:
        prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1 * dt) * noise               # :60-67
    prev_sample = prev_sample.float()
    log_prob = (-((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1 * dt)) ** 2))
                - torch.log(std_dev_t * torch.sqrt(-1 * dt))
                - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi))))                         # :73-77
    return (prev_sample, log_prob.mean(dim=tuple(range(1, log_prob.ndim))), prev_sample_mean,
            std_dev_t * torch.sqrt(-1 * dt))


def predict_velocity(transformer, latents, timestep, embeds, neg_embeds, guidance_scale, model_dtype,
                     autocast_dtype):
    """Classifier-free guidance as wan_pipeline_with_logprob.py:262-282: the latents cast to the model dtype, one
    transformer call with the prompt, one with the negative prompt, combined in float32 (flow_grpo casts both to the
    embeddings' dtype, float32 there, :272). Under autocast in `autocast_dtype` (flow_grpo: Accelerate bf16)."""
    import contextlib

    import torch

    ctx = (torch.autocast(latents.device.type, dtype=autocast_dtype) if autocast_dtype is not None
           else contextlib.nullcontext())
    x = latents.to(model_dtype)
    with ctx:
        cond = transformer(hidden_states=x, timestep=timestep, encoder_hidden_states=embeds,
                           return_dict=False)[0].float()
        uncond = transformer(hidden_states=x, timestep=timestep, encoder_hidden_states=neg_embeds,
                             return_dict=False)[0].float()
    return uncond + guidance_scale * (cond - uncond)


def latent_shape(transformer, vae, height, width, num_frames):
    """pipeline_wan.py prepare_latents: (C, (F - 1) // 4 + 1, H // 8, W // 8)."""
    t, s = vae.config.scale_factor_temporal, vae.config.scale_factor_spatial
    if (num_frames - 1) % t or height % 16 or width % 16:
        raise ValueError(f"num_frames - 1 must be divisible by {t} and height, width by 16 "
                         f"({height} x {width} x {num_frames})")
    return (transformer.config.in_channels, (num_frames - 1) // t + 1, height // s, width // s)


def decode_video(vae, latents):
    """wan_pipeline_with_logprob.py:352-364 for one video (1, C, f, h, w): latents / (1 / latents_std) +
    latents_mean, VAE decode in float32, VideoProcessor.postprocess_video(output_type='pt') = (x / 2 + 0.5) clamped
    to [0, 1]. Returns (F, 3, H, W) float32 on the CPU."""
    import torch

    z = latents.to(vae.dtype)
    mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(z.device, z.dtype)
    std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(z.device, z.dtype)
    z = z / std + mean
    video = vae.decode(z, return_dict=False)[0].float()
    video = (video / 2 + 0.5).clamp(0, 1)
    return video[0].permute(1, 0, 2, 3).cpu()


def sample_batch(transformer, vae, scheduler, embeds, neg_embeds, seeds, num_steps, guidance_scale, height, width,
                 num_frames, device, model_dtype, autocast_dtype, deterministic=True, keep_chain=True, decode=True):
    """One batch of wan_pipeline_with_logprob.py:228-364: initial latents and every step's noise from a CPU
    torch.Generator seeded with the sample's seed (device independent), the Wan SDE step (deterministic: the Euler
    ODE), CFG with two calls. Returns (videos, latents chain (B, N + 1, ...) or None, log-probs (B, N) or None,
    timesteps); videos is a list of (F, 3, H, W) float32 CPU tensors in [0, 1], decoded one by one (None without
    decode)."""
    import torch

    timesteps, sigmas, step_index = schedule(scheduler, num_steps)
    shape = latent_shape(transformer, vae, height, width, num_frames)
    gens = [torch.Generator("cpu").manual_seed(int(s)) for s in seeds]
    latents = torch.stack([torch.randn(shape, generator=g, dtype=torch.float32) for g in gens]).to(device)
    n = latents.shape[0]
    all_latents, all_logp = [latents], []
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            tb = torch.full((n,), int(t), dtype=torch.int64, device=device)                   # t.expand(B), :263
            v = predict_velocity(transformer, latents, tb, embeds, neg_embeds, guidance_scale, model_dtype,
                                 autocast_dtype)
            noise = None
            if not deterministic:
                noise = torch.stack([torch.randn(shape, generator=g, dtype=torch.float32) for g in gens]).to(device)
            idx = torch.full((n,), step_index[i], dtype=torch.long)
            latents, logp, _, _ = wan_sde_step_with_logprob(sigmas, idx, v, latents, deterministic, noise=noise)
            if keep_chain:
                all_latents.append(latents)
                if logp is not None:
                    all_logp.append(logp)
        videos = [decode_video(vae, latents[k:k + 1]) for k in range(n)] if decode else None
    chain = torch.stack(all_latents, dim=1) if keep_chain else None
    logp = torch.stack(all_logp, dim=1) if keep_chain and all_logp else None
    return videos, chain, logp, timesteps


def video_to_uint8(video):
    """(F, 3, H, W) float in [0, 1] -> (F, H, W, 3) uint8 RGB, as flow_grpo rewards.py:148-153."""
    import torch

    return (video * 255).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy().copy()


def target_text(prompt: str) -> str:
    """flow_grpo ocr.py:86: the text between the first pair of double quotes."""
    return prompt.split('"')[1]


def video_frame_accuracy(recognized: str, target: str) -> float:
    """flow_grpo ocr.py:91, :106-116 (`OcrScorer_video_or_image`): spaces removed and lower case on both sides,
    1 - min(Levenshtein distance, len(target)) / len(target); no substring shortcut (unlike the image reward)."""
    from rapidfuzz.distance.Levenshtein import distance

    recognized = recognized.replace(" ", "").lower()
    target = target.replace(" ", "").lower()
    dist = min(distance(recognized, target), len(target))
    return 1 - dist / len(target)


def video_ocr_reward(frame_scores) -> float:
    """flow_grpo ocr.py:117-123: the mean over the frames with a positive score, 0.0 if there is none."""
    positive = [s for s in frame_scores if s > 0]
    return sum(positive) / len(positive) if positive else 0.0


class PaddleOcrA:
    """OCR engine A, the training reward: flow_grpo ocr.py:65-123 (`PaddleOCR(use_angle_cls=False, lang="en",
    use_gpu=False, show_log=False)`, `.ocr(frame, cls=False)`, text = concatenation of the recognized lines with a
    positive score). Built from paddleocr 2.10.0's own `TextSystem` (the class `PaddleOCR` wraps, paddleocr.py:
    628-694) so that area 5's local, offline model directories are used: `PaddleOCR.__init__` downloads the default
    PP-OCR tar files from paddleocr.bj.bcebos.com when `inference.pdmodel` is absent (ppocr/utils/network.py:
    106-128). `PaddleOCR.ocr` for one ndarray is reproduced (paddleocr.py:744-772). As in flow_grpo, the RGB array is
    passed as is (PaddleOCR treats a 3-channel ndarray as BGR). Paddle 3.0's inference predictor rejects
    `enable_memory_optim()` for PIR (`.json`) models on the CPU; that call, a memory-planning switch with no effect on
    results, is disabled (area 5 README "Changes")."""

    def __init__(self, model_root, cpu_threads=8):
        import paddle.inference as pinf
        from paddleocr import paddleocr as po
        from paddleocr.tools.infer.predict_system import TextSystem

        pinf.Config.enable_memory_optim = lambda self, *a, **k: None
        params = po.parse_args(mMain=False)
        params.__dict__.update(use_angle_cls=False, lang="en", use_gpu=False, show_log=False,
                               det_model_dir=str(Path(model_root) / "paddleocr" / "PP-OCRv4_mobile_det"),
                               rec_model_dir=str(Path(model_root) / "paddleocr" / "en_PP-OCRv4_mobile_rec"),
                               rec_char_dict_path=str(Path(po.__file__).parent / "ppocr" / "utils" / "en_dict.txt"),
                               rec_image_shape="3, 48, 320", cpu_threads=cpu_threads)
        self.po = po
        self.system = TextSystem(params)

    def text(self, arr) -> str:
        img, _, _ = self.po.check_img(arr, (255, 255, 255))
        img = self.po.alpha_to_color(img, (255, 255, 255))
        boxes, rec, _ = self.system(img, False, {})
        if not boxes and not rec:
            return ""
        return "".join(r[0] if r[1] > 0 else "" for r in rec)                                 # ocr.py:106

    def video_reward(self, frames, prompt) -> dict:
        """flow_grpo ocr.py:86-123 for one video (F, H, W, 3) uint8: frames[::4]; an OCR exception on a frame gives
        that frame the maximum distance (:112-114), as in the training reward."""
        target = target_text(prompt)
        per_frame = []
        for f in range(0, len(frames), FRAME_INTERVAL_A):
            try:
                text = self.text(frames[f])
                acc, err = video_frame_accuracy(text, target), None
            except Exception as e:                                                             # noqa: BLE001
                text, acc, err = None, 0.0, f"{type(e).__name__}: {e}"[:300]
            per_frame.append({"frame": f, "text": text, "acc": acc, "error": err})
        return {"frames": per_frame, "reward": video_ocr_reward([r["acc"] for r in per_frame])}


def sampler_doc(num_steps, guidance_scale, height, width, num_frames, max_sequence_length, scheduler_config,
                timesteps, sigmas, step_index, transformer_dtype, encoder_dtype, deterministic=True) -> dict:
    """The deployment sampler (08 section 3.6 'scheduler config'), stored in every cut as sampler.json; the weight
    precisions are part of it because they change the videos."""
    scheduler_config = {k: (sorted(v) if k == "_use_default_values" else v) for k, v in scheduler_config.items()}
    return {"pipeline": "flow_grpo wan_pipeline_with_logprob (ported)",
            "scheduler": "UniPCMultistepScheduler (sigma table and index_for_timestep only)",
            "scheduler_config": scheduler_config, "num_steps": num_steps, "deterministic": deterministic,
            "step": "flow_grpo wan sde_step_with_logprob; deterministic = Euler ODE step sample + dt * v",
            "timesteps": [int(t) for t in timesteps], "sigmas": [round(float(s), 8) for s in sigmas],
            "step_index": list(step_index), "guidance_scale": guidance_scale,
            "cfg": "two transformer calls (prompt, negative prompt), combined in float32",
            "height": height, "width": width, "num_frames": num_frames, "negative_prompt": "",
            "max_sequence_length": max_sequence_length,
            "prompt_embedding": "flow_grpo wan_prompt_embedding.py: no prompt cleaning, cut at the attention mask, "
                                "zero-padded to max_sequence_length",
            "noise": "per sample: torch.Generator('cpu').manual_seed(seed); initial latents, then one draw per step "
                     "when not deterministic", "latents_dtype": "float32", "vae_dtype": "float32",
            "decode": "latents * latents_std + latents_mean, VAE decode, (x / 2 + 0.5) clamped to [0, 1], "
                      "uint8 = round(x * 255)",
            "transformer_dtype": transformer_dtype, "text_encoder_dtype": encoder_dtype,
            "autocast": "bfloat16" if transformer_dtype == "bf16" else None}

# ---------------------------------------------------------------------------
# END SHARED SAMPLER
# ---------------------------------------------------------------------------


def sampler_code_sha256() -> str:
    text = Path(__file__).read_text(encoding="utf-8")
    start = text.index("# BEGIN SHARED SAMPLER")
    end = text.index("# END SHARED SAMPLER")
    return sha256_bytes(text[start:end].encode("utf-8"))


class PerPromptStatTracker:
    """flow_grpo stat_tracking.py:5-36, type 'grpo', cleared after every epoch (train_wan2_1.py:822)."""

    def __init__(self, global_std: bool):
        self.global_std = global_std

    def update(self, prompts, rewards):
        import numpy as np

        prompts, rewards = np.array(prompts), np.array(rewards, dtype=np.float64)
        adv = np.zeros_like(rewards)
        for p in np.unique(prompts):
            pr = rewards[prompts == p]
            mean = np.mean(pr)
            std = (np.std(rewards) if self.global_std else np.std(pr)) + 1e-4
            adv[prompts == p] = (pr - mean) / std
        return adv


def grpo_terms(transformer, lat, nxt, old_lp, adv, timestep, step_idx, embeds, neg_embeds, sigmas, s, model_dtype,
               autocast_dtype):
    """One trained timestep of a micro-batch, train_wan2_1.py:935-960: the velocity with CFG (`compute_log_prob`,
    :190-230), the log-probability of the recorded next latents, the PPO-clipped policy loss per video (:949-956) and
    the KL term per video against the adapter-disabled transformer (:938-941, :959-961: mean squared difference of
    the two step means over twice the step variance (std_dev_t * sqrt(-dt))^2; 0 when beta is 0). Returns (policy
    loss, KL, log-probs, ratios), each (B,)."""
    import torch

    vel = predict_velocity(transformer, lat, timestep, embeds, neg_embeds, s["guidance_scale"], model_dtype,
                           autocast_dtype)
    _, lp, mean, std = wan_sde_step_with_logprob(sigmas, step_idx, vel, lat, False, prev_sample=nxt)
    ratio = torch.exp(lp - old_lp)
    per = torch.maximum(-adv * ratio, -adv * torch.clamp(ratio, 1.0 - s["clip_range"], 1.0 + s["clip_range"]))
    kl = torch.zeros_like(per)
    if s["beta"] > 0:
        with torch.no_grad():
            transformer.disable_adapters()
            try:
                vref = predict_velocity(transformer, lat, timestep, embeds, neg_embeds, s["guidance_scale"],
                                        model_dtype, autocast_dtype)
            finally:
                transformer.enable_adapters()
            _, _, mean_ref, _ = wan_sde_step_with_logprob(sigmas, step_idx, vref, lat, False, prev_sample=nxt)
        kl = ((mean - mean_ref) ** 2).mean(dim=tuple(range(1, mean.ndim))) / (2 * std.view(-1) ** 2)
    return per, kl, lp, ratio


# ---------------------------------------------------------------------------
# Pre-step (MM-33)
# ---------------------------------------------------------------------------

def uv_lock() -> Path:
    """This area has no lock of its own: it runs area 5's venv (README "Proposed trainer")."""
    lock = HERE / "uv.lock"
    return lock if lock.exists() else OCR_RUNTIME / "uv.lock"


def write_env_freeze(out: Path) -> dict:
    pkgs = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            pkgs[name.lower().replace("_", "-")] = dist.version
    lines = [f"{k}=={pkgs[k]}" for k in sorted(pkgs)]
    lines.append(f"# flow_grpo algorithm reference {ALGORITHM_REFERENCE} (ported)")
    (out / "env-freeze.txt").write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    lock = uv_lock()
    runtime = {
        "python": platform.python_version(), "executable": sys.executable,
        "image_cuda_version": os.environ.get("CUDA_VERSION"),
        "uv_lock": str(lock), "uv_lock_sha256": sha256_file(lock) if lock.exists() else None,
        "script_sha256": sha256_file(Path(__file__).resolve()), "sampler_code_sha256": sampler_code_sha256(),
        "uid": os.getuid(), "gid": os.getgid(), "cpu_count": os.cpu_count(),
        "env_freeze_sha256": sha256_file(out / "env-freeze.txt"),
    }
    write_json(out / "env-runtime.json", runtime)
    return runtime


def prepare_out(out: Path) -> int:
    """As areas 3 and 5: a non-empty --out is a rerun after a preemption; earlier contents move to
    `preempted-attempt-<k>/` (never deleted). Returns the attempt number."""
    out.mkdir(parents=True, exist_ok=True)
    existing = [p for p in out.iterdir() if not p.name.startswith("preempted-attempt-")]
    previous = sorted(p for p in out.iterdir() if p.name.startswith("preempted-attempt-"))
    attempt = len(previous) + 1
    if existing:
        attempt = len(previous) + 2
        dest = out / f"preempted-attempt-{attempt - 1}"
        dest.mkdir()
        for p in existing:
            p.rename(dest / p.name)
        if (dest / "attempts.jsonl").exists():
            shutil.copyfile(dest / "attempts.jsonl", out / "attempts.jsonl")
    append_jsonl(out / "attempts.jsonl", {"attempt": attempt, "started_at": utc_now(), "argv": sys.argv})
    return attempt


# ---------------------------------------------------------------------------
# Bindings
# ---------------------------------------------------------------------------

def base_binding(base: Path, verify: bool) -> dict:
    """Base model reference (MM-18, MM-21): repo, 40-hex revision and every file's SHA-256 from models/MODELS.json
    (the evaluator's record; `.cache/` excluded as area 5); with `verify`, every file is re-hashed (about 29 GB)."""
    record = json.loads(MODELS_JSON.read_text(encoding="utf-8"))["models"][BASE_NAME]
    files = record["files"]
    on_disk = []
    for p in sorted(base.rglob("*")):
        rel = p.relative_to(base).as_posix()
        if rel == ".cache" or rel.startswith(".cache/"):
            continue
        if p.is_symlink():
            raise SystemExit(f"{p} is a symbolic link; the base directory must be plain (MM-20)")
        if p.is_file():
            on_disk.append(rel)
    if on_disk != sorted(files):
        raise SystemExit(f"{base} does not hold exactly the files of {MODELS_JSON} [{BASE_NAME}]")
    if verify:
        for rel, want in files.items():
            if sha256_file(base / rel) != want["sha256"]:
                raise SystemExit(f"{base / rel} differs from its SHA-256 in {MODELS_JSON}")
    rev = record["revision"]
    if len(rev) != 40 or any(c not in "0123456789abcdef" for c in rev):
        raise SystemExit(f"base revision {rev!r} is not a 40-hex commit (MM-18)")
    digests = {k: v["sha256"] for k, v in files.items()}
    return {"repo": record["repo"], "revision": rev, "local_dir": str(base), "files": digests,
            "files_sha256": sha256_bytes(canonical(digests)), "verified": verify}


def read_prompts(prompts_dir: Path) -> tuple[list[dict], Path]:
    path = prompts_dir / "train.jsonl"
    rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    ids = [r["prompt_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{path}: prompt ids are not unique")
    return rows, path


def epoch_prompts(prompts: list[dict], epoch: int, per_epoch: int) -> list[dict]:
    """Epoch e (1-based) uses shard rows (e - 1) * P .. e * P - 1 in file order, wrapping at the end of the shard
    (prompts-short: 5,024 rows, full2 reads 800); the P prompts of an epoch always differ."""
    n = len(prompts)
    return [prompts[((epoch - 1) * per_epoch + k) % n] for k in range(per_epoch)]


# ---------------------------------------------------------------------------
# Cuts
# ---------------------------------------------------------------------------

def canonicalize_safetensors_header(path: Path) -> None:
    """As areas 3 and 5: rewrite the safetensors header as sorted, compact JSON padded to 8 bytes (tensor bytes and
    offsets unchanged) and sort the LoraConfig's `target_modules`, so the file hash depends on the weights and
    metadata only."""
    import struct

    raw = path.read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + n])
    meta = header.get("__metadata__", {})
    if "lora_adapter_metadata" in meta:
        cfg = json.loads(meta["lora_adapter_metadata"])
        if isinstance(cfg.get("target_modules"), list):
            cfg["target_modules"] = sorted(cfg["target_modules"])
        meta["lora_adapter_metadata"] = json.dumps(cfg, indent=2, sort_keys=True)
    text = json.dumps(header, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    text += b" " * (-len(text) % 8)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(text)) + text + raw[8 + n:])
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_cut(transformer, out: Path, epoch: int, global_step: int, samples_consumed: int, samples_filtered: int,
             base_ref: dict, sampler: dict, lora_cfg: dict, extra: dict) -> Path:
    """checkpoint-<epoch>/ via .checkpoint-<epoch>.partial/ and an atomic rename; cut.json last (as areas 3, 5)."""
    cut = out / f"checkpoint-{epoch}"
    if cut.exists():
        return cut
    tmp = out / f".checkpoint-{epoch}.partial"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    transformer.save_lora_adapter(str(tmp), adapter_name="default", safe_serialization=True, weight_name=LORA_WEIGHTS)
    canonicalize_safetensors_header(tmp / LORA_WEIGHTS)
    os.chmod(tmp / LORA_WEIGHTS, 0o644)
    write_json(tmp / "adapter_config.json", {**lora_cfg, "base_model_name_or_path": base_ref["repo"],
                                             "base_model_revision": base_ref["revision"], "format": "peft",
                                             "loader": "WanTransformer3DModel.load_lora_adapter(dir, prefix=None)",
                                             "weights": "raw"})
    write_json(tmp / "sampler.json", sampler)
    write_json(tmp / "base_ref.json", {k: base_ref[k] for k in ("repo", "revision", "files", "files_sha256")})
    write_json(tmp / "trainer_state.json", {
        "epoch": epoch, "global_step": global_step, "samples_consumed": samples_consumed,
        "samples_filtered": samples_filtered,
        "note": "resume-only information; the run does not resume from a cut (a preempted run restarts)"})
    files = {name: sha256_file(tmp / name) for name in CUT_FILES}
    write_json(tmp / CUT_MARKER, {
        "schema": "ptc_mm.flow_grpo_wan_cut", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
        "epoch": epoch, "global_step": global_step, "samples_consumed": samples_consumed,
        "samples_filtered": samples_filtered, "files": files, "deployable": list(DEPLOYABLE),
        "created_at": utc_now(), **extra,
    })
    os.replace(tmp, cut)
    return cut


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--profile", choices=sorted(PROFILES), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--base", type=Path, default=DEFAULT_BASE)
    p.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    p.add_argument("--seed", type=int, default=5000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, help="override the profile's epoch count (full: set from the smoke)")
    p.add_argument("--save-every", type=int, help="override the profile's cut interval in epochs (the last epoch "
                                                  "is always cut)")
    p.add_argument("--no-verify-base", action="store_true", help="skip re-hashing the base files (host checks)")
    p.add_argument("--train-batch", type=int, help="memory knob: videos per training forward/backward (default 2)")
    p.add_argument("--sample-batch", type=int, help="memory knob: videos per sampling batch (default 4)")
    p.add_argument("--no-gradient-checkpointing", action="store_true", help="memory knob: faster, more memory")
    p.add_argument("--beta", type=float, help="budget knob: KL coefficient (flow_grpo 0.004; 0 skips the reference "
                                              "forward)")
    return p.parse_args(argv)


def resolve_settings(args) -> dict:
    s = {**COMMON, **PROFILES[args.profile]}
    for key in ("epochs", "save_every", "train_batch", "sample_batch"):
        if getattr(args, key) is not None:
            s[key] = getattr(args, key)
    if args.no_gradient_checkpointing:
        s["gradient_checkpointing"] = False
    if args.beta is not None:
        s["beta"] = args.beta
    for key in ("epochs", "save_every", "train_batch", "sample_batch", "prompts_per_epoch", "images_per_prompt"):
        if not (isinstance(s[key], int) and s[key] >= 1):
            raise SystemExit(f"{key} must be a positive integer, not {s[key]!r}")
    if s["precision"] not in DTYPES:
        raise SystemExit(f"precision {s['precision']!r} is not one of {DTYPES}")
    if s["ema"]:
        raise SystemExit("EMA is off by decision (README \"Decisions\"); this port has no EMA")
    s["seed"] = args.seed
    s["samples_per_epoch"] = s["prompts_per_epoch"] * s["images_per_prompt"]
    nb = s["num_batches_per_epoch"]
    # flow_grpo: samples_per_epoch = train_batch_size x num_batches_per_epoch; training splits the kept samples into
    # num_batches micro-batches (train_wan2_1.py:899) and accumulates over gradient_accumulation_steps of them
    # (config/grpo.py:49: nb // 2 if nb > 1 else 1), so two optimizer steps per epoch.
    if s["samples_per_epoch"] % nb:
        raise SystemExit(f"samples per epoch {s['samples_per_epoch']} must be a multiple of num_batches_per_epoch {nb}")
    s["gradient_accumulation_steps"] = nb // 2 if nb > 1 else 1
    if nb % s["gradient_accumulation_steps"]:
        raise SystemExit("num_batches_per_epoch must be a multiple of gradient_accumulation_steps")
    s["num_train_timesteps"] = int(s["num_steps"] * s["timestep_fraction"])                  # train_wan2_1.py:373
    return s


def main(argv=None) -> int:
    args = parse_args(argv)
    s = resolve_settings(args)
    out = args.out.resolve()
    attempt = prepare_out(out)
    runtime = write_env_freeze(out)                 # before torch is imported (MM-33)
    t_start = time.time()
    base_ref = base_binding(args.base.resolve(), verify=not args.no_verify_base)
    prompts, prompts_file = read_prompts(args.prompts)
    P, G, B, E = s["prompts_per_epoch"], s["images_per_prompt"], s["sample_batch"], s["epochs"]
    if P > len(prompts):
        raise SystemExit(f"{P} prompts per epoch but the shard holds {len(prompts)}")
    plans = {epoch: epoch_prompts(prompts, epoch, P) for epoch in range(1, E + 1)}

    import concurrent.futures
    import gc

    import numpy as np
    import torch
    from peft import LoraConfig

    device = torch.device(args.device)
    cuda = device.type == "cuda"
    model_dtype = torch.bfloat16 if s["precision"] == "bf16" else torch.float32
    autocast_dtype = torch.bfloat16 if s["precision"] == "bf16" else None
    if s["allow_tf32"] and cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    L = s["max_sequence_length"]
    mem_log = {"rss_mb_start": rss_mb()}

    # --- umT5: every prompt of the run once, then freed (GPU probe: 11.3 GB GPU, 21 GB host RSS peak) ---
    t0 = time.time()
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    run_prompts = [""] + [r["prompt"] for e in plans for r in plans[e]]
    emb = encode_all_prompts(args.base, run_prompts, device, model_dtype, L)
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()
        mem_log["peak_allocated_gb_encode"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 3)
        mem_log["peak_reserved_gb_encode"] = round(torch.cuda.max_memory_reserved(device) / 2**30, 3)
        mem_log["allocated_gb_after_encode"] = round(torch.cuda.memory_allocated(device) / 2**30, 3)
        torch.cuda.reset_peak_memory_stats(device)
    mem_log.update(seconds_encode=round(time.time() - t0, 1), rss_mb_after_encode=rss_mb(),
                   rss_now_mb_after_encode=rss_now_mb(), prompts_encoded=len(emb))
    neg = pad_embeds([emb[""]], L).to(device, model_dtype)                     # negative prompt "" (:574)

    transformer, vae, scheduler = load_transformer_vae(args.base, device, model_dtype)
    torch.manual_seed(derive_seed("torch", s["seed"]))   # peft gaussian init draws from the global RNG
    lora_config = LoraConfig(**LORA)
    transformer.add_adapter(lora_config)
    for prm in transformer.parameters():            # LoRA parameters in fp32 (diffusers cast_training_params)
        if prm.requires_grad:
            prm.data = prm.data.float()
    if s["gradient_checkpointing"]:
        transformer.enable_gradient_checkpointing()                            # train_wan2_1.py:475
    params = [prm for prm in transformer.parameters() if prm.requires_grad]
    lora_cfg = json.loads(json.dumps(lora_config.to_dict(), default=lambda o: sorted(o) if isinstance(o, set)
                                     else str(o)))
    optimizer = torch.optim.AdamW(params, lr=s["learning_rate"], betas=(s["adam_beta1"], s["adam_beta2"]),
                                  weight_decay=s["adam_weight_decay"], eps=s["adam_epsilon"])
    # the deployment sampler (the evaluator's: eval_videos.py resolve_sampler builds the same document)
    d_ts, d_sig, d_idx = schedule(scheduler, s["deploy_num_steps"])
    sampler = sampler_doc(s["deploy_num_steps"], s["guidance_scale"], s["height"], s["width"], s["num_frames"], L,
                          dict(scheduler.config), d_ts, d_sig, d_idx, s["precision"], s["precision"],
                          deterministic=True)
    train_ts, train_sig, train_idx = schedule(scheduler, s["num_steps"])
    train_sampler = sampler_doc(s["num_steps"], s["guidance_scale"], s["height"], s["width"], s["num_frames"], L,
                                dict(scheduler.config), train_ts, train_sig, train_idx, s["precision"],
                                s["precision"], deterministic=False)
    reward_a = PaddleOcrA(OCR_MODELS, cpu_threads=s["ocr_cpu_threads"])
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)   # rewards overlap sampling (train_wan2_1.py:708)
    mem_log.update(rss_mb_after_load=rss_mb(), rss_now_mb_after_load=rss_now_mb(),
                   lora_params=sum(p.numel() for p in params))
    if cuda:
        mem_log["allocated_gb_after_load"] = round(torch.cuda.memory_allocated(device) / 2**30, 3)

    def score(frames_list, prompts_list):
        return [reward_a.video_reward(f, p) for f, p in zip(frames_list, prompts_list)]

    ocr_models = json.loads(OCR_MODELS_JSON.read_text(encoding="utf-8"))["models"]
    run_args = {
        "schema": "ptc_mm.flow_grpo_wan_run_args", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
        "profile": args.profile, "attempt": attempt, "settings": s, "lora": lora_cfg, "seed": s["seed"],
        "seed_derivation": ("derive_seed(*parts) = first 8 bytes of SHA-256(':'.join(parts)) >> 1. Epoch e prompts: "
                            "shard rows (e - 1) * P .. e * P - 1 in file order (no random draw); sample i of epoch e: "
                            "noise seed derive_seed('sample', seed, e, i) for a CPU torch.Generator (initial latents, "
                            "then one draw per denoising step); zero-advantage padding: "
                            "torch.Generator('cpu').manual_seed(derive_seed('mask', seed, e)); training order: "
                            "torch.Generator('cpu').manual_seed(derive_seed('perm', seed, e, inner)); LoRA init: "
                            "torch global RNG seeded with derive_seed('torch', seed)"),
        "base_model": base_ref["local_dir"],
        "base_ref": {k: base_ref[k] for k in ("repo", "revision", "files_sha256", "verified")},
        "train_file": str(prompts_file), "train_file_sha256": sha256_file(prompts_file),
        "n_train_prompts": len(prompts),
        "sampler": sampler, "sampler_sha256": sha256_bytes(canonical(sampler)), "training_sampler": train_sampler,
        "reward": {"name": "video_ocr", "engine": "PaddleOCR (paddleocr 2.10.0 TextSystem, paddlepaddle CPU)",
                   "models": {k: ocr_models.get(k, {}).get("revision")
                              for k in ("paddleocr/PP-OCRv4_mobile_det", "paddleocr/en_PP-OCRv4_mobile_rec")},
                   "frames": f"every {FRAME_INTERVAL_A}th frame from 0",
                   "formula": "flow_grpo ocr.py:65-123: per frame 1 - min(lev(text, target), len) / len on space-free "
                              "lower case; video = mean over the frames with a positive score, 0 if none; an OCR "
                              "exception scores its frame 0"},
        "consumption_log": "samples.jsonl", "epoch_plan": "epoch_plan.jsonl", "step_log": "train_steps.jsonl",
        "consumption_semantics": ("samples.jsonl: one line per generated video after the epoch's rewards; every sample "
                                  "of epoch e is either consumed by exactly one optimizer step of epoch e per inner "
                                  "epoch (train_steps.jsonl) or filtered (epoch_plan.jsonl `filtered`: its advantage "
                                  "is 0, train_wan2_1.py:846-873). When every advantage of the epoch is 0, flow_grpo "
                                  "adds 1e-6 to all of them and trains on every sample (epoch_plan.jsonl "
                                  "`all_zero_advantage` true, nothing filtered)"),
        "device": str(device), "dtype": str(model_dtype),
        "gpu": torch.cuda.get_device_name(device) if cuda else None,
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "sampler_code_sha256": runtime["sampler_code_sha256"], "script_sha256": runtime["script_sha256"],
        "memory_at_start": mem_log, "created_at": utc_now(),
    }
    write_json(out / "run_args.json", run_args)

    global_step = 0
    samples_consumed = 0
    samples_filtered = 0
    tracker = PerPromptStatTracker(s["global_std"])
    T = s["num_train_timesteps"]
    nb = s["num_batches_per_epoch"]
    gas = s["gradient_accumulation_steps"]
    for epoch in range(1, E + 1):
        t_ep = time.time()
        if cuda:
            torch.cuda.reset_peak_memory_stats(device)
        # ---------------- sampling (train_wan2_1.py:638-731) ----------------
        transformer.eval()
        plan = [(row, g) for row in plans[epoch] for g in range(G)]
        n = len(plan)
        chains, logps = [], []
        futures, sample_ids, prompt_ids, seeds_all, nonfinite = [], [], [], [], []
        for b0 in range(0, n, B):
            batch = plan[b0:b0 + B]
            idx = list(range(b0, b0 + len(batch)))
            seeds = [derive_seed("sample", s["seed"], epoch, i) for i in idx]
            e = pad_embeds([emb[r["prompt"]] for r, _ in batch], L).to(device, model_dtype)
            videos, chain, logp, _ = sample_batch(transformer, vae, scheduler, e, neg.expand(len(batch), -1, -1),
                                                  seeds, s["num_steps"], s["guidance_scale"], s["height"], s["width"],
                                                  s["num_frames"], device, model_dtype, autocast_dtype,
                                                  deterministic=False, keep_chain=True, decode=True)
            chains.append(chain)
            logps.append(logp)
            nonfinite.extend(not bool(torch.isfinite(v).all()) for v in videos)
            frames = [video_to_uint8(v) for v in videos]              # flow_grpo rewards.py:148-153
            futures.append(pool.submit(score, frames, [r["prompt"] for r, _ in batch]))
            for k, (row, g) in enumerate(batch):
                sample_ids.append(sha256_bytes(canonical([SAMPLE_ID_DOMAIN, s["seed"], attempt, epoch, idx[k],
                                                          row["prompt_id"], seeds[k], g])))
                prompt_ids.append(row["prompt_id"])
                seeds_all.append(seeds[k])
            del videos, frames
        results = [r for f in futures for r in f.result()]
        rewards = [r["reward"] for r in results]
        t_sample = time.time() - t_ep
        peak_sampling = (round(torch.cuda.max_memory_allocated(device) / 2**30, 3) if cuda else None)
        first_step = global_step + 1
        for k, (row, g) in enumerate(plan):
            errs = [f for f in results[k]["frames"] if f["error"]]
            append_jsonl(out / "samples.jsonl", {
                "epoch": epoch, "sample_index": k, "sample_id": sample_ids[k], "prompt_id": row["prompt_id"],
                "prompt": row["prompt"], "target": target_text(row["prompt"]), "member": g, "seed": seeds_all[k],
                "frame_ocr": results[k]["frames"],
                "ocr_error": f"frame {errs[0]['frame']}: {errs[0]['error']}"[:300] if errs else None,
                "ocr_acc": results[k]["reward"], "reward": rewards[k], "nonfinite": nonfinite[k],
                "attempt": attempt, "utc": utc_now()})
        # ---------------- advantages and the zero-advantage filter (train_wan2_1.py:803-873) ----------------
        adv = torch.as_tensor(tracker.update(prompt_ids, rewards), dtype=torch.float32)
        mask = adv != 0                                  # (advantages over timesteps all equal per sample)
        true_count = int(mask.sum())
        all_zero = true_count == 0
        if all_zero:                                     # :853-858: every advantage 0 -> + 1e-6, every sample kept
            adv = adv + 1e-6
            mask = adv != 0
        reincluded = []
        if true_count % nb != 0:                         # :860 tests the count before the all-zero revision
            false_idx = torch.where(~mask)[0]
            need = nb - (true_count % nb)
            if len(false_idx) >= need:
                gm = torch.Generator("cpu").manual_seed(derive_seed("mask", s["seed"], epoch))
                pick = false_idx[torch.randperm(len(false_idx), generator=gm)[:need]]
                mask[pick] = True
                reincluded = sorted(int(x) for x in pick)
        kept = torch.where(mask)[0]
        n_kept = len(kept)
        if n_kept % nb:                                  # cannot happen: the epoch's sample count is a multiple of nb
            raise SystemExit(f"epoch {epoch}: {n_kept} kept samples are not a multiple of {nb}")
        filtered_ids = [sample_ids[i] for i in range(n) if not bool(mask[i])]
        n_steps_epoch = (nb // gas) * s["num_inner_epochs"] if n_kept else 0
        append_jsonl(out / "epoch_plan.jsonl", {
            "epoch": epoch, "kept": [sample_ids[i] for i in kept.tolist()], "filtered": filtered_ids,
            "reincluded": [sample_ids[i] for i in reincluded], "filter_reason": "zero_advantage",
            "all_zero_advantage": all_zero,
            "steps": [first_step, global_step + n_steps_epoch] if n_steps_epoch else [], "utc": utc_now()})
        chain_all = torch.cat(chains)                    # (n, num_steps + 1, C, f, h, w) float32 on the device
        logp_all = torch.cat(logps)                      # (n, num_steps)
        del chains, logps
        # ---------------- training (train_wan2_1.py:878-1001) ----------------
        transformer.train()
        t_tr = time.time()
        if cuda:
            torch.cuda.reset_peak_memory_stats(device)
        if n_kept:
            m = n_kept // nb                                # flow_grpo micro-batch size (train_wan2_1.py:899)
            for inner in range(s["num_inner_epochs"]):
                gp = torch.Generator("cpu").manual_seed(derive_seed("perm", s["seed"], epoch, inner))
                order = kept[torch.randperm(n_kept, generator=gp)].tolist()
                virtual = [order[v * m:(v + 1) * m] for v in range(nb)]
                info, consumed = {"policy_loss": [], "kl_loss": [], "approx_kl": [], "clipfrac": []}, []
                optimizer.zero_grad(set_to_none=True)
                for v, vidx in enumerate(virtual):
                    for c0 in range(0, m, s["train_batch"]):
                        bidx = vidx[c0:c0 + s["train_batch"]]
                        bt = torch.tensor(bidx)
                        e = pad_embeds([emb[plan[i][0]["prompt"]] for i in bidx], L).to(device, model_dtype)
                        ne = neg.expand(len(bidx), -1, -1)
                        a = torch.clamp(adv[bt], -s["adv_clip_max"], s["adv_clip_max"]).to(device)
                        for j in range(T):
                            lat_j = chain_all[bt, j]
                            nxt_j = chain_all[bt, j + 1]
                            old_lp = logp_all[bt, j]
                            tj = torch.full((len(bidx),), int(train_ts[j]), dtype=torch.int64, device=device)
                            sj = torch.full((len(bidx),), train_idx[j], dtype=torch.long)
                            per, kl, lp, ratio = grpo_terms(transformer, lat_j, nxt_j, old_lp, a, tj, sj, e, ne,
                                                            train_sig, s, model_dtype, autocast_dtype)
                            # flow_grpo: loss = mean over the micro-batch of m videos, averaged by accelerate over
                            # the accumulation window of (gradient_accumulation_steps x T) backward calls.
                            loss = (per + s["beta"] * kl).sum() / (m * gas * T)
                            loss.backward()
                            info["policy_loss"].append(float(per.mean()))
                            info["kl_loss"].append(float(kl.mean()))
                            info["approx_kl"].append(float(0.5 * torch.mean((lp - old_lp) ** 2)))
                            info["clipfrac"].append(float(torch.mean((torch.abs(ratio - 1.0) > s["clip_range"])
                                                                     .float())))
                        consumed.extend(sample_ids[i] for i in bidx)
                    if (v + 1) % gas == 0:
                        global_step += 1
                        append_jsonl(out / "train_steps.jsonl", {"step": global_step, "epoch": epoch, "inner": inner,
                                                                 "sample_ids": consumed, "utc": utc_now()})
                        gnorm = float(torch.nn.utils.clip_grad_norm_(params, s["max_grad_norm"]))
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        append_jsonl(out / "train_log.jsonl", {
                            "step": global_step, "epoch": epoch, "grad_norm": gnorm,
                            **{k: float(np.mean(x)) for k, x in info.items()}})
                        info = {k: [] for k in info}
                        consumed = []
            samples_consumed += n_kept
        samples_filtered += n - n_kept
        del chain_all, logp_all
        t_train = time.time() - t_tr
        r = np.array(rewards)
        mem = {"epoch": epoch, "seconds_sampling": round(t_sample, 2), "seconds_training": round(t_train, 2),
               "seconds_epoch": round(time.time() - t_ep, 2), "global_step": global_step,
               "reward_mean": float(r.mean()), "reward_std": float(r.std()), "kept": n_kept,
               "all_zero_advantage": all_zero, "nonfinite": int(sum(nonfinite)),
               "zero_std_groups": int(sum(np.std(r[[k for k in range(n) if prompt_ids[k] == p]]) == 0
                                          for p in dict.fromkeys(prompt_ids))),
               "max_rss_mb": rss_mb(), "rss_now_mb": rss_now_mb(), "utc": utc_now()}
        if cuda:
            mem["peak_allocated_gb_sampling"] = peak_sampling
            mem["peak_allocated_gb_training"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 3)
            mem["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved(device) / 2**30, 3)
        append_jsonl(out / "memory.jsonl", mem)
        print(json.dumps(mem, sort_keys=True), flush=True)
        if epoch % s["save_every"] == 0 or epoch == E:
            save_cut(transformer, out, epoch, global_step, samples_consumed, samples_filtered, base_ref, sampler,
                     lora_cfg, {"script_sha256": runtime["script_sha256"],
                                "sampler_code_sha256": runtime["sampler_code_sha256"], "seed": s["seed"],
                                "profile": args.profile, "ema": False})
    pool.shutdown()
    append_jsonl(out / "attempts.jsonl", {"attempt": attempt, "finished_at": utc_now(), "epochs": E,
                                          "global_step": global_step, "samples_consumed": samples_consumed,
                                          "samples_filtered": samples_filtered,
                                          "wall_seconds": round(time.time() - t_start, 1)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
