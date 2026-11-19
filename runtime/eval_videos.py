#!/usr/bin/env python
"""Per-item OCR evaluation of a Wan2.1-T2V-1.3B + LoRA cut (or the base alone) for prototype area "video generation"
(MULTIMODAL-SPEC row 3; prototype, unpromotable).

An item is one registered prompt (design 05 section 3.3) from flow_grpo's OCR test split, each asking for one quoted
text to be rendered. The item set and the training shard are area 5's files, copied byte for byte (`--copy-items`;
provenance in ITEMS-SOURCE.json; README "Decisions"). Per item and repeat r: one
video (240 x 416, 17 frames) from the deployment sampler (flow-matching Euler ODE, 20 steps, CFG 4.5, bf16; the
deterministic branch of flow_grpo's wan_pipeline_with_logprob.py) with noise seed derive_seed('eval', eval_seeds[r],
item), the same for every arm (common random numbers). Every frame is kept as a lossless PNG with its SHA-256, the
video's frame manifest `frames.json` is the item's output_ref, and every metric is computed from the saved uint8 frames
read back from disk.

Metrics (README "Decisions"):
  ocr_b_acc   deciding (T3, construct-aligned disclosure, R15-1), higher is better, [0, 1]: per frame the approximate-
              substring OCR accuracy of area 5 (`ocr_substring_accuracy`: spaces removed, lower case; 1 - min over
              substrings s of the recognized text of lev(s, target) / len(target)) with OCR engine B = docTR
              (db_resnet50 + crnn_vgg16_bn), averaged over ALL frames of the video, frames reading nothing included
  ocr_a_acc   supporting, T0: flow_grpo's `video_ocr` reward exactly (flow_grpo/ocr.py:65-123): OCR engine A =
              PaddleOCR PP-OCRv4 on every 4th frame (0, 4, 8, 12, 16 for 17 frames), per frame 1 - min(lev(text,
              target), len(target)) / len(target) on space-free lower case (no substring shortcut), the video score is
              the mean over the frames with a positive score, 0 if none
Descriptive sub-metrics only (never deciding), finite numbers only (MM-1): both engines' scores (an engine that
raised is left out), nonfinite, mean absolute difference of consecutive frames (temporal-collapse indicator). The
target and the per-frame recognized texts are in scores.jsonl.

Ports from yifan123/flow_grpo at commit 879042cf5707f8b90daa98d147d7deac2317c5da (MIT License, Copyright (c) 2025
Jie Liu), cited by file:line of that commit where they are used: the shared sampler block (the Wan sampling loop and
SDE step, the prompt encoding and the video OCR reward, byte-identical in train_flow_grpo_wan.py and
train_flow_grpo_wan_v2.py) and the OCR accuracy formulas (ocr.py). Its wan_pipeline_with_logprob.py is derived
from huggingface/diffusers `src/diffusers/pipelines/wan/pipeline_wan.py` (commit
cb8e61ed2f792db8bcb9606c12e9ae7e400e4b49: "Copyright 2025 The Wan Team and The HuggingFace Team. All rights
reserved.", licensed under the Apache License, Version 2.0); the Wan SDE step and the sampling loop ported from it
carry that attribution (THIRD_PARTY_NOTICES.md).
The edit distance comes from rapidfuzz (MIT); flow_grpo and the published 2026-11-15 runs used python-Levenshtein,
which returns the same values (RESULT.md).

Modes:
  python eval_videos.py --copy-items
      copies area 5's items/ and prompts/ byte for byte (refuses differing existing files) and writes ITEMS-SOURCE.json
  python eval_videos.py (--checkpoint CUT | --base-only) --out DIR --eval-seeds 11 [--items DIR] [--limit N]
                        [--batch-size 4] [--model-hash H] [--base-model-hash H] [--item-set-hash H] [--device cuda]
      writes DIR/videos/<item>/r<repeat>/{frame_XX.png, frames.json}, DIR/scores.jsonl, DIR/summary.json,
      DIR/env-runtime.json, and one MM-1/MM-2 arm per metric: DIR/arm-<metric>/{items.jsonl, eval_meta.json,
      videos/ (hard links)}
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import resource
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_NAME = "wan2.1-t2v-1.3b"
DEFAULT_BASE = HERE / "models" / BASE_NAME
MODELS_JSON = HERE / "models" / "MODELS.json"                 # written by fetch_models.py
# The SD3.5 OCR recipe's runtime (OCR engines, their weights and the item source): the override
# GATEDTRAIN_RECIPE_FLOW_GRPO_WAN_VIDEO_OCR_RUNTIME, else the sibling checkout ../gatedtrain-recipe-flow-grpo-sd35-ocr/runtime.
OCR_RUNTIME = Path(os.environ.get("GATEDTRAIN_RECIPE_FLOW_GRPO_WAN_VIDEO_OCR_RUNTIME")
                   or HERE.parent.parent / "gatedtrain-recipe-flow-grpo-sd35-ocr" / "runtime")
OCR_MODELS = OCR_RUNTIME / "ocr-models"
OCR_MODELS_JSON = OCR_RUNTIME / "MODELS.json"
ITEM_SOURCES = {"items": (OCR_RUNTIME / "items", ("items.jsonl", "item_set_meta.json")),
                "prompts": (OCR_RUNTIME / "prompts", ("train.jsonl", "prompts_meta.json"))}
ITEMS_SOURCE_JSON = HERE / "ITEMS-SOURCE.json"
LORA_WEIGHTS = "pytorch_lora_weights.safetensors"
ADAPTER = "flow_grpo_wan"
SCORER_VERSION = "eval_videos@0.1"
# flow_grpo scripts/train_wan2_1.py:451-460 (LoRA r 32, alpha 64, gaussian init at :461-466)
WAN_LORA_TARGETS = ("add_k_proj", "add_q_proj", "add_v_proj", "to_add_out", "to_k", "to_out.0", "to_q", "to_v")
SAMPLER_DEFAULT = {"num_steps": 20, "guidance_scale": 4.5, "height": 240, "width": 416, "num_frames": 17,
                   "max_sequence_length": 512}
FRAME_INTERVAL_A = 4                                            # flow_grpo ocr.py:77
DOCTR = {"det_arch": "db_resnet50", "reco_arch": "crnn_vgg16_bn", "assume_straight_pages": True,
         "preserve_aspect_ratio": True, "symmetric_pad": True, "join": "words in reading order, no separator"}
METRICS = {
    # metric -> evaluator id, direction, range, tier, role
    "ocr_b_acc": ("t2v.wan_ocr_doctr_items@0.1", "higher_is_better", [0, 1], "T3 (construct-aligned disclosure)",
                  "deciding"),
    "ocr_a_acc": ("t2v.wan_ocr_paddle_items@0.1", "higher_is_better", [0, 1], "T0", "supporting"),
}
ERROR_CLASSES = ("scorer_error",)


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
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


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


def substring_edit_distance(text: str, pattern: str) -> int:
    """min over substrings s of `text` (the empty one included) of the Levenshtein distance lev(s, `pattern`): Sellers'
    approximate-substring dynamic programme (copied from area 5's eval_images.py)."""
    prev = [0] * (len(text) + 1)
    for i, pc in enumerate(pattern, start=1):
        cur = [i] + [0] * len(text)
        for j, tc in enumerate(text, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (pc != tc))
        prev = cur
    return min(prev)


def ocr_substring_accuracy(recognized: str, target: str) -> float:
    """Area 5's deciding per-image formula (the SD3.5 OCR recipe's eval_images.py `ocr_substring_accuracy`),
    here per frame: spaces removed and lower case on both sides, then 1 - min over substrings s of the recognized
    text of lev(s, target) / len(target), in [0, 1]; an empty recognized text scores 0. An empty normalized target
    raises (an error, never a score)."""
    recognized = recognized.replace(" ", "").lower()
    target = target.replace(" ", "").lower()
    if not target:
        raise ValueError("the target text is empty after normalization")
    if not recognized:
        return 0.0
    return 1 - substring_edit_distance(recognized, target) / len(target)


def video_substring_accuracy(frame_texts, target) -> float:
    """The deciding metric `ocr_b_acc` (README "Decisions"): the mean of
    `ocr_substring_accuracy` over all frames, frames that read nothing included (as 0)."""
    if not frame_texts:
        raise ValueError("a video without frames")
    return sum(ocr_substring_accuracy(t, target) for t in frame_texts) / len(frame_texts)


def mean_abs_frame_diff(frames) -> float:
    """Descriptive temporal-collapse indicator: mean |frame[i + 1] - frame[i]| over consecutive uint8 frames, / 255."""
    import numpy as np

    if len(frames) < 2:
        return 0.0
    f = np.asarray(frames, dtype=np.float32)
    return float(np.abs(f[1:] - f[:-1]).mean() / 255.0)


# ---------------------------------------------------------------------------
# Items and the training shard (area 5's files, byte for byte)
# ---------------------------------------------------------------------------

def copy_items() -> int:
    """Copy area 5's registered item set and training shard unchanged; a file already present must be identical.
    No provenance file is added inside items/ or prompts/: the item set hash is the manifest hash of the directory,
    so the copies keep area 5's hashes. Provenance goes to ITEMS-SOURCE.json next to this script."""
    record = {"schema": "ptc_mm.video_items_source", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
              "rule": "byte-for-byte copies of area 5's registered item set and training shard "
                      "(README 'Decisions'); not regenerated",
              "copies": {}}
    for dst_name, (src_dir, names) in ITEM_SOURCES.items():
        dst_dir = HERE / dst_name
        dst_dir.mkdir(exist_ok=True)
        extra = sorted({p.name for p in dst_dir.iterdir()} - set(names))
        if extra:
            raise SystemExit(f"{dst_dir} holds files that are not area 5's: {extra}")
        files = {}
        for name in names:
            src, dst = src_dir / name, dst_dir / name
            if dst.exists():
                if dst.read_bytes() != src.read_bytes():
                    raise SystemExit(f"{dst} exists and differs from {src}")
            else:
                shutil.copy2(src, dst)
            files[name] = {"source": str(src), "sha256": sha256_file(src), "bytes": src.stat().st_size,
                           "copy_sha256": sha256_file(dst)}
            if files[name]["sha256"] != files[name]["copy_sha256"]:
                raise SystemExit(f"{dst} differs from {src} after the copy")
        record["copies"][dst_name] = {"source_dir": str(src_dir), "files": files}
    write_json(ITEMS_SOURCE_JSON, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


def read_items(items_dir: Path) -> list[dict]:
    rows = [json.loads(ln) for ln in (items_dir / "items.jsonl").read_text(encoding="utf-8").splitlines()
            if ln.strip()]
    ids = [r["item"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("item ids are not unique")
    return rows


# ---------------------------------------------------------------------------
# OCR engine B
# ---------------------------------------------------------------------------

class DoctrB:
    """OCR engine B: docTR (python-doctr 1.1.0) `ocr_predictor` with db_resnet50 + crnn_vgg16_bn, area 5's weights
    (mindee/doctr GitHub release assets converted to safetensors, the SD3.5 OCR recipe's runtime/fetch_models.py);
    RGB uint8 input."""

    def __init__(self, device, model_root=OCR_MODELS):
        import torch
        from doctr.models import crnn_vgg16_bn, db_resnet50, ocr_predictor
        from safetensors.torch import load_file

        det = db_resnet50(pretrained=False, pretrained_backbone=False)
        det.load_state_dict(load_file(str(Path(model_root) / "doctr" / "db_resnet50.safetensors")))
        rec = crnn_vgg16_bn(pretrained=False, pretrained_backbone=False)
        rec.load_state_dict(load_file(str(Path(model_root) / "doctr" / "crnn_vgg16_bn.safetensors")))
        self.torch = torch
        self.predictor = ocr_predictor(det_arch=det, reco_arch=rec, pretrained=False,
                                       assume_straight_pages=DOCTR["assume_straight_pages"],
                                       preserve_aspect_ratio=DOCTR["preserve_aspect_ratio"],
                                       symmetric_pad=DOCTR["symmetric_pad"]).to(device).eval()

    def texts(self, arrays):
        with self.torch.no_grad():
            doc = self.predictor(list(arrays))
        return ["".join(w.value for b in page.blocks for ln in b.lines for w in ln.words) for page in doc.pages]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def base_record() -> dict:
    return json.loads(MODELS_JSON.read_text(encoding="utf-8"))["models"][BASE_NAME]


def check_cut(cut: Path) -> dict:
    """Cut binding (area 5's layout `checkpoint-<epoch>/`: LoRA weights, adapter_config.json, sampler.json,
    base_ref.json, trainer_state.json, cut.json last): every file's SHA-256 in cut.json, the same shared sampler
    code, the same base (revision and every file's SHA-256 as models/MODELS.json), and flow_grpo's Wan LoRA targets."""
    marker = json.loads((cut / "cut.json").read_text(encoding="utf-8"))
    for name, want in marker["files"].items():
        if sha256_file(cut / name) != want:
            raise SystemExit(f"{cut / name} differs from its SHA-256 in cut.json")
    if marker.get("sampler_code_sha256") != sampler_code_sha256():
        raise SystemExit("the cut was trained with another sampler code than this evaluator's")
    base_ref = json.loads((cut / "base_ref.json").read_text(encoding="utf-8"))
    record = base_record()
    if base_ref["revision"] != record["revision"] or base_ref["files"] != {k: v["sha256"] for k, v in
                                                                           record["files"].items()}:
        raise SystemExit(f"the cut names another base model than models/{BASE_NAME}")
    cfg = json.loads((cut / "adapter_config.json").read_text(encoding="utf-8"))
    if sorted(cfg.get("target_modules") or []) != sorted(WAN_LORA_TARGETS):
        raise SystemExit(f"the cut's LoRA target_modules {cfg.get('target_modules')} are not flow_grpo's Wan targets "
                         f"{sorted(WAN_LORA_TARGETS)} (train_wan2_1.py:451-460)")
    return marker


def resolve_sampler(args, cut: Path | None, scheduler) -> dict:
    want = {k: getattr(args, k) for k in SAMPLER_DEFAULT if getattr(args, k) is not None}

    def build(s):
        ts, sig, idx = schedule(scheduler, s["num_steps"])
        return sampler_doc(s["num_steps"], s["guidance_scale"], s["height"], s["width"], s["num_frames"],
                           s["max_sequence_length"], dict(scheduler.config), ts, sig, idx, args.dtype, args.dtype,
                           deterministic=True)

    if cut is not None:
        doc = json.loads((cut / "sampler.json").read_text(encoding="utf-8"))
        for k, v in want.items():
            if doc[k] != v:
                raise SystemExit(f"--{k.replace('_', '-')} {v} differs from the cut's sampler.json ({doc[k]})")
        if not doc.get("deterministic"):
            raise SystemExit("the cut's sampler.json is not deterministic; the deployment sampler is the Euler ODE")
        if doc != build({k: doc[k] for k in SAMPLER_DEFAULT}):
            raise SystemExit("the cut's sampler.json is not this evaluator's sampler")
        return doc
    return build({**SAMPLER_DEFAULT, **want})


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        dst.write_bytes(src.read_bytes())


def save_video(out: Path, rel_dir: str, frames_u8) -> tuple[dict, list]:
    """Write every frame as a lossless PNG and the manifest frames.json (file names, SHA-256s, shape); read the PNGs
    back. Returns (output_ref of the manifest, the frames as read from disk)."""
    import numpy as np
    from PIL import Image

    vdir = out / rel_dir
    vdir.mkdir(parents=True, exist_ok=True)
    entries, saved = [], []
    for i, fr in enumerate(frames_u8):
        name = f"frame_{i:02d}.png"
        Image.fromarray(fr).save(vdir / name, format="PNG")
        arr = np.asarray(Image.open(vdir / name).convert("RGB"))
        if not np.array_equal(arr, fr):
            raise RuntimeError(f"{vdir / name} does not read back to the written frame")
        saved.append(arr)
        entries.append({"path": name, "sha256": sha256_file(vdir / name)})
    h, w = frames_u8[0].shape[:2]
    write_json(vdir / "frames.json", {"schema": "ptc_mm.video_frames", "schema_version": "1.0",
                                      "num_frames": len(entries), "height": int(h), "width": int(w),
                                      "format": "PNG, RGB uint8, lossless", "frames": entries})
    rel = f"{rel_dir}/frames.json"
    return {"path": rel, "sha256": sha256_file(out / rel)}, saved


def stats(vals):
    vals = [v for v in vals if v is not None]
    n = len(vals)
    if not n:
        return {"n": 0}
    mean = sum(vals) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else None
    return {"n": n, "mean": mean, "sd": sd, "min": min(vals), "max": max(vals),
            "share_at_0": sum(v == 0 for v in vals) / n, "share_at_1": sum(v == 1 for v in vals) / n}


def mm38_checks(item_scores, score_range, errors) -> dict:
    """MM-38's degeneracy rules for a bounded, non-binary metric on the item scores (mean over repeats): the mean lies
    inside the range by at least 5 % of its width, fewer than 50 % of the item scores share one value, the item
    standard deviation is non-zero; plus zero evaluator errors. Advisory here; the smoke review decides."""
    vals = [v for v in item_scores if v is not None]
    lo, hi = score_range
    n = len(vals)
    if not n:
        return {"n": 0, "pass": False}
    mean = sum(vals) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    counts = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    mode_value, mode_count = max(counts.items(), key=lambda kv: kv[1])
    margin = 0.05 * (hi - lo)
    checks = {"mean_inside_by_5pct": lo + margin <= mean <= hi - margin, "mode_share_below_0.5": mode_count / n < 0.5,
              "sd_positive": sd > 0, "zero_errors": errors == 0}
    return {"n": n, "mean": mean, "sd": sd, "mode_value": mode_value, "mode_share": mode_count / n,
            "errors": errors, "checks": checks, "pass": all(checks.values())}


def evaluate(args) -> int:
    import numpy as np
    import torch

    t0 = time.time()
    device = torch.device(args.device)
    model_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    autocast_dtype = torch.bfloat16 if args.dtype == "bf16" else None
    cut = args.checkpoint.resolve() if args.checkpoint else None
    marker = check_cut(cut) if cut else None
    items = read_items(args.items)
    if args.limit:
        items = items[: args.limit]
    seeds = [int(x) for x in args.eval_seeds.split(",")]
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)
    rss = {"start": rss_mb()}
    cuda = device.type == "cuda"
    gpu = {}

    from diffusers import UniPCMultistepScheduler

    # the sampler first (cheap): a cut whose sampler.json differs is refused before the encoder is loaded
    sampler = resolve_sampler(args, cut, UniPCMultistepScheduler.from_pretrained(
        str(args.base), subfolder="scheduler", local_files_only=True))
    max_len = sampler["max_sequence_length"]
    te = time.time()
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    emb = encode_all_prompts(args.base, [""] + [it["prompt"] for it in items], device, model_dtype, max_len)
    t_encode = time.time() - te
    if cuda:
        gpu["encode_peak_allocated_gb"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 3)
        gpu["encode_peak_reserved_gb"] = round(torch.cuda.max_memory_reserved(device) / 2**30, 3)
        torch.cuda.reset_peak_memory_stats(device)
    rss["after_encode"] = rss_mb()
    transformer, vae, scheduler = load_transformer_vae(args.base, device, model_dtype)
    lora = None
    if cut is not None:
        transformer.load_lora_adapter(str(cut), prefix=None, weight_name=LORA_WEIGHTS, adapter_name="flow_grpo",
                                      local_files_only=True)
        cfg = json.loads((cut / "adapter_config.json").read_text(encoding="utf-8"))
        lora = {k: cfg.get(k) for k in ("r", "lora_alpha", "target_modules")}
    transformer.eval()
    if resolve_sampler(args, cut, scheduler) != sampler:
        raise SystemExit("the scheduler loaded with the transformer gives another sampler")
    engine_a = PaddleOcrA(OCR_MODELS, cpu_threads=args.ocr_cpu_threads)
    engine_b = DoctrB(device)
    rss["after_load"] = rss_mb()
    neg = pad_embeds([emb[""]], max_len).to(device, model_dtype)

    jobs = [(r, it) for r in range(len(seeds)) for it in items]
    scores = []
    t_gen = t_save = t_a = t_b = 0.0
    for b0 in range(0, len(jobs), args.batch_size):
        batch = jobs[b0:b0 + args.batch_size]
        noise_seeds = [derive_seed("eval", seeds[r], it["item"]) for r, it in batch]
        e = pad_embeds([emb[it["prompt"]] for _, it in batch], max_len).to(device, model_dtype)
        tg = time.time()
        videos, _, _, _ = sample_batch(transformer, vae, scheduler, e, neg.expand(len(batch), -1, -1), noise_seeds,
                                       sampler["num_steps"], sampler["guidance_scale"], sampler["height"],
                                       sampler["width"], sampler["num_frames"], device, model_dtype, autocast_dtype,
                                       deterministic=True, keep_chain=False)
        t_gen += time.time() - tg
        ts = time.time()
        rows, saved = [], []
        for k, (r, it) in enumerate(batch):
            video = videos[k]
            nonfinite = not bool(torch.isfinite(video).all())
            if nonfinite:
                video = torch.nan_to_num(video, nan=0.0, posinf=1.0, neginf=0.0)
            ref, frames = save_video(out, f"videos/{it['item']}/r{r}", video_to_uint8(video))
            saved.append(frames)
            rows.append({"item": it["item"], "repeat": r, "seed": seeds[r], "noise_seed": noise_seeds[k],
                         "target": it["target"], "output_ref": ref, "nonfinite": nonfinite, "errors": {},
                         "num_frames": len(frames), "mean_abs_frame_diff": mean_abs_frame_diff(frames)})
        t_save += time.time() - ts
        ta = time.time()
        for k, (row, (_, it)) in enumerate(zip(rows, batch)):
            res = engine_a.video_reward(saved[k], it["prompt"])
            row["ocr_a_frames"] = res["frames"]
            failed = [f for f in res["frames"] if f["error"]]
            if failed:                         # scorer failure: an error record (MM-4), never a 0 score
                row["ocr_a_acc"] = None
                row["errors"]["ocr_a_acc"] = {"class": "scorer_error",
                                              "message": f"frame {failed[0]['frame']}: {failed[0]['error']}"[:300]}
            else:
                row["ocr_a_acc"] = res["reward"]
        t_a += time.time() - ta
        tb = time.time()
        for k, row in enumerate(rows):
            row["ocr_b_frames"], row["ocr_b_acc"] = None, None
            try:
                texts = engine_b.texts(saved[k])
                accs = [ocr_substring_accuracy(t, row["target"]) for t in texts]
                row["ocr_b_frames"] = [{"frame": f, "text": t, "acc": a} for f, (t, a) in enumerate(zip(texts, accs))]
                row["ocr_b_acc"] = video_substring_accuracy(texts, row["target"])
            except Exception as ex:                                  # scorer failure: an error record (MM-4)
                row["errors"]["ocr_b_acc"] = {"class": "scorer_error", "message": f"{type(ex).__name__}: {ex}"[:300]}
        t_b += time.time() - tb
        scores.extend(rows)
        print(f"{len(scores)}/{len(jobs)} videos, {time.time() - t0:.0f} s", flush=True)
    wall = time.time() - t0
    write_jsonl(out / "scores.jsonl", scores)

    code_sha = sha256_file(Path(__file__).resolve())
    sampler_hash = sha256_bytes(canonical(sampler))
    ocr_models = json.loads(OCR_MODELS_JSON.read_text(encoding="utf-8"))["models"]
    versions = {k: importlib.metadata.version(k) for k in ("paddlepaddle", "paddleocr", "python-doctr", "rapidfuzz",
                                                           "pillow", "diffusers", "transformers", "peft")}
    scorers = {
        "ocr_b_acc": {"name": "video_ocr_accuracy_doctr",
                      "version": f"{SCORER_VERSION} engine docTR db_resnet50+crnn_vgg16_bn python-doctr "
                                 f"{versions['python-doctr']} {json.dumps(DOCTR, sort_keys=True)} approximate-substring "
                                 "edit distance (Sellers DP), spaces removed, lower case; per frame, mean over all "
                                 "frames (zeros included); frames read back from lossless PNG",
                      "weights_sha256": sha256_bytes(canonical({k: v["sha256"] for k, v in
                                                                ocr_models["doctr"]["files"].items()})),
                      "code_sha256": code_sha},
        "ocr_a_acc": {"name": "video_ocr_reward_paddleocr",
                      "version": f"{SCORER_VERSION} flow_grpo video_ocr (ocr.py:65-123): every {FRAME_INTERVAL_A}th "
                                 "frame, 1 - min(lev, len)/len, mean over positive frames, 0 if none; engine PaddleOCR "
                                 f"PP-OCRv4 paddleocr {versions['paddleocr']} paddlepaddle {versions['paddlepaddle']} "
                                 f"PP-OCRv4_mobile_det@{ocr_models['paddleocr/PP-OCRv4_mobile_det']['revision']} "
                                 "en_PP-OCRv4_mobile_rec@"
                                 f"{ocr_models['paddleocr/en_PP-OCRv4_mobile_rec']['revision']}",
                      "weights_sha256": sha256_bytes(canonical({**{f"det/{k}": v["sha256"] for k, v in ocr_models[
                          "paddleocr/PP-OCRv4_mobile_det"]["files"].items()}, **{f"rec/{k}": v["sha256"] for k, v in
                          ocr_models["paddleocr/en_PP-OCRv4_mobile_rec"]["files"].items()}})),
                      "code_sha256": code_sha},
    }
    for metric, (evaluator_id, direction, rng, tier, role) in METRICS.items():
        arm = out / f"arm-{metric}"
        arm.mkdir()
        lines = []
        for row in scores:
            vdir = Path(row["output_ref"]["path"]).parent
            for f in sorted((out / vdir).iterdir()):
                link_or_copy(f, arm / vdir / f.name)
            err = row["errors"].get(metric)
            # MM-1: sub_metrics hold finite numbers only; texts stay in scores.jsonl; an engine that raised has no
            # score, so its key is left out (its error is the arm line's `error` in that engine's own arm).
            sub = {k: row[k] for k in ("ocr_a_acc", "ocr_b_acc") if row[k] is not None}
            sub["nonfinite"] = int(row["nonfinite"])
            sub["mean_abs_frame_diff"] = row["mean_abs_frame_diff"]
            lines.append({"item": row["item"], "repeat": row["repeat"], "seed": row["seed"],
                          "score": None if err else row[metric], "error": err, "sub_metrics": sub,
                          "output_ref": row["output_ref"]})
        write_jsonl(arm / "items.jsonl", lines)
        write_json(arm / "eval_meta.json", {
            "schema": "gatedtrain.item_eval", "schema_version": "1.0",
            "evaluator_id": evaluator_id, "adapter": ADAPTER, "metric": metric, "direction": direction,
            "score_range": rng,
            "checkpoint": {"model_hash": args.model_hash, "base_model_hash": args.base_model_hash},
            "normalizer_sha256": None, "env_version_sha256": None,
            "item_set_hash": args.item_set_hash, "eval_seeds": seeds,
            "sampler": sampler, "sampler_hash": sampler_hash, "scorer": scorers[metric],
            "framework": {"name": "diffusers", "version": versions["diffusers"]},
            "created_at": utc_now()})

    def item_means(metric):
        by = {}
        for r in scores:
            by.setdefault(r["item"], []).append(r[metric])
        return [None if any(v is None for v in vs) else sum(vs) / len(vs) for vs in by.values()]

    summary = {"checkpoint": str(cut) if cut else None, "base_only": cut is None,
               "epoch": marker["epoch"] if marker else 0, "lora": lora, "items": len(items), "repeats": len(seeds),
               "videos": len(scores), "frames_per_video": sampler["num_frames"],
               "errors": sum(1 for r in scores if r["errors"]),
               "nonfinite": sum(int(r["nonfinite"]) for r in scores),
               "metrics": {m: stats([r[m] for r in scores]) for m in ("ocr_b_acc", "ocr_a_acc")},
               "mean_abs_frame_diff": stats([r["mean_abs_frame_diff"] for r in scores]),
               "ocr_b_frames_at_0_share": (sum(f["acc"] == 0 for r in scores for f in (r["ocr_b_frames"] or []))
                                           / max(1, sum(len(r["ocr_b_frames"] or []) for r in scores))),
               "mm38": {m: {**mm38_checks(item_means(m), METRICS[m][2],
                                          sum(1 for r in scores if m in r["errors"])),
                            "role": METRICS[m][4]} for m in METRICS},
               "agreement_a_b": {"pearson": None},
               "seconds": {"wall": round(wall, 1), "encode": round(t_encode, 1), "generation": round(t_gen, 1),
                           "save": round(t_save, 1), "ocr_a": round(t_a, 1), "ocr_b": round(t_b, 1)},
               "seconds_per_video": round(wall / max(1, len(scores)), 3),
               "generation_seconds_per_video": round(t_gen / max(1, len(scores)), 3),
               "rss_mb": {**rss, "peak": rss_mb()},
               "sampler_hash": sampler_hash, "device": str(device), "dtype": str(model_dtype),
               "batch_size": args.batch_size,
               "arms": {m: {"evaluator_id": v[0], "tier": v[3], "role": v[4],
                            "construct_aligned": m == "ocr_b_acc"} for m, v in METRICS.items()},
               "missing_hashes": [k for k, v in (("model_hash", args.model_hash),
                                                  ("item_set_hash", args.item_set_hash)) if v is None]}
    pairs = [(r["ocr_a_acc"], r["ocr_b_acc"]) for r in scores if r["ocr_a_acc"] is not None
             and r["ocr_b_acc"] is not None]
    if len(pairs) > 2:
        a, b = np.array(pairs).T
        if a.std() > 0 and b.std() > 0:
            summary["agreement_a_b"]["pearson"] = float(np.corrcoef(a, b)[0, 1])
    if cuda:
        gpu["generate_peak_allocated_gb"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 3)
        gpu["generate_peak_reserved_gb"] = round(torch.cuda.max_memory_reserved(device) / 2**30, 3)
        summary["gpu"] = gpu
        summary["peak_allocated_gb"] = max(gpu["encode_peak_allocated_gb"], gpu["generate_peak_allocated_gb"])
        summary["peak_reserved_gb"] = max(gpu["encode_peak_reserved_gb"], gpu["generate_peak_reserved_gb"])
    lock = HERE / "uv.lock"
    if not lock.exists():
        lock = OCR_RUNTIME / "uv.lock"
    write_json(out / "env-runtime.json", {
        "python": sys.version.split()[0], "executable": sys.executable,
        "uv_lock": str(lock), "uv_lock_sha256": sha256_file(lock) if lock.exists() else None,
        "script_sha256": code_sha, "sampler_code_sha256": sampler_code_sha256(), "versions": versions,
        "torch": torch.__version__, "checkpoint": str(cut) if cut else None, "base": str(Path(args.base).resolve()),
        "items": str(args.items.resolve()), "items_sha256": sha256_file(args.items / "items.jsonl"),
        "uid": os.getuid(), "gid": os.getgid()})
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--copy-items", action="store_true")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--base-only", action="store_true")
    p.add_argument("--base", type=Path, default=DEFAULT_BASE)
    p.add_argument("--items", type=Path, default=HERE / "items")
    p.add_argument("--out", type=Path)
    p.add_argument("--eval-seeds", default="11")
    p.add_argument("--model-hash")
    p.add_argument("--base-model-hash")
    p.add_argument("--item-set-hash")
    p.add_argument("--limit", type=int, help="first N items only (smoke and probe arms)")
    p.add_argument("--batch-size", type=int, default=4, help="videos per transformer batch (decoded one by one)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16",
                   help="transformer and text encoder (bf16 = decision; fp32 for CPU tests only)")
    p.add_argument("--ocr-cpu-threads", type=int, default=8)
    for k, v in SAMPLER_DEFAULT.items():
        p.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=None)
    args = p.parse_args(argv)
    if not args.copy_items:
        if args.out is None or (args.checkpoint is None) == (not args.base_only):
            p.error("evaluation needs --out and exactly one of --checkpoint / --base-only")
        if args.limit is not None and args.limit < 1:
            p.error("--limit must be positive")
        if args.batch_size < 1:
            p.error("--batch-size must be positive")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.copy_items:
        return copy_items()
    return evaluate(args)


if __name__ == "__main__":
    sys.exit(main())
