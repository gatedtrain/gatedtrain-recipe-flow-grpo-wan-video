"""Registry rows of adapter `flow_grpo_wan`: Flow-GRPO + LoRA on Wan2.1 video (area 3 video).

The training row, the evaluator rows and the adapter class, loaded by `gatedtrain.ext.registry` through the
`gatedtrain.adapters` entry point (MULTIMODAL-SPEC section 6)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from gatedtrain.ext.rowkit import CONTAINER_PATHS, runtime_dir, venv_runtime_rows

NAME = "flow_grpo_wan"
# The runtime directory: this repository's runtime/ (scripts, the Wan2.1 base, prompts, items; mounted read-only at its
# own host path), or the override; earlier runs lie under the host's output root as `{output_root}/<OUT_NAME>`.
RUNTIME_ENV = "GATEDTRAIN_RECIPE_FLOW_GRPO_WAN_VIDEO_RUNTIME"
OUT_NAME = "flow-grpo-wan-video"
RUNTIME = runtime_dir(RUNTIME_ENV, __file__)
# Cross-recipe dependency (README "Dependency on the SD3.5 OCR recipe"): every
# launch runs the SD3.5 OCR recipe's runtime venv (its uv.lock, pinned by the adapter) and reads its OCR models; the
# directory providing them (mounted read-only at its own host path) is the override, or the sibling checkout's
# runtime/. The interpreter tree comes from its `.venv/pyvenv.cfg`. No Python import of that recipe.
OCR_RUNTIME_ENV = "GATEDTRAIN_RECIPE_FLOW_GRPO_WAN_VIDEO_OCR_RUNTIME"
OCR_RUNTIME_DEFAULT = str(Path(__file__).resolve().parents[2].parent / "gatedtrain-recipe-flow-grpo-sd35-ocr" / "runtime")
OCR_RUNTIME = os.environ.get(OCR_RUNTIME_ENV) or OCR_RUNTIME_DEFAULT
HOST_ROWS = venv_runtime_rows(OCR_RUNTIME, OUT_NAME)
# The interpreter install directory (the parent of the tree; None without the OCR runtime's venv).
PYTHON_DIR = str(Path(HOST_ROWS["interpreter_trees"][0]).parent) if HOST_ROWS["interpreter_trees"] else None

TRAINER_ROWS: dict[str, dict[str, Any]] = {}
EVALUATOR_ROWS: dict[str, dict[str, Any]] = {}

# Area "video generation" (MULTIMODAL-SPEC section 6 rows 3 and 3s and the row 3 deviations; runtime/README.md
# "Decisions", 2026-11-14; runtime/ (README "Trainer", "Evaluator"; train_flow_grpo_wan.py,
# eval_videos.py, read 2026-11-14) and its GPU probes probe-train-1 (2-epoch smoke) and probe-base-1 (20-item base evaluation); tests/README.md section 11.25, EXT-214 to EXT-228). The
# Flow-GRPO port of flow_grpo's train_wan2_1.py on area 5's stack (same image, venv and lock), LoRA cuts on
# Wan2.1-T2V-1.3B in area 5's layout; a sibling of `flow_grpo_sd35` with its own `uses` id (`trainer_ids` maps an id
# to one adapter, so the id `rl.flow_grpo@0.1` cannot name both).
TRAINER_ROWS["flow_grpo_wan"] = {
    "uses": ("rl.flow_grpo_wan@0.1",),
    # from env-freeze.txt (MM-33): area 5's pins (one venv, the OCR runtime's uv.lock); python-doctr is the evaluator's.
    "pins": ("diffusers", "peft", "torch", "transformers", "paddleocr", "paddlepaddle"),
    "outputs": ("checkpoint-<epoch>/ LoRA cuts with cut.json written last", "run_args.json", "samples.jsonl",
                "epoch_plan.jsonl", "train_steps.jsonl"),
    # EXT-215: train_flow_grpo_wan.py DEPLOYABLE (cut.json `deployable`), as areas 3 and 5.
    "components": ("pytorch_lora_weights.safetensors", "adapter_config.json", "sampler.json", "base_ref.json"),
    # EXT-218: observed per-sample log with filtered outcomes and all-zero-advantage epochs; no fallback.
    "consumption": {"target": "row", "fallback": None},
    # estimate from the GPU probes: smoke 0.21 h (probe-train-1 wall 747 s), full 60 epochs x about 205 s = 3.4 h,
    # four evaluations (smoke 40 items x 2, full 400 items x 2) at about 3.9 s per video plus about 3 min of encoding.
    "gpu_h_estimate": 4.8,
    "evaluators": ("t2v.wan_ocr_doctr_items@0.1", "t2v.wan_ocr_paddle_items@0.1"),   # deciding first (MM-38)
    "framework": "diffusers",
    # EXT-224, in this order: the interpreter install directory, the OCR runtime (venv, lock, OCR models), the runtime
    # (scripts, base, prompts, items) and the training runs' outputs (an evaluation mounts a cut read-only at
    # /inputs/cut).
    "input_roots": HOST_ROWS["input_roots"][:-1] + (RUNTIME,) + HOST_ROWS["input_roots"][-1:],
    "spec_binding": False,                           # the smoke launch precedes the registration (sigma from the smoke)
    # README "Smoke launch" and area 5's records: every name, none a credential (HF_TOKEN is never allowed);
    # PYTORCH_CUDA_ALLOC_CONF and DOCTR_CACHE_DIR are optional. OCR_RUNTIME_ENV tells the scripts where the OCR runtime
    # is mounted when it is not the sibling checkout (the scripts read the same variable).
    "env_names": ("DOCTR_CACHE_DIR", "HF_HUB_OFFLINE", "HOME", OCR_RUNTIME_ENV, "PYTHONDONTWRITEBYTECODE",
                  "PYTHONUNBUFFERED", "PYTORCH_CUDA_ALLOC_CONF", "TOKENIZERS_PARALLELISM", "XDG_CACHE_HOME"),
    "scan_skip": HOST_ROWS["scan_skip"],
    "interpreter_trees": HOST_ROWS["interpreter_trees"],
    "interpreter": HOST_ROWS["interpreter"],          # the OCR runtime's (area 5's) venv (one lock)
    "container_paths": CONTAINER_PATHS,
    "env_paths": ("DOCTR_CACHE_DIR", "HOME", "XDG_CACHE_HOME"),
}

# Area "video generation" evaluators (runtime/README.md "Decisions": deciding `ocr_b_acc` T3, construct-aligned
# under R15-1, blocking; `ocr_a_acc` supporting, T0). Scorer: eval_videos.py (generation and both OCR engines in one GPU
# job), code_sha256 its SHA-256 (of the published file; the gate result was produced by sha256:de9c2e89..., which
# differs in comments, names, how it locates the OCR runtime and its arm schema id: RESULT.md "Published code"); the
# versions and weight hashes are the strings it wrote in probe-base-1 (2026-11-14). An
# item is one registered prompt of flow_grpo's OCR test split (area 5's 400 items, byte copies); its `prompt` is the
# exact-overlap key; `source_line` is an integer and not a payload field.
EVAL_VIDEOS_SHA256 = "sha256:61873167b855e84df94edf99492ee52ccb6f3147127502cbfaf157b2bee59b4f"
_WAN = {
    "direction": "higher_is_better",
    "score_range": (0, 1),
    "binary": False,
    "error_classes": ("scorer_error",),                  # MM-4: an OCR engine raised (eval_videos.py)
    "score_outcomes": ("nonfinite_video",),              # MM-4: a non-finite video is scored on the NaN-zeroed frames
    "payload_text_field": "prompt",
    "item_payload": ("prompt", "target"),
    "normalizer": False,
    "env_version": False,
    # EXT-222: the deployment sampler of the cuts (both scripts' `sampler_doc`), exactly these keys.
    "sampler_keys": ("autocast", "cfg", "decode", "deterministic", "guidance_scale", "height", "latents_dtype",
                     "max_sequence_length", "negative_prompt", "noise", "num_frames", "num_steps", "pipeline",
                     "prompt_embedding", "scheduler", "scheduler_config", "sigmas", "step", "step_index",
                     "text_encoder_dtype", "timesteps", "transformer_dtype", "vae_dtype", "width"),
    "framework": "diffusers",
    "samples": "400, 1",
}
EVALUATOR_ROWS["t2v.wan_ocr_doctr_items@0.1"] = {
    **_WAN, "metric": "ocr_b_acc",
    # 05 section 6.1, R15-1: another OCR engine (docTR) than the reward's (PaddleOCR PP-OCRv4), another formula (the
    # best-matching substring) and another frame set and aggregate (all 17 frames, zeros included, instead of every 4th
    # frame's positive scores), so T3; the same construct, so construct-aligned, stated in the registration and report.
    "tier": "T3",
    "construct_aligned": {
        "reward": "video_ocr_paddleocr",
        "statement": ("ocr_b_acc measures the construct of the training reward video_ocr_paddleocr (the prompt's quoted "
                      "text is legibly rendered in the video) with another OCR engine (docTR db_resnet50 + "
                      "crnn_vgg16_bn, not PaddleOCR PP-OCRv4), another formula (edit distance of the best-matching "
                      "substring of the recognized text) and another aggregate (the mean over all 17 frames, frames "
                      "that read nothing included, not the mean of the positive scores of every 4th frame): a pass "
                      "shows that the reward's gain transfers to an independent OCR engine over the whole clip, not "
                      "that the videos are better otherwise (05 section 6.1, R15-1)")},
    "scorer": {"name": "video_ocr_accuracy_doctr",
               "version": ('eval_videos@0.1 engine docTR db_resnet50+crnn_vgg16_bn python-doctr 1.1.0 '
                           '{"assume_straight_pages": true, "det_arch": "db_resnet50", "join": "words in reading '
                           'order, no separator", "preserve_aspect_ratio": true, "reco_arch": "crnn_vgg16_bn", '
                           '"symmetric_pad": true} approximate-substring edit distance (Sellers DP), spaces removed, '
                           'lower case; per frame, mean over all frames (zeros included); frames read back from '
                           'lossless PNG'),
               "weights_sha256": "sha256:7380ad4e2df54bd2b610b1e19e7cde2b4a35d06c9fa00eda70f7f6b23947004c",
               "code_sha256": EVAL_VIDEOS_SHA256}}
EVALUATOR_ROWS["t2v.wan_ocr_paddle_items@0.1"] = {
    **_WAN, "metric": "ocr_a_acc", "tier": "T0",       # the training reward itself: supporting, never deciding
    "scorer": {"name": "video_ocr_reward_paddleocr",
               "version": ("eval_videos@0.1 flow_grpo video_ocr (ocr.py:65-123): every 4th frame, 1 - min(lev, "
                           "len)/len, mean over positive frames, 0 if none; engine PaddleOCR PP-OCRv4 paddleocr 2.10.0 "
                           "paddlepaddle 3.0.0 PP-OCRv4_mobile_det@3cc09f3a5b424e8e010abc7a4271aea12999c2f7 "
                           "en_PP-OCRv4_mobile_rec@f97b62fdc0eb71c689393a19a4b21baa1795c9ab"),
               "weights_sha256": "sha256:7783ef1516fc76bcd590b6fac074b4942f15a9111c9d55506256a8067d13f07a",
               "code_sha256": EVAL_VIDEOS_SHA256}}

PLUGIN = {"adapters": {"flow_grpo_wan": "gatedtrain_recipe_flow_grpo_wan_video.adapter:FlowGrpoWanAdapter"}, "trainer_rows": TRAINER_ROWS,
          "evaluator_rows": EVALUATOR_ROWS}
