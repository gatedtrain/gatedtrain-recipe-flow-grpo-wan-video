# Area "video generation": Flow-GRPO + LoRA on Wan2.1-T2V-1.3B, video OCR (design notes, prototype, unpromotable)

Multimodal prototype, scope row 3 ("Video generation | flow_grpo | Wan2.1-T2V-1.3B + LoRA, reduced resolution and
frames"), gatedtrain's `MULTIMODAL-SPEC.md` §6 row 3 and its row 3 deviations. Runtime kind `venv_in_container` (MM-32), as
the SD1.5 DDPO and SD3.5 OCR recipes.

These are the design notes written while the recipe was built (2026-11-14 to 2026-11-15), kept for the reasoning
behind it. The sections up to "Profiles and cost" were written before any download or GPU run: facts read from code
or from the Hugging Face API carry a file:line or source, every number marked **estimate** was unmeasured then, every
claim marked **unverified** was not checked then. The trainer and evaluator sections were updated as they were built;
"Decisions" records what was chosen, "Attempt 2 (amendment)" and "Result" what happened.

Names used below: "area 5" and `area5-sd35/` are the SD3.5 OCR recipe and its `runtime/` (`<sd35-ocr-runtime>`),
"area 3" (SD1.5) the DDPO SD1.5 recipe, `area3-video/` this recipe's `runtime/` (`<runtime>`); `out/`, `runs/` and
`records/` are the run outputs, collected runs and launch records under the host's gatedtrain output root (`<output-root>`).

## Naming

- The scope table numbers this area 3; the launch request's `area` field is 3 (scope row 3), and the `out/`, `runs/`
  and `records/` subdirectories are named `area3-video` (distinct from the SD1.5 recipe's `area3-sd15`); the
  convention break is stated in spec row 3's deviations.
- No new reference clone is needed: the flow_grpo clone used for the SD3.5 recipe (commit
  `879042cf5707f8b90daa98d147d7deac2317c5da`, 2026-05-07, MIT) already contains the Wan2.1 path (below).

## Model facts (Hugging Face API, read 2026-11-14 with `env -u HF_TOKEN`, no token)

| Repo | Revision (`sha`) | Licence | Gated | Size (all files) |
|---|---|---|---|---|
| `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | `0fad780a534b6463e45facd96134c9f345acfa5b` (lastModified 2025-04-04) | apache-2.0 (card `license`) | false | 28.94 GB |
| `Wan-AI/Wan2.1-T2V-1.3B` (original layout) | `37ec512624d61f7aa208f7ea8140a131f93afc9a` (2025-03-01) | apache-2.0, `LICENSE.txt` | false | 17.57 GB |

Diffusers repo components (`model_index.json`: `WanPipeline`, `_diffusers_version` 0.33.0.dev0):

| Component | Class | Files | Size | Notes |
|---|---|---|---|---|
| `text_encoder/` | `UMT5EncoderModel` (`google/umt5-xxl` encoder, d_model 4096, 24 layers, d_ff 10240, vocab 256,384, `torch_dtype` float32) | 5 safetensors shards + index | **22.72 GB (fp32)** → ≈ 5.68 B params | bf16 in memory ≈ 11.4 GB; transformers keeps `wo` in fp32 (`modeling_umt5.py:520` `_keep_in_fp32_modules = ["wo"]`, area-5 venv), so ≈ 13.4 GB on the GPU (**estimate**) |
| `transformer/` | `WanTransformer3DModel` (30 layers, 12 heads × 128, ffn 8960, patch (1, 2, 2), in/out 16 ch, text_dim 4096, `added_kv_proj_dim` null) | 2 safetensors shards + index | 5.68 GB (fp32) → ≈ 1.42 B params | bf16 ≈ 2.84 GB |
| `vae/` | `AutoencoderKLWan` (z_dim 16, temporal downsample ×4, spatial ×8) | 1 safetensors | 0.51 GB (fp32) | |
| `scheduler/` | `UniPCMultistepScheduler`, `use_flow_sigmas` true, `flow_shift` 3.0, `final_sigmas_type` zero, `prediction_type` flow_prediction | json | — | flow_grpo uses only its sigma table (below) |
| `tokenizer/` | `T5TokenizerFast` (`spiece.model`, `tokenizer.json`) | 4 files | 21 MB | |
| `assets/`, `examples/` | images | 10 files | 6.8 MB | not needed |

The original repo holds the same transformer in one 5.68 GB safetensors file with original key names, the umT5
encoder as a **bf16 pickle** `models_t5_umt5-xxl-enc-bf16.pth` (11.36 GB) and the VAE as `Wan2.1_VAE.pth`; using it
means converting keys and unpickling, so it is not proposed.

**Download needed:** the Diffusers repo at `0fad780a…` without `assets/` and `examples/`:
**≈ 28.93 GB**, of which 22.72 GB is the fp32 umT5 encoder. Method as area 5 (its README "Models":
`token=False`, `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`, `HF_TOKEN` removed, fresh `HF_HOME` outside the area, every LFS
file's SHA-256 checked against the API's `lfs.sha256`), into `area3-video/models/Wan2.1-T2V-1.3B-Diffusers/`, then
`MODELS.json` with per-file SHA-256.

**Can the text encoder be kept off the training GPU?** Yes. In flow_grpo the umT5 encoder stays on the GPU for the
whole run (`scripts/train_wan2_1.py:440`) and encodes every sampling batch (`:652-658`). The prompt embedding is a
pure function of (base, prompt): flow_grpo pads to 512 tokens with zeros after cutting at the attention mask
(`flow_grpo/diffusers_patch/wan_prompt_embedding.py:17-33`), so a cached embedding trimmed to its length and
re-padded with zeros is bit-identical. The port encodes every prompt the run will use at start (the prompt schedule
is fixed by the seed, as area 5's `encode_all_prompts`, `area5-sd35/train_flow_grpo.py:181`), frees the encoder, then
loads transformer and VAE. Size: ≈ 60 tokens × 4096 × 2 bytes ≈ 0.5 MB per prompt (**estimate** from the OCR prompt
lengths), ≈ 0.1-0.2 GB for a run, kept on the CPU. GPU peak of the encode phase ≈ 13.4 GB alone (**estimate**). Host
RSS: `from_pretrained(torch_dtype=bfloat16, device_map="cuda")` streams the fp32 shards; area 5's T5-XXL load
peaked at 12.2 GiB host RAM in its smoke (`host_ram_peak_gb` 12.19); ≤ 14
GiB expected here (**estimate**, measured by the smoke).

## flow_grpo's Wan2.1 path (clone of `yifan123/flow_grpo` @ `879042cf`, read-only)

| Item | Finding |
|---|---|
| Entry | `README.md:37-40` (2025-08-15): `accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=1 … scripts/train_wan2_1.py --config config/grpo.py:general_ocr_wan2_1` |
| Script | `scripts/train_wan2_1.py` (1,012 lines), pipeline patch `flow_grpo/diffusers_patch/wan_pipeline_with_logprob.py` (373 lines), prompt encoder `flow_grpo/diffusers_patch/wan_prompt_embedding.py` |
| Config | `config/grpo.py:29-81` `general_ocr_wan2_1`: base `hf_cache/Wan2.1-T2V-1.3B-Diffusers` (:34); dataset `dataset/ocr` (:31, same OCR prompts as SD3.5); `sample.num_steps` 20 (:35), `eval_num_steps` 50 (:36), CFG 4.5 (:37); **240 × 416, 33 frames** (:40-42); `train_batch_size` 8, `num_image_per_prompt` 4 (:43-44), `num_batches_per_epoch` 2 (:45) → 16 videos (4 prompts × 4) per epoch; `timestep_fraction` 0.99 (:52) → int(20 × 0.99) = **19 trained steps** per video (`train_wan2_1.py:373`); KL β 0.004 (:54); lr 1e-4 (:55); PPO clip 1e-3 (:56); `global_std` False (:64, per-group std); EMA on (:65); **bf16** (:66); `save_freq` 60, `eval_freq` 30 (:70-71); reward `{"video_ocr": 1.0}` (:74-76) |
| Base settings inherited | `config/base.py`: AdamW β 0.9/0.999, wd 1e-4, ε 1e-8 (:76-84), grad clip 1.0 (:89), adv clip 5 (:97), `train.cfg` True (:95), seed 42 (:14) |
| LoRA | r 32, α 64, gaussian init (`train_wan2_1.py:461-466`), targets `add_k_proj, add_q_proj, add_v_proj, to_add_out, to_k, to_out.0, to_q, to_v` (:451-460). Wan T2V has no added KV projections (`transformer/config.json` `added_kv_proj_dim` null; diffusers 0.42.1 `transformer_wan.py:211`), so the adapter lands on `to_q/k/v/out.0` of both attentions of 30 blocks: 240 linears × r(1536 + 1536) ≈ **23.6 M params, ≈ 94 MB fp32** per cut (**estimate** from the shapes) |
| Precision and placement | pipeline loaded without `torch_dtype` (`:407-409`, fp32), VAE fp32 on GPU (`:439`), umT5 bf16 on GPU (`:440`), transformer fp32 on GPU (`:445`), gradient checkpointing on (`:475`); README.md:388: Wan needs bf16, fp16 inference "cannot produce valid images or videos" |
| Sampler (training) | `wan_pipeline_with_logprob.py:255-341`: SDE on **every** step (no SDE window), CFG by two separate transformer calls (`:265-282`); SDE step `:10-84`: std_dev_t = σ_min + (σ_max − σ_min)·σ with **σ_max = `sigmas[1]`** and σ_min = `sigmas[-1]` (:47-51; 0 with `final_sigmas_type` zero); log-prob averaged over all latent elements (:80). The UniPC scheduler serves only as a sigma table (`index_for_timestep`, `sigmas`; diffusers 0.42.1 `scheduling_unipc_multistep.py:1100`, `:428-465`) |
| Sampler (evaluation) | `train_wan2_1.py:232-331` `eval`: `determistic=True` (Euler ODE, patch :70-71), 50 steps, CFG 4.5, EMA weights swapped in |
| Training loop | `:879-1005`: per micro-batch of 8 videos, for each of 19 steps: cond + uncond forward (`compute_log_prob` :190-230), KL reference forward with the adapter disabled (`:937-941`), PPO-clip loss, Accelerate accumulation over 19 steps (`:382`); EMA decay 0.9, update every 8 steps (`:478`, `:1004-1005`), cuts save EMA weights (`:337-346`) |
| Cuts | `save_ckpt` → `<save_dir>/checkpoints/checkpoint-<global_step>/lora/` (PEFT `save_pretrained`, `:337-344`); only at `epoch % 60 == 0 and epoch > 0` (`:668-669`), so spec row 3's "`checkpoint-<step>/lora/` (unverified for Wan)" is confirmed for upstream; a port writes area 5's layout instead |
| Reward `video_ocr` | `flow_grpo/rewards.py:143-158` → `flow_grpo/ocr.py:65-123` `OcrScorer_video_or_image`: PaddleOCR (`use_angle_cls` False, `lang` en, CPU), frames `video[::4]` (:77, :96; 17 frames → 5: 0, 4, 8, 12, 16), per frame 1 − min(lev(text, target), len) / len on space-free lower case (:106-116; **no substring shortcut**, unlike the image reward `ocr.py:46-47`), and the video reward is the **mean over frames with a positive score only**, 0 if none (:117-123) |
| Quirks | the first two epochs sample nothing (`:670-672`, `:731-732`, "group size bug"); `transformer.module.disable_adapter()` (`:940`) assumes a DDP wrapper (**unverified** whether it raises with `--num_processes=1`); in-loop wandb (`:24`, `:394-400`) and video logging; a `DistributedKRepeatSampler` on its own `torch.Generator` (`:81-124`) |

### Does it run as-is on one 24 GB GPU? No (same blockers as area 5, plus video specifics)

- **Dependencies:** `setup.py:9-51` pins torch 2.6.0, diffusers 0.33.1, transformers 4.40.0, accelerate 1.4.0, peft
  0.10.0, numpy 1.26.4, deepspeed, xformers, wandb; paddleocr 2.9.1 needs `numpy<2` (area 5 README "Trainer choice");
  `config/grpo.py:2,5` uses `imp` (removed in Python 3.12). Area 5's pins (diffusers 0.42.1, transformers 5.21.0,
  peft 0.21.2, torch 2.15.0) already contain everything the Wan path calls: `WanPipeline`
  (`diffusers/pipelines/wan/pipeline_wan.py`), `WanTransformer3DModel` (`models/transformers/transformer_wan.py:507`,
  forward with `hidden_states, timestep, encoder_hidden_states, attention_kwargs, return_dict` as flow_grpo calls it),
  `AutoencoderKLWan` (`models/autoencoders/autoencoder_kl_wan.py`), `UniPCMultistepScheduler.index_for_timestep`
  (`:1100`), `UMT5EncoderModel` (`transformers/models/umt5/modeling_umt5.py:1091`), `imageio` and `cv2` (all in
  `area5-sd35/.venv`).
- **Network:** wandb and Hub ids; the container runs with network `none`.
- **Memory (estimate):** flow_grpo keeps umT5 bf16 (≈ 13.4 GB with fp32 `wo`), the transformer in fp32 (5.7 GB), the
  VAE fp32 (0.5 GB) resident, and trains micro-batches of 8 videos with CFG → ≈ 20 GB of weights before LoRA, Adam,
  stored chains and activations: does not fit 24 GB.
- **EMA:** decay 0.9 every 8 steps with this budget's ≈ 80 optimizer steps saves a lagging LoRA (area 5's finding;
  area 5 decided EMA off).

## Proposed trainer: a minimal port `train_flow_grpo_wan.py` (area 5 precedent)

Built from `area5-sd35/train_flow_grpo.py` (same run layout, seeds, logs, cuts, OCR engine A) with the Wan pieces
taken line by line from `train_wan2_1.py` and `wan_pipeline_with_logprob.py`. One venv: area 5's `pyproject.toml`
and `uv.lock` copied unchanged (110 packages, sha256 `8243b02f…`; since python-Levenshtein's removal and the rename,
109 packages, `502c7ce7…`, `RESULT.md`), so the framework pins are identical to area 5;
runtime image `gatedtrain-a5-runtime:1` reused (it adds the X11/GL libraries that `import cv2` needs; built from
`container/Containerfile`, a byte-identical copy of the SD3.5 OCR recipe's). No new image.

Same as flow_grpo `general_ocr_wan2_1`: SDE sampler on every step with the Wan σ rule (σ_max = `sigmas[1]`), 20
sampling steps, CFG 4.5 with separate cond/uncond calls, umT5 at 512 tokens with zero padding, negative prompt "",
19 trained steps per video with CFG, per-prompt advantages with the group's std, adv clip 5, PPO clip 1e-3, KL β
0.004 against the adapter-disabled transformer, AdamW lr 1e-4, grad clip 1.0, LoRA r 32 / α 64 gaussian on the
attention projections, 4 prompts × 4 videos per epoch, two optimizer steps of 8 videos, zero-advantage filter with
its padding rule and the all-zero `+1e-6` rule (`train_wan2_1.py:845-871`), bf16, VAE fp32, `video_ocr` reward
formula. Different, with reason:

| Change | Reason |
|---|---|
| **Frames 17** (spec row 3: 240 × 416 × 17) instead of 33 | latent 5 × 30 × 52 → 1,950 video tokens (33 frames: 9 × 30 × 52 → 3,510); ≈ half the cost and activation memory; 17 = 4k + 1 as the VAE requires (`pipeline_wan.py:494-498`) |
| umT5 encodes all prompts of the run at start, then is freed | memory (above); bit-identical embeddings |
| Transformer stored in bf16, LoRA fp32, autocast bf16 | flow_grpo stores fp32 and autocasts (Accelerate); same matmul precision, half the weight memory |
| Training micro-batch 2 (accumulated to flow_grpo's 8) | memory; the gradient equals flow_grpo's mean over 8 videos × 19 steps (as area 5) |
| EMA off (cuts hold the raw LoRA) | area 5 decision; `--ema` refused at launch |
| No skipped first two epochs, no in-run eval, no wandb, cuts every N epochs incl. the last | upstream quirks; offline; evaluation is the separate per-item evaluator |
| Prompts drawn without replacement from the registered shard with seeded generators; per-video noise seed (CPU generator) for the initial latents and every SDE step | observed row-level consumption (MM-24, MM-29), common random numbers at evaluation |
| `samples.jsonl` with per-frame OCR texts and scores, `epoch_plan.jsonl`, `train_steps.jsonl` | MM-24, MM-28 (as area 5) |
| OCR engine A = area 5's PaddleOCR 2.10.0 `TextSystem` build (PP-OCRv4 mobile det + en PP-OCRv4 mobile rec, local models, paddlepaddle 3.0.0 CPU) | offline; same engine as area 5's reward (`area5-sd35/README.md` "Changes") |
| Frames passed to OCR as RGB uint8 (as flow_grpo `rewards.py:149-153`) | unchanged behaviour |

The flow_grpo `DistributedKRepeatSampler` shuffles the 8 slots of a batch across 2 prompts (`:106-110`); the port
keeps groups contiguous (one prompt per 4 slots). No effect on the algorithm; stated as a deviation.

### Memory and time (estimates, to be replaced by the smoke)

- **FLOPs per transformer forward per video at 240 × 416 × 17:** linear layers ≈ 2 × 1.39 B × 1,950 ≈ 5.4 TFLOP,
  self-attention ≈ 4 × 1,950² × 1,536 × 30 ≈ 0.7 TFLOP, cross-attention to 512 text tokens ≈ 0.2 TFLOP → ≈ 6.3
  TFLOP, about the same as SD3.5 Medium at 512 px in area 5 (2.47 B × 2 × 1,229 ≈ 6.1 TFLOP).
- **Calibration from area 5 (measured, its full run's `memory.jsonl`):** 64 samples per epoch, sampling 42 s
  (10 steps × CFG), training 128 s (9 steps, KL reference, checkpointing, micro-batch 2) → 0.066 s per sample-step
  sampling, 0.22 s per sample-step training. Wan's README states 4 min for 81 frames 832 × 480 at 50 steps on a
  4090 (`README.md:28` of the Diffusers repo), ≈ 120 effective TFLOPS at 32,760 tokens; at 1,950 tokens efficiency is
  lower, so per-forward cost is taken as 1.2-1.7 × area 5's.
- **Per video:** sampling 20 steps ≈ 1.6-2.3 s + VAE decode of 17 frames ≈ 0.5-1.5 s; training 19 steps × 0.26-0.37
  s ≈ 5-7 s; PaddleOCR on 5 frames on the CPU ≈ 0.5-2 s → **≈ 8-12 s per video, ≈ 2.2-3.2 min per 16-video epoch**.
  At 33 frames ≈ 2 ×.
- **Peak GPU memory:** encode phase ≈ 13.4 GB (alone). Training phase: transformer bf16 2.8 + VAE fp32 0.5 +
  LoRA/Adam/grads 0.4 + stored chains (16 videos × 21 latents × 2 × 0.5 MB + 16 × 4 MB embeddings ≈ 0.4 GB) +
  sampling batch activations 1-2 + VAE decode of one video 2-4 (decoded one by one) + training micro-batch 2 with
  checkpointing and CFG 2-4 → **≈ 9-14 GB**; MM-38 bound 22.5 GB leaves room for micro-batch 4 or 33 frames.
- **Host RAM:** encode phase ≤ 14 GiB, then ≈ 4-6 GiB + PaddleOCR (≈ 1 GiB) (**estimate**); a 32 GiB machine is
  enough unless another job shares it.

## Reward, deciding metric and tier (proposal)

| # | Training reward | Deciding metric | Tier (05 §6.1) | Verdict |
|---|---|---|---|---|
| 1 | flow_grpo `video_ocr`: PaddleOCR (engine A), frames 0, 4, 8, 12, 16, mean of positive frame scores | **`ocr_b_acc`**: docTR (engine B, db_resnet50 + crnn_vgg16_bn, area 5's weights), approximate-substring accuracy per frame (area 5's `ocr_substring_accuracy`), **mean over all 17 frames, zeros included** | **T3, construct-aligned (R15-1)**, as area 5: another engine, another formula, another frame set and aggregate | **recommended** |
| 2 | same | the reward itself (`ocr_a_acc`, T0) | T0: supporting only, never deciding (05 §6.1; every T0 row of the spec is "supporting, never deciding") | not recommended: spec row 3 says T0, but a T0 deciding metric cannot gate, so the area could not PASS |
| 3 | frame-level PickScore / aesthetic / CLIP | a different learned scorer | T2 at best, needs human-preference validity (C6) | not for the prototype |
| 4 | per-frame JPEG / H.264 compressibility | another codec | T3 construct-aligned (area 3's construct) | fallback only (repeats area 3, ignores flow_grpo's Wan task) |

Supporting arm: `ocr_a_acc` = the reward formula exactly (T0). Both OCR engines, their weights and hashes are area
5's (`area5-sd35/MODELS.json`, Apache-2.0; PaddleOCR 2 × < 20 MB, docTR 2 files): copy into `area3-video/ocr-models/`
(no download).

**MM-38 degeneracy risk (the main risk of this area).** Wan2.1 claims English text rendering (`README.md:30` of the
Diffusers repo), but at 1.3 B, 240 × 416 and 17 frames legibility is unknown. If most base videos read nothing,
(a) MM-38 fails on `ocr_b_acc` (≥ 50 % of items at 0) and (b) most 4-video groups have equal (zero) rewards and are
filtered, so training barely moves. Mitigations in order: (1) **render probe before any code is written**: 20 base
videos of training prompts through diffusers' plain `WanPipeline` at 240 × 416 × 17, 20 steps, CFG 4.5, scored by
both engines on the CPU (≈ 5 GPU-min, **estimate**); (2) if the share at 0 exceeds ≈ 0.5: restrict both the shard
and the items to short targets (for example ≤ 10 characters without spaces), decided before registration and
disclosed; (3) else raise the resolution to 480 × 832 with 9 frames (3 × 30 × 52 = 4,680 tokens, ≈ 2.5 × cost) —
Wan's recommended 480P (`README.md:93`); (4) last resort reward 4. The mean over 17 frames makes the deciding metric
continuous, which helps the "fewer than 50 % share one value" rule as long as text appears at all.

**Hacking risk to report.** As in area 5, OCR rewards trade visual quality for legible text; a video may also freeze
on a text card (temporal collapse). Not gated; reported descriptively (sub-metric: mean absolute frame difference).

## Items, training shard and power (proposal)

- **Reuse area 5's files unchanged** (copied, same bytes and hashes): items `area5-sd35/items/items.jsonl` (400
  flow_grpo OCR test prompts, sha256 `3884dcff…20f3e8`) and shard `area5-sd35/prompts/train.jsonl` (18,359 rows,
  `75361afd…585fb8`), disjoint by prompt and by normalized target (area 5 README "Items"). flow_grpo's Wan config
  uses the same `dataset/ocr` (`config/grpo.py:31`). If the probe forces a short-target subset, both files are
  filtered by a recorded rule.
- **N = 400, K = 1**, `eval_seeds [11]` (spec row 3's "50 prompts" estimate gives MDE 0.40 σ_d, too coarse).
  Smoke: 40 items per arm.
- **Power:** paired, superiority δ 0, α 0.025 one-sided, power 0.8: MDE = 2.8016 σ_d / √N = **0.140 σ_d at N 400**
  (0.162 at 300, 0.198 at 200).
- **Sigma rule (as areas 5-7):** `sigma_plan: fixed` = max(1.5 × ucb80(s_pilot), √2 × ucb80(s_base)) from the smoke's
  40 items; ucb80(s) = s × √((n − 1) / χ²₀.₂₀,ₙ₋₁) = **1.114 s at n 40** (1.177 at n 20; scipy, area-5 venv).
  Example: s_pilot 0.20, s_base 0.25 → σ_plan = max(0.334, 0.394) = 0.394 → planning MDE 0.055 at N 400. Area 5
  observed Δ̂ 0.045 with σ_d 0.25 (its RESULT.md); the video run has fewer samples (≈ 640 vs 1,920), so the
  effect may be smaller: the full run's epochs are set from the smoke so that the run is not the bottleneck.

## Evaluator: `eval_videos.py` (built 2026-11-14; CPU-tested, not yet run on the GPU)

Generation and scoring in one GPU job, as area 5's `eval_images.py` (copied and adapted, not imported). Decisions:
"Decisions" below.

- **Items and shard:** `python eval_videos.py --copy-items` copied area 5's `items/{items.jsonl,item_set_meta.json}` and
  `prompts/{train.jsonl,prompts_meta.json}` byte for byte (refuses a differing or extra file). Nothing is added inside
  `items/` or `prompts/`, because the item set hash is the manifest hash of the directory: the copies keep area 5's
  hashes (items `3884dcff…20f3e8`, shard `75361afd…585fb8`; directory manifest `sha256:0b01553c…8ad4a7` for `items/`
  and `sha256:4483a9f2…d3b457` for `prompts/`, the same as area 5's, measured with `gatedtrain.canonical.manifest_hash`).
  Source paths and SHA-256s are in `ITEMS-SOURCE.json`.
- **Generation (deployment sampler, shared block `BEGIN/END SHARED SAMPLER`, `sampler_code_sha256`
  `sha256:dad91e41…d90300` as run; `sha256:41606b0b…dcb1f17` since its edit-distance import moved to rapidfuzz):** umT5 loaded in bf16 straight to the GPU, every module cast to bf16 as flow_grpo
  (`train_wan2_1.py:440`), encodes `""` and every item prompt once exactly as `wan_prompt_embedding.py:4-34` (no prompt
  cleaning, unlike diffusers' `WanPipeline`; cut at the mask, zero-padded to 512; kept trimmed on the CPU), then is
  freed; then transformer bf16 (diffusers keeps its `_keep_in_fp32_modules` in fp32) under bf16 autocast, VAE fp32,
  UniPC scheduler as sigma table only. Sampler: the deterministic branch of `wan_pipeline_with_logprob.py` (Euler ODE
  `x + dt·v`, `:70-71`), 20 steps (int64 timesteps 999 … 138, sigmas 0.999999 … 0.138715, 0 from the base's
  scheduler config, flow_shift 3.0), CFG 4.5 with two transformer calls (prompt, negative prompt `""`, as flow_grpo
  `train_wan2_1.py:235`), combined in float32, 240 × 416 × 17 (latent 16 × 5 × 30 × 52), initial noise from a CPU
  generator seeded `derive_seed('eval', seed, item)` (identical in every arm), decode per video (`latents ·
  latents_std + latents_mean`, `(x / 2 + 0.5)` clamped, uint8 = `round(x · 255)`). `--batch-size` default 4.
  Base-only `sampler_hash` with the defaults: `sha256:997c2bf0…5ff9e3` (computed on the CPU from the base's
  scheduler config). A cut is `checkpoint-<epoch>/` as area 5 (`pytorch_lora_weights.safetensors`,
  `adapter_config.json`, `sampler.json`, `base_ref.json`, `trainer_state.json`, `cut.json` last); the evaluator checks
  every file hash, the shared sampler code hash, the base (revision and every file against `models/MODELS.json`),
  the LoRA `target_modules` = flow_grpo's Wan list (`train_wan2_1.py:451-460`) and that the cut's `sampler.json`
  equals this evaluator's, then loads the LoRA with `load_lora_adapter(prefix=None)`. The trainer port must copy the
  shared block byte for byte and write `sampler.json` with `sampler_doc`.
- **Outputs (`/out`):** `videos/<item>/r<k>/frame_00.png … frame_16.png` (lossless RGB PNGs, each read back and
  compared with the written frame) and `frames.json` (file names, SHA-256s, shape) whose path and SHA-256 are the
  item's `output_ref`; `scores.jsonl` (per-frame texts and scores of both engines, noise seed, target, mean absolute
  frame difference); `arm-ocr_b_acc/` (deciding, `t2v.wan_ocr_doctr_items@0.1`) and `arm-ocr_a_acc/` (supporting T0,
  `t2v.wan_ocr_paddle_items@0.1`), each `items.jsonl` + `eval_meta.json` with exactly the MM-2 keys (adapter
  `flow_grpo_wan`, framework diffusers) + hard-linked `videos/`; `summary.json` (means, sd, shares at 0 and 1, per-arm
  advisory MM-38 checks on item means: mean inside [0.05, 0.95], mode share < 0.5, sd > 0, zero errors; timing per
  stage, peak GPU memory of the encode and the generate phase from `torch.cuda.max_memory_allocated`/`reserved`, host
  RSS at start, after encode, after load and peak); `env-runtime.json`.
- **Metrics (from the saved uint8 frames):** `ocr_b_acc` = mean over all 17 frames of area 5's
  `ocr_substring_accuracy` with docTR (engine B, area 5's weights, GPU), frames that read nothing count 0;
  `ocr_a_acc` = flow_grpo's `video_ocr` exactly (`ocr.py:65-123`): PaddleOCR (engine A, area 5's `TextSystem` build,
  CPU) on frames 0, 4, 8, 12, 16, per frame `1 − min(lev, len) / len` without the substring shortcut, mean over frames
  with a positive score, 0 if none. MM-4: an OCR exception is a `scorer_error` record (in the reward port it is a 0
  frame, as flow_grpo); a non-finite video is scored on the NaN-zeroed frames and counted in `nonfinite`.
- **OCR models:** read from area 5's runtime (`<sd35-ocr-runtime>/ocr-models/`, hashes from its `MODELS.json`;
  located by `GATEDTRAIN_RECIPE_FLOW_GRPO_WAN_VIDEO_OCR_RUNTIME`, else the sibling checkout), so the launch mounts it
  read-only (it also holds the venv).
- **CPU tests** (`tests/test_eval_videos.py`, area 5's venv, 15 tests, ≈ 40 s): argument rules; item copies equal
  area 5's bytes; `derive_seed` and `ocr_substring_accuracy` equal area 5's; hand cases of both formulas; the
  `video_ocr` port equals flow_grpo's own `OcrScorer_video_or_image` on 40 synthetic videos (random recognized lines,
  zero-score lines, OCR exceptions); the prompt encoding is bit-identical to flow_grpo's `_get_t5_prompt_embeds` (tiny
  random UMT5, stub tokenizer); the sampler and decode equal flow_grpo's `wan_pipeline_with_logprob(determistic=True)`
  within 1e-5 (tiny random transformer and VAE, fp32, CPU); the real OCR engines on a rendered 17-frame text video
  (A: frames 0/4/8 "OPEN LATE" 1.0, 12/16 0 → reward 1.0; B: frames 0-8 1.0, 9-16 0); `evaluate()` end to end on a
  tiny random base with the real OCR engines (outputs, MM-1/MM-2 arm keys, frame and manifest hashes, hard links,
  identical frames in two runs, a LoRA cut with the same `sampler_hash` and different frames, refusals of a changed
  cut file, another sampler code, other LoRA targets, another `--num-frames`, a non-empty `--out`).
- **Time per arm (estimate, unmeasured):** 400 × (1.6-2.3 s generation + 0.5-1.5 s decode) ≈ 14-25 min + OCR A on 5
  frames per video 3-13 min (CPU) + OCR B 17 frames per video ≈ 3-6 min + load/encode ≈ 3 min → ≈ 25-45 min.

## Trainer: `train_flow_grpo_wan.py` (built 2026-11-14; CPU-tested, not yet run on the GPU)

Minimal port of flow_grpo's `scripts/train_wan2_1.py` (`general_ocr_wan2_1`), built from area 5's
`train_flow_grpo.py` (copied and adapted, not imported) and aligned to the evaluator: the `BEGIN/END SHARED SAMPLER`
block is copied byte for byte from `eval_videos.py` (`sampler_code_sha256`
`sha256:dad91e418fe717bf2ad0b3fe2727dfdea36a316b87c2240efc0aa14955d90300` as run, now
`sha256:41606b0b53ad637da4614aaa38fd52b5e9bf0b2147da7ed71c3eaad92dcb1f17`, `RESULT.md`), so the trainer and the evaluator share
`derive_seed`, the umT5 encoding, `load_transformer_vae`, the Wan SDE step, CFG, the sampling loop, decode, the
`video_ocr` reward (`PaddleOcrA.video_reward`) and `sampler_doc`. Decisions: "Decisions" below, and area 5's (minimal
port, EMA off, cut layout, `filtered` zero-advantage outcomes).

**Same as flow_grpo:** SDE sampler on every step with σ_max = `sigmas[1]` (`wan_pipeline_with_logprob.py:10-84`),
20 sampling steps, CFG 4.5 by two transformer calls with negative prompt `""`, umT5 512 tokens cut at the mask and
zero-padded, 19 trained steps of 20 (`timestep_fraction` 0.99) with CFG, per-prompt advantages with the group's own
std (`global_std` False) + 1e-4, advantage clip 5, PPO clip 1e-3, KL β 0.004 = mean squared difference of the step
means of the LoRA and the adapter-disabled transformer over 2 (σ_t·√−dt)² (`train_wan2_1.py:938-961`), AdamW lr 1e-4
(β 0.9/0.999, wd 1e-4, ε 1e-8), grad-norm clip 1.0, LoRA r 32 / α 64 gaussian on flow_grpo's Wan list (matches
`to_q/k/v/out.0` of `attn1` and `attn2` in 30 blocks; 23.6 M parameters expected, logged as
`run_args.memory_at_start.lora_params`), 4 prompts × 4 videos per epoch, `num_batches_per_epoch` 2 and
`gradient_accumulation_steps` 1 (two optimizer steps of the kept videos / 2), the zero-advantage filter with its
padding to a multiple of 2 **and the all-zero rule** (every advantage 0 → all + 1e-6, every video trained; the
padding test uses the count before the revision, as `:853-860`), 240 × 416 (decision 17 frames), bf16, VAE fp32,
`video_ocr` = PaddleOCR on frames 0, 4, 8, 12, 16, mean of the positive frame scores, an OCR exception scores its
frame 0, frames as RGB uint8 from the in-memory video (`rewards.py:148-153`).

**CPU tests** (`tests/test_train_flow_grpo_wan.py`, area 5's venv, 9 tests, ≈ 15 s): the shared block equals the
evaluator's and has the pinned hash; the deployment `sampler.json` from the real scheduler config survives JSON and
has the GPU probe's `sampler_hash` `sha256:997c2bf0…5ff9e3`; profile settings; file-order prompts; the advantage
tracker equals flow_grpo's `PerPromptStatTracker`; one trained timestep (policy loss, KL) and the LoRA gradient equal
flow_grpo's formula with flow_grpo's own `sde_step_with_logprob` (micro-batch 2 accumulated = flow_grpo's micro-batch
4, relative gradient difference < 1e-5, tiny random transformer, fp32); the real PaddleOCR reward on a rendered
17-frame video (frames 0/4/8 "OPEN LATE" 1.0, 12/16 0, reward 1.0); `main()` end to end on a tiny random base (3
epochs, 2 prompts × 4 videos, 32 × 48 × 5 frames, 4 steps, scripted stub reward exercising a filtered group, the
padding re-inclusion, an OCR frame error and the all-zero rule): run records, `samples.jsonl` derived fields,
`epoch_plan.jsonl`, `train_steps.jsonl`, cut counts and hashes, `cut.json` last, `eval_videos.check_cut` accepts both
cuts, `eval_videos.resolve_sampler` returns the cut's `sampler.json`, `eval_videos.evaluate` loads the LoRA and
generates, a rerun into the same `--out` moves the attempt aside; `main()` for 1 epoch with the real PaddleOCR reward
(random tiny videos read nothing → all-zero rule).

### Deviations from upstream (`train_wan2_1.py` @ `879042cf`)

| # | flow_grpo | This port | Reason |
|---|---|---|---|
| 1 | 33 frames (`config/grpo.py:42`) | 17 frames (latent 5 × 30 × 52, 1,950 video tokens) | decision; ≈ half the cost |
| 2 | umT5 bf16 resident on the GPU, encodes every sampling batch (`:440`, `:652-658`) | every prompt of the run encoded once at start (`""` + 4 per epoch), embeddings trimmed on the CPU, encoder freed (`gc`, `empty_cache`) before the transformer loads | memory: probe 11.3 GB GPU / 21 GB host RSS for the encoder; embeddings bit-identical (evaluator test) |
| 3 | transformer fp32 weights, accelerate bf16 autocast (`:445`, `:383`) | transformer stored bf16 (diffusers keeps `_keep_in_fp32_modules` fp32), `torch.autocast(bf16)`, LoRA fp32 | memory; same matmul precision |
| 4 | training micro-batch = kept / 2 (8), accelerate accumulation over 19 steps | `train_batch` (default 2) videos per forward/backward, loss summed per video / (kept/2 × 1 × 19) | memory; gradient identical (tested) |
| 5 | sampling batch 8 = 2 prompts × 4 shuffled across slots (`DistributedKRepeatSampler`, `:81-124`) | sampling batch `sample_batch` (default 4) = one prompt group, groups contiguous | consumption by row; no algorithmic effect |
| 6 | prompts: `randperm` of the shard per batch on a seeded generator, prompts may repeat across batches and epochs (and two batches of one epoch may share a prompt, merging two groups) | shard rows in file order: epoch e uses rows 4(e − 1) … 4e − 1, no prompt repeats within the run (18,359 rows) | chosen for exact row-level consumption |
| 7 | noise from the global device RNG (`set_seed(42)`) | per-video CPU `torch.Generator` seeded `derive_seed('sample', seed, e, i)` (initial latents and every step) | recorded seeds (MM-29), device independent |
| 8 | padding and training order from `torch.randperm` on the global RNG (`:864`, `:881`) | seeded CPU generators `derive_seed('mask', seed, e)`, `derive_seed('perm', seed, e, inner)` | reproducible, recorded |
| 9 | advantages a float64 tensor (from numpy) | float32 | as area 5; negligible |
| 10 | EMA decay 0.9 every 8 steps, cuts save EMA (`:478`, `:1004`, `:337-346`) | no EMA; cuts hold the raw LoRA; no `--ema` option | decision |
| 11 | the first two epochs sample nothing (`:670-672`, `:731-732`, "group size bug") | every epoch samples and trains | upstream quirk (a DataLoader-worker artifact) |
| 12 | `transformer.module.disable_adapter()` (`:940`; assumes a DDP-wrapped PeftModel) | diffusers `disable_adapters()` / `enable_adapters()` on the single-process transformer | same reference (base without LoRA); single GPU |
| 13 | LoRA by `get_peft_model` (PeftModel), cut = `save_pretrained` → `checkpoint-<step>/lora/adapter_model.safetensors` (`:337-344`) | diffusers `add_adapter(LoraConfig)`, cut = `save_lora_adapter` → `pytorch_lora_weights.safetensors` (diffusers keys `blocks.N.attnK.to_q.lora_A.weight`, header canonicalized) | the evaluator's loader `load_lora_adapter(prefix=None)`; area 5 layout |
| 14 | cut every 60 epochs (`epoch > 0`, at the start of the epoch, `:668-669`), `num_epochs` 100000 stopped by hand | cuts after every `save_every` epochs and after the last; profile epochs | budget; MM-20 atomic cuts |
| 15 | in-run evaluation every 30 epochs (50-step ODE, EMA), wandb, video logging, `imp`/`ml_collections`/`absl`/`accelerate` | none | offline; evaluation is `eval_videos.py`; area 5's stack |
| 16 | paddleocr 2.9.1 `PaddleOCR()` default models (downloaded) | paddleocr 2.10.0 `TextSystem` with area 5's local PP-OCRv4 mobile det + en rec, CPU, `enable_memory_optim` off | offline; area 5's engine A (area 5 README "Changes") |
| 17 | `kl_reward` (0 in the config) computes zeros during sampling | not ported (0) | no effect at 0 |
| 18 | advantages grouped by the tokenized-and-decoded prompt text | grouped by `prompt_id` | same groups (prompts unique per epoch) |
| 19 | TF32 on matmul only | TF32 on matmul and cuDNN (as area 5) | no effect in bf16 |

### Outputs (`/out`) and differences from area 5's schemas (for the `flow_grpo_wan` adapter)

Same files as area 5 (`env-freeze.txt` with the reference line, `env-runtime.json`, `run_args.json`, `samples.jsonl`,
`epoch_plan.jsonl`, `train_steps.jsonl`, `train_log.jsonl`, `memory.jsonl`, `attempts.jsonl`, `checkpoint-<E>/` via
`.checkpoint-<E>.partial/` + rename with `pytorch_lora_weights.safetensors`, `adapter_config.json`, `sampler.json`,
`base_ref.json`, `trainer_state.json`, `cut.json` last; `prepare_out` preemption rule; `samples_consumed` /
`samples_filtered` / `global_step` in `cut.json`). Differences:

| File | Difference from area 5 |
|---|---|
| `run_args.json` | schema `ptc_mm.flow_grpo_wan_run_args`; settings `height`, `width`, `num_frames` (no `resolution`), no `noise_level` / `deploy_noise_level` / `encoder_precision` (one `precision` for transformer and umT5, `bf16`), `num_steps` 20, `deploy_num_steps` 20, `max_sequence_length` 512, `num_batches_per_epoch` 2 is a constant (area 5 derived it as samples_per_epoch / sample_batch), `gradient_accumulation_steps` 1, `sample_batch` a free memory knob (area 5 required it = `images_per_prompt`), `images_per_prompt` (key kept) = videos per prompt 4, `global_std` false, `reward` `"video_ocr"` (area 5 `"ocr"`), new `prompt_order` `"file"`, `ema` always false; `reward` object adds `models` (det + rec revisions) and `frames`; `seed_derivation` has no `'prompts'` seed (file order); `memory_at_start` adds GPU encode peaks, `rss_now_mb_*`, `lora_params` |
| `env-runtime.json` | adds `uv_lock` (path): area 3 has no lock, area 5's `uv.lock` is used and hashed |
| `samples.jsonl` | no `ocr_text`; new `frame_ocr` `[{frame, text, acc, error}]` (frames 0, 4, …), `nonfinite`; `ocr_acc` = `reward` = mean of the positive `frame_ocr[].acc` (0 if none); `ocr_error` = `"frame <f>: <message>"` of the first failed frame or null, and **an error does not force `ocr_acc` 0** (other frames still count; area 5: error ⇒ acc 0, text null); `sample_id` domain `"flow-grpo-wan"`; `prompt_id` of sample i in epoch e = shard row 4(e − 1) + i // 4 (re-derivable) |
| `epoch_plan.jsonl` | new `all_zero_advantage` (bool). When true: every group has equal rewards, `kept` = all 16, `filtered` = `reincluded` = [], two steps (area 5's rule "a group with equal rewards is never plainly kept" does not hold in that epoch). Padding to a multiple of 2, so `reincluded` has at most 1 id |
| `train_steps.jsonl` | unchanged; 2 steps per epoch with kept / 2 ids each |
| `train_log.jsonl` | no `scale` (no gradient scaler in bf16) |
| `memory.jsonl` | `peak_allocated_gb_sampling` and `peak_allocated_gb_training` instead of `peak_allocated_gb`; adds `all_zero_advantage`, `nonfinite`, `rss_now_mb` |
| cut | `cut.json` schema `ptc_mm.flow_grpo_wan_cut`, `ema` always false; `adapter_config.json` `loader` `WanTransformer3DModel.load_lora_adapter(dir, prefix=None)`, `weights` `raw`, `target_modules` sorted flow_grpo Wan list (the evaluator requires exactly that list); `sampler.json` = the evaluator's Wan document (`pipeline`, `timesteps`, `step_index`, `deterministic`, `height`, `width`, `num_frames`, `cfg`, `prompt_embedding`, `decode`, `autocast` bfloat16; no `noise_level`, `resolution`); `base_ref.json` from `models/MODELS.json` key `wan2.1-t2v-1.3b` (base directory `models/wan2.1-t2v-1.3b/`, not the `models/Wan2.1-T2V-1.3B-Diffusers/` this README proposed earlier) |
| argv | options `--profile {smoke,full}`, `--out`, `--base`, `--prompts`, `--seed` (default 5000), `--device`, `--epochs`, **`--save-every`** (new), `--no-verify-base`, `--train-batch`, **`--sample-batch`** (new), `--no-gradient-checkpointing`, `--beta`; no `--ema` |

**Seeds (MM-29).** As area 5 except the prompts: epoch e's prompts are shard rows 4(e − 1) … 4e − 1 in file order;
noise `derive_seed('sample', seed, e, i)`; padding `derive_seed('mask', seed, e)`; order `derive_seed('perm', seed, e,
inner)`; LoRA init `derive_seed('torch', seed)` on the torch global RNG; `sample_id` = SHA-256 of canonical
`["flow-grpo-wan", seed, attempt, epoch, sample_index, prompt_id, noise_seed, member]`.

### Profiles, memory and time (estimates, to be replaced by the smoke's `memory.jsonl`)

| Profile | Epochs | Videos | Optimizer steps | Cuts |
|---|---|---|---|---|
| `smoke` | 2 | 32 | 4 | `checkpoint-2` |
| `full` | `FULL_EPOCHS` = 40 (module constant; `--epochs` overrides, set from the smoke) | 640 | 80 | every 10 epochs (`--save-every`) and the last |

- **GPU memory, training phase (estimate).** Resident: transformer bf16 2.8 GB, VAE fp32 0.5 GB, LoRA + grads +
  Adam 0.4 GB, stored chains 16 × 21 latents × 0.5 MB = 0.17 GB. Sampling batch 4: the probe measured 5.3 GB peak
  (batch 4, ODE, decode one by one) → ≈ 6 GB with the LoRA and the optimizer state. Training micro-batch b (CFG = 2
  forwards per video, gradient checkpointing per block): checkpointed block inputs 30 × 1,950 × 1,536 × 2 B ≈ 0.18 GB
  per forward → 0.36 GB per video, plus one block's recomputed activations (FFN 8,960 wide, fp32 norms; SDPA, no
  N² buffer) ≈ 0.3 GB per forward → **≈ 1 GB per video**; b = 2 → **≈ 6-8 GB peak**, b = 4 ≈ 8-10 GB, b = 8
  (flow_grpo's) ≈ 12-16 GB. Without checkpointing ≈ 9 GB activations per forward: not usable. **Default
  `train_batch` 2** (safe; area 5 precedent); if the smoke's `peak_allocated_gb_training` is below ≈ 10 GB, the
  full run may pass `--train-batch 4` (same gradient, fewer kernel launches). Encode phase (measured by the probe):
  11.3 GB GPU, 21 GB host RSS peak, freed before the transformer loads.
- **Time (estimate).** Sampling ≈ 16 × 3.2 s (probe: 3.2 s per video at batch 4 with decode) ≈ 50 s per epoch, OCR
  overlapped (probe 0.23 s per video). Training per video and trained step ≈ 10 forward equivalents (2 with grad, 2
  recomputed, ≈ 4 backward, 2 reference) × ≈ 0.07-0.08 s ≈ 0.7-0.8 s → 16 × 19 × 0.75 ≈ 210-245 s. **≈ 4.5-5 min
  per epoch** (more than the 2.2-3.2 min estimated above). Smoke ≈ 10 min of epochs + start (29 GB base re-hash ≈ 1-3
  min, encode ≈ 2.6 min measured, load ≈ 1 min) ≈ **0.25-0.3 GPU-h**. Full at 40 epochs ≈ 3-3.3 h; set
  `FULL_EPOCHS` / `--epochs` from the smoke.

### Smoke launch (area 5's runtime image; a GPU job)

```
# <python-dir>: the interpreter install the venv links to; <sd35-ocr-runtime>, <runtime>: see the top of this file
mkdir -p <output-root>/area3-video/smoke-1          # new, empty, owned by the launching user
docker run --rm --network none --user 1000:1000 --gpus all --cap-drop ALL \
  --security-opt no-new-privileges --shm-size=8g --tmpfs /tmp:rw,exec,nosuid,nodev,size=8g \
  -v <python-dir>:<python-dir>:ro \
  -v <sd35-ocr-runtime>:<sd35-ocr-runtime>:ro \
  -v <runtime>:<runtime>:ro \
  -v <output-root>/area3-video/smoke-1:/out:rw \
  -e HF_HUB_OFFLINE=1 -e HOME=/tmp/home -e XDG_CACHE_HOME=/tmp/cache -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONUNBUFFERED=1 -e TOKENIZERS_PARALLELISM=false -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  gatedtrain-a5-runtime:1 \
  <sd35-ocr-runtime>/.venv/bin/python \
  <runtime>/train_flow_grpo_wan.py --profile smoke --out /out --seed 5000
```

`gatedtrain-a5-runtime:1` must resolve to image id `sha256:52070d08298e4ed72cfe49bf94cbc2061623cb01e972e1ab84a6e135b4b057a1`
(check with `docker image inspect --format '{{.Id}}' gatedtrain-a5-runtime:1`, or pass that id instead of the tag). No
`HF_TOKEN`. The area 5 mount holds the venv, the lock and the OCR models; the area 3 mount holds the scripts, the
base model and the shard. Then evaluate `checkpoint-2` with `eval_videos.py --checkpoint /inputs/cut … --limit 40`
(cut mounted read-only at `/inputs/cut`) next to the 40-item base arm. Read from the smoke: `memory.jsonl`
(`seconds_epoch`, `peak_allocated_gb_sampling`, `peak_allocated_gb_training`, `max_rss_mb`), `run_args.json`
`memory_at_start` (`lora_params` = 23,592,960 expected, encode peaks), `epoch_plan.jsonl` (`all_zero_advantage`
share: the probe's reward was 0 on 45 % of base videos), `train_log.jsonl` (finite losses, `clipfrac`, `approx_kl`).

**Unverified (no GPU run yet):** memory and time above; bf16 numerics of the training forward (ratio ≈ 1 at the
first step is expected because sampling and training use the same bf16 weights, LoRA B = 0); gradient checkpointing
on the real 30-block model under autocast; the LoRA parameter count on the real model; host RSS after the encoder is
freed (`ru_maxrss` keeps the 21 GB peak; `rss_now_mb` shows the current value); diffusers'
`save_lora_adapter` logs a harmless "should be kept in float32" warning (it calls `self.to(dtype=None)`, no cast;
diffusers `loaders/peft.py:428`); whether upstream's `transformer.module` raises with one process (not needed here).

## Profiles and cost (estimates)

| Run | Content | GPU-h (estimate) |
|---|---|---|
| render probe | 20 base videos, plain `WanPipeline`, CPU scoring | 0.1 |
| smoke | 3 epochs × 16 videos (6 optimizer steps), cuts 1-3, plus 40-item base and smoke-cut evaluations | 0.4-0.5 |
| full | E epochs × 16 videos, E from the smoke so that training ≈ 2 h (≈ 40 epochs, 640 videos, 80 optimizer steps), cuts every 10 epochs | ≈ 2.0 |
| evaluation | base + candidate, 400 items each | 0.8-1.5 |
| **area total** | | **≈ 3.3-4.1** (spec row 3: 2.5) |

Host RAM: the umT5 load is the peak (≤ 14 GiB estimate); no concurrent GPU or large-RAM job during the runs.

## What the gatedtrain adapter needs (written before the adapter existed)

Spec row 3 says "`flow_grpo` (not built), same id, profile `wan`". area 5's adapter `flow_grpo_sd35.py` is
SD3.5-specific at module level: `BASE_REQUIRED` and `BASE_WEIGHT_DIRS` name three text encoders (`:66-70`),
`SAMPLER_CODE_SHA256` (`:76`), `DEFAULT_BASE_REL` (`:79`), the scripts' option tables (`:83-94`), the `sample_id`
domain `"flow-grpo-sd35"` (`:308`), and the registry row is keyed by adapter name with area-5 `input_roots`,
`scan_skip` and `interpreter` (`registry.py:346-375`); the evaluator rows' `sampler_keys` lack frames
(`registry.py:385-399`). A "profile" would mean turning those globals into per-profile data. **Proposal:** a
sibling adapter `flow_grpo_wan` that subclasses `FlowGrpoSd35Adapter` (as that one subclasses `DdpoSd15Adapter`,
`flow_grpo_sd35.py:133`), same `uses` id `rl.flow_grpo@0.1`, overriding:

1. **Cuts (MM-20):** as area 5 (`checkpoint-<E>/`, `cut.json` last, `samples_filtered`, `ema` false).
2. **Identity (MM-17, MM-18):** components LoRA weights, `adapter_config.json`, `sampler.json`, `base_ref.json`;
   base hash = manifest of `models/Wan2.1-T2V-1.3B-Diffusers/` excluding `.cache/`; `BASE_REQUIRED` =
   `model_index.json`, `scheduler/scheduler_config.json`, `transformer/config.json`, `vae/config.json`,
   `text_encoder/config.json`, `tokenizer/tokenizer_config.json`; weight dirs `transformer`, `vae`, `text_encoder`.
   **No umT5 embedding manifest** (spec row 3 lists one): embeddings are computed in-run from the base, as area 5
   dropped its T5 manifest (`MULTIMODAL-SPEC.md:181`).
3. **Base binding (MM-21):** `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` @ `0fad780a…`, files equal `MODELS.json`.
4. **Sampler:** `sampler_code_sha256` of a shared block byte-identical in trainer and evaluator; sampler keys add
   `height`, `width`, `num_frames`, `flow_shift`.
5. **Consumption (MM-24, MM-28):** `row`, `observed`, `samples.jsonl` ⋈ `epoch_plan.jsonl` ⋈ `train_steps.jsonl`;
   `consumed` and `filtered` (`zero_advantage`) as area 5, with epoch size 16, groups of 4, kept a multiple of 2
   (`num_batches_per_epoch`); `reward` = mean of the positive frame scores of `frame_ocr` (recomputed and checked);
   `sample_id` domain `"flow-grpo-wan"`. Fallback none (as area 5).
6. **Exact overlap (MM-11):** item prompt and normalized target against the shard and the sampled prompts (0 by
   construction).
7. **Arms and registry rows:** two evaluator rows (deciding `ocr_b_acc` T3 construct-aligned, supporting
   `ocr_a_acc` T0), `item_payload ("prompt", "target")`, error class `scorer_error`, `score_outcomes
   ("nonfinite_video",)`; trainer row with pins `diffusers, peft, torch, transformers, paddleocr, paddlepaddle`,
   `input_roots` the Python directory, `<runtime>`, `<output-root>/area3-video`;
   `scan_skip` the area's `.venv`; `env_names` and `env_paths` as area 5; `mount_binding_problems` (the gap areas 6
   and 7 hit).
8. **argv rules:** as area 5 (no `--no-verify-base`, no `--ema`, profiles smoke/full, evaluator `--base`/`--items`
   defaults only, `--limit` refused under full).

## Launch request facts (proposal, as area 5)

Mounts: the Python directory ro; `<runtime>` ro (scripts, lock, venv, `models/`,
`ocr-models/`, `prompts/`, `items/`, `MODELS.json`); a cut at `/inputs/cut` ro for candidate evaluations;
`<output-root>/area3-video/<run>` at `/out` rw; tmpfs `/tmp`. Image `gatedtrain-a5-runtime:1`, network
`none`, `gpus: true`, user `1000:1000`, `--cap-drop ALL`, `--security-opt no-new-privileges`, `--shm-size=8g`.
Environment: `HF_HUB_OFFLINE=1`, `HOME=/tmp/home`, `XDG_CACHE_HOME=/tmp/cache`, `DOCTR_CACHE_DIR=/tmp/cache/doctr`,
`PYTHONDONTWRITEBYTECODE=1`, `PYTHONUNBUFFERED=1`, `TOKENIZERS_PARALLELISM=false`; never `HF_TOKEN`.

## Changes to spec row 3 that followed from the decisions

Trainer `train_flow_grpo_wan.py` (port) instead of `train_wan2_1.py`; adapter `flow_grpo_wan` (sibling) instead of a
profile; cuts `checkpoint-<epoch>/` instead of `checkpoint-<step>/lora/`; no umT5 embedding manifest; evaluator ids
and tier (T3 deciding + T0 supporting) instead of `t2v.flow_grpo_items@0.1` T0; N 400 instead of 50; frames 17
(flow_grpo uses 33); GPU-h ≈ 3.5-4 instead of 2.5.

## Decisions

Taken on 2026-11-14 before the trainer and evaluator were built (option (a), the recommended one, in each case):

1. **Base download:** `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` @ `0fad780a…` (Apache-2.0, ungated), ≈ 28.9 GB without
   `assets/`/`examples/`: one repository, safetensors only, loads with `WanPipeline`. Not taken: (b) the original
   repository, 17.6 GB, with a bf16 pickle encoder and key conversion; (c) deferring the area.
2. **Trainer:** the minimal port `train_flow_grpo_wan.py` on area 5's stack (the same reasons as area 5). Not taken:
   (b) flow_grpo's own `train_wan2_1.py` in a legacy stack with patches for 24 GB.
3. **Deciding metric and tier:** docTR `ocr_b_acc`, substring formula, mean over all 17 frames, T3 construct-aligned,
   blocking; the PaddleOCR reward as T0 supporting (area 5 precedent). Not taken: (b) the reward as T0 as spec row 3
   said (cannot gate); (c) the compressibility pair (repeats area 3).
4. **Resolution, frames and sampler:** 240 × 416 × 17, 20 sampling steps, 19 trained, evaluation Euler ODE 20 steps
   CFG 4.5 (spec row 3, ≈ half the cost of 33 frames). Not taken: (b) flow_grpo's 33 frames (≈ 2 × training cost);
   (c) flow_grpo's 50-step evaluation (≈ 2.5 × evaluation cost).
5. **Further choices:** render probe before the port; short-target restriction or 480 × 832 × 9 as pre-registered
   fallbacks if the probe or the smoke fails MM-38; EMA off; N 400, K 1, 40 smoke items; sigma rule as areas 5-7;
   epochs from the smoke; sibling adapter `flow_grpo_wan`; directory name `area3-video`.

## Attempt 2 (amendment)

**Why.** Attempt 1 (run `run-20261115T060531Z-718ea124`) failed its gate: ocr_b_acc 0.1554 (base) vs
0.1469 (checkpoint-60), delta -0.0086, 95 % interval [-0.020, 0.002]. Cause: too little training signal (60 epochs x
4 prompts x 4 videos = 120 small optimizer steps, approx_kl about 0). Decided 2026-11-15: retry as an amendment
with **short target texts only** (the pre-registered first MM-38 fallback, "Decisions" item 5) and **a larger batch** (8 prompts x 4 videos = 32 samples per epoch, 100 epochs); training and held-out
items both restricted to short targets. Nothing of attempt 1 changes: `items/`, `prompts/`, `gate/`,
`train_flow_grpo_wan.py` (pinned by attempt 1's launch records, which ext-collect re-hashes, EXT-83) and
`eval_videos.py` were byte-identical, so attempt 1 stayed collectable. (The published
copies of the scripts differ from these only in how they locate area 5's runtime; the versions that ran are not
published, see `../RESULT.md` "Published code".)

**Data (`make_short_split.py`).** Same sources and rules as area 5's split
(`<sd35-ocr-runtime>/data-source/ocr/{train,test}.txt` = flow_grpo `dataset/ocr` @ `879042cf`; target = the first double-quoted
span; numpy permutation `SeedSequence([0x5d35a5, 20261113])`), plus one filter: **short = the target has at most 2
whitespace-separated words** (`len(target.split()) <= 2`).

| Rule (alternatives, `--dry-run`) | held-out items (of 1,017) | short unique train prompts | training shard |
|---|---|---|---|
| words <= 1 | 44 | 884 | 838 |
| **words <= 2 (chosen)** | **288** | **5,677** | **5,024** |
| words <= 3 | 730 | 14,220 | 12,285 |
| chars <= 8 | 55 | 1,188 | 1,021 |
| chars <= 12 | 203 | 3,997 | 3,444 |
| chars <= 15 | 407 | 8,019 | 6,695 |

"At most 2 words" gives 288 >= 250 items, so no alternative was needed.
- `items-short/items.jsonl`: 288 items, every short-target prompt of the 1,017 usable test prompts (not only the
  first attempt's 400; 106 of them are among those 400, with identical rows), in the order of area 5's permutation
  (not sorted: `--limit 40` takes a random sample). Same row schema as `items/items.jsonl` (`item` =
  `ocr-test-<source line>`, `prompt`, `target`, `source_line`). Its manifest hash, the registration's item set hash, was
  computed with the original `item_set_meta.json` (not published; `../RESULT.md` "Redacted copies").
- `prompts-short/train.jsonl`: 5,024 rows = the 5,677 short-target unique train prompts (first occurrence, file
  order) minus 653 whose normalized target (spaces removed, lower case) equals an item's target (no item prompt occurs
  in train.txt). Exact prompt overlap 0, same-target overlap 0. Its manifest hash was computed with the original
  `prompts_meta.json` (not published).
- `item_set_meta.json` / `prompts_meta.json`: the first attempt's schema with the rule (`short_rule`, `max_words`) and
  the counts in `split`.

**Trainer `train_flow_grpo_wan_v2.py`**: a copy of `train_flow_grpo_wan.py` that differs only in
`PROFILES`, `FULL_EPOCHS`, `DEFAULT_PROMPTS` (`prompts-short/`) and docstrings (checked as syntax trees by
`tests/test_train_flow_grpo_wan_v2.py`); the SHARED SAMPLER block is identical (`dad91e41...` as run, now `41606b0b...`).

| Profile | Epochs | P x G per epoch | Optimizer steps per epoch | Train micro-batch | Sampling batch | Cuts |
|---|---|---|---|---|---|---|
| `smoke2` | 2 | 8 x 4 = 32 | 2 (16 samples each; `num_batches_per_epoch` 2, accumulation 1) | 2 | 4 | 2 |
| `full2` | 100 | 8 x 4 = 32 | 2 | 2 | 4 | every 25 (25, 50, 75, 100) |

The full run reads 800 of the 5,024 shard rows (no wrap). Every other setting is attempt 1's (lr 1e-4, beta 0.004,
clip 1e-3, 20 SDE steps, 19 trained, CFG 4.5, 240 x 416 x 17, EMA off). Memory: the micro-batch stays 2; the only new
memory is the epoch's latent chains (32 x 21 x 16 x 5 x 30 x 52 float32, about 0.34 GB on the GPU, was 0.17 GB).

**Evaluator.** `eval_videos.py` unchanged; it already takes `--items`, so attempt 2 passes
`--items <runtime>/items-short`. Same evaluator ids and code hash; the arm's item set is
bound by `env-runtime.json` `items_sha256` (gatedtrain checks it against the registered items.jsonl). For
`gatedtrain ext-arm-fill`, pass `--items <runtime>/items-short`.

**Cost (estimates from attempt 1's measured rates: 186-223 s per 16-sample epoch, about 337 s start-up for encode and
load, evaluation about 4.5 s per video plus about 3 min of encoding):**

| Run | Content | GPU-h |
|---|---|---|
| smoke2 training | 2 epochs x 32 samples (4 steps), cut 2 | 0.30-0.35 |
| smoke evaluations | first 40 short items, base and cut 2 (2 jobs x (40 x 4.5 s + 3 min)) | about 0.2 |
| full2 training | 100 epochs x 32 samples (200 steps, 3,200 videos), 372-446 s per epoch | 10.4-12.5 |
| gate evaluations | 288 items, base and last cut (2 jobs x (288 x 4.5 s + 3 min)) | about 0.8 |
| **attempt 2 total** | | **about 11.7-13.9** |

**Smoke commands** (through `gatedtrain ext-launch --adapter flow_grpo_wan --profile smoke`; image, mounts and environment
names as attempt 1's requests: `<python-dir>`, `<sd35-ocr-runtime>`, `<runtime>` read-only at their
own paths, the new empty output directory at `/out`; a cut evaluation adds the cut read-only at `/inputs/cut`; the
evaluations add `DOCTR_CACHE_DIR=/tmp/cache/doctr`):

```
# training, /out = out/area3-video/smoke2-1
<sd35-ocr-runtime>/.venv/bin/python <runtime>/train_flow_grpo_wan_v2.py \
  --profile smoke2 --out /out --seed 5000
# base, first 40 short items, /out = out/area3-video/eval-smoke2-base-1
<sd35-ocr-runtime>/.venv/bin/python <runtime>/eval_videos.py \
  --base-only --items <runtime>/items-short --out /out --eval-seeds 11 --limit 40
# smoke cut, /inputs/cut = out/area3-video/smoke2-1/checkpoint-2, /out = out/area3-video/eval-smoke2-cut-1
<sd35-ocr-runtime>/.venv/bin/python <runtime>/eval_videos.py \
  --checkpoint /inputs/cut --items <runtime>/items-short --out /out --eval-seeds 11 --limit 40
```

Full run: the same training argv with `--profile full2` under `ext-launch --profile full`; gate evaluations as above
without `--limit` (288 items). A new spec (`prompts-short/`, `items-short/`) and a registration with `amends` go with
it (`../results/2026-11-15/gate/`).

**Tests (CPU, area 5's venv):** `tests/test_make_short_split.py` (5: the files are the derivation, rule, order,
disjointness, metas, attempt-1 files untouched), `tests/test_train_flow_grpo_wan_v2.py` (5: attempt-1 script hash,
v2 = v1 except profiles and prompts, profiles, the short shard, one epoch of the smoke2 batch shape end to end on a
tiny base); the attempt-1 suites are unchanged.

## Result (2026-11-15)

- **Attempt 1** (`gate/registration.yaml`, 400 items, full-1 checkpoint-60): FAIL, ocr_b_acc 0.1554 -> 0.1469
  (delta -0.0086, CI [-0.020, 0.002]); too little training signal (120 steps, approx_kl near 0).
- **Attempt 2** (`gate/registration-v2.yaml`, amends attempt 1; `items-short/` 288 items, `prompts-short/`; full2-1:
  8 prompts x 4 videos, 100 epochs, 200 optimizer steps, 11.22 GPU-h): **PASS**, ocr_b_acc 0.1456 -> 0.1774 (delta
  +0.0318, 95% cluster-bootstrap CI [0.0152, 0.0494], p <= 7.3e-05). Run `run-20261115T231401Z-bd967291`;
  consumption precision row (2,756 consumed, 444 filtered); impact maps cuts 25/50/75/100 to steps 50/100/150/200.
