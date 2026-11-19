"""Flow-GRPO Wan2.1-T2V-1.3B + LoRA adapter (area "video generation": the Flow-GRPO port `train_flow_grpo_wan.py` of
flow_grpo's `scripts/train_wan2_1.py` on area 5's stack, video OCR; MULTIMODAL-SPEC.md section 6 rows 3 and 3s and the
row 3 deviations, runtime/README.md "Decisions"; tests/README.md section 11.25, EXT-214 to
EXT-228, and attempt 2, EXT-229 to EXT-232).

MM-12: a pure-Python class that imports no framework, torch, OCR engine or pickle loader. It reads JSON, JSONL, the text
`env-freeze.txt` and safetensors headers, re-derives the per-frame OCR accuracies with its own Levenshtein distance, and
hashes everything else.

Run layout, read from the runtime's code `runtime/train_flow_grpo_wan.py` (sha256
`da064300...`, read 2026-11-14) and its README "Trainer" (table "Outputs (`/out`) and differences from area 5's
schemas"), verified against the GPU smoke `probe-train-1/` (2 epochs, cut 2). The layout is area 5's
(`FlowGrpoAdapter`, the flow_grpo family base it extends), with these differences:

- `run_args.json`: schema `ptc_mm.flow_grpo_wan_run_args`; settings `reward` `"video_ocr"`, `prompt_order` `"file"`,
  `prompts_per_epoch` P, `images_per_prompt` G (videos per prompt), `num_frames`, `height`, `width`;
  `num_batches_per_epoch` 2 and `gradient_accumulation_steps` 1 (two optimizer steps per epoch);
- `samples.jsonl`: no `ocr_text`; `frame_ocr` `[{frame, text, acc, error}]` for frames 0, 4, 8, ... (flow_grpo
  `video[::4]`); `ocr_acc` = `reward` = the mean of the positive frame accuracies (0 if none, flow_grpo ocr.py:117-123);
  `ocr_error` the first failed frame's `"frame <f>: <message>"` or null, and an error does not force `ocr_acc` 0 (the
  failed frame scores 0, the others still count); `nonfinite`; `sample_id` domain `"flow-grpo-wan"`; the prompt of sample
  i of epoch e is shard row ((e - 1) P + i // G) mod n in file order (no random draw);
- `epoch_plan.jsonl`: `all_zero_advantage` (bool). When every advantage of the epoch is 0 (every prompt group has equal
  rewards), flow_grpo adds 1e-6 to all of them and trains on every sample (train_wan2_1.py:853-858): `kept` = all
  samples, `filtered` = `reincluded` = [], two steps; otherwise area 5's zero-advantage filter and padding;
- cuts `checkpoint-<epoch>/` as area 5 (`cut.json` schema `ptc_mm.flow_grpo_wan_cut`, `ema` false; `adapter_config.json`
  `target_modules` flow_grpo's Wan list, which the evaluator requires).

Consumption (MM-24, MM-28): as area 5, one `consumed` record per (kept sample, applying step) and one `filtered` record
per zero-advantage sample the filter dropped; an all-zero epoch has no filtered record. Identity (MM-17, MM-18): as area
5; the base `models/wan2.1-t2v-1.3b/` (21 files, Wan-AI/Wan2.1-T2V-1.3B-Diffusers @ 0fad780a...) has no `.cache/`, its
model hash is its manifest hash.

Attempt 2 (amendment, decided 2026-11-15; area README "Attempt 2 (amendment)"; EXT-229 to EXT-232): short target
texts only and 8 prompts x 4 videos per epoch. `train_flow_grpo_wan.py` stayed byte-identical (the first attempt's launch
records pin its SHA-256, EXT-83, so a changed file would make that run uncollectable; the published copy differs
only in comments, names and how it locates the OCR runtime, RESULT.md "Published code"); the amendment's trainer is the
copy `train_flow_grpo_wan_v2.py` (profiles `smoke2`, `full2`; default `--prompts` `prompts-short/`), pinned by path like
the first. `eval_videos.py` is unchanged (same code hash, same evaluator ids) and evaluates `items-short/` through its
`--items` option. Each training script binds its profiles and its prompt directory: launch profile smoke/full means
`--profile smoke/full` for the first script and `smoke2/full2` for the second, `--prompts` only the script's default, and
at collect `run_args.json` `profile` and `train_file` must be one script's pair (`prompts/` with smoke/full,
`prompts-short/` with smoke2/full2; the launch profile stands for the launched script's profile in area 5's profile
rule). The spec's data and items and the registration's item set hash keep the two attempts apart (generic checks), and
an arm's evaluation `env-runtime.json` (`items_sha256`, the items.jsonl eval_videos.py read) must be the registered
item set's (`arm_problems`), so an arm of the other attempt's items refuses even with the registered hash claimed.

Launch: all three uses (training, base evaluation, cut evaluation) run area 5's runtime image `gatedtrain-a5-runtime:1` and
area 5's venv (the OCR runtime's `.venv/bin/python` and `uv.lock`, both pinned; `rows.OCR_RUNTIME`, the SD3.5 OCR
recipe's runtime directory) with the scripts of this recipe's runtime directory; the scripts are pinned by their container paths (the interpreter's directory is not the scripts', so area
5's rule "the script beside the interpreter" does not apply) and the mount set is fixed (`launch_problems`).
"""

from __future__ import annotations

import posixpath
from pathlib import Path
from typing import Any

from gatedtrain.canonical import file_sha256
from gatedtrain.errors import Problem
from gatedtrain.ext.registry import TRAINER_ROWS
from gatedtrain.ext.sdk import is_num as _num, read_cut_marker, read_json as _read_json, read_run_args, resolve_option
from gatedtrain.ext.sdk import RUNTIME_FILE
from gatedtrain.families.flow_grpo import CUT_COMPLETENESS, PLAN, SAMPLES, FlowGrpoAdapter
from gatedtrain_recipe_flow_grpo_wan_video.rows import OCR_RUNTIME, PYTHON_DIR, RUNTIME

NAME = "flow_grpo_wan"
ROW = TRAINER_ROWS[NAME]
AREA3V = RUNTIME                                             # the runtime directory, mounted read-only at its own path
AREA5 = OCR_RUNTIME                                          # the SD3.5 OCR runtime: venv, lock, OCR models (ro, own path)
# PYTHON_DIR (rows): the uv-managed CPython install directory the OCR runtime's venv links to (read-only)
TRAIN_SCRIPT = "train_flow_grpo_wan.py"                     # attempt 1 (byte-identical since its launch)
TRAIN_SCRIPT_V2 = "train_flow_grpo_wan_v2.py"                # attempt 2 (amendment): only profiles and prompts differ
TRAIN_SCRIPTS = (TRAIN_SCRIPT, TRAIN_SCRIPT_V2)
EVAL_SCRIPT = "eval_videos.py"
SCRIPT_PATHS = {name: f"{AREA3V}/{name}" for name in (*TRAIN_SCRIPTS, EVAL_SCRIPT)}
BASE_REL = "models/wan2.1-t2v-1.3b"                          # both scripts: DEFAULT_BASE = HERE / models / BASE_NAME
ITEMS_REL = "items"                                          # eval_videos.py --items default HERE / items
ITEMS_SHORT_REL = "items-short"                              # attempt 2's item set (make_short_split.py)
ITEMS_RELS = (ITEMS_REL, ITEMS_SHORT_REL)
# Each training script's default `--prompts` (DEFAULT_PROMPTS) and its profiles by launch profile (PROFILES).
PROMPTS_REL = {TRAIN_SCRIPT: "prompts", TRAIN_SCRIPT_V2: "prompts-short"}
SCRIPT_PROFILES = {TRAIN_SCRIPT: {"smoke": "smoke", "full": "full"},
                   TRAIN_SCRIPT_V2: {"smoke": "smoke2", "full": "full2"}}
EVAL_CUT_CONTAINER = "/inputs/cut"                           # a cut evaluation mounts the cut read-only here
# The runtime image (area 5's `gatedtrain-a5-runtime:1`, README "Smoke launch") and area 5's venv lock (109 packages), pinned.
# The published runs recorded area 5's lock before python-Levenshtein was removed and its project renamed
# (sha256:8243b02f..., 110 packages; RESULT.md); only this pin is accepted.
IMAGE = "sha256:52070d08298e4ed72cfe49bf94cbc2061623cb01e972e1ab84a6e135b4b057a1"
VENV_LOCK_SHA256 = "sha256:502c7ce70e0950cf36cbd108cd9566e0193cbcabe8914c8d0c5bbd231caecda1"
# The SHARED SAMPLER block of train_flow_grpo_wan.py and eval_videos.py (byte-identical; README "Evaluator", "Trainer";
# every cut's and run's `sampler_code_sha256`). The probes and the published runs recorded the block before its
# edit-distance import moved from python-Levenshtein to rapidfuzz (sha256:dad91e41...; RESULT.md); only this pin is
# accepted.
SAMPLER_CODE_SHA256 = "sha256:41606b0b53ad637da4614aaa38fd52b5e9bf0b2147da7ed71c3eaad92dcb1f17"
# The Wan2.1 Diffusers directory (README "Model facts"; `models/MODELS.json`: 21 files, no `.cache/`).
BASE_REQUIRED = ("model_index.json", "scheduler/scheduler_config.json", "transformer/config.json", "vae/config.json",
                 "text_encoder/config.json", "tokenizer/tokenizer_config.json")
BASE_WEIGHT_DIRS = ("transformer", "vae", "text_encoder")
# flow_grpo scripts/train_wan2_1.py:451-460 (train_flow_grpo_wan.py LORA, eval_videos.py WAN_LORA_TARGETS).
WAN_LORA_TARGETS = ("add_k_proj", "add_q_proj", "add_v_proj", "to_add_out", "to_k", "to_out.0", "to_q", "to_v")
FRAME_INTERVAL = 4                                           # flow_grpo ocr.py:77 (`video[::4]`)
RUN_ARGS_SCHEMA = "ptc_mm.flow_grpo_wan_run_args"
CUT_SCHEMA = "ptc_mm.flow_grpo_wan_cut"
REWARD = "video_ocr"
PROFILES = ("smoke", "full")                                 # launch profiles = train_flow_grpo_wan.py's
# argparse options of the two scripts (train_flow_grpo_wan.py parse_args :712-729, eval_videos.py parse_args :887-915):
# option -> takes a value. argparse accepts a unique prefix of an option (allow_abbrev), resolved the same way here.
TRAIN_OPTIONS = {"--profile": True, "--out": True, "--base": True, "--prompts": True, "--seed": True, "--device": True,
                 "--epochs": True, "--save-every": True, "--no-verify-base": False, "--train-batch": True,
                 "--sample-batch": True, "--no-gradient-checkpointing": False, "--beta": True}
EVAL_OPTIONS = {"--copy-items": False, "--checkpoint": True, "--base-only": False, "--base": True, "--items": True,
                "--out": True, "--eval-seeds": True, "--model-hash": True, "--base-model-hash": True,
                "--item-set-hash": True, "--limit": True, "--batch-size": True, "--device": True, "--dtype": True,
                "--ocr-cpu-threads": True, "--num-steps": True, "--guidance-scale": True, "--height": True,
                "--width": True, "--num-frames": True, "--max-sequence-length": True}
OPTIONS = {TRAIN_SCRIPT: TRAIN_OPTIONS, TRAIN_SCRIPT_V2: TRAIN_OPTIONS, EVAL_SCRIPT: EVAL_OPTIONS}
# The read-only mounts of every launch (README "Smoke launch"): the interpreter tree, area 5 (venv, lock, OCR models) and
# the area 3 runtime, each at its own path; the output at /out.
MOUNTS = tuple((p, "ro") for p in (PYTHON_DIR, AREA5, AREA3V) if p) + (("/out", "rw"),)

CHECKPOINT_COMPLETENESS = {
    "library": ("the Flow-GRPO Wan port train_flow_grpo_wan.py on diffusers + peft (cuts by its save_cut, cut.json "
                "written last)"),
    **CUT_COMPLETENESS,
    "sources": ["runtime/train_flow_grpo_wan.py:674-705 save_cut (cut.json last, os.replace)",
                "runtime/train_flow_grpo_wan.py:123-126 CUT_MARKER, CUT_FILES, DEPLOYABLE",
                "runtime/README.md \"Outputs (/out) and differences from area 5's schemas\", cut row",
                "probe-train-1/checkpoint-2 (GPU smoke, 2026-11-14; tests/fixtures/flow_grpo_wan/probe-train-1)"],
    "verified": True,
}
SEED_DERIVATION = ("train_flow_grpo_wan.py derive_seed(*parts) = first 8 bytes of SHA-256(':'.join(parts)) >> 1: epoch "
                   "e's prompts are shard rows (e - 1) P .. e P - 1 in file order (no random draw), sample i's noise "
                   "from a CPU torch.Generator seeded with derive_seed('sample', seed, e, i) (initial latents and every "
                   "SDE step; recorded per sample in samples.jsonl and in the ledger's seed_inputs), the zero-advantage "
                   "padding from derive_seed('mask', seed, e), the training order from derive_seed('perm', seed, e, "
                   "inner), the LoRA init from derive_seed('torch', seed). The training seed is run_args.json `seed`, "
                   "which must equal the launch record's training_seed and every cut's cut.json `seed`; 05 section 14 "
                   "H(run_seed, step, data_ref, sample_index) is not claimed.")


def levenshtein(a: str, b: str) -> int:
    """Unit-cost edit distance (the `Levenshtein.distance` flow_grpo's reward calls), two rows of the DP table."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def video_frame_accuracy(recognized: str, target: str) -> float:
    """train_flow_grpo_wan.py `video_frame_accuracy` (flow_grpo ocr.py:91, :106-116): spaces removed and lower case on
    both sides, 1 - min(lev, len(target)) / len(target); the same expression, so the same float."""
    recognized = recognized.replace(" ", "").lower()
    target = target.replace(" ", "").lower()
    dist = min(levenshtein(recognized, target), len(target))
    return 1 - dist / len(target)


def video_ocr_reward(frame_scores: list) -> float:
    """train_flow_grpo_wan.py `video_ocr_reward` (flow_grpo ocr.py:117-123): the mean over the frames with a positive
    score, 0.0 if there is none (summed in frame order, as the runtime)."""
    positive = [s for s in frame_scores if s > 0]
    return sum(positive) / len(positive) if positive else 0.0


def _norm_path(p: Any) -> Any:
    return posixpath.normpath(p) if isinstance(p, str) and p.startswith("/") else p


class FlowGrpoWanAdapter(FlowGrpoAdapter):
    """The MM-12 interface for Flow-GRPO + LoRA runs on Wan2.1-T2V-1.3B (EXT-214 to EXT-228). Cuts, identity, the
    epoch plan, steps, exact overlap and the sampler check are the flow_grpo family's (`FlowGrpoAdapter`); the base directory,
    the video reward fields, the file-order prompts, the all-zero-advantage epochs, the pinned scripts, image, lock and
    mounts and the run checks are this area's."""

    # Layout constants, read through self so a family base class reads the subclass's values.
    AREA3V = AREA3V
    BASE_REL = BASE_REL
    CUT_SCHEMA = CUT_SCHEMA
    EVAL_CUT_CONTAINER = EVAL_CUT_CONTAINER
    EVAL_SCRIPT = EVAL_SCRIPT
    FRAME_INTERVAL = FRAME_INTERVAL
    IMAGE = IMAGE
    ITEMS_RELS = ITEMS_RELS
    MOUNTS = MOUNTS
    OPTIONS = OPTIONS
    PLAN = PLAN
    PROMPTS_REL = PROMPTS_REL
    REWARD = REWARD
    RUNTIME_FILE = RUNTIME_FILE
    RUN_ARGS_SCHEMA = RUN_ARGS_SCHEMA
    SAMPLES = SAMPLES
    SCRIPT_PATHS = SCRIPT_PATHS
    SCRIPT_PROFILES = SCRIPT_PROFILES
    TRAIN_SCRIPTS = TRAIN_SCRIPTS
    VENV_LOCK_SHA256 = VENV_LOCK_SHA256
    WAN_LORA_TARGETS = WAN_LORA_TARGETS

    name = NAME
    trainer_id = ROW["uses"][0]
    trainer_ids = tuple(ROW["uses"])
    evaluator_ids = tuple(ROW["evaluators"])
    checkpoint_completeness = CHECKPOINT_COMPLETENESS
    seed_derivation = SEED_DERIVATION
    binds_spec = bool(ROW["spec_binding"])
    base_required = BASE_REQUIRED
    base_weight_dirs = BASE_WEIGHT_DIRS
    base_kind = "a Wan2.1-T2V-1.3B Diffusers directory"
    pins = tuple(ROW["pins"])
    sampler_code_sha256 = SAMPLER_CODE_SHA256
    sampler_code_source = "eval_videos.py and train_flow_grpo_wan.py SHARED SAMPLER"
    sample_id_domain = "flow-grpo-wan"
    arm_needs_items_dir = True                          # arm_problems() receives the registered item set (EXT-231)

    BASE_FIX = ("train from the spec's Wan2.1 directory inside a read-only mount (train_flow_grpo_wan.py --base, default "
                f"{AREA3V}/{BASE_REL}, re-hashed at start against models/MODELS.json), and record that mount in the "
                "launch record")
    DATA_FIX = ("train on the spec's prompt shard inside a read-only mount (train_flow_grpo_wan.py --prompts, recorded in "
                "run_args.json train_file, train_file_sha256 and n_train_prompts)")

    # --- consumption hooks (MM-24, MM-28, MM-29; EXT-218) ---

    def _settings_problems(self, s: dict) -> list[str]:
        """The settings the video rules need: the `video_ocr` reward, file-order prompts with P x G = samples_per_epoch,
        a positive integer frame count (train_flow_grpo_wan.py COMMON, PROFILES, resolve_settings)."""
        P, G, spe, frames = (s.get(k) for k in ("prompts_per_epoch", "images_per_prompt", "samples_per_epoch",
                                                 "num_frames"))
        out = []
        if s.get("reward") != self.REWARD:
            out.append(f"run_args.json settings.reward is {s.get('reward')!r}, not {self.REWARD!r}: the frame rule of "
                       "samples.jsonl is flow_grpo's video_ocr (train_flow_grpo_wan.py COMMON)")
        if s.get("prompt_order") != "file":
            out.append(f"run_args.json settings.prompt_order is {s.get('prompt_order')!r}, not 'file': the prompts of "
                       "an epoch are re-derived as shard rows in file order (train_flow_grpo_wan.py epoch_prompts)")
        if not (isinstance(P, int) and not isinstance(P, bool) and P > 0 and P * G == spe):
            out.append(f"run_args.json settings.prompts_per_epoch {P!r} x images_per_prompt {G!r} is not "
                       f"samples_per_epoch {spe!r}")
        if not (isinstance(frames, int) and not isinstance(frames, bool) and frames > 0):
            out.append(f"run_args.json settings.num_frames {frames!r} is not a positive integer")
        return out

    def _reward_field_problems(self, x: dict, n: int, s: dict, ocr_reward: bool) -> list[str]:
        """train_flow_grpo_wan.py :924-932 and PaddleOcrA.video_reward (:449-463): `frame_ocr` holds frames 0, 4, 8, ... of the
        video; a frame without error has a text and the accuracy flow_grpo's formula gives for it (re-derived, exact);
        a failed frame has no text, accuracy 0 and an error message; `ocr_acc` = `reward` = the mean of the positive
        frame accuracies (exact); `ocr_error` names the first failed frame or is null; `nonfinite` is a boolean."""
        where = f"{self.SAMPLES} line {n}"
        frames = x.get("frame_ocr")
        want = list(range(0, s["num_frames"], self.FRAME_INTERVAL))
        if not (isinstance(frames, list) and all(isinstance(f, dict) for f in frames)
                and [f.get("frame") for f in frames] == want):
            return [f"{where}: frame_ocr is not one {{frame, text, acc, error}} per frame {want} (every "
                    f"{self.FRAME_INTERVAL}th of settings.num_frames {s['num_frames']}, flow_grpo ocr.py:77)"]
        out = []
        target = x["target"]
        if not target.replace(" ", ""):
            return [f"{where}: target {target!r} is empty without spaces (the reward divides by its length)"]
        first_error = None
        for f in frames:
            acc, err, text = f.get("acc"), f.get("error"), f.get("text")
            if not (_num(acc) and 0 <= acc <= 1):
                out.append(f"{where} frame {f['frame']}: acc {acc!r} is not a number in [0, 1]")
                continue
            if err is not None:
                if not (isinstance(err, str) and err and text is None and acc == 0):
                    out.append(f"{where} frame {f['frame']}: a failed frame has a non-empty error message, no text and "
                               "acc 0 (an OCR exception scores its frame 0, train_flow_grpo_wan.py video_reward)")
                if first_error is None:
                    first_error = f
            elif not isinstance(text, str):
                out.append(f"{where} frame {f['frame']}: text {text!r} is not a string (no error recorded)")
            elif acc != video_frame_accuracy(text, target):
                out.append(f"{where} frame {f['frame']}: acc {acc!r} is not flow_grpo's frame accuracy "
                           f"{video_frame_accuracy(text, target)!r} of text {text!r} against target {target!r} "
                           "(ocr.py:106-116)")
        if out:
            return out
        reward, acc = x.get("reward"), x.get("ocr_acc")
        want_r = video_ocr_reward([f["acc"] for f in frames])
        if not (reward == acc == want_r):
            out.append(f"{where}: reward {reward!r} and ocr_acc {acc!r} are not the mean of the positive frame "
                       f"accuracies {want_r!r} (flow_grpo ocr.py:117-123)")
        want_e = (f"frame {first_error['frame']}: {first_error['error']}"[:300] if first_error is not None else None)
        if x.get("ocr_error") != want_e:
            out.append(f"{where}: ocr_error {x.get('ocr_error')!r} is not {want_e!r} (the first failed frame, or null)")
        if not isinstance(x.get("nonfinite"), bool):
            out.append(f"{where}: nonfinite {x.get('nonfinite')!r} is not a boolean")
        return out

    def _epoch_prompt_problems(self, e: int, got: list[dict], spe: int, group: int, s: dict,
                               rows: list[dict]) -> list[str]:
        """train_flow_grpo_wan.py epoch_prompts: sample i of epoch e (1-based) uses shard row ((e - 1) P + i // G) mod
        n in file order; so a group shares one prompt and an epoch's groups differ (P <= n)."""
        P, n = s["prompts_per_epoch"], len(rows)
        if not n:
            return [f"the training shard has no rows, but {self.SAMPLES} epoch {e} logs samples"]
        out = []
        for x in got:
            k = ((e - 1) * P + x["sample_index"] // group) % n
            if x["prompt_id"] != rows[k].get("prompt_id"):
                out.append(f"{self.SAMPLES} epoch {e} sample_index {x['sample_index']}: prompt_id {x['prompt_id']!r} is not "
                           f"shard row {k} ({rows[k].get('prompt_id')!r}): epoch {e} uses rows {(e - 1) * P % n} .. "
                           f"of the shard in file order (settings.prompt_order file)")
                if len(out) >= 3:
                    break
        return out

    def _flat_group_problems(self, e: int, plan: dict, xs: list[dict], group: int) -> list[str]:
        """`all_zero_advantage` (train_wan2_1.py:853-858): true exactly when every prompt group of the epoch has equal
        rewards (every advantage 0); then every sample is kept, in sample order, none filtered or re-included.
        Otherwise area 5's rule (a group with equal rewards is never plainly kept)."""
        az = plan.get("all_zero_advantage")
        if not isinstance(az, bool):
            return [f"{self.PLAN} epoch {e}: all_zero_advantage {az!r} is not a boolean"]
        flat = [len({m["reward"] for m in xs[g0:g0 + group]}) == 1 for g0 in range(0, len(xs), group)]
        if az:
            out = []
            if not all(flat):
                out.append(f"{self.PLAN} epoch {e}: all_zero_advantage is true, but prompt group {flat.index(False)} has "
                           "unequal rewards (a nonzero advantage)")
            if plan["kept"] != [m["sample_id"] for m in xs] or plan["filtered"] or plan["reincluded"]:
                out.append(f"{self.PLAN} epoch {e}: all_zero_advantage is true, so every sample is kept in sample order and "
                           "none is filtered or re-included (flow_grpo adds 1e-6 to every advantage, "
                           "train_wan2_1.py:853-858)")
            return out
        if xs and all(flat):
            return [f"{self.PLAN} epoch {e}: every prompt group has equal rewards (every advantage 0), but "
                    "all_zero_advantage is false (the runtime then keeps every sample, train_wan2_1.py:853-858)"]
        return super()._flat_group_problems(e, plan, xs, group)

    # --- launch rules (MM-31 to MM-36; EXT-223, EXT-224) ---

    @staticmethod
    def _script(argv: Any) -> str | None:
        if isinstance(argv, list) and len(argv) > 1 and isinstance(argv[1], str):
            return posixpath.basename(argv[1])
        return None

    @staticmethod
    def _options(args: list, options: dict[str, bool]) -> tuple[list[tuple[str, str | None, str]], list[str]]:
        """([(option, value, token)], [tokens argparse would refuse]): a token that is no option nor a unique prefix
        of one, a positional argument (neither script takes one), or a value option without its value."""
        parsed, odd, i = [], [], 0
        while i < len(args):
            tok = args[i]
            opt, value = resolve_option(tok, options)
            if opt is None:
                odd.append(tok if isinstance(tok, str) else repr(tok))
            else:
                if options[opt] and value is None:
                    value = args[i + 1] if i + 1 < len(args) and isinstance(args[i + 1], str) else None
                    i += 1
                    if value is None:
                        odd.append(tok)
                        continue
                parsed.append((opt, value, tok))
            i += 1
        return parsed, odd

    def argv_problems(self, argv: Any, *, profile: str | None = None) -> list[str]:
        """EXT-223 (as EXT-173 for area 5), EXT-230, ext-launch, --recheck and ext-collect. argv[1] is
        train_flow_grpo_wan.py, train_flow_grpo_wan_v2.py or eval_videos.py at its path in the read-only area 3 mount
        (the interpreter is area 5's venv, so the script directory is pinned, not derived from argv[0]); every token is
        an option of that script (a unique prefix counts, as argparse resolves it) and `--out` is /out. Training:
        `--profile` the script's profile of the launch profile (smoke/full, or smoke2/full2 for the v2 script),
        `--prompts` only the script's default inside the mount (prompts/, prompts-short/), `--no-verify-base` refused
        (collect refuses a run that did not re-hash its base). eval_videos.py: exactly one of `--base-only` and
        `--checkpoint /inputs/cut`; `--base` only the default and `--items` only items/ or items-short/ inside the
        mount (the registration's item set hash and the arms' item lines decide which one an arm may use);
        `--copy-items` refused (a host step that writes into the runtime); `--limit` refused under `--profile full`."""
        if not (isinstance(argv, list) and len(argv) > 1 and all(isinstance(a, str) for a in argv)):
            return []                                     # bind_argv reports the shape
        script = self._script(argv)
        if script not in self.OPTIONS or argv[1] != self.SCRIPT_PATHS[script]:
            return [f"argv[1] {argv[1]!r} is not one of the runtime's scripts {sorted(self.SCRIPT_PATHS.values())} (a copy "
                    "elsewhere brings its own default base, prompts and items; EXT-223)"]
        parsed, odd = self._options(argv[2:], self.OPTIONS[script])
        out = [f"{script} argument {tok!r} is not one of its options (argparse refuses it; a unique prefix counts)"
               for tok in odd]
        given: dict[str, str | None] = {}
        for opt, value, _ in parsed:
            given[opt] = value                            # argparse keeps the last value
        if given.get("--out") != "/out":
            out.append(f"{script} --out {given.get('--out')!r} is not /out (the one read-write mount)")
        if script in self.TRAIN_SCRIPTS:
            if "--no-verify-base" in given:
                out.append(f"{script} --no-verify-base skips re-hashing the base, and ext-collect refuses such a "
                           "run (run_args.json base_ref.verified false, EXT-219)")
            names = self.SCRIPT_PROFILES[script]
            want = (names[profile],) if profile in names else tuple(names.values())
            if given.get("--profile") not in want:
                out.append(f"{script} --profile {given.get('--profile')!r} is not "
                           + (f"{want[0]!r}, its profile for the launch profile {profile!r}" if profile in names
                              else " or ".join(names.values()))
                           + " (EXT-223, EXT-230)")
            prompts = f"{self.AREA3V}/{self.PROMPTS_REL[script]}"
            if "--prompts" in given and posixpath.normpath(given["--prompts"] or "") != prompts:
                out.append(f"{script} --prompts {given['--prompts']!r} is not its default {prompts}: each training "
                           "script reads its own attempt's shard (EXT-230)")
        else:
            defaults = {"--base": (f"{self.AREA3V}/{self.BASE_REL}",), "--items": tuple(f"{self.AREA3V}/{r}" for r in self.ITEMS_RELS)}
            for opt, value in given.items():
                if opt in defaults and value not in defaults[opt]:
                    out.append(f"{self.EVAL_SCRIPT} {opt} {value!r} is not " + " or ".join(defaults[opt]) + ": the "
                               f"evaluator does not hash it, so an arm is bound to the registered {opt[2:]} only "
                               "through a fixed directory inside the read-only runtime mount (EXT-223, EXT-230)")
            if "--copy-items" in given:
                out.append(f"{self.EVAL_SCRIPT} --copy-items writes the item copies into the runtime directory, a host step "
                           "(the runtime is mounted read-only)")
            elif ("--base-only" in given) == ("--checkpoint" in given):
                out.append(f"{self.EVAL_SCRIPT} evaluates exactly one of --base-only and --checkpoint {self.EVAL_CUT_CONTAINER}")
            elif "--checkpoint" in given and given["--checkpoint"] != self.EVAL_CUT_CONTAINER:
                out.append(f"{self.EVAL_SCRIPT} --checkpoint {given['--checkpoint']!r} is not {self.EVAL_CUT_CONTAINER} (the cut's "
                           "read-only mount)")
            if "--limit" in given and profile == "full":
                out.append(f"{self.EVAL_SCRIPT} --limit {given['--limit']!r} under --profile full: a gate arm evaluates the "
                           "whole registered item set; --limit is for the smoke arms (EXT-223)")
        return out

    def launch_problems(self, doc: dict, *, kind: str) -> list[tuple[str, str]]:
        """EXT-224: [(field, message)] of this adapter's launch rules (request or record). Every use runs area 5's
        runtime image (`venv_in_container`, the pinned image ID) with area 5's venv lock (pinned SHA-256), on the GPU,
        with network none, framework diffusers; the training `--seed` is the record's training_seed; the mounts are
        exactly the interpreter tree, area 5 and area 3 read-only at their own paths and /out read-write, plus the cut
        read-only at /inputs/cut for a cut evaluation."""
        out: list[tuple[str, str]] = []
        script = self._script(doc.get("argv"))
        if script not in self.OPTIONS:
            return out                                    # bind_argv and argv_problems report the script
        rt = doc.get("runtime") if isinstance(doc.get("runtime"), dict) else {}
        if rt.get("kind") != "venv_in_container" or rt.get("image") != self.IMAGE:
            out.append(("runtime", f"{script} runs in area 5's runtime image: runtime {{kind: venv_in_container, image: "
                        f"{self.IMAGE}}} (gatedtrain-a5-runtime:1), not {rt.get('kind')!r} {rt.get('image')!r}"))
        if kind == "request":
            lock = rt.get("venv_lock")
            try:
                got = file_sha256(Path(lock)) if isinstance(lock, str) and Path(lock).is_file() else None
            except OSError:
                got = None
        else:
            got = rt.get("venv_lock_sha256")
        if got != self.VENV_LOCK_SHA256:
            out.append(("runtime", f"the venv lock is {got!r}, not area 5's uv.lock {self.VENV_LOCK_SHA256} (the venv the "
                        "interpreter belongs to; MM-32)"))
        fw = doc.get("framework") if isinstance(doc.get("framework"), dict) else {}
        if fw.get("name") != "diffusers":
            out.append(("framework", f"framework {fw.get('name')!r} is not diffusers"))
        if doc.get("network") != "none":
            out.append(("network", f"{script} runs with network none (every input is mounted), not "
                        f"{doc.get('network')!r}"))
        if doc.get("gpus") is not True:
            out.append(("gpus", f"{script} needs the GPU (gpus true)"))
        given = {o: v for o, v, _ in self._options(doc["argv"][2:], self.OPTIONS[script])[0]}
        if script in self.TRAIN_SCRIPTS:
            seed = doc.get("training_seed")
            if not (isinstance(seed, int) and not isinstance(seed, bool) and given.get("--seed") == str(seed)):
                out.append(("argv", f"{script} --seed {given.get('--seed')!r} is not the training_seed {seed!r} "
                            "(MM-29; the script's default 5000 is not taken implicitly)"))
        want = list(self.MOUNTS)
        if script == self.EVAL_SCRIPT and "--checkpoint" in given:
            want.append((self.EVAL_CUT_CONTAINER, "ro"))
        mounts = doc.get("mounts") if isinstance(doc.get("mounts"), list) else []
        got_m = sorted((_norm_path(m.get("container")), m.get("mode")) for m in mounts if isinstance(m, dict))
        if got_m != sorted(want) or len(got_m) != len(mounts):
            out.append(("mounts", f"{script} mounts {got_m}, not exactly {sorted(want)}"))
        return out

    # --- run checks (MM-21, MM-29, MM-33; EXT-219) ---

    def _reward_setting_problems(self, s: dict) -> list[Problem]:
        if s.get("reward") != self.REWARD:
            return [Problem("EXT-STAGE", "--stage-out", f"run_args.json settings.reward is {s.get('reward')!r}, not "
                            f"{self.REWARD!r} (flow_grpo's video OCR reward with PaddleOCR, runtime/README.md "
                            "\"Decisions\")", "collect a smoke or full run of train_flow_grpo_wan.py")]
        return []

    # --- arms (EXT-231) ---

    def arm_problems(self, arm: Path, records: list[dict], item_rows: list[dict], items_dir: Path | None = None,
                     model_dir: Path | None = None) -> list[str]:
        """EXT-231: eval_videos.py takes `--items` (items/ or items-short/) without hashing it against the registration,
        but its `env-runtime.json` beside the arm (the evaluation output root) records `items_sha256`, the SHA-256 of
        the items.jsonl it read: that must be the registered item set's items.jsonl, so an arm of one attempt's items
        is refused under the other attempt's registration even when its eval_meta.json claims the registered hash."""
        if items_dir is None:
            return []
        rt_path = Path(arm).parent / self.RUNTIME_FILE
        rt, why = _read_json(rt_path)
        if not isinstance(rt, dict):
            return [f"{rt_path} (eval_videos.py writes it beside the arms) cannot be read: {why}"]
        want = file_sha256(Path(items_dir) / "items.jsonl")
        got = rt.get("items_sha256")
        if got != want:
            return [f"{rt_path} items_sha256 {got!r} is not the registered item set's items.jsonl {want}: the "
                    f"evaluation read {rt.get('items')!r} (eval_videos.py --items), not the registered items (EXT-231)"]
        return []

    def checks(self, *, last_cut: Path, base: Path, launch: dict, base_field: str, stage_out: Path | None = None,
               data: Path | None = None, data_field: str = "data", **kw: Any) -> list[Problem]:
        """Area 5's checks (EXT-219), plus: run_args.json and every cut.json carry the Wan port's schemas
        (another area's run is refused before its logs are read), and every cut's LoRA `target_modules` is flow_grpo's
        Wan list (eval_videos.py check_cut refuses any other). EXT-231: run_args.json `profile` is one script's
        (smoke/full of train_flow_grpo_wan.py, smoke2/full2 of train_flow_grpo_wan_v2.py), the launch record's
        script is that script, and `train_file` is that script's shard (prompts/ or prompts-short/), so neither
        attempt's run is collected with the other's data, script or profile."""
        # Area 5's profile rule compares run_args.json and cut.json `profile` with the launch record's; here the launch
        # profile (smoke, full) stands for the launched script's profile (smoke2, full2 for the v2 script, EXT-231).
        names = self.SCRIPT_PROFILES.get(posixpath.basename(str(launch.get("script_path"))), {})
        script_launch = dict(launch, profile=names.get(launch.get("profile"), launch.get("profile")))
        out = super().checks(last_cut=last_cut, base=base, launch=script_launch, base_field=base_field,
                             stage_out=stage_out, data=data, data_field=data_field, **kw)
        stage_out = Path(stage_out) if stage_out is not None else Path(last_cut).parent
        args, _ = read_run_args(stage_out)
        if args is None:
            return out
        fix = "collect a run of train_flow_grpo_wan.py as it wrote it"
        out += self._attempt_problems(args, launch)
        if args.get("schema") != self.RUN_ARGS_SCHEMA:
            out.append(Problem("EXT-STAGE", "--stage-out", f"run_args.json schema is {args.get('schema')!r}, not "
                               f"{self.RUN_ARGS_SCHEMA} (train_flow_grpo_wan.py)", fix))
        cuts = [d for _, d in self.find_cuts(stage_out)[0]] or [Path(last_cut)]
        for d in cuts:
            doc, _ = read_cut_marker(d)
            if (doc or {}).get("schema") != self.CUT_SCHEMA:
                out.append(Problem("EXT-STAGE", "--stage-out", f"{Path(d).name}/cut.json schema is "
                                   f"{(doc or {}).get('schema')!r}, not {self.CUT_SCHEMA} (train_flow_grpo_wan.py save_cut)",
                                   fix))
            cfg, _ = _read_json(Path(d) / "adapter_config.json")
            tm = cfg.get("target_modules") if isinstance(cfg, dict) else None
            if not (isinstance(tm, list) and all(isinstance(t, str) for t in tm)
                    and sorted(tm) == sorted(self.WAN_LORA_TARGETS)):
                out.append(Problem("EXT-STAGE", "--stage-out", f"{Path(d).name}/adapter_config.json target_modules "
                                   f"{tm!r} are not flow_grpo's Wan targets {sorted(self.WAN_LORA_TARGETS)} "
                                   "(train_wan2_1.py:451-460; eval_videos.py check_cut refuses any other)", fix))
        return out

    @staticmethod
    def _attempt_problems(args: dict, launch: dict) -> list[Problem]:
        """EXT-231: the run's profile names its training script (attempt 1 or 2); the launch record ran that script and
        the run read that script's shard."""
        profile = args.get("profile")
        script = next((s for s, names in SCRIPT_PROFILES.items() if profile in names.values()), None)
        if script is None:
            allowed = [p for names in SCRIPT_PROFILES.values() for p in names.values()]
            return [Problem("EXT-STAGE", "--stage-out", f"run_args.json profile {profile!r} is not one of {allowed} "
                            "(cpu test profiles are never collected; EXT-231)",
                            "collect a smoke or full run of train_flow_grpo_wan.py or train_flow_grpo_wan_v2.py")]
        out = []
        launched = launch.get("script_path")
        if launched != SCRIPT_PATHS[script]:
            out.append(Problem("EXT-LAUNCH", "--launch-record.script_path", f"run_args.json profile {profile!r} is "
                               f"{script}'s, but the launch record ran {launched!r} (EXT-231)",
                               "collect the run with the launch record of the script that wrote it"))
        want = f"{AREA3V}/{PROMPTS_REL[script]}/train.jsonl"
        got = args.get("train_file")
        if not (isinstance(got, str) and posixpath.normpath(got) == want):
            out.append(Problem("EXT-DATA", "--stage-out", f"run_args.json train_file {got!r} is not {want}: a "
                               f"{profile!r} run of {script} trains on {PROMPTS_REL[script]}/ (attempt "
                               f"{1 if script == TRAIN_SCRIPT else 2}; EXT-231)",
                               "collect each attempt's run with its own spec, script and shard"))
        return out

