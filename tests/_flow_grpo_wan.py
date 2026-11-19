"""Area "video generation" test helpers (verifier-owned; tests/README.md section 11.25, EXT-214 to EXT-228).

Written from the runtime (`runtime/`: README.md "Trainer" (its table "Outputs
(`/out`) and differences from area 5's schemas"), "Evaluator", "Smoke launch"; `train_flow_grpo_wan.py` `save_cut`,
`base_binding`, `epoch_prompts`, `resolve_settings`, `main`; `eval_videos.py` `evaluate`, `save_video`; read
2026-11-14) and its GPU probes, not from gatedtrain. Real pieces are copied into `fixtures/flow_grpo_wan/`:
- `probe-train-1/`: the 2-epoch GPU smoke `probe-train-1/` (run_args.json, env-freeze.txt,
  env-runtime.json, attempts.jsonl, samples.jsonl, epoch_plan.jsonl, train_steps.jsonl, train_log.jsonl, memory.jsonl
  byte copies, except that the host paths in run_args.json, env-runtime.json and attempts.jsonl (here and in
  probe-base-1/env-runtime.json) are replaced by the neutral `/srv/ptc/flow-grpo-wan-video/runtime` and
  `/srv/ptc/flow-grpo-sd35-ocr/runtime`; `checkpoint-2/` with its small files copied, the 94 MB LoRA replaced by a minimal safetensors file with
  the real `__metadata__` and two of the real tensor names, and cut.json's `files` re-hashed for it). That run used
  `FULL_EPOCHS` 40 and `save_every` 10 for the full profile; the script now says 60 and 15, so its `script_sha256`
  (`ad8db67d...`) is not the current script's;
- `probe-base-1/`: the 20-item base evaluation `probe-base-1/` (both arms' eval_meta.json and
  items.jsonl, every `videos/<item>/r0/frames.json`, summary.json, env-runtime.json; the 340 PNG frames are left out);
- `area5-uv.lock`: area 5's `uv.lock` (the OCR runtime's venv, which every launch of this recipe runs), whose SHA-256
  the adapter pins.
The first 40 items and 64 shard rows are area 5's fixtures (`fixtures/flow_grpo/items.jsonl`, `train.jsonl`, moved from
gatedtrain with this recipe): the runtime's copies are byte copies of area 5's files (runtime/ITEMS-SOURCE.json), so these are
their prefixes too (EXT-227).

Paths: the runtime is the recipe's (`rows.RUNTIME`), the OCR runtime `rows.OCR_RUNTIME` (the SD3.5 OCR recipe's
runtime). When the OCR runtime has no venv (a fresh checkout), the tests set `GATEDTRAIN_RECIPE_FLOW_GRPO_WAN_VIDEO_OCR_RUNTIME`
at a stand-in (a temporary `tests/.stand-in-ocr-runtime-*/` holding only `.venv/pyvenv.cfg`, whose interpreter tree is
the neutral `/srv/python/cpython-3.12.16-linux-x86_64-gnu`) before the recipe is imported; the test CLI subprocesses
inherit it. Real-file tests read the real runtime and OCR runtime (`REAL_AREA3V`, `REAL_AREA5`) and the GPU runs
(`GATEDTRAIN_RECIPE_FLOW_GRPO_WAN_VIDEO_OUTPUTS`, the directory holding probe-train-1/, probe-base-1/ and full-1/;
`GATEDTRAIN_RECIPE_FLOW_GRPO_WAN_VIDEO_RECORDS`, the directory holding their launch records; set the runtime overrides to the
runtimes those runs used as well) and skip where absent.

`WanScenario(root)` builds the rest on a mirrored host (`GATEDTRAIN_EXT_MIRROR_ROOT` = `<root>/host` stands for the
container's `/`, EXT-112), so every container path is the real one:
- the interpreter tree's `bin/python3.12` (stand-in) and the OCR runtime's `.venv/bin/python` linking to it (as on the
  host), area 5's `uv.lock` (the real bytes);
- the runtime: the two scripts (stand-ins, never executed), `models/wan2.1-t2v-1.3b/` (the 21 file names of the
  real base, stand-in contents) and `models/MODELS.json`, `prompts/train.jsonl` (64 rows), `items/` (40 items);
- `<output root>/area3-video/full-1/`: a training run (seed 5000, 3 epochs of P = 4 prompts x G = 4 videos in file order,
  2 optimizer steps per epoch, cuts at epochs 2 and 3). Epoch 1 filters one never-rendered group (4 samples, as the
  GPU smoke's epoch 1) and has one sample with an OCR frame error whose reward stays 1.0; epoch 2 has a group with one
  member at the group mean and an always-rendered group: 5 zero-advantage samples, 11 nonzero, so the padding re-includes
  one (12 kept, 4 filtered); epoch 3 renders nothing at all: every advantage is 0, so all 16 are kept
  (`all_zero_advantage`);
- `<output root>/area3-video/eval-{base,cand}/`: two evaluations as eval_videos.py writes them (17 frames per video with
  `frames.json` as the output reference, `scores.jsonl`, `arm-ocr_b_acc/` and `arm-ocr_a_acc/` with hard-linked videos);
- `<root>/`: spec, registration, trainer.yaml, the launch record of the full run, a smoke record, the host work's GPU
  log.

Attempt 2 (amendment, EXT-229 to EXT-232): every scenario's runtime also holds `train_flow_grpo_wan_v2.py` (stand-in),
`prompts-short/train.jsonl` (the 22 fixture shard rows with a target of at most 2 words) and `items-short/` (fixture
items 4 .. 39 in reverse order: another item set, so another manifest hash). `WanScenario(root, attempt=2)` points the
spec at those, trains with the v2 script and profile `full2` (8 prompts x 4 videos per epoch, 3 epochs, cuts 2 and 3;
`DEFAULT_PLAN_V2`) and evaluates the short items.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import pwd
import random
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable

import yaml

REPO = Path(__file__).resolve().parents[1]
RUNTIME_ENV = "GATEDTRAIN_RECIPE_FLOW_GRPO_WAN_VIDEO_RUNTIME"
OCR_RUNTIME_ENV = "GATEDTRAIN_RECIPE_FLOW_GRPO_WAN_VIDEO_OCR_RUNTIME"
OUTPUTS_ENV = "GATEDTRAIN_RECIPE_FLOW_GRPO_WAN_VIDEO_OUTPUTS"      # tests only: the directory holding the GPU runs' outputs
RECORDS_ENV = "GATEDTRAIN_RECIPE_FLOW_GRPO_WAN_VIDEO_RECORDS"      # tests only: the directory holding their launch records
NEUTRAL_TREE = "/srv/python/cpython-3.12.16-linux-x86_64-gnu"
# The real runtime and OCR runtime (the overrides, else this checkout's runtime/ and the sibling SD3.5 OCR checkout's
# runtime/): the real-file tests read their committed scripts, lock, manifests and item sets.
REAL_AREA3V = Path(os.environ.get(RUNTIME_ENV) or REPO / "runtime")
REAL_AREA5 = Path(os.environ.get(OCR_RUNTIME_ENV) or REPO.parent / "gatedtrain-recipe-flow-grpo-sd35-ocr" / "runtime")
# The GPU runs (probe-train-1/, probe-base-1/, full-1/) and their launch records, when the host has them.
REAL_OUT = Path(os.environ.get(OUTPUTS_ENV) or "/nonexistent/flow-grpo-wan-video-outputs")
REAL_RECORDS = Path(os.environ.get(RECORDS_ENV) or "/nonexistent/flow-grpo-wan-video-records")


def run_user() -> str:
    """The container's `<uid>:<gid>` in fixture requests and records: the uid running the tests and its primary group,
    so the output directories the tests create are owned by the container's uid (EXT-63) whoever runs them. The launch
    rules refuse root, so a test that needs it skips under uid 0."""
    uid = os.getuid()
    if uid == 0:
        raise unittest.SkipTest("the launch rules refuse a container running as root (uid 0); run the tests as a "
                                "non-root user")
    return f"{uid}:{pwd.getpwuid(uid).pw_gid}"


def _stand_in_ocr_runtime() -> None:
    """Point the OCR runtime override at a stand-in when the configured one has no venv (a fresh checkout): a temporary
    `tests/.stand-in-ocr-runtime-*/` holding only `.venv/pyvenv.cfg`, whose interpreter tree is the neutral
    NEUTRAL_TREE. The test CLI subprocesses inherit the override."""
    if (REAL_AREA5 / ".venv" / "pyvenv.cfg").is_file():
        return
    if "gatedtrain_recipe_flow_grpo_wan_video.rows" in sys.modules:
        raise RuntimeError(f"_flow_grpo_wan must be imported before the recipe's rows (it sets {OCR_RUNTIME_ENV})")
    # Not under /tmp: the container's /tmp is a tmpfs, which no mount may nest in (EXT-88).
    tmp = Path(tempfile.mkdtemp(prefix=".stand-in-ocr-runtime-", dir=Path(__file__).resolve().parent))
    atexit.register(shutil.rmtree, tmp, True)
    (tmp / ".venv").mkdir()
    (tmp / ".venv" / "pyvenv.cfg").write_text(f"home = {NEUTRAL_TREE}/bin\nversion_info = 3.12.16\n", encoding="utf-8")
    os.environ[OCR_RUNTIME_ENV] = str(tmp)


_stand_in_ocr_runtime()
from gatedtrain_recipe_flow_grpo_wan_video import rows as R  # noqa: E402  (after the OCR runtime override)

from gatedtrain.testing.ext import ENV_HOST_WORK, ENV_INPUT_ROOTS, ENV_INTERPRETER_TREES, ENV_MIRROR_ROOT, ENV_OUTPUT_ROOT, gpu_step_script, host_config_sha256, launch_label, safetensors_bytes, sha_of, write_canonical
EXT = Path(__file__).resolve().parent / "fixtures"     # the recipe's fixtures (gatedtrain: tests/fixtures/ext)
from gatedtrain.testing.support import file_sha256, manifest_hash, run_cli

FGW = EXT / "flow_grpo_wan"
PROBE = FGW / "probe-train-1"
PROBE_EVAL = FGW / "probe-base-1"
FG = EXT / "flow_grpo"                                 # area 5's item and shard prefixes (byte copies, EXT-227)
LOCK_FIXTURE = FGW / "area5-uv.lock"
ADAPTER = "flow_grpo_wan"
TRAINER_ID = "rl.flow_grpo_wan@0.1"
EVALUATOR_ID = "t2v.wan_ocr_doctr_items@0.1"
SUPPORT_ID = "t2v.wan_ocr_paddle_items@0.1"
METRIC, SUPPORT_METRIC = "ocr_b_acc", "ocr_a_acc"
CPYTHON = R.HOST_ROWS["interpreter_trees"][0]         # the interpreter tree the OCR runtime's venv links to
PYTHON_DIR = R.PYTHON_DIR                             # its parent, mounted at its own path
AREA5 = R.OCR_RUNTIME                                 # the OCR runtime (venv, lock), mounted at its own path
AREA3V = R.RUNTIME                                    # the runtime, mounted at its own path
OUT_ROOT = "/srv/gatedtrain/out"                             # the scenario's output root (mirrored)
INTERPRETER = f"{AREA5}/.venv/bin/python"
TRAIN_PY = f"{AREA3V}/train_flow_grpo_wan.py"
TRAIN_PY_V2 = f"{AREA3V}/train_flow_grpo_wan_v2.py"                # attempt 2 (amendment)
EVAL_PY = f"{AREA3V}/eval_videos.py"
BASE_REL = "models/wan2.1-t2v-1.3b"
IMAGE = "sha256:52070d08298e4ed72cfe49bf94cbc2061623cb01e972e1ab84a6e135b4b057a1"     # gatedtrain-a5-runtime:1
LOCK_SHA = "sha256:502c7ce70e0950cf36cbd108cd9566e0193cbcabe8914c8d0c5bbd231caecda1"  # the SD3.5 OCR recipe's runtime/uv.lock
LOCK_SHA_RUN = "sha256:8243b02f8287944026511dff3489f2cb23c91b88a0fd0d7a1b8f22678ece20d7"  # the same lock as the probes and published runs recorded it
EVAL_VIDEOS_SHA256 = "sha256:61873167b855e84df94edf99492ee52ccb6f3147127502cbfaf157b2bee59b4f"   # the published eval_videos.py
PROBE_EVAL_VIDEOS_SHA256 = "sha256:de9c2e89f94b53d05ced25f2aba4c28bed1b295f64c2bcdbb6d3b57dde179641"   # the eval_videos.py the GPU probe and the gate ran
SAMPLER_CODE_SHA = "sha256:41606b0b53ad637da4614aaa38fd52b5e9bf0b2147da7ed71c3eaad92dcb1f17"
# The block the probes (fixtures) and the published runs recorded, before its edit-distance import moved to rapidfuzz.
SAMPLER_CODE_SHA_RUN = "sha256:dad91e418fe717bf2ad0b3fe2727dfdea36a316b87c2240efc0aa14955d90300"
# The registry's scorers: the probe's, with the published eval_videos.py's code hash.
SCORER = {**json.loads((PROBE_EVAL / f"arm-{METRIC}" / "eval_meta.json").read_text(encoding="utf-8"))["scorer"],
          "code_sha256": EVAL_VIDEOS_SHA256}
SUPPORT_SCORER = {**json.loads((PROBE_EVAL / f"arm-{SUPPORT_METRIC}" / "eval_meta.json").read_text(encoding="utf-8"))["scorer"],
                  "code_sha256": EVAL_VIDEOS_SHA256}
SAMPLER = json.loads((PROBE / "checkpoint-2" / "sampler.json").read_text(encoding="utf-8"))
SAMPLER_KEYS = tuple(sorted(SAMPLER))
SEEDS = [11]
LORA_WEIGHTS = "pytorch_lora_weights.safetensors"
CUT_FILES = (LORA_WEIGHTS, "adapter_config.json", "sampler.json", "base_ref.json", "trainer_state.json")
DEPLOYABLE = (LORA_WEIGHTS, "adapter_config.json", "sampler.json", "base_ref.json")
WAN_TARGETS = ("add_k_proj", "add_q_proj", "add_v_proj", "to_add_out", "to_k", "to_out.0", "to_q", "to_v")
_PROBE_BASE_REF = json.loads((PROBE / "checkpoint-2" / "base_ref.json").read_text(encoding="utf-8"))
BASE_FILES = tuple(sorted(_PROBE_BASE_REF["files"]))   # the 21 files of models/wan2.1-t2v-1.3b (MODELS.json)
REPO, REVISION = _PROBE_BASE_REF["repo"], _PROBE_BASE_REF["revision"]
SEED = 5000
FRAMES = 17
OCR_FRAMES = tuple(range(0, FRAMES, 4))                # flow_grpo `video[::4]`: 0, 4, 8, 12, 16
SCRIPT_TEXT = {"train_flow_grpo_wan.py": f"# fixture stand-in for {TRAIN_PY}; never executed\n",
               "train_flow_grpo_wan_v2.py": f"# fixture stand-in for {TRAIN_PY_V2}; never executed\n",
               "eval_videos.py": f"# fixture stand-in for {EVAL_PY}; never executed\n"}
LABEL = "mm-a3-video-full"
AREA = 3                                              # MULTIMODAL-SPEC row 3 (the directory keeps the number)
ENV = {"HF_HUB_OFFLINE": "1", "HOME": "/tmp/home", "XDG_CACHE_HOME": "/tmp/cache", "PYTHONDONTWRITEBYTECODE": "1",
       "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
       "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
ENV_NAMES = sorted(ENV)
EVAL_ENV = {**ENV, "DOCTR_CACHE_DIR": "/tmp/cache/doctr"}
PROFILE = {"epochs": 3, "save_every": 2}
# Per-member frame plans of a prompt group (G = 4). A member is "none" (no frame reads anything: reward 0), "full"
# (every OCR frame reads the target: reward 1), ("part", k) (frame 0 reads the target with k characters replaced, the
# others nothing: reward 1 - k / L), or "full_err" (as "full", frame 8 raises: reward still 1.0, `ocr_error` set).
# "mean1" puts exactly member 2 at the group mean (k1 = 3 k2 - L; L >= 9).
PATTERNS = {"varied": ["none", ("part", 2), ("part", 1), "full"], "varied_err": ["none", ("part", 2), ("part", 1),
                                                                                 "full_err"],
            "flat0": ["none"] * 4, "flat1": ["full"] * 4, "mean1": "mean1"}
DEFAULT_PLAN = {1: ["varied_err", "varied", "varied", "flat0"], 2: ["varied", "mean1", "flat1", "varied"],
                3: ["flat0"] * 4}
# Attempt 2 (P = 8): epoch 1 as attempt 1's with four more groups, epoch 2 mixed, epoch 3 renders nothing (all-zero).
DEFAULT_PLAN_V2 = {1: ["varied_err", "varied", "varied", "flat0", "varied", "flat1", "varied", "flat0"],
                   2: ["varied", "varied", "flat1", "varied", "varied", "flat0", "varied", "varied"],
                   3: ["flat0"] * 8}
# attempt -> (training script, its full profile, prompt directory, item directory) in the runtime
ATTEMPTS = {1: (TRAIN_PY, "full", "prompts", "items"), 2: (TRAIN_PY_V2, "full2", "prompts-short", "items-short")}


def sha256_text(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def derive_seed(*parts) -> int:
    """train_flow_grpo_wan.py / eval_videos.py `derive_seed` (SHARED SAMPLER block)."""
    return int.from_bytes(hashlib.sha256(":".join(str(p) for p in parts).encode("utf-8")).digest()[:8], "big") >> 1


def compact_hash(obj: Any) -> str:
    """The runtime's own hash encoding (`canonical` in both scripts): compact sorted JSON, not RFC 8785."""
    return sha256_text(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def write_json(path: Path, doc: Any) -> Path:
    """train_flow_grpo_wan.py / eval_videos.py `write_json`: indent 2, sorted keys, final newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_jsonl(path: Path, docs: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(d, sort_keys=True) + "\n" for d in docs), encoding="utf-8")
    return path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def target_text(prompt: str) -> str:
    """train_flow_grpo_wan.py `target_text` (flow_grpo ocr.py:86)."""
    return prompt.split('"')[1]


def lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def frame_accuracy(recognized: str, target: str) -> float:
    """train_flow_grpo_wan.py `video_frame_accuracy` (flow_grpo ocr.py:91, :106-116), the same expression."""
    recognized = recognized.replace(" ", "").lower()
    target = target.replace(" ", "").lower()
    return 1 - min(lev(recognized, target), len(target)) / len(target)


def video_reward(accs: list[float]) -> float:
    """train_flow_grpo_wan.py `video_ocr_reward` (flow_grpo ocr.py:117-123)."""
    positive = [s for s in accs if s > 0]
    return sum(positive) / len(positive) if positive else 0.0


def damaged(target: str, k: int) -> str:
    """The target (spaces removed) with its first k characters replaced by '#'."""
    t = target.replace(" ", "")
    return "#" * k + t[k:]


def frame_ocr(member: Any, target: str) -> tuple[list[dict], str | None]:
    """(frame_ocr, ocr_error) of one video (PaddleOcrA.video_reward on frames 0, 4, ..., 16)."""
    out, error = [], None
    for f in OCR_FRAMES:
        if member in ("full", "full_err"):
            text = target
        elif isinstance(member, tuple) and f == 0:
            text = damaged(target, member[1])
        else:
            text = ""
        if member == "full_err" and f == 8:
            out.append({"frame": f, "text": None, "acc": 0.0, "error": "RuntimeError: fixture OCR failure"})
            error = error or f"frame {f}: RuntimeError: fixture OCR failure"[:300]
            continue
        out.append({"frame": f, "text": text, "acc": frame_accuracy(text, target), "error": None})
    return out, error


def members_of(pattern: str, target: str) -> list[Any]:
    if pattern != "mean1":
        return list(PATTERNS[pattern])
    L = len(target.replace(" ", ""))
    assert L >= 9, f"mean1 needs a target of at least 9 characters: {target!r}"
    k2 = L // 3 + 1
    return ["none", ("part", 3 * k2 - L), ("part", k2), "full"]


# ---------------------------------------------------------------------------
# Base model and cuts (MM-17, MM-18, MM-20)
# ---------------------------------------------------------------------------

def build_base(d: Path, tag: str = "") -> Path:
    """The Wan2.1 Diffusers directory with the real base's 21 file names (stand-in contents)."""
    for rel in BASE_FILES:
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith(".safetensors"):
            p.write_bytes(safetensors_bytes(f"{rel} {tag}"))
        elif rel.endswith(".json"):
            p.write_text(json.dumps({"_fixture": rel, "tag": tag}, indent=2) + "\n", encoding="utf-8")
        else:
            p.write_text(f"fixture stand-in {rel} {tag}\n", encoding="utf-8")
    return d


def base_files(base: Path) -> dict[str, str]:
    return {rel: file_sha256(base / rel) for rel in BASE_FILES}


def models_json(base: Path) -> dict:
    """models/MODELS.json as fetch_models.py writes it (the base entry)."""
    return {"schema": "ptc_mm.video_models", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
            "models": {"wan2.1-t2v-1.3b": {"repo": REPO, "revision": REVISION, "license": "apache-2.0", "gated": False,
                                           "files": {rel: {"sha256": file_sha256(base / rel),
                                                           "size": (base / rel).stat().st_size} for rel in BASE_FILES}}}}


def base_ref_doc(base: Path, *, revision: str = REVISION) -> dict:
    """train_flow_grpo_wan.py `base_binding` reduced to the cut's base_ref.json keys."""
    files = base_files(base)
    return {"repo": REPO, "revision": revision, "files": files, "files_sha256": compact_hash(files)}


def lora_bytes(tag: str) -> bytes:
    """`save_lora_adapter` + `canonicalize_safetensors_header` with the real cut's `__metadata__` and two of its tensor
    names (the fixture cut's stand-in)."""
    raw = (PROBE / "checkpoint-2" / LORA_WEIGHTS).read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + n])
    text = json.dumps(header, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    text += b" " * (-len(text) % 8)
    return struct.pack("<Q", len(text)) + text + hashlib.sha256(tag.encode()).digest()[:16]


def build_cut(out: Path, epoch: int, *, global_step: int, samples_consumed: int, samples_filtered: int, base: Path,
              script_sha256: str, profile: str = "full", seed: int = SEED, tag: str = "") -> Path:
    """`checkpoint-<epoch>/` as train_flow_grpo_wan.py save_cut writes it, `cut.json` last."""
    d = out / f"checkpoint-{epoch}"
    d.mkdir(parents=True)
    (d / LORA_WEIGHTS).write_bytes(lora_bytes(f"lora epoch {epoch} {tag}"))
    shutil.copyfile(PROBE / "checkpoint-2" / "adapter_config.json", d / "adapter_config.json")
    write_json(d / "sampler.json", SAMPLER)
    write_json(d / "base_ref.json", base_ref_doc(base))
    write_json(d / "trainer_state.json", {"epoch": epoch, "global_step": global_step,
                                          "note": "resume-only information; the run does not resume from a cut "
                                                  "(a preempted run restarts)",
                                          "samples_consumed": samples_consumed, "samples_filtered": samples_filtered})
    files = {n: file_sha256(d / n) for n in CUT_FILES}
    write_json(d / "cut.json", {
        "schema": "ptc_mm.flow_grpo_wan_cut", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
        "epoch": epoch, "global_step": global_step, "samples_consumed": samples_consumed,
        "samples_filtered": samples_filtered, "files": files, "deployable": list(DEPLOYABLE),
        "created_at": "2026-11-14T22:00:00Z", "script_sha256": script_sha256, "sampler_code_sha256": SAMPLER_CODE_SHA,
        "seed": seed, "profile": profile, "ema": False})
    return d


def adapter_hash(cut: Path) -> str:
    from gatedtrain.testing.support import canonical_bytes, manifest_entries
    entries = [e for e in manifest_entries(cut) if e["path"] in DEPLOYABLE]
    return "sha256:" + hashlib.sha256(canonical_bytes(entries)).hexdigest()


def lora_model_hash(cut: Path, base: Path) -> str:
    """MM-18: sha256(canonical_json({base: base manifest hash, adapter: adapter hash})); the base has no `.cache/`."""
    return sha_of({"base": manifest_hash(base), "adapter": adapter_hash(cut)})


# ---------------------------------------------------------------------------
# A training run (train_flow_grpo_wan.py main)
# ---------------------------------------------------------------------------

def settings(seed: int = SEED, **over) -> dict:
    """train_flow_grpo_wan.py `resolve_settings` of the GPU smoke with the fixture profile."""
    s = json.loads((PROBE / "run_args.json").read_text(encoding="utf-8"))["settings"]
    s.update(PROFILE)
    s.update(over)
    s["seed"] = seed
    s["samples_per_epoch"] = s["prompts_per_epoch"] * s["images_per_prompt"]
    return s


def simulate(prompts: list[dict], s: dict, *, attempt: int = 1, plan: dict[int, list[str]] | None = None
             ) -> tuple[list[dict], list[dict], list[dict], dict[int, tuple]]:
    """(samples.jsonl, epoch_plan.jsonl, train_steps.jsonl lines, {epoch: (global_step, consumed, filtered)}).
    Prompts in file order (`epoch_prompts`); the zero-advantage filter, the all-zero rule and the padding as
    train_flow_grpo_wan.py :933-960 (a sample whose reward equals its group's mean has advantage 0; the padding test
    uses the count before the all-zero revision); the training order as :964-1013. Random draws use `random.Random` (the
    runtime uses torch generators; the adapter does not re-derive these draws)."""
    plan = plan or DEFAULT_PLAN
    seed, P, G, E = s["seed"], s["prompts_per_epoch"], s["images_per_prompt"], s["epochs"]
    nb, gas, inner_n = s["num_batches_per_epoch"], s["gradient_accumulation_steps"], s["num_inner_epochs"]
    samples, plans, steps, state = [], [], [], {}
    gs = consumed = filtered = 0
    for e in range(1, E + 1):
        rows = [(prompts[((e - 1) * P + k) % len(prompts)], g) for k in range(P) for g in range(G)]
        pats = plan.get(e, ["varied"] * P)
        ids, rewards = [], []
        for i, (row, g) in enumerate(rows):
            noise = derive_seed("sample", seed, e, i)
            sid = compact_hash(["flow-grpo-wan", seed, attempt, e, i, row["prompt_id"], noise, g])
            target = target_text(row["prompt"])
            fo, err = frame_ocr(members_of(pats[i // G], target)[g], target)
            r = video_reward([f["acc"] for f in fo])
            ids.append(sid)
            rewards.append(r)
            samples.append({"attempt": attempt, "epoch": e, "frame_ocr": fo, "member": g, "nonfinite": False,
                            "ocr_acc": r, "ocr_error": err, "prompt": row["prompt"], "prompt_id": row["prompt_id"],
                            "reward": r, "sample_id": sid, "sample_index": i, "seed": noise, "target": target,
                            "utc": "2026-11-14T21:00:00Z"})
        n = len(rows)
        means = [sum(rewards[k * G:(k + 1) * G]) / G for k in range(P)]
        mask = [abs(rewards[i] - means[i // G]) > 1e-12 for i in range(n)]
        true_count = sum(mask)
        all_zero = true_count == 0
        if all_zero:
            mask = [True] * n
        reincluded = []
        if true_count % nb:
            false_idx = [i for i in range(n) if not mask[i]]
            need = nb - true_count % nb
            if len(false_idx) >= need:
                reincluded = sorted(random.Random(derive_seed("mask", seed, e)).sample(false_idx, need))
                for i in reincluded:
                    mask[i] = True
        kept = [i for i in range(n) if mask[i]]
        n_steps = (nb // gas) * inner_n if kept else 0
        plans.append({"all_zero_advantage": all_zero, "epoch": e, "filter_reason": "zero_advantage",
                      "filtered": [ids[i] for i in range(n) if not mask[i]], "kept": [ids[i] for i in kept],
                      "reincluded": [ids[i] for i in reincluded], "steps": [gs + 1, gs + n_steps] if n_steps else [],
                      "utc": "2026-11-14T21:00:01Z"})
        if kept:
            m = len(kept) // nb
            for inner in range(inner_n):
                order = list(kept)
                random.Random(derive_seed("perm", seed, e, inner)).shuffle(order)
                virtual = [order[v * m:(v + 1) * m] for v in range(nb)]
                acc = []
                for v, vidx in enumerate(virtual):
                    acc += [ids[i] for i in vidx]
                    if (v + 1) % gas == 0:
                        gs += 1
                        steps.append({"epoch": e, "inner": inner, "sample_ids": acc, "step": gs,
                                      "utc": "2026-11-14T21:00:05Z"})
                        acc = []
            consumed += len(kept)
        filtered += n - len(kept)
        state[e] = (gs, consumed, filtered)
    return samples, plans, steps, state


def run_args(*, s: dict, profile: str, attempt: int, n_train: int, train_sha256: str, files_sha256: str,
             script_sha256: str, data_rel: str = "prompts") -> dict:
    """train_flow_grpo_wan.py run_args.json: the GPU smoke's document with this run's values."""
    doc = json.loads((PROBE / "run_args.json").read_text(encoding="utf-8"))
    doc.update({"profile": profile, "attempt": attempt, "settings": s, "seed": s["seed"],
                "base_model": f"{AREA3V}/{BASE_REL}",
                "base_ref": {"repo": REPO, "revision": REVISION, "files_sha256": files_sha256, "verified": True},
                "train_file": f"{AREA3V}/{data_rel}/train.jsonl", "train_file_sha256": train_sha256,
                "n_train_prompts": n_train, "sampler": SAMPLER, "sampler_sha256": compact_hash(SAMPLER),
                "sampler_code_sha256": SAMPLER_CODE_SHA, "script_sha256": script_sha256,
                "created_at": "2026-11-14T21:00:00Z"})
    return doc


def build_run(out: Path, *, runtime: Path, base: Path, script_sha256: str, profile: str = "full",
              s: dict | None = None, plan: dict[int, list[str]] | None = None, data_rel: str = "prompts"
              ) -> tuple[Path, dict[int, tuple]]:
    s = s or settings()
    out.mkdir(parents=True)
    prompts = read_jsonl(runtime / data_rel / "train.jsonl")
    samples, plans, steps, state = simulate(prompts, s, plan=plan)
    shutil.copyfile(PROBE / "env-freeze.txt", out / "env-freeze.txt")
    rt = json.loads((PROBE / "env-runtime.json").read_text(encoding="utf-8"))
    rt.update({"script_sha256": script_sha256, "env_freeze_sha256": file_sha256(out / "env-freeze.txt"),
               "sampler_code_sha256": SAMPLER_CODE_SHA, "uv_lock_sha256": LOCK_SHA})   # this code's, not the probe's
    write_json(out / "env-runtime.json", rt)
    write_json(out / "run_args.json", run_args(
        s=s, profile=profile, attempt=1, n_train=len(prompts),
        train_sha256=file_sha256(runtime / data_rel / "train.jsonl"), files_sha256=base_ref_doc(base)["files_sha256"],
        script_sha256=script_sha256, data_rel=data_rel))
    write_jsonl(out / "samples.jsonl", samples)
    write_jsonl(out / "epoch_plan.jsonl", plans)
    write_jsonl(out / "train_steps.jsonl", steps)
    write_jsonl(out / "train_log.jsonl", [{"step": st["step"], "epoch": st["epoch"], "policy_loss": 0.0}
                                         for st in steps])
    write_jsonl(out / "memory.jsonl", [{"epoch": e, "seconds_epoch": 205.0} for e in state])
    write_jsonl(out / "attempts.jsonl", [{"attempt": 1, "started_at": "2026-11-14T21:00:00Z"}])
    for e, (gs, consumed, filtered) in state.items():
        if e % s["save_every"] == 0 or e == s["epochs"]:
            build_cut(out, e, global_step=gs, samples_consumed=consumed, samples_filtered=filtered, base=base,
                      script_sha256=script_sha256, profile=profile, seed=s["seed"])
    return out, state


# ---------------------------------------------------------------------------
# Evaluation runs (eval_videos.py evaluate: videos, scores.jsonl, one arm per metric)
# ---------------------------------------------------------------------------

def baseline_score(i: int, r: int = 0) -> float:
    return ((i * 7) % 9) / 16 + 1 / 17


def candidate_score(i: int, r: int = 0) -> float:
    return min(1.0, baseline_score(i, r) + 0.125 + 0.0625 * (i % 3))


def save_video(out: Path, rel_dir: str, tag: str) -> dict:
    """eval_videos.py `save_video`: 17 PNG frames (stand-in bytes) and frames.json, whose path and SHA-256 are the
    output reference."""
    d = out / rel_dir
    d.mkdir(parents=True)
    frames = []
    for f in range(FRAMES):
        p = d / f"frame_{f:02d}.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\n" + f"stand-in {tag} {f}".encode())
        frames.append({"path": p.name, "sha256": file_sha256(p)})
    write_json(d / "frames.json", {"format": "PNG, RGB uint8, lossless", "frames": frames, "height": 240,
                                   "num_frames": FRAMES, "schema": "ptc_mm.video_frames", "schema_version": "1.0",
                                   "width": 416})
    return {"path": f"{rel_dir}/frames.json", "sha256": file_sha256(d / "frames.json")}


def build_eval(out: Path, items: list[dict], score_fn: Callable[[int, int], float], *, tag: str,
               model: str | None, base_model: str | None, item_set: str | None, filled: bool = True,
               metrics: tuple[str, ...] = (METRIC, SUPPORT_METRIC), items_sha256: str | None = None) -> Path:
    """`<out>/videos/<item>/r<r>/`, `<out>/scores.jsonl` and `<out>/arm-<metric>/` per metric with the video
    directories hard-linked, as eval_videos.py writes them. `filled`: the gatedtrain hashes are set (as after
    ext-arm-fill); otherwise the evaluator's own output (nulls, compact-JSON sampler hash). `env-runtime.json` names the
    SHA-256 of the items.jsonl the evaluation read (`items_sha256`; default: these rows as `write_jsonl` writes them),
    which the adapter binds to the registered item set (EXT-231)."""
    out.mkdir(parents=True)
    rt = json.loads((PROBE_EVAL / "env-runtime.json").read_text(encoding="utf-8"))
    rt.update({"sampler_code_sha256": SAMPLER_CODE_SHA, "uv_lock_sha256": LOCK_SHA})   # this code's, not the probe's
    rt["items_sha256"] = items_sha256 or sha256_text("".join(json.dumps(d, sort_keys=True) + "\n"
                                                             for d in items).encode("utf-8"))
    write_json(out / "env-runtime.json", rt)
    rows = []
    for r, seed in enumerate(SEEDS):
        for i, it in enumerate(items):
            ref = save_video(out, f"videos/{it['item']}/r{r}", f"{tag} {it['item']} {r}")
            b = score_fn(i, r)
            rows.append({"item": it["item"], "repeat": r, "seed": seed,
                         "noise_seed": derive_seed("eval", seed, it["item"]), "target": it["target"],
                         "output_ref": ref, "nonfinite": False, "errors": {}, "num_frames": FRAMES,
                         "mean_abs_frame_diff": 0.05, "ocr_a_acc": b, "ocr_b_acc": b})
    write_jsonl(out / "scores.jsonl", rows)
    scorers = {METRIC: (EVALUATOR_ID, SCORER), SUPPORT_METRIC: (SUPPORT_ID, SUPPORT_SCORER)}
    for metric in metrics:
        arm = out / f"arm-{metric}"
        arm.mkdir()
        lines = []
        for row in rows:
            vdir = Path(row["output_ref"]["path"]).parent
            (arm / vdir).mkdir(parents=True)
            for f in sorted((out / vdir).iterdir()):
                os.link(f, arm / vdir / f.name)
            sub = {k: row[k] for k in ("ocr_a_acc", "ocr_b_acc") if row[k] is not None}
            sub.update(nonfinite=int(row["nonfinite"]), mean_abs_frame_diff=row["mean_abs_frame_diff"])
            lines.append({"item": row["item"], "repeat": row["repeat"], "seed": row["seed"], "score": row[metric],
                          "error": None, "sub_metrics": sub, "output_ref": row["output_ref"]})
        write_jsonl(arm / "items.jsonl", lines)
        evaluator, scorer = scorers[metric]
        write_json(arm / "eval_meta.json", {
            "schema": "gatedtrain.item_eval", "schema_version": "1.0", "evaluator_id": evaluator, "adapter": ADAPTER,
            "metric": metric, "direction": "higher_is_better", "score_range": [0, 1],
            "checkpoint": {"model_hash": model if filled else None, "base_model_hash": base_model if filled else None},
            "normalizer_sha256": None, "env_version_sha256": None,
            "item_set_hash": item_set if filled else None, "eval_seeds": list(SEEDS),
            "sampler": SAMPLER, "sampler_hash": sha_of(SAMPLER) if filled else compact_hash(SAMPLER),
            "scorer": dict(scorer), "framework": {"name": "diffusers", "version": "0.42.1"},
            "created_at": "2026-11-14T22:30:00Z"})
    return out


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

def real_baseline_scores() -> list[float]:
    """The 20 deciding-metric scores of the GPU base evaluation probe-base-1 (fixture copy)."""
    return [l["score"] for l in read_jsonl(PROBE_EVAL / f"arm-{METRIC}" / "items.jsonl")]


def smoke_record(*, base_model: str, item_set: str, **overrides) -> dict[str, Any]:
    """A PASS smoke.json (MM-38) on the deciding metric: 40 baseline ocr_b_acc scores (the real probe's 20 twice),
    measurements of the GPU smoke (peak 5.5 GB, 25 GB host RSS at the encode, 205 s per epoch-step pair)."""
    rec = {"schema": "gatedtrain.smoke_record", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
           "adapter": ADAPTER, "area": AREA, "base_model_hash": base_model, "item_set_hash": item_set,
           "peak_gpu_mem_gb": 11.3, "host_ram_peak_gb": 24.2, "swap_peak_gb": 0.0, "seconds_per_step": 102.0,
           "projected_gpu_h": 4.8, "gpu_log_total_h": 6.0,
           "baseline_scores": real_baseline_scores() * 2, "error_count": 0, "verdict": "PASS", "reasons": []}
    rec.update(overrides)
    return rec


class WanScenario:
    """A whole area-3 video run collected against the Wan2.1 base (module docstring)."""

    def __init__(self, root: Path, *, n_items: int = 40, overlap: int = 0, filled: bool = True,
                 inner_epochs: int = 1, plan: dict[int, list[str]] | None = None, attempt: int = 1):
        self.root = root
        self.attempt = attempt
        self.train_py, self.full_profile, data_rel, items_rel = ATTEMPTS[attempt]
        self.mirror = root / "host"
        self.cpython = self.mirror / CPYTHON.lstrip("/")
        self.area5 = self.mirror / AREA5.lstrip("/")
        self.runtime = self.mirror / AREA3V.lstrip("/")
        self.output_root = self.mirror / OUT_ROOT.lstrip("/")
        self.host_work = root / "host-work"
        # the interpreter tree and area 5's venv (the link names the container path, as on the host)
        py = self.cpython / "bin" / "python3.12"
        py.parent.mkdir(parents=True)
        py.write_text("# fixture stand-in for the uv-managed CPython 3.12.16; never executed\n", encoding="utf-8")
        venv = self.area5 / ".venv" / "bin"
        venv.mkdir(parents=True)
        (venv / "python").symlink_to(f"{CPYTHON}/bin/python3.12")
        shutil.copyfile(LOCK_FIXTURE, self.area5 / "uv.lock")
        self.lock = self.area5 / "uv.lock"
        # the area 3 runtime
        rt = self.runtime
        rt.mkdir(parents=True)
        for name, text in SCRIPT_TEXT.items():
            (rt / name).write_text(text, encoding="utf-8")
        self.script = rt / Path(self.train_py).name
        # both attempts' data in every runtime (attempt 2: short-target shard rows, another item set)
        fixture_items = read_jsonl(FG / "items.jsonl")
        for rel, rows in (("items", fixture_items[:n_items]), ("items-short", fixture_items[4:n_items][::-1])):
            (rt / rel).mkdir()
            write_jsonl(rt / rel / "items.jsonl", rows)
            write_json(rt / rel / "item_set_meta.json", {"schema": "ptc_mm.sd35_ocr_item_set", "n_items": len(rows)})
        (rt / "prompts").mkdir()
        shutil.copyfile(FG / "train.jsonl", rt / "prompts" / "train.jsonl")
        (rt / "prompts-short").mkdir()
        write_jsonl(rt / "prompts-short" / "train.jsonl",
                    [r for r in read_jsonl(FG / "train.jsonl") if len(r["target"].split()) <= 2])
        self.data = rt / data_rel
        self.items_dir = rt / items_rel
        all_items = read_jsonl(self.items_dir / "items.jsonl")
        prompts = read_jsonl(self.data / "train.jsonl")
        for row, item in zip(prompts[:overlap], all_items):          # exact overlap: a shard prompt equals an item's
            row["prompt"] = item["prompt"]
        if overlap:
            write_jsonl(self.data / "train.jsonl", prompts)
        self.train_rows = prompts
        self.item_rows = all_items
        self.items = [r["item"] for r in all_items]
        self.base = build_base(rt / BASE_REL)
        write_json(rt / "models" / "MODELS.json", models_json(self.base))
        (root / "trainer.yaml").write_text("# area 3 video trainer config (fixture)\nbackend: flow_grpo_wan\n"
                                           "profile: full\n", encoding="utf-8")
        gpu_step_script(self.host_work)
        (self.host_work / "gpu-log.md").write_bytes((EXT / "gpu-log.fixture.md").read_bytes())
        runs = self.output_root / "area3-video"
        s = settings(num_inner_epochs=inner_epochs, **({"prompts_per_epoch": 8} if attempt == 2 else {}))
        self.stage_out, self.state = build_run(runs / f"full-{attempt}", runtime=rt, base=self.base,
                                               script_sha256=file_sha256(self.script), profile=self.full_profile,
                                               s=s, plan=plan or (DEFAULT_PLAN_V2 if attempt == 2 else None),
                                               data_rel=data_rel)
        self.cuts = sorted(self.stage_out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        self.final_cut = self.cuts[-1]
        self.base_model = manifest_hash(self.base)
        self.cand_model = lora_model_hash(self.final_cut, self.base)
        self.item_set_hash = manifest_hash(self.items_dir)
        self.render()
        self.base_eval = build_eval(runs / "eval-base", self.item_rows, baseline_score, tag="base",
                                    model=self.base_model, base_model=self.base_model, item_set=self.item_set_hash,
                                    filled=filled)
        self.cand_eval = build_eval(runs / "eval-cand", self.item_rows, candidate_score, tag="cand",
                                    model=self.cand_model, base_model=self.base_model, item_set=self.item_set_hash,
                                    filled=filled)
        self.baseline = self.base_eval / f"arm-{METRIC}"
        self.candidate = self.cand_eval / f"arm-{METRIC}"
        rec = self.launch_record()
        self.launch = write_canonical(root / "launch.json", rec)
        self.label = rec["label"]
        self.smoke = write_canonical(root / "smoke.json", smoke_record(base_model=self.base_model,
                                                                         item_set=self.item_set_hash))
        self.guard_log()

    # --- paths and environment ---

    def input_roots(self) -> list[Path]:
        return [self.mirror / PYTHON_DIR.lstrip("/"), self.area5, self.runtime, self.output_root / "area3-video"]

    def env(self) -> dict[str, str]:
        return {ENV_HOST_WORK: str(self.host_work), ENV_OUTPUT_ROOT: str(self.output_root),
                ENV_INPUT_ROOTS: os.pathsep.join(str(p) for p in self.input_roots()),
                ENV_MIRROR_ROOT: str(self.mirror), ENV_INTERPRETER_TREES: str(self.cpython)}

    def config_sha(self) -> str:
        return host_config_sha256(self.host_work, self.output_root, self.input_roots(), mirror_root=self.mirror,
                                  interpreter=INTERPRETER, interpreter_trees=[self.cpython])

    def guard_log(self) -> None:
        lines = [f"2026-11-11T09:00:00Z START {self.label} attempt=1",
                 f"2026-11-11T10:00:00Z END {self.label} attempt=1 rc=0"]
        (self.host_work / "gpu-guard.log").write_text("".join(l + "\n" for l in lines), encoding="utf-8")

    @property
    def paths(self) -> dict[str, Path]:
        return {"spec": self.root / "spec.yaml", "registration": self.root / "registration.yaml"}

    def render(self, reg_mutate: Callable[[dict], None] | None = None,
               spec_mutate: Callable[[dict], None] | None = None) -> None:
        text = (FGW / "registration.template.yaml").read_text(encoding="utf-8")
        text = text.replace("{item_set_hash}", self.item_set_hash).replace("{sampler}", json.dumps(SAMPLER))
        text = text.replace("{evaluator_hash}", sha_of({"evaluator": EVALUATOR_ID, "metric": METRIC,
                                                         "direction": "higher_is_better", "scorer": SCORER}))
        if reg_mutate is not None:
            reg = yaml.safe_load(text)
            reg_mutate(reg)
            text = yaml.safe_dump(reg, sort_keys=False)
        (self.root / "registration.yaml").write_text(text, encoding="utf-8")
        spec = (FGW / "spec.template.yaml").read_text(encoding="utf-8")
        spec = spec.replace("{registration_sha256}", file_sha256(self.root / "registration.yaml")[len("sha256:"):])
        spec = spec.replace("{base}", str(self.base)).replace("{data}", str(self.data))
        spec = spec.replace("{items}", str(self.items_dir))
        if spec_mutate is not None:
            doc = yaml.safe_load(spec)
            spec_mutate(doc)
            spec = yaml.safe_dump(doc, sort_keys=False)
        (self.root / "spec.yaml").write_text(spec, encoding="utf-8")

    # --- launch ---

    def mounts(self, out: Path, cut: Path | None = None) -> list[dict]:
        ms = [{"host": str(self.mirror / PYTHON_DIR.lstrip("/")), "container": PYTHON_DIR, "mode": "ro"},
              {"host": str(self.area5), "container": AREA5, "mode": "ro"},
              {"host": str(self.runtime), "container": AREA3V, "mode": "ro"}]
        if cut is not None:
            ms.append({"host": str(cut), "container": "/inputs/cut", "mode": "ro"})
        return ms + [{"host": str(out), "container": "/out", "mode": "rw"}]

    def argv(self, profile: str | None = None, seed: int = SEED) -> list[str]:
        """The training argv of this scenario's attempt; `profile` is the script's (default its full profile)."""
        return [INTERPRETER, self.train_py, "--profile", profile or self.full_profile, "--out", "/out", "--seed",
                str(seed)]

    def launch_record(self, **overrides) -> dict[str, Any]:
        """ext-launch's record of the full training run (README "Smoke launch" with --profile full)."""
        rec = {"schema": "gatedtrain.launch_record", "schema_version": "2.0", "tags": ["prototype", "unpromotable"],
               "adapter": ADAPTER, "area": AREA, "profile": "full", "label": LABEL, "host_work": str(self.host_work),
               "runtime": {"kind": "venv_in_container", "image": IMAGE, "venv_lock_sha256": file_sha256(self.lock)},
               "framework": {"name": "diffusers", "version": "0.42.1", "commit": None},
               "training_seed": SEED, "gpus": True, "argv": self.argv() + ["--epochs", "3"],
               "mounts": self.mounts(self.stage_out), "tmpfs": ["/tmp"], "network": "none", "network_reason": None,
               "user": run_user(), "env_names": list(ENV_NAMES), "gpu_est_h": 3.5, "gpu_used_h": 6.0,
               "output_dir": str(self.stage_out), "output_root": str(self.output_root),
               "gpu_step_sha256": file_sha256(self.host_work / "bin" / "gpu-step.sh"), "allow_network": False,
               "interpreter_path": f"{CPYTHON}/bin/python3.12",
               "interpreter_sha256": file_sha256(self.cpython / "bin" / "python3.12"),
               "script_path": self.train_py, "script_sha256": file_sha256(self.script),
               "host_config_sha256": self.config_sha()}
        rec.update(overrides)
        rec["label"] = launch_label(rec)
        return rec

    def rewrite_launch(self, **overrides) -> Path:
        rec = self.launch_record(**overrides)
        p = write_canonical(self.root / "launch.json", rec)
        self.label = rec["label"]
        self.guard_log()
        return p

    def new_out(self, name: str) -> Path:
        p = self.output_root / "area3-video" / name
        p.mkdir(parents=True)
        return p

    def request(self, out: Path, *, profile: str = "smoke", **overrides) -> dict[str, Any]:
        """An ext-launch request of the trainer into `out` (README "Smoke launch")."""
        req = {"area": AREA, "label": f"mm-a3-video-{profile}", "host_work": str(self.host_work),
               "runtime": {"kind": "venv_in_container", "image": IMAGE, "venv_lock": str(self.lock)},
               "framework": {"name": "diffusers", "version": "0.42.1", "commit": None}, "training_seed": SEED,
               "gpus": True, "network": "none", "network_reason": None, "user": run_user(), "env": dict(ENV),
               "mounts": self.mounts(out), "tmpfs": ["/tmp"], "output_dir": str(out), "gpu_est_h": 0.3,
               "argv": self.argv(profile if self.attempt == 1 else {"smoke": "smoke2", "full": "full2"}[profile])}
        req.update(overrides)
        return req

    def eval_request(self, out: Path, *, cut: Path | None = None, limit: int | None = 40, **overrides) -> dict[str, Any]:
        """An evaluation request (eval_videos.py, one GPU job: generation and both OCR engines); a cut is mounted
        read-only at /inputs/cut (area 5's eval-cand-1 shape)."""
        argv = [INTERPRETER, EVAL_PY] + (["--checkpoint", "/inputs/cut"] if cut is not None else ["--base-only"])
        argv += ["--out", "/out", "--eval-seeds", "11"] + (["--limit", str(limit)] if limit else [])
        if self.attempt == 2:
            argv += ["--items", f"{AREA3V}/items-short"]
        req = self.request(out, label="mm-a3-video-eval", env=dict(EVAL_ENV), mounts=self.mounts(out, cut),
                           gpu_est_h=0.1, argv=argv)
        req.update(overrides)
        return req

    def launch_request(self, req: dict, name: str, *, profile: str = "smoke", smoke: Path | None = None,
                       extra: list[str] = ()):
        path = self.root / f"{name}.request.yaml"
        path.write_text(yaml.safe_dump(req, sort_keys=False), encoding="utf-8")
        out = self.root / f"{name}.launch.json"
        args = ["ext-launch", "--adapter", ADAPTER, "--profile", profile, "--request", str(path), "--gpu-log",
                str(self.host_work / "gpu-log.md"), "--out-record", str(out)]
        if smoke is not None:
            args += ["--smoke-record", str(smoke)]
        return run_cli(args + list(extra), cwd=self.root, env=self.env()), out

    def recheck(self, record: Path, extra: list[str] = ()):
        return run_cli(["ext-launch", "--adapter", ADAPTER, "--recheck", str(record), *extra], cwd=self.root,
                       env=self.env())

    # --- edits and collect ---

    def edit_json(self, path: Path, fn: Callable[[dict], None]) -> None:
        doc = json.loads(path.read_text(encoding="utf-8"))
        fn(doc)
        write_json(path, doc)

    def logs(self) -> tuple[list[dict], list[dict], list[dict]]:
        return (read_jsonl(self.stage_out / "samples.jsonl"), read_jsonl(self.stage_out / "epoch_plan.jsonl"),
                read_jsonl(self.stage_out / "train_steps.jsonl"))

    def edit_logs(self, fn: Callable[[list[dict], list[dict], list[dict]], None]) -> None:
        samples, plans, steps = self.logs()
        fn(samples, plans, steps)
        write_jsonl(self.stage_out / "samples.jsonl", samples)
        write_jsonl(self.stage_out / "epoch_plan.jsonl", plans)
        write_jsonl(self.stage_out / "train_steps.jsonl", steps)

    def reseal(self, cut: Path) -> None:
        """Rewrite cut.json's file hashes after a cut file was edited (the cut stays complete)."""
        self.edit_json(cut / "cut.json", lambda d: d.update(files={n: file_sha256(cut / n) for n in CUT_FILES}))

    def collect(self, runs_root: Path, *, baseline: Path | None = None, candidate: Path | None = None,
                launch: Path | None = None, smoke: Path | None = None, timeout: int = 300):
        args = ["ext-collect", str(self.paths["spec"]), "--adapter", ADAPTER, "--stage-out", str(self.stage_out),
                "--launch-record", str(launch or self.launch), "--smoke-record", str(smoke or self.smoke),
                "--eval-baseline", str(baseline or self.baseline), "--eval-candidate", str(candidate or self.candidate),
                "--runs-root", str(runs_root)]
        return run_cli(args, cwd=self.root, timeout=timeout, env=self.env())

    def fill(self, arm: Path, checkpoint: Path, *, base: Path | None = None):
        args = ["ext-arm-fill", "--adapter", ADAPTER, "--checkpoint", str(checkpoint), "--items", str(self.items_dir)]
        if base is not None:
            args += ["--base", str(base)]
        return run_cli(args + [str(arm)], cwd=self.root, env=self.env())


def probe_copy(dst: Path) -> tuple[Path, Path]:
    """(a writable copy of `fixtures/flow_grpo_wan/probe-train-1/`, a prompt directory holding the first 64 shard
    rows): the GPU smoke's logs against the rows it read (epochs 1 and 2 use rows 0 .. 7)."""
    out = dst / "probe-train-1"
    shutil.copytree(PROBE, out)
    data = dst / "prompts"
    data.mkdir()
    shutil.copyfile(FG / "train.jsonl", data / "train.jsonl")
    return out, data


def shared_sampler_sha256(script: Path) -> str:
    """Both scripts' `sampler_code_sha256`: SHA-256 of the text from `# BEGIN SHARED SAMPLER` up to `# END SHARED
    SAMPLER`."""
    text = script.read_text(encoding="utf-8")
    return sha256_text(text[text.index("# BEGIN SHARED SAMPLER"):text.index("# END SHARED SAMPLER")].encode("utf-8"))
