"""The runtime scripts that run inside the GPU container (runtime/train_flow_grpo_wan.py and its attempt-2 copy
train_flow_grpo_wan_v2.py, runtime/eval_videos.py, runtime/make_short_split.py), on CPU and without torch, diffusers,
PaddleOCR or docTR, in CI: the video OCR reward against the adapter's own re-derivation, the deciding metric, the
settings, the file-order prompts, the advantages and the zero-advantage rules, the cut layout and the run files,
checked against what the adapter reads at collect and against the committed outputs of the GPU smoke runs
(fixtures/flow_grpo_wan/probe-train-1 and probe-base-1) and the registrations.

Why: these scripts run only on a GPU, so a fault in a part that needs no GPU was found only after GPU time, or never
(a wrong reward trains silently). runtime/tests/ holds deeper tests of the same scripts, but they need the SD3.5 OCR
recipe's runtime venv (torch, diffusers, PaddleOCR) and a flow_grpo clone, so CI does not run them.

The scripts import only the standard library at module level, so they load as they are. The edit distance comes from
rapidfuzz, which CI does not install: when it is missing, `rapidfuzz.distance.Levenshtein.distance` is a plain
dynamic-programming Levenshtein distance defined here. Nothing in runtime/ is changed: the scripts' SHA-256s are
recorded in RESULT.md, eval_videos.py's is the registered scorer pin and the shared sampler block's is the adapter's
SAMPLER_CODE_SHA256.
"""

from __future__ import annotations

import ast
import contextlib
import importlib.machinery
import importlib.util
import io
import json
import os
import random
import re
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import types
import unittest
import unittest.mock
from pathlib import Path

import yaml

from gatedtrain.ext.sdk import compact_sha256, derive_seed as sdk_derive_seed, safetensors_problem
from gatedtrain.families.flow_grpo import REFERENCE_LINE_RE

from gatedtrain_recipe_flow_grpo_wan_video import adapter as A
from gatedtrain_recipe_flow_grpo_wan_video import rows as R

REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "runtime"
FW = Path(__file__).resolve().parent / "fixtures" / "flow_grpo_wan"
TRAIN_RUN = FW / "probe-train-1"
BASE_EVAL = FW / "probe-base-1"
SHARD_HEAD = Path(__file__).resolve().parent / "fixtures" / "flow_grpo" / "train.jsonl"   # first 64 shard rows
GATE = REPO / "results" / "2026-11-15" / "gate"


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


HAVE_RAPIDFUZZ = importlib.util.find_spec("rapidfuzz") is not None


def rapidfuzz_stand_in() -> dict:
    """`rapidfuzz.distance.Levenshtein.distance` as the scripts import it (inside the reward functions), backed by
    levenshtein() above; each module has a spec, as an imported module does."""
    mods = {n: types.ModuleType(n) for n in ("rapidfuzz", "rapidfuzz.distance", "rapidfuzz.distance.Levenshtein")}
    for name, m in mods.items():
        m.__spec__ = importlib.machinery.ModuleSpec(name, None)
    mods["rapidfuzz.distance.Levenshtein"].distance = levenshtein
    mods["rapidfuzz"].distance = mods["rapidfuzz.distance"]
    mods["rapidfuzz.distance"].Levenshtein = mods["rapidfuzz.distance.Levenshtein"]
    return mods


_STAND_INS: dict = {}


def setUpModule():
    # installed for this module's tests only; tearDownModule removes exactly these entries, so no other test module
    # (or a later find_spec/import) sees a fake rapidfuzz
    if not HAVE_RAPIDFUZZ:
        _STAND_INS.update(rapidfuzz_stand_in())
        sys.modules.update(_STAND_INS)


def tearDownModule():
    for name, m in _STAND_INS.items():
        if sys.modules.get(name) is m:
            del sys.modules[name]
    _STAND_INS.clear()


def load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


T = load_script(RUNTIME / "train_flow_grpo_wan.py", "_wan_runtime_train")
T2 = load_script(RUNTIME / "train_flow_grpo_wan_v2.py", "_wan_runtime_train_v2")
E = load_script(RUNTIME / "eval_videos.py", "_wan_runtime_eval")
M = load_script(RUNTIME / "make_short_split.py", "_wan_runtime_short")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def help_options(parse_args) -> set[str]:
    buf = io.StringIO()
    with contextlib.suppress(SystemExit), contextlib.redirect_stdout(buf):
        parse_args(["--help"])
    return set(re.findall(r"(--[a-z][a-z-]*)", buf.getvalue())) - {"--help"}


def sample_line(T_, x_rewards: dict, *, epoch, k, sid, row, g, seed, attempt, nonfinite=False) -> dict:
    """A samples.jsonl line as train_flow_grpo_wan.py main writes it (:924-932) from PaddleOcrA.video_reward's result."""
    errs = [f for f in x_rewards["frames"] if f["error"]]
    return {"epoch": epoch, "sample_index": k, "sample_id": sid, "prompt_id": row["prompt_id"], "prompt": row["prompt"],
            "target": T_.target_text(row["prompt"]), "member": g, "seed": seed, "frame_ocr": x_rewards["frames"],
            "ocr_error": f"frame {errs[0]['frame']}: {errs[0]['error']}"[:300] if errs else None,
            "ocr_acc": x_rewards["reward"], "reward": x_rewards["reward"], "nonfinite": nonfinite, "attempt": attempt,
            "utc": T_.utc_now()}


class FakeTransformer:
    def __init__(self, order_seed: int = 0):
        self.order_seed = order_seed

    def save_lora_adapter(self, save_directory, adapter_name, safe_serialization, weight_name):
        rnd = random.Random(self.order_seed)
        targets = list(T.LORA["target_modules"])
        rnd.shuffle(targets)
        names = [f"blocks.0.attn{a}.to_q.lora_{ab}.weight" for a in (1, 2) for ab in "AB"]
        tensors = {n: {"dtype": "F32", "shape": [2], "data_offsets": [8 * i, 8 * i + 8]} for i, n in enumerate(names)}
        keys = list(tensors)
        rnd.shuffle(keys)
        meta = {"format": "pt", "lora_adapter_metadata": json.dumps({"r": 32, "target_modules": targets})}
        text = json.dumps({"__metadata__": meta, **{k: tensors[k] for k in keys}}, separators=(",", ":")).encode()
        path = Path(save_directory) / weight_name
        path.write_bytes(struct.pack("<Q", len(text)) + text + bytes(range(8 * len(names))))
        os.chmod(path, 0o600)


class Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="wan-runtime-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def patch(self, obj, name, value):
        old = getattr(obj, name)
        setattr(obj, name, value)
        self.addCleanup(setattr, obj, name, old)

    def base(self) -> Path:
        d = self.tmp / "base"
        files = {"model_index.json": b"{}\n", "transformer/diffusion_pytorch_model.safetensors": b"weights",
                 "scheduler/scheduler_config.json": b"{}\n"}
        for rel, data in {**files, ".cache/huggingface/x.lock": b""}.items():
            (d / rel).parent.mkdir(parents=True, exist_ok=True)
            (d / rel).write_bytes(data)
        record = {"repo": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "revision": "0" * 39 + "1",
                  "files": {rel: {"sha256": T.sha256_bytes(data)} for rel, data in files.items()}}
        mj = self.tmp / "MODELS.json"
        mj.write_text(json.dumps({"models": {T.BASE_NAME: record}}), encoding="utf-8")
        for mod in (T, T2, E):
            self.patch(mod, "MODELS_JSON", mj)
        return d


# ---------------------------------------------------------------------------
# The training reward (flow_grpo video_ocr) and the adapter's re-derivation of it
# ---------------------------------------------------------------------------

class TestVideoReward(unittest.TestCase):
    def test_frame_accuracy_is_the_adapters(self):
        rnd = random.Random(5)
        alpha = "abAB 1,é"
        for _ in range(400):
            a = "".join(rnd.choice(alpha) for _ in range(rnd.randint(0, 12)))
            b = "".join(rnd.choice(alpha) for _ in range(rnd.randint(1, 8)))
            if not b.replace(" ", ""):
                continue
            got = T.video_frame_accuracy(a, b)
            self.assertEqual(got, A.video_frame_accuracy(a, b), (a, b))   # collect re-derives it exactly
            self.assertEqual(got, E.video_frame_accuracy(a, b))
            self.assertEqual(got, T2.video_frame_accuracy(a, b))
            ra, rb = a.replace(" ", "").lower(), b.replace(" ", "").lower()
            self.assertEqual(got, 1 - min(levenshtein(ra, rb), len(rb)) / len(rb))
        # no substring shortcut, unlike the image reward
        self.assertEqual(T.video_frame_accuracy("xx open yy", "open"), 0.0)
        self.assertEqual(T.video_frame_accuracy("OPEN", "open"), 1.0)

    def test_video_reward_is_the_mean_of_positive_frames(self):
        for scores in ([0.0, 0.5, 0.25, 0.0, 1.0], [0.0] * 5, [0.1], [], [1 / 3] * 5):
            want = sum(s for s in scores if s > 0) / max(1, sum(s > 0 for s in scores))
            self.assertEqual(T.video_ocr_reward(scores), want)
            self.assertEqual(T.video_ocr_reward(scores), A.video_ocr_reward(scores))

    def test_recorded_training_rewards_are_reproduced(self):
        # The GPU smoke's 32 samples: every frame accuracy and every video reward from the recorded OCR texts.
        samples = read_jsonl(TRAIN_RUN / "samples.jsonl")
        self.assertEqual(len(samples), 32)
        for x in samples:
            self.assertEqual(x["target"], T.target_text(x["prompt"]))
            self.assertEqual([f["frame"] for f in x["frame_ocr"]], list(range(0, 17, T.FRAME_INTERVAL_A)))
            for f in x["frame_ocr"]:
                self.assertEqual(f["acc"], T.video_frame_accuracy(f["text"], x["target"]))
            self.assertEqual(x["reward"], T.video_ocr_reward([f["acc"] for f in x["frame_ocr"]]))
            self.assertEqual(x["reward"], x["ocr_acc"])

    def test_paddle_engine_frames_and_error_path(self):
        # PaddleOcrA.video_reward with the recognizer stubbed: frames 0, 4, ..., 16 of a 17-frame video; a failing
        # frame scores 0 with no text; the line main writes from it passes the adapter's frame rule.
        texts = {0: "OPEN", 4: "0PEN", 8: None, 12: "", 16: "open late"}

        def text(frame):
            if texts[frame] is None:
                raise RuntimeError("paddle failed")
            return texts[frame]

        eng = object.__new__(T.PaddleOcrA)
        eng.text = text
        row = {"prompt_id": "ocr-train-00001", "prompt": 'a sign that reads "Open"'}
        res = eng.video_reward(list(range(17)), row["prompt"])
        self.assertEqual([f["frame"] for f in res["frames"]], [0, 4, 8, 12, 16])
        self.assertEqual([f["acc"] for f in res["frames"]], [1.0, 0.75, 0.0, 0.0, 0.0])
        self.assertEqual(res["frames"][2], {"frame": 8, "text": None, "acc": 0.0, "error": "RuntimeError: paddle failed"})
        self.assertEqual(res["reward"], (1.0 + 0.75) / 2)
        x = sample_line(T, res, epoch=1, k=0, sid="sha256:x", row=row, g=0, seed=1, attempt=1)
        s = T.resolve_settings(T.parse_args(["--profile", "smoke", "--out", "x"]))
        self.assertEqual(A.FlowGrpoWanAdapter()._reward_field_problems(x, 1, s, True), [])
        self.assertEqual(x["ocr_error"], "frame 8: RuntimeError: paddle failed")

    def test_every_prompt_has_a_scorable_target(self):
        prompts = [r["prompt"] for d in ("items", "items-short") for r in read_jsonl(RUNTIME / d / "items.jsonl")]
        prompts += [r["prompt"] for r in read_jsonl(SHARD_HEAD)]
        for p in prompts:
            t = T.target_text(p)
            self.assertEqual(t, E.target_text(p))
            self.assertEqual(t, M.target_text(p))
            self.assertTrue(t.replace(" ", ""), p)
        self.assertEqual(T.target_text('"A" and "B"'), "A")
        self.assertEqual(M.target_text('x " A b " y'), " A b ")


class TestDecidingMetric(unittest.TestCase):
    def test_substring_distance_and_video_mean(self):
        rnd = random.Random(6)
        for _ in range(200):
            text = "".join(rnd.choice("abc") for _ in range(rnd.randint(0, 9)))
            pat = "".join(rnd.choice("abc") for _ in range(rnd.randint(0, 6)))
            subs = {text[i:j] for i in range(len(text) + 1) for j in range(i, len(text) + 1)}
            self.assertEqual(E.substring_edit_distance(text, pat), min(levenshtein(s, pat) for s in subs))
        self.assertEqual(E.ocr_substring_accuracy("noise OPEN late", "open late"), 1.0)
        self.assertEqual(E.ocr_substring_accuracy("", "open"), 0.0)
        with self.assertRaises(ValueError):
            E.ocr_substring_accuracy("x", " ")
        # the mean over all frames, frames that read nothing included as 0
        self.assertEqual(E.video_substring_accuracy(["open", "", "opan", "zzzz"], "open"), (1 + 0 + 0.75 + 0) / 4)
        with self.assertRaises(ValueError):
            E.video_substring_accuracy([], "open")

    def test_recorded_base_evaluation_statistics(self):
        # probe-base-1: summary.json metrics and the MM-38 checks, from its arms' item scores.
        summary = json.loads((BASE_EVAL / "summary.json").read_text(encoding="utf-8"))
        for metric in ("ocr_a_acc", "ocr_b_acc"):
            scores = [x["score"] for x in read_jsonl(BASE_EVAL / f"arm-{metric}" / "items.jsonl")]
            self.assertEqual(E.stats(scores), summary["metrics"][metric])
            want = {k: v for k, v in summary["mm38"][metric].items() if k != "role"}
            self.assertEqual(E.mm38_checks(scores, [0, 1], 0), want)
        def failed(scores, errors=0):
            return sorted(k for k, ok in E.mm38_checks(scores, [0, 1], errors)["checks"].items() if not ok)

        self.assertEqual(failed([0.1, 0.2, 0.3, 0.4]), [])
        self.assertEqual(failed([0.0] * 10 + [0.5] * 9 + [0.3]), ["mode_share_below_0.5"])
        self.assertEqual(failed([0.2, 0.3, 0.4], errors=1), ["zero_errors"])
        self.assertEqual(failed([0.01, 0.02, 0.03, 0.04]), ["mean_inside_by_5pct"])     # mean within 5 % of 0
        self.assertEqual(failed([0.96, 0.97, 0.98, 0.99]), ["mean_inside_by_5pct"])
        self.assertEqual(failed([0.06, 0.07, 0.08]), [])
        self.assertIn("sd_positive", failed([0.3, 0.3, 0.3]))
        self.assertFalse(E.mm38_checks([0.2, 0.3], [0, 1], 1)["pass"])

    def test_metrics_are_the_registered_evaluators(self):
        for metric, spec in E.METRICS.items():
            row = R.EVALUATOR_ROWS[spec[0]]
            self.assertEqual((row["metric"], row["direction"], list(row["score_range"])), (metric, spec[1], spec[2]))


# ---------------------------------------------------------------------------
# Settings, prompts, sampler, shared code
# ---------------------------------------------------------------------------

class TestSettings(unittest.TestCase):
    def settings(self, mod, *argv):
        return mod.resolve_settings(mod.parse_args(["--out", "/tmp/x", *argv]))

    def test_smoke_settings_are_the_recorded_runs(self):
        ra = json.loads((TRAIN_RUN / "run_args.json").read_text(encoding="utf-8"))
        self.assertEqual(self.settings(T, "--profile", "smoke"), ra["settings"])

    def test_flow_grpo_wan_settings(self):
        # flow_grpo config/grpo.py general_ocr_wan2_1, with the recorded deviations (17 frames, EMA off, file order).
        s = self.settings(T, "--profile", "full")
        want = {"num_steps": 20, "deploy_num_steps": 20, "guidance_scale": 4.5, "height": 240, "width": 416,
                "num_frames": 17, "max_sequence_length": 512, "learning_rate": 1.0e-4, "clip_range": 1.0e-3,
                "beta": 0.004, "global_std": False, "ema": False, "timestep_fraction": 0.99, "num_train_timesteps": 19,
                "num_batches_per_epoch": 2, "gradient_accumulation_steps": 1, "reward": "video_ocr",
                "prompt_order": "file", "precision": "bf16", "epochs": 60, "save_every": 15, "prompts_per_epoch": 4,
                "images_per_prompt": 4, "samples_per_epoch": 16}
        self.assertEqual({k: s[k] for k in want}, want)
        self.assertEqual(sorted(T.LORA["target_modules"]), sorted(E.WAN_LORA_TARGETS))
        self.assertEqual(sorted(T.LORA["target_modules"]), sorted(A.WAN_LORA_TARGETS))
        self.assertEqual((T.LORA["r"], T.LORA["lora_alpha"]), (32, 64))

    def test_attempt_two_profiles(self):
        s = self.settings(T2, "--profile", "full2")
        self.assertEqual((s["epochs"], s["save_every"], s["prompts_per_epoch"], s["images_per_prompt"],
                          s["samples_per_epoch"], s["num_batches_per_epoch"], s["gradient_accumulation_steps"]),
                         (100, 25, 8, 4, 32, 2, 1))
        self.assertEqual(self.settings(T2, "--profile", "smoke2")["epochs"], 2)
        self.assertEqual(T2.DEFAULT_PROMPTS.name, "prompts-short")
        common = {k: v for k, v in self.settings(T, "--profile", "full").items() if k not in
                  ("epochs", "save_every", "prompts_per_epoch", "samples_per_epoch")}
        self.assertEqual({k: s[k] for k in common}, common)

    def test_v2_differs_from_v1_only_where_recorded(self):
        # train_flow_grpo_wan_v2.py header: only the profiles, FULL_EPOCHS and the default prompt directory differ
        # (docstrings aside); every function and every other module constant is the same code.
        def parts(path):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            out = {}
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    body = node.body[1:] if (node.body and isinstance(node.body[0], ast.Expr)
                                             and isinstance(node.body[0].value, ast.Constant)) else node.body
                    out[node.name] = ast.dump(ast.Module(body=body, type_ignores=[]))
                elif isinstance(node, ast.Assign):
                    out[ast.unparse(node.targets[0])] = ast.dump(node.value)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    out[ast.unparse(node)] = ""
            return out

        a, b = parts(RUNTIME / "train_flow_grpo_wan.py"), parts(RUNTIME / "train_flow_grpo_wan_v2.py")
        self.assertEqual(set(a), set(b))
        self.assertEqual({k for k in a if a[k] != b[k]}, {"PROFILES", "FULL_EPOCHS", "DEFAULT_PROMPTS"})

    def test_refused_settings(self):
        for argv in (["--epochs", "0"], ["--train-batch", "0"], ["--sample-batch", "-1"]):
            with self.assertRaises(SystemExit):
                self.settings(T, "--profile", "smoke", *argv)
        with unittest.mock.patch.dict(T.COMMON, {"ema": True}), self.assertRaises(SystemExit):
            self.settings(T, "--profile", "smoke")
        with unittest.mock.patch.dict(T.COMMON, {"precision": "fp16"}), self.assertRaises(SystemExit):
            self.settings(T, "--profile", "smoke")
        with unittest.mock.patch.dict(T.COMMON, {"num_batches_per_epoch": 3}), self.assertRaises(SystemExit):
            self.settings(T, "--profile", "smoke")                   # 16 % 3

    def test_adapter_knows_every_option(self):
        # The adapter's argv rules resolve options as argparse does; an option it does not know is unchecked.
        for script, mod in ((A.TRAIN_SCRIPTS[0], T), (A.TRAIN_SCRIPTS[1], T2), (A.EVAL_SCRIPT, E)):
            self.assertEqual(help_options(mod.parse_args), set(A.OPTIONS[script]), script)
        self.assertEqual(set(A.SCRIPT_PROFILES[A.TRAIN_SCRIPTS[0]].values()), set(T.PROFILES))
        self.assertEqual(set(A.SCRIPT_PROFILES[A.TRAIN_SCRIPTS[1]].values()), set(T2.PROFILES))

    def test_epoch_prompts_in_file_order(self):
        rows = read_jsonl(SHARD_HEAD)
        self.assertEqual([r["prompt_id"] for r in T.epoch_prompts(rows, 1, 4)], [r["prompt_id"] for r in rows[:4]])
        self.assertEqual([r["prompt_id"] for r in T.epoch_prompts(rows, 3, 4)], [r["prompt_id"] for r in rows[8:12]])
        wrap = T.epoch_prompts(rows[:6], 2, 4)
        self.assertEqual([r["prompt_id"] for r in wrap], [rows[i]["prompt_id"] for i in (4, 5, 0, 1)])
        # The adapter's rule, on the recorded GPU smoke.
        ra = json.loads((TRAIN_RUN / "run_args.json").read_text(encoding="utf-8"))
        s = ra["settings"]
        for x in read_jsonl(TRAIN_RUN / "samples.jsonl"):
            plan = T.epoch_prompts(rows, x["epoch"], s["prompts_per_epoch"])
            self.assertEqual(x["prompt_id"], plan[x["sample_index"] // s["images_per_prompt"]]["prompt_id"])

    def test_samplers_are_the_recorded_and_registered_ones(self):
        ra = json.loads((TRAIN_RUN / "run_args.json").read_text(encoding="utf-8"))
        smp = ra["sampler"]
        cfg = dict(smp["scheduler_config"])
        cfg["_use_default_values"] = list(reversed(cfg.get("_use_default_values", [])))
        c = T.COMMON
        doc = T.sampler_doc(c["deploy_num_steps"], c["guidance_scale"], c["height"], c["width"], c["num_frames"],
                            c["max_sequence_length"], cfg, smp["timesteps"], smp["sigmas"], smp["step_index"],
                            c["precision"], c["precision"], deterministic=True)
        self.assertEqual(doc, smp)
        self.assertEqual(T.sha256_bytes(T.canonical(doc)), ra["sampler_sha256"])
        for name in ("registration-gatedtrain.yaml", "registration-v2-gatedtrain.yaml"):
            self.assertEqual(yaml.safe_load((GATE / name).read_text(encoding="utf-8"))["sampler"], doc, name)
        meta = json.loads((BASE_EVAL / "arm-ocr_b_acc" / "eval_meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["sampler"], doc)
        self.assertEqual({k: doc[k] for k in E.SAMPLER_DEFAULT}, E.SAMPLER_DEFAULT)

    def test_shared_code_and_pins(self):
        def block(path):
            text = path.read_text(encoding="utf-8")
            return text[text.index("# BEGIN SHARED SAMPLER"):text.index("# END SHARED SAMPLER")]

        b = block(RUNTIME / "eval_videos.py")
        self.assertEqual(block(RUNTIME / "train_flow_grpo_wan.py"), b)
        self.assertEqual(block(RUNTIME / "train_flow_grpo_wan_v2.py"), b)
        self.assertEqual(T.sampler_code_sha256(), A.SAMPLER_CODE_SHA256)
        self.assertEqual(E.sampler_code_sha256(), A.SAMPLER_CODE_SHA256)
        self.assertEqual(T.SAMPLE_ID_DOMAIN, A.FlowGrpoWanAdapter.sample_id_domain)
        self.assertEqual(T.sha256_file(RUNTIME / "eval_videos.py"), R.EVAL_VIDEOS_SHA256)
        for parts in [("sample", 5000, 1, 0), ("eval", 11, "ocr-test-0001"), ("mask", 5000, 2)]:
            self.assertEqual(T.derive_seed(*parts), sdk_derive_seed(*parts))


# ---------------------------------------------------------------------------
# Advantages and the zero-advantage rules (flow_grpo train_wan2_1.py:803-873)
# ---------------------------------------------------------------------------

def zero_advantage_plan(T_, s, sample_ids, prompt_ids, rewards, pick_seed):
    """train_flow_grpo_wan.py main :936-962 (the mask, the all-zero +1e-6 rule, the padding), with a numpy
    permutation standing in for the torch.randperm of the padding (the adapter does not re-derive it)."""
    import numpy as np

    adv = np.asarray(T_.PerPromptStatTracker(s["global_std"]).update(prompt_ids, rewards), dtype=np.float32)
    nb = s["num_batches_per_epoch"]
    mask = adv != 0
    true_count = int(mask.sum())
    all_zero = true_count == 0
    if all_zero:
        adv = adv + np.float32(1e-6)
        mask = adv != 0
    reincluded = []
    if true_count % nb != 0:
        false_idx = np.where(~mask)[0]
        need = nb - (true_count % nb)
        if len(false_idx) >= need:
            pick = false_idx[np.random.default_rng(pick_seed).permutation(len(false_idx))[:need]]
            mask[pick] = True
            reincluded = sorted(int(x) for x in pick)
    kept = [int(i) for i in np.where(mask)[0]]
    return {"kept": [sample_ids[i] for i in kept], "filtered": [sample_ids[i] for i in range(len(rewards)) if not mask[i]],
            "reincluded": [sample_ids[i] for i in reincluded], "all_zero_advantage": all_zero}


class TestAdvantages(unittest.TestCase):
    def test_group_std_advantages_and_exact_zeros(self):
        import numpy as np

        tr = T.PerPromptStatTracker(False)
        r = [0.1, 0.3, 0.5, 0.9, 0.2, 0.2, 0.2, 0.2]
        adv = tr.update(["a"] * 4 + ["b"] * 4, r)
        a = np.array(r[:4])
        np.testing.assert_allclose(adv[:4], (a - a.mean()) / (a.std() + 1e-4))
        self.assertTrue(all(x == 0.0 for x in adv[4:]))
        rnd = random.Random(7)
        for _ in range(200):                                   # G = 4: a flat group is exactly 0, whatever the value
            v = rnd.random()
            self.assertTrue(all(x == 0.0 for x in tr.update(["f"] * 4 + ["o"] * 4, [v] * 4 + [rnd.random()] * 4)[:4]))

    def test_recorded_epoch_plans_are_reproduced(self):
        # probe-train-1: kept, filtered and the all-zero flag from the recorded rewards (the re-included samples are a
        # seeded torch permutation, so only their number and their zero advantage are compared).
        ra = json.loads((TRAIN_RUN / "run_args.json").read_text(encoding="utf-8"))
        samples = read_jsonl(TRAIN_RUN / "samples.jsonl")
        for p in read_jsonl(TRAIN_RUN / "epoch_plan.jsonl"):
            xs = [x for x in samples if x["epoch"] == p["epoch"]]
            got = zero_advantage_plan(T, ra["settings"], [x["sample_id"] for x in xs], [x["prompt_id"] for x in xs],
                                      [x["reward"] for x in xs], 0)
            self.assertEqual(got["all_zero_advantage"], p["all_zero_advantage"])
            self.assertEqual(len(got["reincluded"]), len(p["reincluded"]))
            self.assertEqual(set(got["filtered"]) | set(got["reincluded"]), set(p["filtered"]) | set(p["reincluded"]))
            for x in xs:
                self.assertEqual(x["seed"], T.derive_seed("sample", ra["seed"], x["epoch"], x["sample_index"]))
                self.assertEqual(x["sample_id"], T.sha256_bytes(T.canonical(
                    [T.SAMPLE_ID_DOMAIN, ra["seed"], x["attempt"], x["epoch"], x["sample_index"], x["prompt_id"],
                     x["seed"], x["member"]])))
        self.assertTrue(any(p["filtered"] for p in read_jsonl(TRAIN_RUN / "epoch_plan.jsonl")))


# ---------------------------------------------------------------------------
# Cuts, base binding, run files, the short split
# ---------------------------------------------------------------------------

class TestCutsAndFiles(Tmp):
    def cut(self, transformer=None):
        base = self.base()
        ref = T.base_binding(base, verify=True)
        ra = json.loads((TRAIN_RUN / "run_args.json").read_text(encoding="utf-8"))
        out = self.tmp / "out"
        out.mkdir(exist_ok=True)
        d = T.save_cut(transformer or FakeTransformer(), out, 2, 4, 28, 4, ref, ra["sampler"],
                       {"r": 32, "target_modules": sorted(T.LORA["target_modules"])},
                       {"script_sha256": "sha256:" + "1" * 64, "sampler_code_sha256": T.sampler_code_sha256(),
                        "seed": 5000, "profile": "smoke", "ema": False})
        return d, ref, base

    def test_cut_is_what_the_adapter_and_the_evaluator_accept(self):
        d, ref, base = self.cut()
        ad = A.FlowGrpoWanAdapter()
        self.assertIsNone(ad.cut_problem(d, 2))
        doc = json.loads((d / "cut.json").read_text(encoding="utf-8"))
        recorded = json.loads((TRAIN_RUN / "checkpoint-2" / "cut.json").read_text(encoding="utf-8"))
        self.assertEqual(sorted(doc), sorted(recorded))
        self.assertEqual(doc["schema"], recorded["schema"])
        self.assertEqual(sorted(doc["deployable"]), sorted(R.TRAINER_ROWS["flow_grpo_wan"]["components"]))
        self.assertEqual(stat.S_IMODE((d / T.LORA_WEIGHTS).stat().st_mode), 0o644)
        self.assertIsNone(safetensors_problem(d / T.LORA_WEIGHTS))
        b = json.loads((d / "base_ref.json").read_text(encoding="utf-8"))
        self.assertEqual(b["files"], ad._base_files(base))
        self.assertEqual(b["files_sha256"], compact_sha256(b["files"]))
        self.assertEqual(E.check_cut(d)["epoch"], 2)

    def test_evaluator_refuses_other_lora_targets_and_sampler_code(self):
        d, _, _ = self.cut()
        cfg = json.loads((d / "adapter_config.json").read_text(encoding="utf-8"))
        self.patch(E, "WAN_LORA_TARGETS", E.WAN_LORA_TARGETS[:-1])
        with self.assertRaises(SystemExit):
            E.check_cut(d)
        setattr(E, "WAN_LORA_TARGETS", tuple(cfg["target_modules"]))
        self.patch(E, "sampler_code_sha256", lambda: "sha256:" + "0" * 64)
        with self.assertRaises(SystemExit):
            E.check_cut(d)

    def test_cut_bytes_do_not_depend_on_the_header_order(self):
        a, _, _ = self.cut(FakeTransformer(1))
        shutil.move(str(a), str(self.tmp / "first"))
        b, _, _ = self.cut(FakeTransformer(8))
        self.assertEqual((self.tmp / "first" / T.LORA_WEIGHTS).read_bytes(), (b / T.LORA_WEIGHTS).read_bytes())

    def test_run_files(self):
        out = self.tmp / "out"
        self.assertEqual(T.prepare_out(out), 1)
        rt = T.write_env_freeze(out)
        lines = (out / "env-freeze.txt").read_text(encoding="utf-8").splitlines()
        self.assertTrue(any(REFERENCE_LINE_RE.fullmatch(x) for x in lines))      # the family's freeze rule
        recorded = json.loads((TRAIN_RUN / "env-runtime.json").read_text(encoding="utf-8"))
        self.assertEqual(sorted(rt), sorted(recorded))
        self.assertEqual(rt["script_sha256"], T.sha256_file(RUNTIME / "train_flow_grpo_wan.py"))
        self.assertEqual(T.prepare_out(out), 2)
        (out / "x").write_text("", encoding="utf-8")
        self.assertEqual(T.prepare_out(out), 3)
        self.assertEqual(sorted(p.name for p in out.glob("preempted-attempt-*")),
                         ["preempted-attempt-1", "preempted-attempt-2"])

    def test_base_binding_refusals(self):
        base = self.base()
        (base / "extra").write_text("x", encoding="utf-8")
        with self.assertRaises(SystemExit):
            T.base_binding(base, verify=False)
        (base / "extra").unlink()
        (base / "model_index.json").write_bytes(b"changed")
        with self.assertRaises(SystemExit):
            T.base_binding(base, verify=True)


class TestShortSplit(unittest.TestCase):
    def test_committed_short_split(self):
        items = read_jsonl(RUNTIME / "items-short" / "items.jsonl")
        meta = json.loads((RUNTIME / "items-short" / "item_set_meta.json").read_text(encoding="utf-8"))
        self.assertEqual(len(items), meta["split"]["n_items"])
        self.assertEqual(meta["items_sha256"], M.sha256_file(RUNTIME / "items-short" / "items.jsonl"))
        self.assertEqual(meta["split"]["first_attempt_items_sha256"], M.sha256_file(M.OLD_ITEMS))
        self.assertEqual(meta["split"]["max_words"], M.MAX_WORDS)
        old = {r["item"]: r for r in read_jsonl(M.OLD_ITEMS)}
        for r in items:
            self.assertTrue(M.is_short(r["target"]), r)
            self.assertEqual(r["target"], M.target_text(r["prompt"]))
            if r["item"] in old:
                self.assertEqual(old[r["item"]], r)
        self.assertEqual(sum(r["item"] in old for r in items), meta["split"]["n_items_in_first_attempt_items"])

    def test_split_rules(self):
        cands = [(0, 'a "One Two"'), (1, 'b "One Two Three"'), (2, 'c "Solo"'), (3, 'd "Open"')]
        train_first = {'x "one two"': 7, 'y "Other"': 3, 'z "Long Long Target"': 5, 'c "Solo"': 9}
        items, shard, n_short, dropped_prompt, dropped_target = M.split(M.is_short, train_first, cands, [3, 1, 0, 2])
        self.assertEqual([r["item"] for r in items], ["ocr-test-0003", "ocr-test-0000", "ocr-test-0002"])
        self.assertEqual([r["prompt_id"] for r in shard], ["ocr-train-00003"])     # file order, disjoint by target
        self.assertEqual((n_short, dropped_prompt, dropped_target), (3, 1, 1))


class TestRapidfuzzStandIn(unittest.TestCase):
    """The rapidfuzz stand-in exists only while this module's tests run (setUpModule to tearDownModule): importing
    this file leaves sys.modules alone, and the stand-in has a module spec, so importlib.util.find_spec works on it."""

    @unittest.skipIf(HAVE_RAPIDFUZZ, "rapidfuzz is installed: no stand-in")
    def test_scoped_to_this_module(self):
        self.assertIsNotNone(importlib.util.find_spec("rapidfuzz.distance.Levenshtein"))
        code = "import sys, test_runtime; print(sorted(n for n in sys.modules if n.split('.')[0] == 'rapidfuzz'))"
        got = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parent, capture_output=True,
                             text=True, check=True)
        self.assertEqual(got.stdout.strip(), "[]")
        tearDownModule()
        try:
            self.assertEqual([n for n in sys.modules if n.split(".")[0] == "rapidfuzz"], [])
        finally:
            setUpModule()


if __name__ == "__main__":
    unittest.main()
