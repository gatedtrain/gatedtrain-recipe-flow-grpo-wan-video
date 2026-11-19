#!/usr/bin/env python
"""CPU tests of train_flow_grpo_wan_v2.py, the attempt-2 (amendment) trainer (README "Attempt 2 (amendment)").

Run with area 5's venv, from anywhere:
  env -u HF_TOKEN CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 HOME=<scratch> \
      <sd35-ocr-runtime>/.venv/bin/python -I -B tests/test_train_flow_grpo_wan_v2.py

What is checked:
  - train_flow_grpo_wan.py is unchanged (its pinned SHA-256; RESULT.md "Published code" relates it to the first
    attempt's launch records' `script_sha256`);
  - v2 differs from it only in the module docstring, `DEFAULT_PROMPTS`, `FULL_EPOCHS`, `PROFILES` and the docstring of
    `epoch_prompts` (compared as syntax trees, so comments do not count); the SHARED SAMPLER block is identical;
  - the profiles smoke2 and full2: 8 prompts x 4 videos = 32 samples per epoch, 2 optimizer steps of 16, training
    micro-batch 2, sampling batch 4, 2 epochs (cut 2) and 100 epochs (cuts every 25); the old profile names refused;
  - the default prompt directory is prompts-short/ (5,024 rows) and epoch e reads rows 8(e - 1) .. 8e - 1;
  - main() for one epoch of the smoke2 batch shape on a tiny random base with the scripted reward of
    test_train_flow_grpo_wan.py: 32 samples, kept and filtered, two steps of 16 samples, the cut.
"""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
AREA = HERE.parent
sys.path.insert(0, str(AREA))
sys.path.insert(0, str(HERE))

import eval_videos as ev  # noqa: E402
import test_eval_videos as tev  # noqa: E402
import test_train_flow_grpo_wan as ttw  # noqa: E402  (StubReward, fake encoder, TINY)
import train_flow_grpo_wan as tw  # noqa: E402
import train_flow_grpo_wan_v2 as tw2  # noqa: E402

V1_SHA256 = "bac3ea77d62a1beedcb9b2c305e2a26f9490cc5cc0847dbfd3c68564fc27f0b8"   # see RESULT.md "Published code"
CHANGED = {"DEFAULT_PROMPTS", "FULL_EPOCHS", "PROFILES"}


def stripped_tree(path: Path) -> str:
    """The module's syntax tree without its docstring, the attempt-specific assignments and epoch_prompts' docstring."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = tree.body[1:] if isinstance(tree.body[0], ast.Expr) else tree.body
    keep = []
    for node in body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in CHANGED for t in node.targets):
            continue
        if isinstance(node, ast.FunctionDef) and node.name == "epoch_prompts":
            node.body = node.body[1:]
        keep.append(node)
    tree.body = keep
    return ast.dump(tree)


class TestV2Diff(unittest.TestCase):
    def test_v1_unchanged(self):
        self.assertEqual(hashlib.sha256((AREA / "train_flow_grpo_wan.py").read_bytes()).hexdigest(), V1_SHA256)

    def test_only_profiles_and_prompts_differ(self):
        self.assertEqual(stripped_tree(AREA / "train_flow_grpo_wan_v2.py"), stripped_tree(AREA / "train_flow_grpo_wan.py"))
        self.assertEqual(ttw.shared_block(AREA / "train_flow_grpo_wan_v2.py"), ttw.shared_block(AREA / "eval_videos.py"))
        self.assertEqual(tw2.sampler_code_sha256(), ttw.SAMPLER_SHA)
        self.assertEqual(tw2.COMMON, tw.COMMON)
        self.assertEqual(tw2.DEFAULT_PROMPTS, AREA / "prompts-short")
        self.assertEqual(tw.DEFAULT_PROMPTS, AREA / "prompts")


class TestV2Profiles(unittest.TestCase):
    def test_profiles(self):
        self.assertEqual(sorted(tw2.PROFILES), ["full2", "smoke2"])
        s = tw2.resolve_settings(tw2.parse_args(["--profile", "smoke2", "--out", "/x"]))
        self.assertEqual((s["prompts_per_epoch"], s["images_per_prompt"], s["samples_per_epoch"],
                          s["num_batches_per_epoch"], s["gradient_accumulation_steps"], s["num_train_timesteps"]),
                         (8, 4, 32, 2, 1, 19))
        self.assertEqual((s["epochs"], s["save_every"], s["train_batch"], s["sample_batch"], s["seed"]),
                         (2, 2, 2, 4, 5000))
        f = tw2.resolve_settings(tw2.parse_args(["--profile", "full2", "--out", "/x"]))
        self.assertEqual((f["epochs"], f["save_every"], f["samples_per_epoch"], f["train_batch"], f["sample_batch"]),
                         (100, 25, 32, 2, 4))
        self.assertEqual(tw2.FULL_EPOCHS, 100)
        # every other setting is the first attempt's
        old = tw.resolve_settings(tw.parse_args(["--profile", "full", "--out", "/x"]))
        diff = {k for k in set(old) | set(f) if old.get(k) != f.get(k)}
        self.assertEqual(diff, {"epochs", "save_every", "prompts_per_epoch", "samples_per_epoch"})
        for bad in (["--profile", "smoke", "--out", "/x"], ["--profile", "full", "--out", "/x"]):
            with self.assertRaises(SystemExit):
                tw2.parse_args(bad)
        self.assertEqual(tw2.parse_args(["--profile", "smoke2", "--out", "/x"]).prompts, AREA / "prompts-short")

    def test_short_shard_order(self):
        shard, path = tw2.read_prompts(tw2.DEFAULT_PROMPTS)
        self.assertEqual((len(shard), path), (5024, AREA / "prompts-short" / "train.jsonl"))
        self.assertTrue(all(len(r["target"].split()) <= 2 for r in shard))
        e2 = tw2.epoch_prompts(shard, 2, 8)
        self.assertEqual([r["prompt_id"] for r in e2], [r["prompt_id"] for r in shard[8:16]])
        self.assertEqual(len({r["prompt_id"] for e in range(1, 101) for r in tw2.epoch_prompts(shard, e, 8)}), 800)


class TestV2MainOneEpoch(unittest.TestCase):
    """main() of v2 for one epoch of the smoke2 batch shape (8 prompts x 4 videos) on a tiny random base."""

    @classmethod
    def setUpClass(cls):
        cls.root = ttw.SCRATCH / "train-v2-e2e"
        if cls.root.exists():
            shutil.rmtree(cls.root)
        cls.base = cls.root / "base" / "tiny"
        cls.base.mkdir(parents=True)
        cls.models_json = tev.make_tiny_base(cls.base)
        cls.prompts = cls.root / "prompts"
        cls.prompts.mkdir()
        markers = ["VARY", "CONST", "ODD", "CONST", "VARY", "CONST", "CONST", "VARY", "CONST"]
        rows = [{"prompt": f'{m} poster {i} reading "Text {i}"', "prompt_id": f"t-{i:02d}", "target": f"Text {i}"}
                for i, m in enumerate(markers)]
        (cls.prompts / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        cls.saved = (tw2.MODELS_JSON, tw2.encode_all_prompts, tw2.PaddleOcrA, dict(tw2.PROFILES), dict(tw2.COMMON),
                     ev.MODELS_JSON)
        tw2.MODELS_JSON = ev.MODELS_JSON = cls.models_json
        tw2.encode_all_prompts = fake_encode_all_prompts
        tw2.PaddleOcrA = ttw.StubReward
        tw2.PROFILES["cpu-smoke2-1"] = {**tw2.PROFILES["smoke2"], "epochs": 1, "save_every": 1}
        tw2.COMMON.update(ttw.TINY)

    @classmethod
    def tearDownClass(cls):
        tw2.MODELS_JSON, tw2.encode_all_prompts, tw2.PaddleOcrA, profiles, common, ev.MODELS_JSON = cls.saved
        tw2.PROFILES.clear()
        tw2.PROFILES.update(profiles)
        tw2.COMMON.clear()
        tw2.COMMON.update(common)

    def jl(self, path):
        return [json.loads(ln) for ln in path.read_text().splitlines()]

    def test_one_epoch_32_samples(self):
        out = self.root / "run"
        self.assertEqual(tw2.main(["--profile", "cpu-smoke2-1", "--out", str(out), "--base", str(self.base),
                                   "--prompts", str(self.prompts), "--device", "cpu", "--seed", "77",
                                   "--no-verify-base"]), 0)
        args = json.loads((out / "run_args.json").read_text())
        s = args["settings"]
        self.assertEqual((s["prompts_per_epoch"], s["samples_per_epoch"], s["train_batch"], s["sample_batch"]),
                         (8, 32, 2, 4))
        self.assertEqual(args["profile"], "cpu-smoke2-1")
        rt = json.loads((out / "env-runtime.json").read_text())
        self.assertEqual(rt["script_sha256"], tw2.sha256_file(AREA / "train_flow_grpo_wan_v2.py"))
        self.assertEqual(rt["sampler_code_sha256"], ttw.SAMPLER_SHA)
        samples = self.jl(out / "samples.jsonl")
        self.assertEqual(len(samples), 32)
        self.assertEqual([x["prompt_id"] for x in samples], [f"t-{i // 4:02d}" for i in range(32)])
        plan = self.jl(out / "epoch_plan.jsonl")[0]
        # VARY x3 and ODD (3 nonzero + 1 at the group mean) give 15 nonzero advantages: the padding re-includes one
        self.assertEqual((len(plan["kept"]), len(plan["filtered"]), len(plan["reincluded"]), plan["steps"]),
                         (16, 16, 1, [1, 2]))
        steps = self.jl(out / "train_steps.jsonl")
        self.assertEqual([len(st["sample_ids"]) for st in steps], [8, 8])
        self.assertEqual(sorted(i for st in steps for i in st["sample_ids"]), sorted(plan["kept"]))
        marker = json.loads((out / "checkpoint-1" / "cut.json").read_text())
        self.assertEqual((marker["global_step"], marker["samples_consumed"], marker["samples_filtered"],
                          marker["profile"]), (2, 16, 16, "cpu-smoke2-1"))
        ev.check_cut(out / "checkpoint-1")


def fake_encode_all_prompts(base_dir, prompts, device, dtype, max_sequence_length, batch_size=8):
    enc = tev.tiny_encoder().to(device=device, dtype=dtype)
    return tw2.encode_prompts(tev.StubTokenizer(), enc, prompts, max_sequence_length, batch_size)


if __name__ == "__main__":
    unittest.main(verbosity=2)
