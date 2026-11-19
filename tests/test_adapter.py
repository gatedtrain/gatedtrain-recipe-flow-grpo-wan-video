"""External-run adapters, slice 8: area "video generation" (Flow-GRPO + LoRA on Wan2.1-T2V-1.3B, video OCR; tests/README.md
section 11.25, EXT-214 to EXT-228).

Attempt 2 (amendment, decided 2026-11-15; EXT-229 to EXT-232): the v2 trainer `train_flow_grpo_wan_v2.py`
(profiles smoke2/full2, prompts-short/), `eval_videos.py --items items-short`, the two attempts kept apart, the GPU cap.

Sources: MULTIMODAL-SPEC.md MM-1 to MM-39 (section 6 rows 3 and 3s and the row 3 deviations, MM-11, MM-17, MM-18, MM-20,
MM-21, MM-24, MM-27, MM-28, MM-29, MM-31 to MM-36, MM-38), design revision 16: 05 section 6.1 (R15-1), sections 11,
11.2, 12.1 to 12.3; runtime/README.md "Decisions" (2026-11-14); the area runtime
`runtime/` (README "Trainer", "Evaluator", "Smoke launch"; `train_flow_grpo_wan.py`,
`eval_videos.py`; read 2026-11-14) and its GPU probes `probe-train-1` and `probe-base-1`. Helpers:
`tests/_flow_grpo_wan.py`.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

import _flow_grpo_wan as W
from gatedtrain.testing.ext import sha_of, write_canonical
from gatedtrain.testing.support import EXIT_OK, EXIT_REFUSED, describe, file_sha256, manifest_hash, parse_errors, parse_run_dir, run_cli

NOT_PASS = 1
REAL_AREA = W.REAL_AREA3V
REAL_TRAIN = REAL_AREA / "train_flow_grpo_wan.py"
REAL_EVAL = REAL_AREA / "eval_videos.py"
REAL_PROBE = W.REAL_OUT / "probe-train-1"
REAL_PROBE_EVAL = W.REAL_OUT / "probe-base-1"
REAL_LOCK = W.REAL_AREA5 / "uv.lock"


def registry():
    from gatedtrain.ext import registry as reg
    return reg


def adapter():
    return registry().adapter(W.ADAPTER)


def _root_can_read() -> bool:
    return os.geteuid() == 0


class _Names(ast.NodeTransformer):
    """Replaces a name bound to an earlier literal constant by that constant."""

    def __init__(self, known: dict):
        self.known = known

    def visit_Name(self, node: ast.Name):
        return ast.Constant(self.known[node.id]) if node.id in self.known else node


def module_constants(path: Path, names: tuple[str, ...]) -> dict:
    """Literal module-level constants of a runtime script, read with `ast` (the script is never imported or compiled
    into the area, so no byte code is written there)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    known: dict = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            value = _Names(known).visit(node.value)       # literals and earlier constants (DEPLOYABLE, PROFILES)
            try:
                known[node.targets[0].id] = ast.literal_eval(value)
            except ValueError:
                pass
    return {k: v for k, v in known.items() if k in names}


def argparse_options(path: Path) -> set[str]:
    """The `--option` strings of a script's `p.add_argument(...)` calls (literal first arguments) plus the f-string
    loop over SAMPLER_DEFAULT of eval_videos.py, resolved from the constant."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
                and node.args and isinstance(node.args[0], ast.Constant)):
            out.add(node.args[0].value)
    consts = module_constants(path, ("SAMPLER_DEFAULT",))
    out |= {"--" + k.replace("_", "-") for k in consts.get("SAMPLER_DEFAULT", {})}
    return out


class WanCase(unittest.TestCase):
    """One scenario per test that changes it (`fresh`)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-wan-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.n = 0

    def fresh(self, **kw) -> W.WanScenario:
        type(self).n += 1
        return W.WanScenario(self.tmp / f"sc-{type(self).n}", **kw)

    def runs(self) -> Path:
        type(self).n += 1
        return self.tmp / f"runs-{type(self).n}"

    def passes(self, sc: W.WanScenario, **kw) -> Path:
        proc = sc.collect(self.runs(), **kw)
        self.assertIn(proc.returncode, (EXIT_OK, NOT_PASS), describe(proc))
        return parse_run_dir(proc.stdout)

    def refused(self, rule: str, field_prefix: str | None = None, *, sc: W.WanScenario, text: str | None = None,
                **kw):
        runs = self.runs()
        proc = sc.collect(runs, **kw)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertNotIn("Traceback", proc.stderr)
        errs = parse_errors(proc.stderr)
        hits = [e for e in errs if e["rule"] == rule and (field_prefix is None or e["field"].startswith(field_prefix))
                and (text is None or text in e["message"])]
        self.assertTrue(hits, f"expected error[{rule}] {field_prefix}... {text!r}; got {errs}\n{describe(proc)}")
        self.assertFalse(runs.exists() and any(runs.iterdir()), "a refusal writes no run directory")
        return hits


# ---------------------------------------------------------------------------
# EXT-214: registry rows, docs and the `uses` id
# ---------------------------------------------------------------------------

class TestRegistryRows(unittest.TestCase):

    def test_trainer_row(self):
        row = registry().TRAINER_ROWS[W.ADAPTER]
        self.assertEqual(list(row["uses"]), [W.TRAINER_ID])
        self.assertEqual(list(row["evaluators"]), [W.EVALUATOR_ID, W.SUPPORT_ID])     # deciding first (MM-38)
        self.assertEqual(row["consumption"], {"target": "row", "fallback": None})
        self.assertEqual(set(row["pins"]), {"diffusers", "peft", "torch", "transformers", "paddleocr", "paddlepaddle"})
        self.assertEqual((row["framework"], tuple(row["components"])), ("diffusers", W.DEPLOYABLE))
        self.assertEqual(row["gpu_h_estimate"], 4.8)
        self.assertFalse(row["spec_binding"])
        self.assertIsNone(registry().sidecar(W.ADAPTER))
        self.assertEqual(registry().image_interpreters(W.ADAPTER), {})
        self.assertEqual(registry().interpreter_flags(W.ADAPTER), {})

    def test_host_rows(self):
        """Area 5's venv and interpreter tree; area 3's runtime and outputs; no credential environment name."""
        reg = registry()
        self.assertEqual(reg.input_roots(W.ADAPTER), (W.PYTHON_DIR, W.AREA5, W.AREA3V, "{output_root}/flow-grpo-wan-video"))
        self.assertEqual(reg.interpreter(W.ADAPTER), W.INTERPRETER)
        self.assertEqual(reg.interpreter_trees(W.ADAPTER), (W.CPYTHON,))
        self.assertEqual(reg.scan_skip(W.ADAPTER)[0], f"{W.AREA5}/.venv")
        self.assertEqual(set(reg.env_names(W.ADAPTER)), set(W.EVAL_ENV) | {W.OCR_RUNTIME_ENV})
        self.assertNotIn("HF_TOKEN", reg.env_names(W.ADAPTER))
        self.assertEqual(set(reg.env_paths(W.ADAPTER)), {"HOME", "XDG_CACHE_HOME", "DOCTR_CACHE_DIR"})

    def test_uses_id_names_one_adapter(self):
        """MM-13: `trainer_ids()` maps a spec `uses` to one adapter, so area 3's sibling has its own id (the decision's
        `rl.flow_grpo@0.1` stays area 5's); no two rows share an id."""
        ids = [u for row in registry().TRAINER_ROWS.values() for u in row["uses"]]
        self.assertEqual(len(ids), len(set(ids)), ids)
        self.assertEqual(registry().trainer_ids()[W.TRAINER_ID], W.ADAPTER)
        self.assertNotEqual(registry().trainer_ids().get("rl.flow_grpo@0.1"), W.ADAPTER)
        from gatedtrain.spec import TRAINER_BACKENDS
        self.assertEqual(TRAINER_BACKENDS[W.TRAINER_ID], "ext")

    def test_evaluator_rows(self):
        rows = registry().EVALUATOR_ROWS
        for eid, metric, tier, scorer in ((W.EVALUATOR_ID, W.METRIC, "T3", W.SCORER),
                                          (W.SUPPORT_ID, W.SUPPORT_METRIC, "T0", W.SUPPORT_SCORER)):
            with self.subTest(evaluator=eid):
                row = rows[eid]
                self.assertEqual((row["metric"], row["direction"], tuple(row["score_range"]), row["tier"]),
                                 (metric, "higher_is_better", (0, 1), tier))
                self.assertFalse(row["binary"])
                self.assertEqual(tuple(row["error_classes"]), ("scorer_error",))
                self.assertEqual(tuple(row["score_outcomes"]), ("nonfinite_video",))
                self.assertEqual((row["payload_text_field"], tuple(row["item_payload"])), ("prompt", ("prompt", "target")))
                self.assertFalse(row["normalizer"] or row["env_version"])
                self.assertEqual(set(row["sampler_keys"]), set(W.SAMPLER_KEYS))
                self.assertEqual(row["scorer"], scorer)                       # the probe-base-1 arms' scorers
                self.assertEqual(row["scorer"]["code_sha256"], W.EVAL_VIDEOS_SHA256)
                self.assertEqual(row["samples"], "400, 1")

    def test_construct_alignment_declared(self):
        """R15-1: docTR over all 17 frames is T3 and construct-aligned with the PaddleOCR video reward; the reward arm
        is T0 and declares nothing; no shared annotation."""
        rows = registry().EVALUATOR_ROWS
        ca = rows[W.EVALUATOR_ID]["construct_aligned"]
        self.assertEqual(ca["reward"], "video_ocr_paddleocr")
        for phrase in ("another OCR engine", "another aggregate", "all 17 frames", "not that"):
            self.assertIn(phrase, ca["statement"])
        self.assertIsNone(rows[W.SUPPORT_ID].get("construct_aligned"))
        self.assertIsNone(rows[W.EVALUATOR_ID].get("shared_annotation"))
        self.assertIn("mean over all frames (zeros included)", rows[W.EVALUATOR_ID]["scorer"]["version"])
        self.assertIn("mean over positive frames", rows[W.SUPPORT_ID]["scorer"]["version"])

    def test_identity_hash_and_items_table(self):
        from gatedtrain.items import EVALUATORS
        for eid, metric, scorer in ((W.EVALUATOR_ID, W.METRIC, W.SCORER), (W.SUPPORT_ID, W.SUPPORT_METRIC,
                                                                            W.SUPPORT_SCORER)):
            self.assertEqual(registry().evaluator_identity_hash(eid),
                             sha_of({"evaluator": eid, "metric": metric, "direction": "higher_is_better",
                                     "scorer": scorer}))
            self.assertEqual((EVALUATORS[eid]["direction"], tuple(EVALUATORS[eid]["score_range"])),
                             ("higher_is_better", (0, 1)))

    def test_item_set_rows(self):
        from gatedtrain.ext.collect import read_item_set
        row = registry().EVALUATOR_ROWS[W.EVALUATOR_ID]
        tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-wan-items-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        shutil.copyfile(W.FG / "items.jsonl", tmp / "items.jsonl")
        rows, bad = read_item_set(tmp, tuple(row["item_payload"]), tuple(row.get("item_payload_objects", ())))
        self.assertEqual((len(rows), bad), (40, []))

    def test_adapter_imports_no_framework(self):
        src = (Path(__file__).resolve().parents[1] / "src" / "gatedtrain_recipe_flow_grpo_wan_video" / "adapter.py").read_text()
        for mod in ("torch", "diffusers", "peft", "paddle", "doctr", "Levenshtein", "pickle", "numpy"):
            self.assertNotIn(f"import {mod}", src)


# ---------------------------------------------------------------------------
# EXT-215: cuts and the base directory
# ---------------------------------------------------------------------------

class TestCuts(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-wan-cuts-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.base = W.build_base(self.tmp / "base")
        self.out = self.tmp / "run"
        for e, gs in ((15, 30), (30, 60), (45, 90)):
            W.build_cut(self.out, e, global_step=gs, samples_consumed=gs * 7, samples_filtered=e,
                        base=self.base, script_sha256="sha256:" + "1" * 64)

    def found(self):
        complete, incomplete = adapter().find_cuts(self.out)
        return [s for s, _ in complete], sorted(Path(p).name for p in incomplete)

    def edit(self, d: Path, fn):
        doc = json.loads((d / "cut.json").read_text())
        fn(doc)
        (d / "cut.json").write_text(json.dumps(doc))

    def test_complete_by_global_step(self):
        self.assertEqual(self.found(), ([30, 60, 90], []))
        shutil.copytree(self.out / "checkpoint-45", self.out / ".checkpoint-60.partial")
        self.assertEqual(self.found(), ([30, 60, 90], []))

    def test_incomplete_cuts(self):
        d = self.out / "checkpoint-30"
        cases = {
            "no marker": lambda: (d / "cut.json").unlink(),
            "changed file": lambda: (d / "sampler.json").write_text("{}"),
            "wrong epoch": lambda: self.edit(d, lambda doc: doc.update(epoch=31)),
            "extra file": lambda: (d / "lora" ).mkdir(),
            "no samples_filtered": lambda: self.edit(d, lambda doc: doc.pop("samples_filtered")),
            "ema not bool": lambda: self.edit(d, lambda doc: doc.update(ema=0)),
        }
        for name, fn in cases.items():
            with self.subTest(case=name):
                shutil.rmtree(d)
                W.build_cut(self.out, 30, global_step=60, samples_consumed=420, samples_filtered=30, base=self.base,
                            script_sha256="sha256:" + "1" * 64)
                fn()
                self.assertEqual(self.found(), ([30, 90], ["checkpoint-30"]))

    def test_base_directory_is_a_model_directory(self):
        """The Wan2.1 Diffusers base (21 files: umT5 shards, transformer shards, VAE) is a model directory; a truncated
        transformer shard or a missing text encoder config is not."""
        a = adapter()
        self.assertIsNone(a.model_dir_problem(self.base))
        self.assertIsNone(a.model_dir_problem(self.out / "checkpoint-45"))
        w = self.base / "transformer" / "diffusion_pytorch_model-00002-of-00002.safetensors"
        w.write_bytes(w.read_bytes()[:-3])
        self.assertIn("transformer/", a.model_dir_problem(self.base))
        other = W.build_base(self.tmp / "base2")
        (other / "text_encoder" / "config.json").unlink()
        self.assertIn("not a Wan2.1-T2V-1.3B Diffusers directory", a.model_dir_problem(other))

    def test_probe_cut_complete(self):
        """The GPU smoke's cut (fixture: LoRA replaced, re-hashed) is a complete cut at global step 4."""
        self.assertEqual([(s, d.name) for s, d in adapter().find_cuts(W.PROBE)[0]], [(4, "checkpoint-2")])


# ---------------------------------------------------------------------------
# EXT-216: identity (MM-17, MM-18)
# ---------------------------------------------------------------------------

class TestIdentity(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-wan-id-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.base = W.build_base(self.tmp / "base")
        self.cut = W.build_cut(self.tmp / "run", 2, global_step=4, samples_consumed=28, samples_filtered=4,
                               base=self.base, script_sha256="sha256:" + "1" * 64)

    def test_base_alone(self):
        ident = adapter().identity(self.base)
        self.assertEqual((ident["model_hash"], ident["cut_hash"]), (manifest_hash(self.base), manifest_hash(self.base)))
        self.assertEqual(sorted(ident["components"]), sorted(W.BASE_FILES))
        self.assertEqual(len(ident["components"]), 21)
        self.assertIsNone(ident["base_ref"])

    def test_lora_cut(self):
        ident = adapter().identity(self.cut, self.base)
        self.assertEqual(ident["model_hash"], W.lora_model_hash(self.cut, self.base))
        self.assertEqual(ident["adapter_hash"], W.adapter_hash(self.cut))
        self.assertEqual(sorted(ident["components"]), sorted(W.DEPLOYABLE))
        self.assertEqual(ident["base_ref"], {"repo": W.REPO, "revision": W.REVISION,
                                             "manifest_hash": manifest_hash(self.base)})

    def test_resume_state_not_a_component(self):
        before = adapter().identity(self.cut, self.base)["model_hash"]
        (self.cut / "trainer_state.json").write_text('{"epoch": 2, "other": 1}\n')
        self.assertEqual(adapter().identity(self.cut, self.base)["model_hash"], before)
        (self.cut / "sampler.json").write_text('{"num_frames": 33}\n')       # the deployment sampler is a component
        self.assertNotEqual(adapter().identity(self.cut, self.base)["model_hash"], before)
        with self.assertRaises(ValueError):
            adapter().identity(self.cut)


# ---------------------------------------------------------------------------
# EXT-217: the whole run collected
# ---------------------------------------------------------------------------

class TestCollectPass(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-wan-pass-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.sc = W.WanScenario(cls.tmp / "sc")
        cls.proc = cls.sc.collect(cls.tmp / "runs")
        cls.run_dir = parse_run_dir(cls.proc.stdout) if cls.proc.returncode == EXIT_OK else None

    def setUp(self):
        self.assertEqual(self.proc.returncode, EXIT_OK, describe(self.proc))

    def report(self) -> dict:
        return json.loads((self.run_dir / "report" / "gate_report.json").read_text(encoding="utf-8"))

    def ledger(self) -> list[dict]:
        import pyarrow.parquet as pq
        return pq.read_table(self.run_dir / "ledger" / "consumption.parquet").to_pylist()

    def test_pass_and_hashes(self):
        rep = self.report()
        self.assertEqual(rep["status"], "PASS")
        self.assertEqual((rep["adapter"], rep["independence_tier"], rep["direction"]),
                         (W.ADAPTER, "T3", "higher_is_better"))
        self.assertEqual((rep["baseline_model_hash"], rep["candidate_model_hash"]), (self.sc.base_model,
                                                                                     self.sc.cand_model))
        self.assertEqual(rep["contamination"], {"exact_overlap": 0, "allowed": False})
        self.assertEqual((rep["consumption_precision"], rep["n_items"], rep["samples_per_item"]), ("row", 40, 1))

    def test_construct_alignment_reported(self):
        ca = self.report()["construct_alignment"]
        self.assertEqual(ca["reward"], "video_ocr_paddleocr")
        self.assertIn("whole clip", ca["statement"])

    def test_plan_shapes(self):
        """The scenario: epoch 1 filters a never-rendered group; epoch 2 filters 4 and re-includes 1 (padding);
        epoch 3 renders nothing, every advantage is 0, so all 16 are kept."""
        _, plans, steps = self.sc.logs()
        self.assertEqual([(len(p["kept"]), len(p["filtered"]), len(p["reincluded"]), p["all_zero_advantage"])
                          for p in plans], [(12, 4, 0, False), (12, 4, 1, False), (16, 0, 0, True)])
        self.assertEqual([p["steps"] for p in plans], [[1, 2], [3, 4], [5, 6]])
        self.assertEqual(len(steps), 6)

    def test_ledger_counts(self):
        rows = self.ledger()
        self.assertEqual(sum(r["outcome"] == "consumed" for r in rows), 40)       # 12 + 12 + 16
        self.assertEqual(sum(r["outcome"] == "filtered" for r in rows), 8)        # 4 + 4 + 0
        ids = [r["sample_id"] for r in rows]
        self.assertEqual(len(set(ids)), len(ids))

    def test_all_zero_epoch_consumed(self):
        """Every sample of the all-zero epoch is consumed once (steps 5, 6), none filtered, rewards 0."""
        samples, _, steps = self.sc.logs()
        seeds = {s["seed"] for s in samples if s["epoch"] == 3}
        rows = [r for r in self.ledger() if r["seed_inputs"]["seed"] in seeds]
        self.assertEqual(len(rows), 16)
        self.assertEqual({(r["outcome"], r["step"] in (5, 6), r["rewards"][0]) for r in rows},
                         {("consumed", True, 0.0)})

    def test_filtered_and_reincluded(self):
        samples, plans, _ = self.sc.logs()
        by_id = {s["sample_id"]: s for s in samples}
        filtered = [by_id[i] for p in plans for i in p["filtered"]]
        rows = [r for r in self.ledger() if r["outcome"] == "filtered"]
        for r, s in zip(rows, filtered):
            self.assertEqual((r["reason"], r["step"], r["rewards"]), ("zero_advantage", None, [s["reward"]]))
            key = "\x00".join(str(x) for x in (r["run_id"], r["attempt_id"], s["epoch"], "filtered", s["prompt_id"],
                                               s["seed"], s["member"]))
            self.assertEqual(r["sample_id"], "sha256:" + hashlib.sha256(key.encode()).hexdigest())
        reinc = by_id[plans[1]["reincluded"][0]]
        hits = [r for r in self.ledger() if r["seed_inputs"] == {"seed": reinc["seed"]}]
        self.assertEqual([r["outcome"] for r in hits], ["consumed"])

    def test_ocr_error_keeps_the_reward(self):
        """An OCR frame error scores that frame 0; the other frames still count, so the reward stays 1.0 (README
        "Outputs": unlike area 5, an error does not force ocr_acc 0)."""
        samples, _, _ = self.sc.logs()
        err = [s for s in samples if s["ocr_error"]]
        self.assertEqual(len(err), 1)
        self.assertEqual((err[0]["ocr_acc"], err[0]["ocr_error"]), (1.0, "frame 8: RuntimeError: fixture OCR failure"))
        rows = [r for r in self.ledger() if r["seed_inputs"] == {"seed": err[0]["seed"]}]
        self.assertEqual([(r["outcome"], r["rewards"]) for r in rows], [("consumed", [1.0])])

    def test_data_ref_and_sampler(self):
        samples, _, _ = self.sc.logs()
        r = [x for x in self.ledger() if x["outcome"] == "consumed"][0]
        self.assertEqual(r["data_ref"]["shard"], "train.jsonl")
        self.assertEqual(r["data_ref"]["snapshot"], manifest_hash(self.sc.data))
        self.assertEqual(r["sampler_config_hash"], sha_of(W.SAMPLER))
        self.assertIn(r["data_ref"]["row"], {"sha256:" + hashlib.sha256(s["prompt_id"].encode()).hexdigest()
                                             for s in samples})

    def test_lineage(self):
        edges = json.loads((self.run_dir / "lineage" / "edges.json").read_text(encoding="utf-8"))
        train = [a for a in edges["activities"] if a["kind"] == "training_stage"][0]
        self.assertEqual([g["last_step_included"] for g in train["generated"]], [4, 6])
        compose = [a for a in edges["activities"] if a["kind"] == "lora_compose"]
        self.assertEqual(len(compose), 2)

    def test_arm_output_ref_copied(self):
        """The arm's output reference is the video's frames.json (17 frames listed with their SHA-256)."""
        line = W.read_jsonl(self.run_dir / "eval" / "candidate" / "items.jsonl")[0]
        ref = self.run_dir / "eval" / "candidate" / line["output_ref"]["path"]
        self.assertTrue(line["output_ref"]["path"].endswith("/r0/frames.json"))
        self.assertEqual(file_sha256(ref), line["output_ref"]["sha256"])
        self.assertEqual(len(json.loads(ref.read_text())["frames"]), 17)

    def test_consumption_note(self):
        notes = " ".join(self.report()["notes"])
        self.assertIn("8 filtered", notes)


# ---------------------------------------------------------------------------
# EXT-218: the observed consumption log refuses (MM-24, MM-27, MM-28, MM-29)
# ---------------------------------------------------------------------------

class TestConsumptionLog(WanCase):

    def check(self, fn, text, **kw):
        sc = self.fresh(**kw)
        sc.edit_logs(fn)
        return self.refused("EXT-CONSUMPTION", "--stage-out", sc=sc, text=text)

    @staticmethod
    def first(samples, **match):
        return next(s for s in samples if all(s[k] == v for k, v in match.items()))

    def test_frames_not_every_fourth(self):
        def fn(s, p, st):
            s[0]["frame_ocr"] = s[0]["frame_ocr"][:-1]
        self.check(fn, "frame_ocr is not one")

    def test_frame_accuracy_recomputed(self):
        """A frame's acc is flow_grpo's formula of its text against the target (re-derived exactly)."""
        def fn(s, p, st):
            x = self.first(s, epoch=1, sample_index=1)                 # ("part", 2): frame 0 reads 2 errors
            x["frame_ocr"][0]["acc"] = 0.9
        self.check(fn, "not flow_grpo's frame accuracy")

    def test_reward_is_the_mean_of_positive_frames(self):
        def fn(s, p, st):
            x = self.first(s, epoch=1, sample_index=3)                 # "full_err": reward 1.0
            x["reward"] = x["ocr_acc"] = 0.8                           # the mean over all five frames
        self.check(fn, "mean of the positive frame accuracies")

    def test_reward_differs_from_ocr_acc(self):
        def fn(s, p, st):
            self.first(s, epoch=2, sample_index=0)["ocr_acc"] = 0.0 + 1e-9
        self.check(fn, "mean of the positive frame accuracies")

    def test_failed_frame_shape(self):
        for name, edit in (("text", {"text": "OPEN"}), ("acc", {"acc": 0.5}), ("empty error", {"error": ""})):
            with self.subTest(case=name):
                def fn(s, p, st, edit=edit):
                    x = self.first(s, epoch=1, sample_index=3)
                    x["frame_ocr"][2].update(edit)
                self.check(fn, "a failed frame")

    def test_ocr_error_names_the_first_failed_frame(self):
        for value in (None, "frame 4: RuntimeError: fixture OCR failure"):
            with self.subTest(ocr_error=value):
                self.check(lambda s, p, st, v=value: self.first(s, epoch=1, sample_index=3).update(ocr_error=v),
                           "ocr_error")
        self.check(lambda s, p, st: self.first(s, epoch=1, sample_index=2).update(ocr_error="frame 0: x"), "ocr_error")

    def test_nonfinite_boolean(self):
        self.check(lambda s, p, st: s[5].update(nonfinite=0), "nonfinite")

    def test_prompts_in_file_order(self):
        """Epoch e uses shard rows 4(e - 1) .. 4e - 1 in file order: a run that read the shard in another order (two
        rows swapped, run_args.json naming that file's hash) refuses."""
        sc = self.fresh()
        rows = W.read_jsonl(sc.data / "train.jsonl")
        rows[0], rows[1] = rows[1], rows[0]
        W.write_jsonl(sc.data / "train.jsonl", rows)
        sc.render()
        sc.edit_json(sc.stage_out / "run_args.json",
                     lambda a: a.update(train_file_sha256=file_sha256(sc.data / "train.jsonl")))
        self.refused("EXT-CONSUMPTION", "--stage-out", sc=sc, text="is not shard row 0")

    def test_sample_id_domain(self):
        """The sample id hashes the domain "flow-grpo-wan" (area 5's "flow-grpo-sd35" refuses)."""
        def fn(s, p, st):
            x = s[0]
            x["sample_id"] = W.compact_hash(["flow-grpo-sd35", W.SEED, 1, x["epoch"], x["sample_index"],
                                              x["prompt_id"], x["seed"], x["member"]])
        self.check(fn, "sample_id is not the hash")

    def test_all_zero_flag_on_a_varied_epoch(self):
        self.check(lambda s, p, st: p[0].update(all_zero_advantage=True), "all_zero_advantage is true")

    def test_all_zero_flag_missing_or_false(self):
        self.check(lambda s, p, st: p[2].pop("all_zero_advantage"), "not a boolean")
        self.check(lambda s, p, st: p[2].update(all_zero_advantage=False), "every prompt group has equal rewards")

    def test_all_zero_epoch_filters_nothing(self):
        """An all-zero epoch keeps every sample: a plan that filters two of them (and trains 14) refuses."""
        def fn(s, p, st):
            gone = p[2]["kept"][:2]
            p[2]["kept"] = p[2]["kept"][2:]
            p[2]["filtered"] = gone
            for x in st:
                x["sample_ids"] = [i for i in x["sample_ids"] if i not in gone]
        self.check(fn, "all_zero_advantage is true, so every sample is kept")

    def test_flat_group_kept_in_a_mixed_epoch(self):
        """Area 5's rule still holds when not every group is flat: epoch 1's never-rendered group kept plainly."""
        def fn(s, p, st):
            flat = p[0]["filtered"]
            order = {x["sample_id"]: x["sample_index"] for x in s if x["epoch"] == 1}
            p[0]["kept"] = sorted(p[0]["kept"] + flat, key=order.get)
            p[0]["filtered"] = []
            st[0]["sample_ids"] += flat[:2]
            st[1]["sample_ids"] += flat[2:]
        self.check(fn, "has equal rewards (zero advantage), but 4 of its samples are kept")

    def test_settings(self):
        for key, value, text in (("reward", "ocr", "not 'video_ocr'"), ("prompt_order", "random", "prompt_order"),
                                 ("prompts_per_epoch", 8, "prompts_per_epoch"), ("num_frames", 0, "num_frames")):
            with self.subTest(key=key):
                sc = self.fresh()
                sc.edit_json(sc.stage_out / "run_args.json", lambda a: a["settings"].update({key: value}))
                self.refused("EXT-CONSUMPTION", "--stage-out", sc=sc, text=text)

    def test_area5_rules_still_apply(self):
        self.check(lambda s, p, st: p[1]["filtered"].append(p[1]["kept"][0]), "both kept and filtered")
        self.check(lambda s, p, st: p[1]["reincluded"].append(p[1]["filtered"][0]), "reincluded")

    def test_cut_counts(self):
        sc = self.fresh()
        sc.edit_json(sc.final_cut / "cut.json", lambda d: d.update(samples_filtered=0))
        self.refused("EXT-CONSUMPTION", "--stage-out", sc=sc, text="samples_filtered")

    def test_two_inner_epochs(self):
        import pyarrow.parquet as pq
        sc = self.fresh(inner_epochs=2)
        run = self.passes(sc)
        rows = pq.read_table(run / "ledger" / "consumption.parquet").to_pylist()
        self.assertEqual((sum(r["outcome"] == "consumed" for r in rows), sum(r["outcome"] == "filtered" for r in rows)),
                         (80, 8))

    def test_unreadable_samples(self):
        if _root_can_read():
            self.skipTest("running as root: chmod 0 does not make a file unreadable")
        sc = self.fresh()
        p = sc.stage_out / "samples.jsonl"
        p.chmod(0)
        self.addCleanup(p.chmod, 0o644)
        proc = sc.collect(self.runs())
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertNotIn("Traceback", proc.stderr)


# ---------------------------------------------------------------------------
# EXT-219: bindings and run checks
# ---------------------------------------------------------------------------

class TestBindings(WanCase):

    def test_base_ref_files_differ(self):
        sc = self.fresh()
        def other(d):
            d["files"]["model_index.json"] = "sha256:" + "0" * 64
            d["files_sha256"] = W.compact_hash(d["files"])
        for cut in sc.cuts:
            sc.edit_json(cut / "base_ref.json", other)
            sc.reseal(cut)
        fs = json.loads((sc.final_cut / "base_ref.json").read_text())["files_sha256"]
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a["base_ref"].update(files_sha256=fs))
        self.refused("EXT-BASE", "artifacts.base.uri", sc=sc, text="not the SHA-256 of the spec's base")

    def test_base_not_verified_or_elsewhere(self):
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a["base_ref"].update(verified=False))
        self.refused("EXT-BASE", "artifacts.base.uri", sc=sc, text="verified")
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json",
                     lambda a: a.update(base_model=f"{W.AREA5}/models/stable-diffusion-3.5-medium"))
        self.refused("EXT-BASE", "artifacts.base.uri", sc=sc, text="base_model")

    def test_data_binding(self):
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a.update(n_train_prompts=18359))
        self.refused("EXT-DATA", "artifacts.prompts.uri", sc=sc, text="n_train_prompts")
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a.update(train_file=f"{W.AREA5}/prompts/train.jsonl"))
        self.refused("EXT-DATA", "artifacts.prompts.uri", sc=sc, text="train_file")

    def test_seed_and_script(self):
        sc = self.fresh()
        sc.edit_json(sc.final_cut / "cut.json", lambda d: d.update(seed=5001))
        self.refused("EXT-LAUNCH", "--launch-record.training_seed", sc=sc, text="cut.json seed")
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a.update(script_sha256="sha256:" + "2" * 64))
        self.refused("EXT-LAUNCH", "--launch-record.script_sha256", sc=sc, text="run_args.json script_sha256")

    def test_sampler_code_pinned(self):
        """A run whose cuts, run_args.json and env-runtime.json agree on another SHARED SAMPLER block refuses: every
        cut must name the evaluator's block (`41606b0b...`)."""
        sc = self.fresh()
        other = "sha256:" + "3" * 64
        for p in (sc.stage_out / "run_args.json", sc.stage_out / "env-runtime.json", *[c / "cut.json" for c in sc.cuts]):
            sc.edit_json(p, lambda d: d.update(sampler_code_sha256=other))
        self.refused("EXT-SAMPLER", "--stage-out", sc=sc, text=W.SAMPLER_CODE_SHA)

    def test_reward_setting(self):
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a["settings"].update(reward="ocr"))
        self.refused("EXT-STAGE", "--stage-out", sc=sc, text="'video_ocr'")

    def test_schemas(self):
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a.update(schema="ptc_mm.flow_grpo_run_args"))
        self.refused("EXT-STAGE", "--stage-out", sc=sc, text="ptc_mm.flow_grpo_wan_run_args")
        sc = self.fresh()
        sc.edit_json(sc.final_cut / "cut.json", lambda d: d.update(schema="ptc_mm.flow_grpo_sd35_cut"))
        self.refused("EXT-STAGE", "--stage-out", sc=sc, text="ptc_mm.flow_grpo_wan_cut")

    def test_lora_targets(self):
        sc = self.fresh()
        sc.edit_json(sc.final_cut / "adapter_config.json", lambda d: d.update(target_modules=["to_q", "to_v"]))
        sc.reseal(sc.final_cut)
        self.refused("EXT-STAGE", "--stage-out", sc=sc, text="Wan targets")

    def test_ema_refused(self):
        sc = self.fresh()
        sc.edit_json(sc.final_cut / "cut.json", lambda d: d.update(ema=True))
        self.refused("EXT-STAGE", "--stage-out", sc=sc, text="EMA is off")

    def test_freeze_pins(self):
        sc = self.fresh()
        p = sc.stage_out / "env-freeze.txt"
        p.write_text("".join(l + "\n" for l in p.read_text().splitlines() if not l.startswith("paddleocr==")))
        sc.edit_json(sc.stage_out / "env-runtime.json", lambda d: d.update(env_freeze_sha256=file_sha256(p)))
        self.refused("EXT-FREEZE", "--stage-out", sc=sc, text="paddleocr")

    def test_last_epoch_not_cut(self):
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a["settings"].update(epochs=60))
        self.refused("EXT-STAGE", "--stage-out", sc=sc, text="settings.epochs")

    def test_unreadable_cut_file(self):
        if _root_can_read():
            self.skipTest("running as root: chmod 0 does not make a file unreadable")
        sc = self.fresh()
        p = sc.final_cut / "trainer_state.json"
        p.chmod(0)
        self.addCleanup(p.chmod, 0o644)
        self.refused("EXT-STAGE", "--stage-out", sc=sc, text="cannot be read")


# ---------------------------------------------------------------------------
# EXT-220: exact overlap (MM-11)
# ---------------------------------------------------------------------------

class TestOverlap(WanCase):

    def test_prompt_overlap_refused(self):
        sc = self.fresh(overlap=1)
        self.refused("EXT-CONTAMINATION", sc=sc)

    def test_target_overlap_refused(self):
        """An item's quoted text equal to a training row's after removing spaces and lower-casing refuses."""
        sc = self.fresh()
        rows = W.read_jsonl(sc.data / "train.jsonl")
        tgt = sc.item_rows[3]["target"]
        rows[40]["prompt"] = f'A storefront sign that reads "{tgt.upper()}" at dusk'
        W.write_jsonl(sc.data / "train.jsonl", rows)
        sc.edit_json(sc.stage_out / "run_args.json",
                     lambda a: a.update(train_file_sha256=file_sha256(sc.data / "train.jsonl")))
        sc.render()
        self.refused("EXT-CONTAMINATION", sc=sc, text="target text")


# ---------------------------------------------------------------------------
# EXT-221: arms (ext-arm-fill, MM-1 lines, the real probe arms)
# ---------------------------------------------------------------------------

class TestArms(WanCase):

    def meta(self, arm: Path) -> dict:
        return json.loads((arm / "eval_meta.json").read_text(encoding="utf-8"))

    def test_fill_base_and_cut(self):
        """The base arm gets the base's manifest hash; a cut arm needs `--base` and gets the composed hash; the
        evaluator's compact-JSON sampler hash is converted (floats such as 4.5)."""
        sc = self.fresh(filled=False)
        self.assertNotEqual(W.compact_hash(W.SAMPLER), sha_of(W.SAMPLER))
        proc = sc.fill(sc.baseline, sc.base, base=sc.base)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        meta = self.meta(sc.baseline)
        self.assertEqual(meta["checkpoint"], {"model_hash": sc.base_model, "base_model_hash": sc.base_model})
        self.assertEqual((meta["item_set_hash"], meta["sampler_hash"]), (sc.item_set_hash, sha_of(W.SAMPLER)))
        proc = sc.fill(sc.candidate, sc.final_cut)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        proc = sc.fill(sc.candidate, sc.final_cut, base=sc.base)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        self.assertEqual(self.meta(sc.candidate)["checkpoint"], {"model_hash": sc.cand_model,
                                                                  "base_model_hash": sc.base_model})
        support = sc.cand_eval / f"arm-{W.SUPPORT_METRIC}"
        self.assertEqual(sc.fill(support, sc.final_cut, base=sc.base).returncode, EXIT_OK)
        self.passes(sc)

    def test_other_scorer_code_refused(self):
        """An arm of another eval_videos.py (the evaluator identity differs) refuses."""
        sc = self.fresh()
        sc.edit_json(sc.candidate / "eval_meta.json", lambda d: d["scorer"].update(code_sha256="sha256:" + "4" * 64))
        self.refused("EXT-ARM", "--eval-candidate", sc=sc, text=W.EVAL_VIDEOS_SHA256)

    def test_supporting_arm_cannot_decide(self):
        sc = self.fresh()
        self.refused("EXT-ARM", "--eval-candidate", sc=sc, text=f"evaluator_id is '{W.SUPPORT_ID}'",
                     baseline=sc.base_eval / f"arm-{W.SUPPORT_METRIC}", candidate=sc.cand_eval / f"arm-{W.SUPPORT_METRIC}")

    def real_arm(self, metric: str) -> Path:
        """A copy of the probe arm with the videos' frames.json files (the PNG frames are not in the fixtures)."""
        arm = self.tmp / f"real-{metric}-{type(self).n}"
        type(self).n += 1
        shutil.copytree(W.PROBE_EVAL / f"arm-{metric}", arm)
        shutil.copytree(W.PROBE_EVAL / "videos", arm / "videos")
        return arm

    def test_real_arm_lines_accepted(self):
        """The 20 lines of each probe-base-1 arm pass the MM-1 rules with their frames.json output references."""
        from gatedtrain.ext.arm import read_lines
        for metric in (W.METRIC, W.SUPPORT_METRIC):
            with self.subTest(metric=metric):
                arm = self.real_arm(metric)
                recs, why = read_lines(arm, seeds=[11], score_range=(0, 1), error_classes=("scorer_error",))
                self.assertEqual((len(recs), why), (20, []))
                self.assertEqual(sorted(recs[0]["sub_metrics"]), ["mean_abs_frame_diff", "nonfinite", "ocr_a_acc",
                                                                  "ocr_b_acc"])

    def test_real_arm_changed_frames_index_refused(self):
        from gatedtrain.ext.arm import read_lines
        arm = self.real_arm(W.METRIC)
        ref = arm / "videos" / "ocr-test-0000" / "r0" / "frames.json"
        ref.write_text(ref.read_text().replace("frame_00.png", "frame_99.png"))
        _, why = read_lines(arm, seeds=[11], score_range=(0, 1), error_classes=("scorer_error",))
        self.assertTrue([w for w in why if "differs from its sha256" in w], why)


# ---------------------------------------------------------------------------
# EXT-222: registration (sampler keys, construct alignment)
# ---------------------------------------------------------------------------

class TestRegistration(unittest.TestCase):

    def reg(self) -> dict:
        text = (W.FGW / "registration.template.yaml").read_text(encoding="utf-8")
        text = text.replace("{item_set_hash}", "sha256:" + "a" * 64).replace("{sampler}", json.dumps(W.SAMPLER))
        return yaml.safe_load(text.replace("{evaluator_hash}", "sha256:" + "b" * 64))

    def fields(self, reg: dict, evaluator: str = W.EVALUATOR_ID) -> list[str]:
        from gatedtrain.registration import check_contents
        return [p.field for p in check_contents(reg, evaluator_id=evaluator)]

    def test_template_accepted(self):
        self.assertEqual(self.fields(self.reg()), [])

    def test_sampler_keys(self):
        """The video sampler's keys: area 5's image sampler keys (`resolution`, `noise_level`) refuse, and so does a
        sampler without `num_frames`."""
        reg = self.reg()
        reg["sampler"]["resolution"] = 512
        del reg["sampler"]["num_frames"]
        f = self.fields(reg)
        self.assertIn("registration.sampler.resolution", f)
        self.assertIn("registration.sampler.num_frames", f)

    def test_construct_alignment(self):
        reg = self.reg()
        del reg["construct_alignment"]
        self.assertIn("registration.construct_alignment", self.fields(reg))
        reg = self.reg()
        reg["evaluators"] = [{"id": W.SUPPORT_ID, "manifest_hash": "sha256:" + "c" * 64}]
        reg["clause"]["metric"] = W.SUPPORT_METRIC
        self.assertIn("registration.construct_alignment", self.fields(reg, W.SUPPORT_ID))
        reg = self.reg()
        reg["shared_annotation"] = "none"
        self.assertIn("registration.shared_annotation", self.fields(reg))


# ---------------------------------------------------------------------------
# EXT-223: argv rules
# ---------------------------------------------------------------------------

class TestArgv(unittest.TestCase):

    def problems(self, argv: list, profile: str | None = None) -> list[str]:
        return adapter().argv_problems(argv, profile=profile)

    def test_training(self):
        ok = [W.INTERPRETER, W.TRAIN_PY, "--profile", "smoke", "--out", "/out", "--seed", "5000"]
        self.assertEqual(self.problems(ok, "smoke"), [])
        self.assertEqual(self.problems([*ok[:3], "full", *ok[4:], "--epochs", "60", "--save-every", "15",
                                        "--train-batch", "4"], "full"), [])
        self.assertEqual(self.problems(ok + ["--save-ev", "2"], "smoke"), [])          # a unique prefix
        for bad in (["--no-verify-base"], ["--no-verify"], ["--ema"], ["extra"], ["--device"]):
            with self.subTest(bad=bad):
                self.assertTrue(self.problems(ok + bad, "smoke"), bad)
        self.assertTrue(self.problems(ok, "full"))                                     # the launch profile differs
        self.assertTrue(self.problems([*ok[:3], "cpu-sanity", *ok[4:]]))
        self.assertTrue(self.problems([*ok[:5], "/tmp/out", *ok[6:]], "smoke"))

    def test_scripts_pinned(self):
        for bad in (f"{W.AREA5}/train_flow_grpo.py", f"{W.AREA5}/train_flow_grpo_wan.py",
                    f"{W.OUT_ROOT}/area3-video/train_flow_grpo_wan.py", f"{W.AREA3V}/fetch_models.py",
                    f"{W.AREA3V}/tests/test_train_flow_grpo_wan.py"):
            with self.subTest(script=bad):
                self.assertTrue(self.problems([W.INTERPRETER, bad, "--profile", "smoke", "--out", "/out"]))

    def test_evaluation(self):
        base = [W.INTERPRETER, W.EVAL_PY, "--base-only", "--out", "/out", "--eval-seeds", "11"]
        cut = [W.INTERPRETER, W.EVAL_PY, "--checkpoint", "/inputs/cut", "--out", "/out", "--eval-seeds", "11"]
        self.assertEqual(self.problems(base, "full"), [])
        self.assertEqual(self.problems(cut, "full"), [])
        self.assertEqual(self.problems(base + ["--base", f"{W.AREA3V}/{W.BASE_REL}", "--items", f"{W.AREA3V}/items"]),
                         [])
        self.assertEqual(self.problems(base + ["--limit", "40"], "smoke"), [])
        for bad, prof in ((["--base", f"{W.AREA5}/models/stable-diffusion-3.5-medium"], None),
                          (["--items", f"{W.AREA5}/items"], None), (["--items=/inputs/x"], None),
                          (["--limit", "40"], "full"), (["--limit=40"], "full"), (["--copy-items"], None),
                          (["--checkpoint", "/inputs/cut"], None)):
            with self.subTest(bad=bad):
                self.assertTrue(self.problems(base + bad, prof))
        self.assertTrue(self.problems([*cut[:3], "/inputs/other", *cut[4:]]))
        self.assertTrue(self.problems([W.INTERPRETER, W.EVAL_PY, "--out", "/out"]))        # neither mode


# ---------------------------------------------------------------------------
# EXT-224: launch requests and records; EXT-225: --spec; EXT-226: smoke
# ---------------------------------------------------------------------------

class LaunchCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-wan-launch-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.sc = W.WanScenario(cls.tmp / "sc", n_items=12)
        cls.k = 0

    def out(self) -> Path:
        type(self).k += 1
        return self.sc.new_out(f"run-{type(self).k}")

    def launch(self, req: dict, **kw):
        type(self).k += 1
        return self.sc.launch_request(req, f"req-{type(self).k}", **kw)

    def refused(self, proc, out, rule, field, text: str | None = None):
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        errs = parse_errors(proc.stderr)
        self.assertTrue([e for e in errs if e["rule"] == rule and e["field"] == field
                         and (text is None or text in e["message"])], f"{rule} {field} {text!r}: {errs}")
        self.assertFalse(out.exists())


class TestLaunchRules(LaunchCase):

    def test_training_request(self):
        proc, out = self.launch(self.sc.request(self.out()))
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        self.assertIn("--runtime=nvidia", proc.stdout)
        self.assertIn(f"{W.IMAGE} {W.INTERPRETER} {W.TRAIN_PY} --profile smoke", proc.stdout)
        for c in (W.PYTHON_DIR, W.AREA5, W.AREA3V):
            self.assertIn(f":{c}:ro", proc.stdout)
        rec = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual((rec["adapter"], rec["runtime"]["venv_lock_sha256"], rec["script_path"]),
                         (W.ADAPTER, W.LOCK_SHA, W.TRAIN_PY))
        self.assertEqual(rec["interpreter_path"], f"{W.CPYTHON}/bin/python3.12")
        proc = self.sc.recheck(out)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))

    def test_evaluation_requests(self):
        """Base and cut evaluations (one GPU job each); the cut is mounted read-only at /inputs/cut."""
        for cut in (None, self.sc.final_cut):
            with self.subTest(cut=cut):
                proc, out = self.launch(self.sc.eval_request(self.out(), cut=cut))
                self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
                rec = json.loads(out.read_text(encoding="utf-8"))
                self.assertEqual(rec["script_path"], W.EVAL_PY)
                self.assertIn("DOCTR_CACHE_DIR", rec["env_names"])
        o = self.out()
        req = self.sc.eval_request(o, cut=self.sc.final_cut)
        req["mounts"] = self.sc.mounts(o)                               # --checkpoint without the cut's mount
        proc, out = self.launch(req)
        self.refused(proc, out, "EXT-LAUNCH", "--request.mounts")

    def test_wrong_image(self):
        proc, out = self.launch(self.sc.request(self.out(), runtime={"kind": "venv_in_container",
                                                                     "image": "sha256:" + "ab" * 32,
                                                                     "venv_lock": str(self.sc.lock)}))
        self.refused(proc, out, "EXT-LAUNCH", "--request.runtime", "gatedtrain-a5-runtime:1")

    def test_other_lock(self):
        lock = self.tmp / "other-uv.lock"
        lock.write_text("# another lock\n")
        proc, out = self.launch(self.sc.request(self.out(), runtime={"kind": "venv_in_container", "image": W.IMAGE,
                                                                     "venv_lock": str(lock)}))
        self.refused(proc, out, "EXT-LAUNCH", "--request.runtime", "uv.lock")

    def test_root_user(self):
        proc, out = self.launch(self.sc.request(self.out(), user="0:0"))
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertTrue([e for e in parse_errors(proc.stderr) if e["field"] == "--request.user"])
        self.assertFalse(out.exists())

    def test_network(self):
        proc, out = self.launch(self.sc.request(self.out(), network="bridge", network_reason="download"))
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertTrue([e for e in parse_errors(proc.stderr) if e["field"].startswith("--request.network")])

    def test_credential_and_other_env_names(self):
        req = self.sc.request(self.out())
        req["env"]["HF_TOKEN"] = "hf_" + "y" * 20
        proc, out = self.launch(req)
        self.refused(proc, out, "EXT-LAUNCH", "--request.env.HF_TOKEN")
        self.assertNotIn("hf_" + "y" * 20, proc.stdout + proc.stderr)
        req = self.sc.request(self.out())
        req["env"]["HF_HOME"] = "/tmp/hf"
        proc, out = self.launch(req)
        self.refused(proc, out, "EXT-LAUNCH", "--request.env.HF_HOME")
        req = self.sc.request(self.out())
        req["env"]["DOCTR_CACHE_DIR"] = "/srv/home/.cache/doctr"
        proc, out = self.launch(req)
        self.refused(proc, out, "EXT-LAUNCH", "--request.env.DOCTR_CACHE_DIR")

    def test_mount_set(self):
        o = self.out()
        req = self.sc.request(o)
        req["mounts"] = [m for m in req["mounts"] if m["container"] != W.AREA5]
        proc, out = self.launch(req)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))         # the interpreter is gone too
        req = self.sc.request(o)
        req["mounts"].insert(0, {"host": str(self.sc.final_cut), "container": "/inputs/cut", "mode": "ro"})
        proc, out = self.launch(req)
        self.refused(proc, out, "EXT-LAUNCH", "--request.mounts")

    def test_seed_and_gpus(self):
        req = self.sc.request(self.out())
        req["argv"] = [a for a in req["argv"]][:-2]                     # no --seed: the script's default
        proc, out = self.launch(req)
        self.refused(proc, out, "EXT-LAUNCH", "--request.argv", "training_seed")
        proc, out = self.launch(self.sc.request(self.out(), training_seed=7))
        self.refused(proc, out, "EXT-LAUNCH", "--request.argv", "training_seed")
        proc, out = self.launch(self.sc.request(self.out(), gpus=False, gpu_est_h=0))
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertTrue([e for e in parse_errors(proc.stderr) if e["field"] == "--request.gpus"])

    def test_bad_argv(self):
        for argv in ([W.INTERPRETER, W.TRAIN_PY, "--profile", "smoke", "--out", "/out", "--seed", "5000",
                      "--no-verify-base"],
                     [W.INTERPRETER, f"{W.AREA5}/train_flow_grpo.py", "--profile", "smoke", "--out", "/out", "--seed",
                      "5000"],
                     [f"{W.CPYTHON}/bin/python3.12", W.TRAIN_PY, "--profile", "smoke", "--out", "/out", "--seed",
                      "5000"]):
            with self.subTest(argv=argv[:2]):
                proc, out = self.launch(self.sc.request(self.out(), argv=argv))
                self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
                self.assertTrue([e for e in parse_errors(proc.stderr) if e["field"] == "--request.argv"])
                self.assertFalse(out.exists())

    def test_record_rules_at_collect(self):
        """The adapter's launch rules hold for the record too: a record naming another image refuses at collect."""
        sc = W.WanScenario(self.tmp / f"sc-rec-{type(self).k}")
        type(self).k += 1
        launch = sc.rewrite_launch(runtime={"kind": "venv_in_container", "image": "sha256:" + "cd" * 32,
                                            "venv_lock_sha256": W.LOCK_SHA})
        proc = sc.collect(self.tmp / f"runs-{type(self).k}", launch=launch)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertTrue([e for e in parse_errors(proc.stderr) if e["field"] == "--launch-record.runtime"])


class TestSpecBinding(LaunchCase):
    """EXT-225: `--spec` is optional (the smoke precedes the registration); given, the spec's base and prompts must lie
    in read-only mounts (the inherited `mount_binding_problems`)."""

    def test_spec_accepted(self):
        proc, out = self.launch(self.sc.request(self.out()), extra=["--spec", str(self.sc.paths["spec"])])
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        proc = self.sc.recheck(out, extra=["--spec", str(self.sc.paths["spec"])])
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))

    def test_spec_without_the_runtime_mount(self):
        req = self.sc.request(self.out())
        req["mounts"] = [m for m in req["mounts"] if m["container"] != W.AREA3V]
        proc, out = self.launch(req, extra=["--spec", str(self.sc.paths["spec"])])
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        fields = [e["field"] for e in parse_errors(proc.stderr)]
        self.assertIn("artifacts.base.uri", fields)
        self.assertIn("artifacts.prompts.uri", fields)


class TestSmoke(LaunchCase):
    """EXT-226 (MM-38): the deciding metric is not binary, so the mean, mode and sd rules judge the smoke baseline."""

    def smoke(self, name: str, **over) -> Path:
        return write_canonical(self.tmp / f"{name}.smoke.json", W.smoke_record(
            base_model=self.sc.base_model, item_set=self.sc.item_set_hash, **over))

    def full(self, smoke: Path):
        return self.launch(self.sc.request(self.out(), profile="full", gpu_est_h=3.5,
                                           argv=self.sc.argv("full") + ["--epochs", "60"]), profile="full", smoke=smoke)

    def test_real_probe_baseline_passes(self):
        """The 20 real ocr_b_acc scores of probe-base-1 (mean 0.169, no shared value) pass."""
        scores = W.real_baseline_scores()
        self.assertEqual(len(scores), 20)
        self.assertAlmostEqual(sum(scores) / 20, 0.1689537824456773, places=12)
        proc, out = self.full(self.smoke("real"))
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))

    def test_degenerate_baselines(self):
        for name, scores, reason in (("mode", [0.0] * 20 + [0.1 + 0.01 * i for i in range(20)], "degenerate_mode"),
                                     ("mean", [0.001 * (i + 1) for i in range(40)], "degenerate_mean"),
                                     ("sd", [0.3] * 40, "degenerate_sd")):
            with self.subTest(case=name):
                proc, out = self.full(self.smoke(name, baseline_scores=scores))
                self.refused(proc, out, "EXT-SMOKE", "--smoke-record")
                self.assertIn(reason, proc.stderr)

    def test_other_adapter_and_collect(self):
        proc, out = self.full(self.smoke("other", adapter="other_adapter"))
        self.refused(proc, out, "EXT-SMOKE", "--smoke-record")
        sc = W.WanScenario(self.tmp / f"sc-smoke-{type(self).k}")
        type(self).k += 1
        bad = write_canonical(self.tmp / "flat.smoke.json", W.smoke_record(
            base_model=sc.base_model, item_set=sc.item_set_hash, baseline_scores=[0.0] * 40))
        proc = sc.collect(self.tmp / f"runs-smoke-{type(self).k}", smoke=bad)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertTrue([e for e in parse_errors(proc.stderr) if e["rule"] == "EXT-SMOKE"])


# ---------------------------------------------------------------------------
# EXT-227: the GPU probes (fixtures, always) and the real runtime (skipped where absent)
# ---------------------------------------------------------------------------

class TestProbeFixture(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-wan-probe-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_probe_logs_pass_the_consumption_check(self):
        """probe-train-1 (2 epochs): 28 consumed records over steps 1-4 and the 4 filtered samples of the never-rendered
        group of epoch 1; the prompts are shard rows 0-7 in file order."""
        out, data = W.probe_copy(self.tmp)
        a = adapter()
        cuts = a.find_cuts(out)[0]
        self.assertEqual(a.consumption_problems(out, data, cuts), [])
        rows = a.consumption(out, data, cuts, run_id="r")["rows"]
        self.assertEqual((sum(r["outcome"] == "consumed" for r in rows), sum(r["outcome"] == "filtered" for r in rows)),
                         (28, 4))
        self.assertEqual(sorted({r["step"] for r in rows if r["step"] is not None}), [1, 2, 3, 4])
        prompts = [s["prompt_id"] for s in W.read_jsonl(out / "samples.jsonl")][::4]
        self.assertEqual(prompts, [r["prompt_id"] for r in W.read_jsonl(data / "train.jsonl")[:8]])

    def test_probe_edits_refused(self):
        for name, edit in (("frame acc", lambda x: x["frame_ocr"][0].update(acc=0.5)),
                           ("reward", lambda x: x.update(reward=0.5, ocr_acc=0.5)),
                           ("sample id domain", lambda x: x.update(sample_id=W.compact_hash(
                               ["flow-grpo-sd35", 5000, 1, x["epoch"], x["sample_index"], x["prompt_id"], x["seed"],
                                x["member"]])))):
            with self.subTest(case=name):
                d = self.tmp / name.replace(" ", "-")
                out, data = W.probe_copy(d)
                samples = W.read_jsonl(out / "samples.jsonl")
                edit(samples[0])
                W.write_jsonl(out / "samples.jsonl", samples)
                self.assertTrue(adapter().consumption_problems(out, data, adapter().find_cuts(out)[0]))

    def test_probe_marker_and_records(self):
        doc = json.loads((W.PROBE / "checkpoint-2" / "cut.json").read_text(encoding="utf-8"))
        self.assertEqual((doc["schema"], doc["epoch"], doc["global_step"], doc["samples_consumed"],
                          doc["samples_filtered"], doc["ema"]), ("ptc_mm.flow_grpo_wan_cut", 2, 4, 28, 4, False))
        self.assertEqual(doc["deployable"], list(W.DEPLOYABLE))
        self.assertEqual(doc["sampler_code_sha256"], W.SAMPLER_CODE_SHA_RUN)
        args = json.loads((W.PROBE / "run_args.json").read_text(encoding="utf-8"))
        self.assertEqual((args["schema"], args["settings"]["reward"], args["settings"]["prompt_order"]),
                         ("ptc_mm.flow_grpo_wan_run_args", "video_ocr", "file"))
        self.assertEqual(args["sampler"], W.SAMPLER)
        cfg = json.loads((W.PROBE / "checkpoint-2" / "adapter_config.json").read_text(encoding="utf-8"))
        self.assertEqual(sorted(cfg["target_modules"]), sorted(W.WAN_TARGETS))
        fw = adapter().framework_identity(W.PROBE, {"framework": {"name": "diffusers", "version": "0.42.1"}})
        self.assertEqual(fw["version"], "0.42.1")
        rt = json.loads((W.PROBE_EVAL / "env-runtime.json").read_text(encoding="utf-8"))
        self.assertEqual((rt["script_sha256"], rt["sampler_code_sha256"], rt["uv_lock_sha256"]),
                         (W.PROBE_EVAL_VIDEOS_SHA256, W.SAMPLER_CODE_SHA_RUN, W.LOCK_SHA_RUN))

    def test_fixture_arms_are_the_registry_s(self):
        for metric, eid in ((W.METRIC, W.EVALUATOR_ID), (W.SUPPORT_METRIC, W.SUPPORT_ID)):
            meta = json.loads((W.PROBE_EVAL / f"arm-{metric}" / "eval_meta.json").read_text(encoding="utf-8"))
            self.assertEqual((meta["evaluator_id"], meta["adapter"], meta["metric"]), (eid, W.ADAPTER, metric))
            self.assertEqual(meta["sampler"], W.SAMPLER)
            self.assertEqual(meta["sampler_hash"], W.compact_hash(W.SAMPLER))
            # the probe ran the eval_videos.py before the publication cleanup (RESULT.md "Published code")
            self.assertEqual(meta["scorer"]["code_sha256"], W.PROBE_EVAL_VIDEOS_SHA256)
            self.assertEqual({**meta["scorer"], "code_sha256": W.EVAL_VIDEOS_SHA256},
                             registry().EVALUATOR_ROWS[eid]["scorer"])

    def test_lock_fixture_is_area5_s(self):
        self.assertEqual(file_sha256(W.LOCK_FIXTURE), W.LOCK_SHA)
        from gatedtrain_recipe_flow_grpo_wan_video import adapter as mod
        self.assertEqual((mod.VENV_LOCK_SHA256, mod.IMAGE, mod.SAMPLER_CODE_SHA256), (W.LOCK_SHA, W.IMAGE,
                                                                                      W.SAMPLER_CODE_SHA))


class TestRuntimeFacts(unittest.TestCase):
    """The real area 3 directory and GPU outputs (read only; skipped where absent)."""

    @unittest.skipUnless(REAL_EVAL.is_file() and REAL_TRAIN.is_file(), "the area-3 video runtime is absent")
    def test_runtime_hashes(self):
        self.assertEqual(file_sha256(REAL_EVAL), W.EVAL_VIDEOS_SHA256)
        self.assertEqual(W.shared_sampler_sha256(REAL_TRAIN), W.SAMPLER_CODE_SHA)
        self.assertEqual(W.shared_sampler_sha256(REAL_EVAL), W.SAMPLER_CODE_SHA)

    @unittest.skipUnless(REAL_EVAL.is_file() and REAL_TRAIN.is_file(), "the area-3 video runtime is absent")
    def test_runtime_constants(self):
        from gatedtrain_recipe_flow_grpo_wan_video import adapter as mod
        t = module_constants(REAL_TRAIN, ("LORA", "COMMON", "PROFILES", "CUT_FILES", "DEPLOYABLE", "SAMPLE_ID_DOMAIN",
                                          "FRAME_INTERVAL_A", "BASE_NAME", "CUT_MARKER", "LORA_WEIGHTS"))
        self.assertEqual(sorted(t["LORA"]["target_modules"]), sorted(mod.WAN_LORA_TARGETS))
        self.assertEqual((t["SAMPLE_ID_DOMAIN"], t["FRAME_INTERVAL_A"]), (adapter().sample_id_domain, mod.FRAME_INTERVAL))
        self.assertEqual((t["COMMON"]["reward"], t["COMMON"]["prompt_order"], t["COMMON"]["num_frames"],
                          t["COMMON"]["num_batches_per_epoch"], t["COMMON"]["ema"]), ("video_ocr", "file", 17, 2, False))
        self.assertEqual(tuple(t["DEPLOYABLE"]), tuple(registry().TRAINER_ROWS[W.ADAPTER]["components"]))
        self.assertEqual(tuple(t["CUT_FILES"]), W.CUT_FILES)
        self.assertEqual(sorted(t["PROFILES"]), sorted(mod.PROFILES))
        self.assertEqual(f"models/{t['BASE_NAME']}", mod.BASE_REL)
        e = module_constants(REAL_EVAL, ("ADAPTER", "METRICS", "WAN_LORA_TARGETS"))
        self.assertEqual(e["ADAPTER"], W.ADAPTER)
        self.assertEqual({m: v[0] for m, v in e["METRICS"].items()}, {W.METRIC: W.EVALUATOR_ID,
                                                                      W.SUPPORT_METRIC: W.SUPPORT_ID})
        self.assertEqual(tuple(e["WAN_LORA_TARGETS"]), mod.WAN_LORA_TARGETS)

    @unittest.skipUnless(REAL_EVAL.is_file() and REAL_TRAIN.is_file(), "the area-3 video runtime is absent")
    def test_argparse_options_are_the_adapter_s(self):
        from gatedtrain_recipe_flow_grpo_wan_video import adapter as mod
        self.assertEqual(argparse_options(REAL_TRAIN), set(mod.TRAIN_OPTIONS))
        self.assertEqual(argparse_options(REAL_EVAL), set(mod.EVAL_OPTIONS))

    @unittest.skipUnless((REAL_PROBE / "samples.jsonl").is_file() and (REAL_AREA / "prompts").is_dir(),
                         "the area-3 video GPU smoke is absent")
    def test_real_probe_consumption(self):
        """The real probe-train-1 outputs (94 MB LoRA included) against the real 18,359-row shard."""
        a = adapter()
        cuts = a.find_cuts(REAL_PROBE)[0]
        self.assertEqual([s for s, _ in cuts], [4])
        self.assertEqual(a.consumption_problems(REAL_PROBE, REAL_AREA / "prompts", cuts), [])

    @unittest.skipUnless((REAL_AREA / W.BASE_REL / "model_index.json").is_file(), "the real Wan2.1 base is absent")
    def test_real_base(self):
        base = REAL_AREA / W.BASE_REL
        self.assertIsNone(adapter().base_dir_problem(base))
        models = json.loads((REAL_AREA / "models" / "MODELS.json").read_text(encoding="utf-8"))["models"]
        rec = models["wan2.1-t2v-1.3b"]
        self.assertEqual((rec["repo"], rec["revision"]), (W.REPO, W.REVISION))
        on_disk = sorted(p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file())
        self.assertEqual(on_disk, sorted(rec["files"]))
        self.assertEqual(tuple(on_disk), W.BASE_FILES)
        self.assertFalse((base / ".cache").exists())

    @unittest.skipUnless((REAL_AREA / "items" / "items.jsonl").is_file()
                         and (REAL_AREA / "prompts" / "train.jsonl").is_file(),
                         "the runtime's items or its regenerated training shard are absent")
    def test_real_items_and_prefixes(self):
        from gatedtrain.ext.collect import read_item_set
        row = registry().EVALUATOR_ROWS[W.EVALUATOR_ID]
        rows, bad = read_item_set(REAL_AREA / "items", tuple(row["item_payload"]))
        self.assertEqual((len(rows), bad), (400, []))
        for fixture, real in ((W.FG / "items.jsonl", REAL_AREA / "items" / "items.jsonl"),
                              (W.FG / "train.jsonl", REAL_AREA / "prompts" / "train.jsonl")):
            data = fixture.read_bytes()
            with open(real, "rb") as f:
                self.assertEqual(f.read(len(data)), data, real)
        self.assertEqual(adapter().item_overlap_problems(rows, REAL_AREA / "prompts"), [])

    @unittest.skipUnless(REAL_LOCK.is_file(), "area 5's lock is absent")
    def test_real_lock(self):
        self.assertEqual(file_sha256(REAL_LOCK), W.LOCK_SHA)
        self.assertEqual(REAL_LOCK.read_bytes(), W.LOCK_FIXTURE.read_bytes())

    @unittest.skipUnless((REAL_PROBE_EVAL / "summary.json").is_file(), "the area-3 video base probe is absent")
    def test_real_probe_eval_fixture_copies(self):
        for rel in (f"arm-{W.METRIC}/eval_meta.json", f"arm-{W.METRIC}/items.jsonl", f"arm-{W.SUPPORT_METRIC}/items.jsonl",
                    "videos/ocr-test-0000/r0/frames.json"):
            self.assertEqual((W.PROBE_EVAL / rel).read_bytes(), (REAL_PROBE_EVAL / rel).read_bytes(), rel)

    @unittest.skipUnless(REAL_AREA.is_dir() and W.REAL_AREA5.is_dir(), "the runtime directories are absent")
    def test_credential_scan_of_the_real_mounts(self):
        from gatedtrain.ext.launch import credential_name_problem
        row = registry().TRAINER_ROWS[W.ADAPTER]
        for root in (str(REAL_AREA), str(W.REAL_AREA5)):
            with self.subTest(root=root):
                self.assertIsNone(credential_name_problem(root, skip=tuple(row["scan_skip"]),
                                                          interpreter_trees=tuple(row["interpreter_trees"])))


# ---------------------------------------------------------------------------
# EXT-228: the sibling adapters stay apart; area 5 keeps its rules
# ---------------------------------------------------------------------------

# Area 5's adapter `flow_grpo_sd35` lives in its recipe repository (gatedtrain-recipe-flow-grpo-sd35-ocr); the two checks that
# need it run where that recipe is installed.
AREA5_INSTALLED = "flow_grpo_sd35" in registry().ADAPTERS


class TestSiblings(WanCase):

    @unittest.skipUnless(AREA5_INSTALLED, "the flow_grpo_sd35 recipe is not installed")
    def test_area5_adapter_refuses_the_wan_run(self):
        sc = self.fresh()
        proc = run_cli(["ext-collect", str(sc.paths["spec"]), "--adapter", "flow_grpo_sd35", "--stage-out",
                        str(sc.stage_out), "--launch-record", str(sc.launch), "--smoke-record", str(sc.smoke),
                        "--eval-baseline", str(sc.baseline), "--eval-candidate", str(sc.candidate), "--runs-root",
                        str(self.runs())], cwd=sc.root, env=sc.env())
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertTrue([e for e in parse_errors(proc.stderr) if e["rule"] == "EXT-SPEC"])

    @unittest.skipUnless(AREA5_INSTALLED, "the flow_grpo_sd35 recipe is not installed")
    def test_wan_spec_needs_the_wan_id(self):
        sc = self.fresh()
        sc.render(spec_mutate=lambda d: d["stages"][0].update(uses="rl.flow_grpo@0.1"))
        self.refused("EXT-SPEC", "stages[0].uses", sc=sc)

    def test_wan_extends_the_family(self):
        """The Wan adapter extends the flow_grpo family base directly, not area 5's adapter (whose facts are checked in
        its recipe repository, gatedtrain-recipe-flow-grpo-sd35-ocr)."""
        from gatedtrain_recipe_flow_grpo_wan_video.adapter import FlowGrpoWanAdapter
        from gatedtrain.families.flow_grpo import FlowGrpoAdapter
        self.assertEqual(FlowGrpoWanAdapter.__bases__, (FlowGrpoAdapter,))



# ---------------------------------------------------------------------------
# Attempt 2 (amendment): EXT-229 runtime facts, EXT-230 argv, EXT-231 the two attempts apart, EXT-232 GPU cap
# ---------------------------------------------------------------------------

REAL_TRAIN_V2 = REAL_AREA / "train_flow_grpo_wan_v2.py"
REAL_RECORDS = W.REAL_RECORDS
REAL_FULL1 = W.REAL_OUT / "full-1"
# The published train_flow_grpo_wan.py. The first attempt's launch records pin the script it ran
# (sha256:da064300..., EXT-83), which differs only in comments, names and how it locates the OCR runtime (RESULT.md
# "Published code").
TRAIN_V1_SHA256 = "sha256:bac3ea77d62a1beedcb9b2c305e2a26f9490cc5cc0847dbfd3c68564fc27f0b8"


class TestAttempt2RuntimeFacts(unittest.TestCase):
    """EXT-229: the real attempt-2 files (read only; skipped where absent)."""

    @unittest.skipUnless(REAL_TRAIN_V2.is_file() and REAL_TRAIN.is_file(), "the attempt-2 trainer is absent")
    def test_v2_script_facts(self):
        from gatedtrain_recipe_flow_grpo_wan_video import adapter as mod
        self.assertEqual(file_sha256(REAL_TRAIN), TRAIN_V1_SHA256)             # attempt 1's script never changes
        self.assertEqual(W.shared_sampler_sha256(REAL_TRAIN_V2), W.SAMPLER_CODE_SHA)
        self.assertEqual(argparse_options(REAL_TRAIN_V2), set(mod.TRAIN_OPTIONS))
        names = ("LORA", "COMMON", "PROFILES", "CUT_FILES", "DEPLOYABLE", "SAMPLE_ID_DOMAIN", "FRAME_INTERVAL_A",
                 "BASE_NAME", "CUT_MARKER", "LORA_WEIGHTS")
        v1, v2 = module_constants(REAL_TRAIN, names), module_constants(REAL_TRAIN_V2, names)
        self.assertEqual({k: v for k, v in v2.items() if k != "PROFILES"}, {k: v for k, v in v1.items() if k != "PROFILES"})
        self.assertEqual(sorted(v1["PROFILES"]), sorted(mod.SCRIPT_PROFILES[mod.TRAIN_SCRIPT].values()))
        self.assertEqual(sorted(v2["PROFILES"]), sorted(mod.SCRIPT_PROFILES[mod.TRAIN_SCRIPT_V2].values()))
        self.assertEqual({k: (v["prompts_per_epoch"], v["images_per_prompt"], v["train_batch"], v["sample_batch"],
                              v["epochs"], v["save_every"]) for k, v in v2["PROFILES"].items()},
                         {"smoke2": (8, 4, 2, 4, 2, 2), "full2": (8, 4, 2, 4, 100, 25)})
        text = REAL_TRAIN_V2.read_text(encoding="utf-8")
        self.assertIn(f'DEFAULT_PROMPTS = HERE / "{mod.PROMPTS_REL[mod.TRAIN_SCRIPT_V2]}"', text)
        self.assertIn(f'DEFAULT_PROMPTS = HERE / "{mod.PROMPTS_REL[mod.TRAIN_SCRIPT]}"\n',
                      REAL_TRAIN.read_text(encoding="utf-8"))

    @unittest.skipUnless((REAL_AREA / "items-short" / "items.jsonl").is_file()
                         and (REAL_AREA / "prompts-short" / "train.jsonl").is_file(),
                         "the attempt-2 items or its regenerated training shard are absent")
    def test_real_short_items_and_shard(self):
        from gatedtrain.ext.collect import read_item_set
        row = registry().EVALUATOR_ROWS[W.EVALUATOR_ID]
        rows, bad = read_item_set(REAL_AREA / "items-short", tuple(row["item_payload"]))
        self.assertEqual((len(rows), bad), (288, []))
        shard = W.read_jsonl(REAL_AREA / "prompts-short" / "train.jsonl")
        self.assertEqual(len(shard), 5024)
        for r in rows + shard:
            self.assertLessEqual(len(r["target"].split()), 2)
        self.assertEqual(adapter().item_overlap_problems(rows, REAL_AREA / "prompts-short"), [])
        self.assertFalse({r["prompt"] for r in rows} & {r["prompt"] for r in shard})
        self.assertNotEqual(manifest_hash(REAL_AREA / "items-short"), manifest_hash(REAL_AREA / "items"))

    @unittest.skipUnless((REAL_FULL1 / "run_args.json").is_file() and REAL_RECORDS.is_dir(),
                         "the first attempt's run is absent")
    def test_first_attempt_still_checks(self):
        """Read only: the first attempt's records still name today's scripts (EXT-83), their argv passes the rules, and
        its run passes the attempt binding and the consumption check against the real 18,359-row shard."""
        from gatedtrain_recipe_flow_grpo_wan_video import adapter as mod
        a = adapter()
        for name in ("train-smoke-1", "train-full-1", "eval-smoke-base-1", "eval-smoke-cut-1", "eval-base-1",
                     "eval-cand-1"):
            with self.subTest(record=name):
                rec = json.loads((REAL_RECORDS / f"{name}.launch.json").read_text(encoding="utf-8"))
                self.assertEqual(file_sha256(Path(rec["script_path"])), rec["script_sha256"])
                self.assertEqual(a.argv_problems(rec["argv"], profile=rec["profile"]), [])
                self.assertEqual(a.launch_problems(rec, kind="record"), [])
        launch = json.loads((REAL_RECORDS / "train-full-1.launch.json").read_text(encoding="utf-8"))
        args = json.loads((REAL_FULL1 / "run_args.json").read_text(encoding="utf-8"))
        self.assertEqual(mod.FlowGrpoWanAdapter._attempt_problems(args, launch), [])
        cuts = a.find_cuts(REAL_FULL1)[0]
        self.assertEqual([s for s, _ in cuts], [30, 60, 90, 120])
        self.assertEqual(a.consumption_problems(REAL_FULL1, REAL_AREA / "prompts", cuts), [])


class TestAttempt2Argv(unittest.TestCase):
    """EXT-230: each training script's profiles and prompt directory; the evaluator's two item directories."""

    def problems(self, argv: list, profile: str | None = None) -> list[str]:
        return adapter().argv_problems(argv, profile=profile)

    def test_training_scripts_and_profiles(self):
        def argv(script, prof, *extra):
            return [W.INTERPRETER, script, "--profile", prof, "--out", "/out", "--seed", "5000", *extra]
        for script, launch, prof in ((W.TRAIN_PY_V2, "smoke", "smoke2"), (W.TRAIN_PY_V2, "full", "full2"),
                                     (W.TRAIN_PY, "smoke", "smoke"), (W.TRAIN_PY, "full", "full")):
            with self.subTest(script=script, launch=launch):
                self.assertEqual(self.problems(argv(script, prof), launch), [])
                self.assertEqual(self.problems(argv(script, prof)), [])
        self.assertEqual(self.problems(argv(W.TRAIN_PY_V2, "full2", "--epochs", "100", "--save-every", "25"), "full"),
                         [])
        for script, launch, prof in ((W.TRAIN_PY_V2, "smoke", "smoke"), (W.TRAIN_PY_V2, "full", "full"),
                                     (W.TRAIN_PY_V2, "smoke", "full2"), (W.TRAIN_PY_V2, "full", "smoke2"),
                                     (W.TRAIN_PY, "smoke", "smoke2"), (W.TRAIN_PY, "full", "full2"),
                                     (W.TRAIN_PY_V2, None, "cpu-test")):
            with self.subTest(bad=(script, launch, prof)):
                self.assertTrue(self.problems(argv(script, prof), launch))
        self.assertTrue(self.problems(argv(W.TRAIN_PY_V2, "smoke2", "--no-verify-base"), "smoke"))

    def test_prompts_only_the_script_s_own(self):
        def argv(script, prof, prompts):
            return [W.INTERPRETER, script, "--profile", prof, "--out", "/out", "--seed", "5000", "--prompts", prompts]
        self.assertEqual(self.problems(argv(W.TRAIN_PY_V2, "smoke2", f"{W.AREA3V}/prompts-short"), "smoke"), [])
        self.assertEqual(self.problems(argv(W.TRAIN_PY, "smoke", f"{W.AREA3V}/prompts"), "smoke"), [])
        self.assertEqual(self.problems(argv(W.TRAIN_PY, "smoke", f"{W.AREA3V}/prompts/"), "smoke"), [])
        for script, prof, prompts in ((W.TRAIN_PY_V2, "smoke2", f"{W.AREA3V}/prompts"),
                                      (W.TRAIN_PY, "smoke", f"{W.AREA3V}/prompts-short"),
                                      (W.TRAIN_PY_V2, "smoke2", f"{W.AREA5}/prompts")):
            with self.subTest(script=script, prompts=prompts):
                msgs = self.problems(argv(script, prof, prompts), "smoke")
                self.assertTrue(msgs)
                self.assertIn("EXT-230", msgs[0])

    def test_evaluator_items(self):
        base = [W.INTERPRETER, W.EVAL_PY, "--base-only", "--out", "/out", "--eval-seeds", "11"]
        for items in ("items", "items-short"):
            with self.subTest(items=items):
                self.assertEqual(self.problems(base + ["--items", f"{W.AREA3V}/{items}"], "full"), [])
                self.assertEqual(self.problems(base + ["--items", f"{W.AREA3V}/{items}", "--limit", "40"], "smoke"),
                                 [])
                self.assertTrue(self.problems(base + ["--items", f"{W.AREA3V}/{items}", "--limit", "40"], "full"))
        for bad in (f"{W.AREA3V}/items-long", f"{W.AREA5}/items", "/inputs/items-short", "items-short"):
            with self.subTest(bad=bad):
                self.assertTrue(self.problems(base + ["--items", bad], "full"))


class TestAttempt2Collect(WanCase):
    """EXT-231: an attempt-2 run (v2 script, profile full2, prompts-short/, items-short/) collects; mixing either
    attempt's script, shard, items or registration with the other's refuses."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.sc2 = W.WanScenario(cls.tmp / "sc-attempt2", attempt=2)

    def test_attempt2_collects(self):
        sc = self.sc2
        args = json.loads((sc.stage_out / "run_args.json").read_text(encoding="utf-8"))
        self.assertEqual((args["profile"], args["settings"]["samples_per_epoch"], args["train_file"]),
                         ("full2", 32, f"{W.AREA3V}/prompts-short/train.jsonl"))
        run_dir = self.passes(sc)
        rep = json.loads((run_dir / "report" / "gate_report.json").read_text(encoding="utf-8"))
        self.assertEqual((rep["status"], rep["n_items"], rep["consumption_precision"]), ("PASS", 36, "row"))
        self.assertEqual(rep["contamination"], {"exact_overlap": 0, "allowed": False})
        _, plans, steps = sc.logs()
        self.assertEqual([p["steps"] for p in plans], [[1, 2], [3, 4], [5, 6]])
        self.assertEqual(sum(len(p["kept"]) + len(p["filtered"]) for p in plans), 96)            # 3 x 32
        self.assertEqual(manifest_hash(sc.items_dir), sc.item_set_hash)
        self.assertNotEqual(sc.item_set_hash, manifest_hash(sc.runtime / "items"))

    def test_attempt2_launches(self):
        sc = self.sc2
        out = sc.new_out("smoke2-1")
        path = self.tmp / "a2-train.request.yaml"
        path.write_text(yaml.safe_dump(sc.request(out), sort_keys=False), encoding="utf-8")
        proc = run_cli(["ext-launch", "--adapter", W.ADAPTER, "--profile", "smoke", "--request", str(path),
                        "--gpu-log", str(sc.host_work / "gpu-log.md"), "--out-record", str(self.tmp / "a2.json")],
                       cwd=sc.root, env=sc.env())
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        self.assertIn(f"{W.INTERPRETER} {W.TRAIN_PY_V2} --profile smoke2", proc.stdout)
        self.assertTrue(proc.stdout.startswith("GATEDTRAIN_GPU_CAP_H=1000 "), proc.stdout)
        out = sc.new_out("eval-smoke-short-1")
        path = self.tmp / "a2-eval.request.yaml"
        path.write_text(yaml.safe_dump(sc.eval_request(out), sort_keys=False), encoding="utf-8")
        proc = run_cli(["ext-launch", "--adapter", W.ADAPTER, "--profile", "smoke", "--request", str(path),
                        "--gpu-log", str(sc.host_work / "gpu-log.md"), "--out-record", str(self.tmp / "a2e.json")],
                       cwd=sc.root, env=sc.env())
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        self.assertIn(f"--items {W.AREA3V}/items-short", proc.stdout)

    def test_old_shard_under_the_new_spec(self):
        """An attempt-1 run (prompts/) collected with an attempt-2 spec (prompts-short/, items-short/) refuses."""
        sc = self.fresh()
        sc.data, sc.items_dir = sc.runtime / "prompts-short", sc.runtime / "items-short"
        sc.render()
        self.refused("EXT-DATA", sc=sc)

    def test_v2_run_on_the_old_shard(self):
        """A full2 run whose train_file is prompts/ (as with `--prompts prompts`) refuses even under a spec naming
        prompts/: each script trains on its own attempt's shard."""
        sc = self.fresh(attempt=2)
        sc.edit_json(sc.stage_out / "run_args.json",
                     lambda a: a.update(train_file=f"{W.AREA3V}/prompts/train.jsonl"))
        self.refused("EXT-DATA", "--stage-out", sc=sc, text="EXT-231")

    def test_profile_script_mismatch(self):
        sc = self.fresh(attempt=2)
        launch = sc.rewrite_launch(script_path=W.TRAIN_PY, script_sha256=file_sha256(sc.runtime / "train_flow_grpo_wan.py"),
                                   argv=[W.INTERPRETER, W.TRAIN_PY, "--profile", "full", "--out", "/out", "--seed",
                                         str(W.SEED)])
        self.refused("EXT-LAUNCH", "--launch-record.script_path", sc=sc, text="EXT-231", launch=launch)
        sc = self.fresh(attempt=2)
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a.update(profile="cpu-test"))
        self.refused("EXT-STAGE", "--stage-out", sc=sc, text="EXT-231")

    def test_short_items_arm_under_the_old_registration(self):
        """Arms evaluated on items-short/ refuse under the attempt-1 registration (items/), whether ext-arm-fill hashed
        the short items or the arm claims the registered hash."""
        sc = self.fresh()
        short = W.read_jsonl(sc.runtime / "items-short" / "items.jsonl")
        for claimed in (manifest_hash(sc.runtime / "items-short"), sc.item_set_hash):
            with self.subTest(claimed=claimed[:16]):
                type(self).n += 1
                ev = W.build_eval(sc.output_root / "area3-video" / f"eval-short-{type(self).n}", short, W.baseline_score,
                                  tag="short", model=sc.base_model, base_model=sc.base_model, item_set=claimed)
                self.refused("EXT-ARM", "--eval-baseline", sc=sc, baseline=ev / f"arm-{W.METRIC}")

    def test_old_items_arm_under_the_new_registration(self):
        sc = self.fresh(attempt=2)
        old = W.read_jsonl(sc.runtime / "items" / "items.jsonl")
        ev = W.build_eval(sc.output_root / "area3-video" / "eval-old-items", old, W.candidate_score, tag="old",
                          model=sc.cand_model, base_model=sc.base_model, item_set=sc.item_set_hash)
        self.refused("EXT-ARM", "--eval-candidate", sc=sc, candidate=ev / f"arm-{W.METRIC}")


class TestGpuCap(unittest.TestCase):
    """EXT-232: the launcher passes a GPU-hour cap of 1000 to gpu-step.sh (in effect no cap; hours are still
    logged) and the smoke budget rule uses the same cap."""

    def test_cap_values(self):
        from gatedtrain.ext import launch, smoke
        self.assertEqual((launch.GPU_CAP_H, smoke.GPU_CAP_H), (1000, 1000))
        v = smoke.smoke_verdict(W.smoke_record(base_model="sha256:" + "0" * 64, item_set="sha256:" + "1" * 64,
                                               projected_gpu_h=12.0, gpu_log_total_h=16.7),
                                score_range=(0, 1), binary=False)
        self.assertEqual(v["reasons"], [])
        v = smoke.smoke_verdict(W.smoke_record(base_model="sha256:" + "0" * 64, item_set="sha256:" + "1" * 64,
                                               projected_gpu_h=12.0, gpu_log_total_h=990.0),
                                score_range=(0, 1), binary=False)
        self.assertIn("budget", v["reasons"])


if __name__ == "__main__":
    unittest.main()
