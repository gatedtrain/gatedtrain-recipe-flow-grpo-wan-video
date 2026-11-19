#!/usr/bin/env python
"""CPU tests of train_flow_grpo_wan.py (no GPU, no network, no Wan weights: of the real model directory only
scheduler/scheduler_config.json is read).

Run with area 5's venv (the runtime of this area), from anywhere:
  env -u HF_TOKEN CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 HOME=<scratch> \
      <sd35-ocr-runtime>/.venv/bin/python -I -B tests/test_train_flow_grpo_wan.py [-k pattern]

What is checked:
  - the shared sampler block is byte-identical to eval_videos.py's and has the pinned SHA-256;
  - the deployment sampler.json from the real scheduler config survives a JSON round trip and has the GPU probe's
    sampler_hash (the GPU probe probe-base-1/summary.json);
  - settings of the profiles (flow_grpo general_ocr_wan2_1 values, 17 frames, EMA off) and argument rules;
  - prompts in the shard's file order; the advantage tracker against flow_grpo's own PerPromptStatTracker;
  - one trained timestep (policy loss with PPO clip, KL against the adapter-disabled transformer) and its gradient
    against flow_grpo's formula (compute_log_prob, train_wan2_1.py:190-230, :936-969) with flow_grpo's own
    sde_step_with_logprob, micro-batch 2 accumulated = flow_grpo's micro-batch 4;
  - the real reward (PaddleOCR engine A, area 5's weights) on a rendered 17-frame text video;
  - main() end to end on a tiny random base (3 epochs, stub reward that exercises the zero-advantage filter, its
    padding and the all-zero +1e-6 rule; and 1 epoch with the real PaddleOCR reward): run records, samples.jsonl,
    epoch_plan.jsonl, train_steps.jsonl, cuts (cut.json last, hashes, counts), sampler.json equal to the evaluator's,
    eval_videos.check_cut accepts the cut and eval_videos.evaluate loads its LoRA.
Only the generation weights are tiny and random; `encode_all_prompts` is replaced (a stub tokenizer and a tiny UMT5
stand in for the umT5 files) and, in the first end-to-end run, `PaddleOcrA` by a scripted stub.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
AREA = HERE.parent
sys.path.insert(0, str(AREA))
sys.path.insert(0, str(HERE))

import eval_videos as ev  # noqa: E402
import test_eval_videos as tev  # noqa: E402  (tiny parts, stub tokenizer)
import train_flow_grpo_wan as tw  # noqa: E402

FLOW_GRPO = tev.FLOW_GRPO
SAMPLER_SHA = "sha256:41606b0b53ad637da4614aaa38fd52b5e9bf0b2147da7ed71c3eaad92dcb1f17"
SCRATCH = Path(os.environ.get("TEST_SCRATCH") or tempfile.mkdtemp(prefix="train_wan_test_"))
TINY = {"num_steps": 4, "deploy_num_steps": 3, "height": 32, "width": 48, "num_frames": 5, "precision": "fp32",
        "ocr_cpu_threads": 4}


def load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def flow_grpo_pipe_module():
    sys.path.insert(0, str(FLOW_GRPO))
    try:
        return load_by_path("flow_grpo_wan_pipe_t", FLOW_GRPO / "flow_grpo" / "diffusers_patch" /
                            "wan_pipeline_with_logprob.py")
    finally:
        sys.path.remove(str(FLOW_GRPO))


def shared_block(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    return text[text.index("# BEGIN SHARED SAMPLER"):text.index("# END SHARED SAMPLER")]


class TestSharedAndSettings(unittest.TestCase):
    def test_shared_block_identical(self):
        self.assertEqual(shared_block(AREA / "train_flow_grpo_wan.py"), shared_block(AREA / "eval_videos.py"))
        self.assertEqual(tw.sampler_code_sha256(), SAMPLER_SHA)
        self.assertEqual(ev.sampler_code_sha256(), SAMPLER_SHA)
        # the cut binding constants the evaluator checks
        self.assertEqual(sorted(tw.LORA["target_modules"]), sorted(ev.WAN_LORA_TARGETS))
        self.assertEqual((tw.LORA_WEIGHTS, tw.BASE_NAME, tw.MODELS_JSON, tw.DEFAULT_BASE),
                         (ev.LORA_WEIGHTS, ev.BASE_NAME, ev.MODELS_JSON, ev.DEFAULT_BASE))
        self.assertEqual({k: tw.COMMON["deploy_num_steps" if k == "num_steps" else k] for k in ev.SAMPLER_DEFAULT},
                         ev.SAMPLER_DEFAULT)
        self.assertEqual(tw.OCR_MODELS, ev.OCR_MODELS)

    def test_deployment_sampler_of_the_real_scheduler(self):
        from diffusers import UniPCMultistepScheduler

        sch = UniPCMultistepScheduler.from_pretrained(str(tw.DEFAULT_BASE), subfolder="scheduler",
                                                      local_files_only=True)
        s = tw.COMMON
        ts, sig, idx = tw.schedule(sch, s["deploy_num_steps"])
        doc = tw.sampler_doc(s["deploy_num_steps"], s["guidance_scale"], s["height"], s["width"], s["num_frames"],
                             s["max_sequence_length"], dict(sch.config), ts, sig, idx, s["precision"], s["precision"],
                             deterministic=True)
        self.assertEqual(json.loads(json.dumps(doc)), doc)
        # the base-only sampler_hash of the GPU render probe (eval_videos.py defaults, README "Evaluator")
        self.assertEqual(tw.sha256_bytes(tw.canonical(doc)),
                         "sha256:997c2bf0a53f49a96ec0168559589b7bbd0f62f9b54166952b5f16ebad5ff9e3")
        self.assertEqual(ts.tolist()[:3] + ts.tolist()[-1:], [999, 982, 964, 138])
        self.assertEqual(idx, list(range(20)))

    def test_profiles(self):
        s = tw.resolve_settings(tw.parse_args(["--profile", "smoke", "--out", "/x"]))
        self.assertEqual((s["samples_per_epoch"], s["num_batches_per_epoch"], s["gradient_accumulation_steps"],
                          s["num_train_timesteps"]), (16, 2, 1, 19))
        self.assertEqual((s["learning_rate"], s["clip_range"], s["beta"], s["global_std"], s["adv_clip_max"],
                          s["guidance_scale"], s["num_steps"], s["max_sequence_length"]),
                         (1e-4, 1e-3, 0.004, False, 5.0, 4.5, 20, 512))
        self.assertEqual((s["height"], s["width"], s["num_frames"], s["ema"], s["precision"]),
                         (240, 416, 17, False, "bf16"))
        self.assertEqual((s["epochs"], s["save_every"], s["train_batch"], s["sample_batch"], s["seed"]),
                         (2, 2, 2, 4, 5000))
        f = tw.resolve_settings(tw.parse_args(["--profile", "full", "--out", "/x", "--epochs", "37",
                                               "--save-every", "5", "--train-batch", "4", "--beta", "0"]))
        self.assertEqual((f["epochs"], f["save_every"], f["train_batch"], f["beta"]), (37, 5, 4, 0.0))
        self.assertEqual(tw.PROFILES["full"]["epochs"], tw.FULL_EPOCHS)
        for bad in (["--profile", "smoke", "--out", "/x", "--ema"], ["--profile", "cpu", "--out", "/x"],
                    ["--profile", "smoke"]):
            with self.assertRaises(SystemExit):
                tw.parse_args(bad)
        with self.assertRaises(SystemExit):
            tw.resolve_settings(tw.parse_args(["--profile", "smoke", "--out", "/x", "--train-batch", "0"]))

    def test_epoch_prompts_file_order(self):
        rows = [{"prompt_id": f"p{i}"} for i in range(10)]
        self.assertEqual([r["prompt_id"] for r in tw.epoch_prompts(rows, 1, 4)], ["p0", "p1", "p2", "p3"])
        self.assertEqual([r["prompt_id"] for r in tw.epoch_prompts(rows, 2, 4)], ["p4", "p5", "p6", "p7"])
        self.assertEqual([r["prompt_id"] for r in tw.epoch_prompts(rows, 3, 4)], ["p8", "p9", "p0", "p1"])
        shard, _ = tw.read_prompts(AREA / "prompts")
        self.assertEqual(len(shard), 18359)
        self.assertEqual(tw.epoch_prompts(shard, 1, 4)[0]["prompt_id"], "ocr-train-00000")

    def test_tracker_matches_flow_grpo(self):
        import numpy as np

        st = load_by_path("flow_grpo_stat", FLOW_GRPO / "flow_grpo" / "stat_tracking.py")
        rng = np.random.default_rng(0)
        prompts = [f"p{i // 4}" for i in range(16)]
        rewards = list(rng.random(16))
        rewards[4:8] = [0.25] * 4                                        # a group with equal rewards
        want = st.PerPromptStatTracker(global_std=False).update(prompts, rewards)
        got = tw.PerPromptStatTracker(False).update(prompts, rewards)
        self.assertTrue(np.array_equal(want, got))
        self.assertTrue((got[4:8] == 0).all())


class TestGrpoStep(unittest.TestCase):
    def test_terms_and_gradient_match_flow_grpo(self):
        import torch

        pipe_mod = flow_grpo_pipe_module()
        transformer, vae, scheduler = tev.tiny_parts()
        from peft import LoraConfig

        torch.manual_seed(5)
        transformer.add_adapter(LoraConfig(**tw.LORA))
        with torch.no_grad():
            for n, p in transformer.named_parameters():
                if "lora_B" in n:
                    p.normal_(0, 0.05)
        params = [p for p in transformer.parameters() if p.requires_grad]
        s = {**tw.COMMON, "num_steps": 4}
        torch.manual_seed(1)
        embeds = torch.randn(4, 512, tev.TEXT_DIM) * (torch.arange(512) < 9).view(1, -1, 1)
        neg = (torch.randn(1, 512, tev.TEXT_DIM) * (torch.arange(512) < 1).view(1, -1, 1)).expand(4, -1, -1)
        seeds = [tw.derive_seed("sample", 1, 1, i) for i in range(4)]
        _, chain, logp, ts = tw.sample_batch(transformer, vae, scheduler, embeds, neg, seeds, 4, 4.5, 32, 48, 5, "cpu",
                                             torch.float32, None, deterministic=False, keep_chain=True, decode=False)
        self.assertEqual(tuple(chain.shape[:2]), (4, 5))
        self.assertEqual(tuple(logp.shape), (4, 4))
        old = logp + torch.tensor([0.0, 2e-3, -2e-3, 5e-4]).view(4, 1)     # exercises both sides of the clip
        adv = torch.tensor([1.5, -0.7, 0.4, -2.0])
        tts, sig, idx = tw.schedule(scheduler, 4)
        T = int(4 * s["timestep_fraction"])
        self.assertEqual(T, 3)

        # flow_grpo, one micro-batch of 4 videos, accelerate averaging over T backward calls
        scheduler.set_timesteps(4)
        timesteps = scheduler.timesteps.repeat(4, 1)
        sample = {"latents": chain[:, :-1], "next_latents": chain[:, 1:], "timesteps": timesteps}
        config = types.SimpleNamespace(train=types.SimpleNamespace(cfg=True),
                                       sample=types.SimpleNamespace(guidance_scale=4.5))
        pipeline = types.SimpleNamespace(scheduler=scheduler)

        def compute_log_prob(sample, j, embeds, negative_embeds):                   # train_wan2_1.py:190-230
            noise_pred_text = transformer(hidden_states=sample["latents"][:, j], timestep=sample["timesteps"][:, j],
                                          encoder_hidden_states=embeds, attention_kwargs=None, return_dict=False)[0]
            noise_pred_uncond = transformer(hidden_states=sample["latents"][:, j],
                                            timestep=sample["timesteps"][:, j], encoder_hidden_states=negative_embeds,
                                            attention_kwargs=None, return_dict=False)[0]
            noise_pred = noise_pred_uncond + config.sample.guidance_scale * (noise_pred_text - noise_pred_uncond)
            return pipe_mod.sde_step_with_logprob(pipeline.scheduler, noise_pred.float(), sample["timesteps"][:, j],
                                                  sample["latents"][:, j].float(),
                                                  prev_sample=sample["next_latents"][:, j].float(),
                                                  return_dt_and_std_dev_t=True)

        for p in params:
            p.grad = None
        ref_terms = []
        for j in range(T):                                                          # :928-976
            prev_sample, log_prob, prev_sample_mean, std_dev_t, dt = compute_log_prob(sample, j, embeds, neg)
            with torch.no_grad():
                transformer.disable_adapters()
                _, _, prev_sample_mean_ref, _, dt_ref = compute_log_prob(sample, j, embeds, neg)
                transformer.enable_adapters()
            advantages = torch.clamp(adv, -s["adv_clip_max"], s["adv_clip_max"])
            ratio = torch.exp(log_prob - old[:, j])
            unclipped_loss = -advantages * ratio
            clipped_loss = -advantages * torch.clamp(ratio, 1.0 - s["clip_range"], 1.0 + s["clip_range"])
            policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
            kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=(1, 2, 3), keepdim=True) / (
                2 * (std_dev_t * dt_ref) ** 2)
            kl_loss = torch.mean(kl_loss)
            loss = policy_loss + s["beta"] * kl_loss
            (loss / T).backward()
            ref_terms.append((float(policy_loss.detach()), float(kl_loss.detach())))
        want = [p.grad.clone() for p in params]

        # this port: micro-batches of 2 accumulated, loss summed per video / (m * gas * T)
        for p in params:
            p.grad = None
        m, gas = 4, 1
        got_terms = []
        for j in range(T):
            pols, kls = [], []
            for c0 in (0, 2):
                bt = torch.arange(c0, c0 + 2)
                tj = torch.full((2,), int(tts[j]), dtype=torch.int64)
                sj = torch.full((2,), idx[j], dtype=torch.long)
                per, kl, lp, ratio = tw.grpo_terms(transformer, chain[bt, j], chain[bt, j + 1], old[bt, j],
                                                   torch.clamp(adv[bt], -5, 5), tj, sj, embeds[bt], neg[bt], sig, s,
                                                   torch.float32, None)
                ((per + s["beta"] * kl).sum() / (m * gas * T)).backward()
                pols.append(per.detach())
                kls.append(kl.detach())
            got_terms.append((float(torch.cat(pols).mean()), float(torch.cat(kls).mean())))
        got = [p.grad.clone() for p in params]
        for (wp, wk), (gp, gk) in zip(ref_terms, got_terms):
            self.assertAlmostEqual(wp, gp, places=6)
            self.assertAlmostEqual(wk, gk, delta=1e-6 * max(1.0, abs(wk)))
        self.assertTrue(any(k > 0 for _, k in got_terms), "the KL term is exercised")
        num = max(float((w - g).abs().max()) for w, g in zip(want, got))
        scale = max(float(w.abs().max()) for w in want)
        self.assertGreater(scale, 0)
        self.assertLess(num, 1e-5 * scale + 1e-9, (num, scale))


class TestRealReward(unittest.TestCase):
    def test_real_paddleocr_on_rendered_video(self):
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont

        engine = tw.PaddleOcrA(tw.OCR_MODELS, cpu_threads=4)
        font = ImageFont.truetype(tev.FONT, 44)
        frames = []
        for i in range(17):
            img = Image.new("RGB", (416, 240), (235, 230, 220))
            if i <= 8:
                ImageDraw.Draw(img).text((40, 90), "OPEN LATE", fill=(20, 20, 20), font=font)
            frames.append(np.asarray(img))
        res = engine.video_reward(np.stack(frames), 'A shop window with a sign that says "Open Late"')
        print("\ntrainer reward frames", [(f["frame"], f["text"], round(f["acc"], 3)) for f in res["frames"]])
        self.assertEqual([f["frame"] for f in res["frames"]], [0, 4, 8, 12, 16])
        self.assertEqual([f["acc"] for f in res["frames"]][3:], [0.0, 0.0])
        self.assertGreaterEqual(res["reward"], 0.85)
        self.assertEqual(res["reward"], tw.video_ocr_reward([f["acc"] for f in res["frames"]]))


class StubReward:
    """Scripted engine-A stand-in: per prompt marker and call order (= member, one prompt group per sampling batch)."""
    script = {"VARY": [0.1, 0.9, 0.3, 0.6], "ODD": [0.0, 0.125, 0.625, 0.25], "CONST": [0.0, 0.0, 0.0, 0.0]}

    def __init__(self, model_root, cpu_threads=8):
        self.calls = {}

    def video_reward(self, frames, prompt):
        assert frames.dtype.name == "uint8" and frames.shape[-1] == 3
        member = self.calls.get(prompt, 0)
        self.calls[prompt] = member + 1
        marker = prompt.split()[0]
        r = self.script[marker][member]
        err = "RuntimeError: synthetic" if (marker == "VARY" and member == 1) else None
        per_frame = [{"frame": f, "text": None if (err and f == 4) else "x",
                      "acc": 0.0 if (err and f == 4) else r, "error": err if f == 4 else None}
                     for f in range(0, len(frames), tw.FRAME_INTERVAL_A)]
        return {"frames": per_frame, "reward": tw.video_ocr_reward([f["acc"] for f in per_frame])}


def fake_encode_all_prompts(base_dir, prompts, device, dtype, max_sequence_length, batch_size=8):
    enc = tev.tiny_encoder().to(device=device, dtype=dtype)
    return tw.encode_prompts(tev.StubTokenizer(), enc, prompts, max_sequence_length, batch_size)


class TestMainEndToEnd(unittest.TestCase):
    """main() on a tiny random base on the CPU; the cut is then checked and evaluated by eval_videos.py."""

    @classmethod
    def setUpClass(cls):
        cls.root = SCRATCH / "train-e2e"
        if cls.root.exists():
            shutil.rmtree(cls.root)
        cls.base = cls.root / "base" / "tiny"
        cls.base.mkdir(parents=True)
        cls.models_json = tev.make_tiny_base(cls.base)
        cls.prompts = cls.root / "prompts"
        cls.prompts.mkdir()
        markers = ["VARY", "CONST", "ODD", "CONST", "CONST", "CONST"]
        rows = [{"prompt": f'{m} poster {i} reading "Text {i}"', "prompt_id": f"t-{i:02d}", "target": f"Text {i}"}
                for i, m in enumerate(markers)]
        (cls.prompts / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        cls.saved = (tw.MODELS_JSON, tw.encode_all_prompts, tw.PaddleOcrA, dict(tw.PROFILES),
                     ev.MODELS_JSON, ev.encode_all_prompts)
        tw.MODELS_JSON = ev.MODELS_JSON = cls.models_json
        tw.encode_all_prompts = fake_encode_all_prompts
        ev.encode_all_prompts = tev.fake_encode_all_prompts
        tw.PROFILES["cpu-test"] = {"epochs": 3, "prompts_per_epoch": 2, "images_per_prompt": 4, "sample_batch": 4,
                                   "train_batch": 2, "save_every": 2}
        cls.orig_common = dict(tw.COMMON)
        tw.COMMON.update(TINY)
        cls.common = ["--base", str(cls.base), "--prompts", str(cls.prompts), "--device", "cpu", "--seed", "77",
                      "--no-verify-base"]

    @classmethod
    def tearDownClass(cls):
        (tw.MODELS_JSON, tw.encode_all_prompts, tw.PaddleOcrA, profiles, ev.MODELS_JSON,
         ev.encode_all_prompts) = cls.saved
        tw.PROFILES.clear()
        tw.PROFILES.update(profiles)
        tw.COMMON.clear()
        tw.COMMON.update(cls.orig_common)

    def jl(self, path):
        return [json.loads(ln) for ln in path.read_text().splitlines()]

    def test_stub_reward_run_then_evaluate(self):
        import torch
        from safetensors.torch import load_file

        tw.PaddleOcrA = StubReward
        out = self.root / "run-stub"
        self.assertEqual(tw.main(["--profile", "cpu-test", "--out", str(out)] + self.common), 0)

        # run records
        freeze = (out / "env-freeze.txt").read_text().splitlines()
        self.assertEqual(freeze[-1], "# flow_grpo algorithm reference yifan123/flow_grpo "
                                     "879042cf5707f8b90daa98d147d7deac2317c5da (ported)")
        self.assertTrue(any(ln.startswith("diffusers==") for ln in freeze))
        rt = json.loads((out / "env-runtime.json").read_text())
        self.assertEqual(rt["sampler_code_sha256"], SAMPLER_SHA)
        self.assertEqual(rt["uv_lock"], str(tw.OCR_RUNTIME / "uv.lock"))
        args = json.loads((out / "run_args.json").read_text())
        s = args["settings"]
        self.assertEqual((args["seed"], s["samples_per_epoch"], s["images_per_prompt"], s["num_batches_per_epoch"],
                          s["gradient_accumulation_steps"], s["num_inner_epochs"], s["global_std"], s["reward"],
                          s["epochs"], s["num_train_timesteps"]),
                         (77, 8, 4, 2, 1, 1, False, "video_ocr", 3, 3))
        self.assertEqual(args["sampler_sha256"], tw.sha256_bytes(tw.canonical(args["sampler"])))
        self.assertFalse(args["training_sampler"]["deterministic"])
        att = self.jl(out / "attempts.jsonl")
        self.assertEqual((len(att), att[0]["attempt"], att[1]["global_step"]), (2, 1, 6))

        # samples.jsonl: 3 epochs x 8 videos, prompts in file order, derived fields
        samples = self.jl(out / "samples.jsonl")
        self.assertEqual(len(samples), 24)
        for x in samples:
            e, i = x["epoch"], x["sample_index"]
            self.assertEqual(x["prompt_id"], f"t-{2 * (e - 1) + i // 4:02d}")
            self.assertEqual(x["member"], i % 4)
            self.assertEqual(x["seed"], tw.derive_seed("sample", 77, e, i))
            self.assertEqual(x["sample_id"], tw.sha256_bytes(tw.canonical(
                ["flow-grpo-wan", 77, 1, e, i, x["prompt_id"], x["seed"], x["member"]])))
            self.assertEqual(x["target"], tw.target_text(x["prompt"]))
            self.assertEqual(x["reward"], x["ocr_acc"])
            self.assertEqual(x["reward"], tw.video_ocr_reward([f["acc"] for f in x["frame_ocr"]]))
            self.assertEqual([f["frame"] for f in x["frame_ocr"]], [0, 4])
            self.assertFalse(x["nonfinite"])
        err = [x for x in samples if x["ocr_error"]]
        self.assertEqual(len(err), 1)
        self.assertEqual((err[0]["epoch"], err[0]["member"], err[0]["reward"]), (1, 1, 0.9))  # frame 0 still counts
        self.assertTrue(err[0]["ocr_error"].startswith("frame 4: "))

        # epoch_plan.jsonl: the filter, the padding, the all-zero rule
        plans = self.jl(out / "epoch_plan.jsonl")
        sid = {(x["epoch"], x["sample_index"]): x["sample_id"] for x in samples}
        self.assertEqual([p["epoch"] for p in plans], [1, 2, 3])
        p1, p2, p3 = plans
        self.assertEqual(p1["kept"], [sid[(1, i)] for i in range(4)])       # VARY kept, CONST filtered
        self.assertEqual(p1["filtered"], [sid[(1, i)] for i in range(4, 8)])
        self.assertEqual((p1["reincluded"], p1["all_zero_advantage"], p1["steps"]), ([], False, [1, 2]))
        # ODD: member 3 has the group mean (advantage 0) -> 3 kept + 1 re-included zero-advantage sample
        self.assertEqual(len(p2["kept"]), 4)
        self.assertEqual(len(p2["reincluded"]), 1)
        self.assertTrue(set(p2["kept"]) >= {sid[(2, 0)], sid[(2, 1)], sid[(2, 2)]})
        self.assertEqual(len(p2["filtered"]), 4)
        self.assertEqual(p2["steps"], [3, 4])
        self.assertEqual((p3["all_zero_advantage"], len(p3["kept"]), p3["filtered"], p3["reincluded"], p3["steps"]),
                         (True, 8, [], [], [5, 6]))
        for p in plans:
            self.assertEqual(p["filter_reason"], "zero_advantage")

        # train_steps.jsonl: two steps per epoch, each kept sample exactly once
        steps = self.jl(out / "train_steps.jsonl")
        self.assertEqual([st["step"] for st in steps], [1, 2, 3, 4, 5, 6])
        for p in plans:
            sts = [st for st in steps if st["epoch"] == p["epoch"]]
            self.assertEqual(len(sts), 2)
            self.assertTrue(all(len(st["sample_ids"]) == len(p["kept"]) // 2 for st in sts))
            self.assertEqual(sorted(i for st in sts for i in st["sample_ids"]), sorted(p["kept"]))
        logs = self.jl(out / "train_log.jsonl")
        self.assertEqual(len(logs), 6)
        self.assertTrue(all(all(isinstance(v, float) and v == v for k, v in lg.items() if k not in ("step", "epoch"))
                            for lg in logs))
        mem = self.jl(out / "memory.jsonl")
        self.assertEqual([m["kept"] for m in mem], [4, 4, 8])
        self.assertEqual([m["all_zero_advantage"] for m in mem], [False, False, True])

        # cuts: epoch 2 (save_every) and 3 (last); cut.json last, hashes, counts
        self.assertEqual(sorted(p.name for p in out.iterdir() if "checkpoint" in p.name),
                         ["checkpoint-2", "checkpoint-3"])
        for name, (gs, consumed, filtered) in {"checkpoint-2": (4, 8, 8), "checkpoint-3": (6, 16, 8)}.items():
            cut = out / name
            marker = json.loads((cut / "cut.json").read_text())
            self.assertEqual(sorted(p.name for p in cut.iterdir()), sorted([*tw.CUT_FILES, "cut.json"]))
            self.assertEqual((marker["global_step"], marker["samples_consumed"], marker["samples_filtered"]),
                             (gs, consumed, filtered))
            self.assertEqual((marker["ema"], marker["seed"], marker["profile"], marker["sampler_code_sha256"]),
                             (False, 77, "cpu-test", SAMPLER_SHA))
            self.assertEqual(sorted(marker["deployable"]), sorted(tw.DEPLOYABLE))
            for f, h in marker["files"].items():
                self.assertEqual(tw.sha256_file(cut / f), h)
                self.assertLessEqual(os.stat(cut / f).st_mtime_ns, os.stat(cut / "cut.json").st_mtime_ns)
            ev.check_cut(cut)                                                  # the evaluator's binding checks
        cut = out / "checkpoint-3"
        cfg = json.loads((cut / "adapter_config.json").read_text())
        self.assertEqual((cfg["r"], cfg["lora_alpha"], cfg["weights"]), (32, 64, "raw"))
        self.assertEqual(cfg["target_modules"], sorted(ev.WAN_LORA_TARGETS))
        w = load_file(str(cut / tw.LORA_WEIGHTS))
        self.assertTrue(all(".lora_A." in k or ".lora_B." in k for k in w))
        self.assertTrue(all(k.startswith("blocks.") for k in w))
        self.assertTrue(any(k.endswith("attn2.to_q.lora_B.weight") for k in w))
        self.assertTrue(all(t.dtype == torch.float32 for t in w.values()))
        self.assertGreater(max(float(t.abs().max()) for k, t in w.items() if "lora_B" in k), 0.0)  # trained
        # sampler.json equals the evaluator's sampler for the same settings
        sampler = json.loads((cut / "sampler.json").read_text())
        eargs = ev.parse_args(["--checkpoint", str(cut), "--out", "/unused", "--dtype", "fp32", "--num-steps", "3",
                               "--height", "32", "--width", "48", "--num-frames", "5"])
        from diffusers import UniPCMultistepScheduler
        sch = UniPCMultistepScheduler.from_pretrained(str(self.base), subfolder="scheduler")
        self.assertEqual(ev.resolve_sampler(eargs, cut, sch), sampler)
        self.assertEqual(ev.resolve_sampler(eargs, None, sch), sampler)
        self.assertEqual(sampler, args["sampler"])
        # the evaluator loads the cut and generates
        ev_out = self.root / "eval-cut"
        self.assertEqual(ev.main(["--checkpoint", str(cut), "--out", str(ev_out), "--base", str(self.base),
                                  "--eval-seeds", "11", "--limit", "2", "--batch-size", "2", "--device", "cpu",
                                  "--dtype", "fp32", "--ocr-cpu-threads", "4"]), 0)
        summary = json.loads((ev_out / "summary.json").read_text())
        self.assertEqual((summary["lora"]["r"], summary["epoch"], summary["errors"]), (32, 3, 0))
        meta = json.loads((ev_out / "arm-ocr_b_acc" / "eval_meta.json").read_text())
        self.assertEqual(meta["sampler_hash"], tw.sha256_bytes(tw.canonical(sampler)))

        # a rerun into the same --out moves the first attempt aside (preemption rule)
        tw.PROFILES["cpu-test-1"] = {**tw.PROFILES["cpu-test"], "epochs": 1}
        self.assertEqual(tw.main(["--profile", "cpu-test-1", "--out", str(out)] + self.common), 0)
        self.assertTrue((out / "preempted-attempt-1" / "checkpoint-3" / "cut.json").exists())
        self.assertEqual([a["attempt"] for a in self.jl(out / "attempts.jsonl")], [1, 1, 2, 2])
        self.assertTrue(all(x["attempt"] == 2 for x in self.jl(out / "samples.jsonl")))

    def test_real_reward_one_epoch(self):
        """The real PaddleOCR reward on tiny random videos: nothing readable, every advantage 0, so flow_grpo's
        all-zero rule trains on every sample with advantage 1e-6."""
        tw.PaddleOcrA = self.saved[2]
        tw.PROFILES["cpu-real"] = {"epochs": 1, "prompts_per_epoch": 2, "images_per_prompt": 4, "sample_batch": 4,
                                   "train_batch": 4, "save_every": 1}
        out = self.root / "run-real"
        self.assertEqual(tw.main(["--profile", "cpu-real", "--out", str(out)] + self.common), 0)
        samples = self.jl(out / "samples.jsonl")
        self.assertEqual(len(samples), 8)
        for x in samples:
            self.assertEqual([f["frame"] for f in x["frame_ocr"]], [0, 4])
            self.assertIsNone(x["ocr_error"])
            self.assertEqual(x["reward"], tw.video_ocr_reward([f["acc"] for f in x["frame_ocr"]]))
        plan = self.jl(out / "epoch_plan.jsonl")[0]
        if all(x["reward"] == 0 for x in samples):
            self.assertTrue(plan["all_zero_advantage"])
            self.assertEqual(len(plan["kept"]), 8)
        self.assertEqual(json.loads((out / "checkpoint-1" / "cut.json").read_text())["global_step"], 2)
        ev.check_cut(out / "checkpoint-1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
