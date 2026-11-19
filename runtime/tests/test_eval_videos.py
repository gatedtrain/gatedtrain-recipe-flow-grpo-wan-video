#!/usr/bin/env python
"""CPU tests of eval_videos.py (no GPU, no network, no Wan weights: the real model directory is never read).

Run with area 5's venv (the SD3.5 OCR recipe's runtime venv, `<sd35-ocr-runtime>` below; set
GATEDTRAIN_RECIPE_FLOW_GRPO_WAN_VIDEO_OCR_RUNTIME when it is not the sibling checkout) and a flow_grpo clone at 879042cf
(FLOW_GRPO_DIR, default ../flow_grpo beside this repository), from anywhere:
  env -u HF_TOKEN CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 HOME=<scratch> \
      <sd35-ocr-runtime>/.venv/bin/python -I tests/test_eval_videos.py [-k pattern]

What is checked:
  - argument parsing; the item and shard copies are byte-identical to area 5's;
  - seed derivation equals area 5's derive_seed;
  - the metric functions on hand cases; the video_ocr port against flow_grpo's own OcrScorer_video_or_image
    (ocr.py:65-123) on synthetic recognized texts;
  - the umT5 prompt encoding against flow_grpo's wan_prompt_embedding._get_t5_prompt_embeds (tiny random UMT5);
  - the Euler-ODE sampler and decode against flow_grpo's wan_pipeline_with_logprob(determistic=True) (tiny random
    WanTransformer3DModel and AutoencoderKLWan, float32 on the CPU);
  - the 20-step schedule of the Wan scheduler config (flow sigmas, shift 3.0);
  - both real OCR engines (area 5's weights) on rendered text frames, on the CPU;
  - evaluate() end to end on a tiny random base (real OCR engines, CPU): outputs, MM-1/MM-2 arms, frame hashes,
    common random numbers across two runs, and a LoRA cut (binding checks, same sampler_hash, different frames).
Only the generation weights are tiny and random; nothing in eval_videos.py's real path is replaced except
`encode_all_prompts` (the stub tokenizer stands in for the umT5 tokenizer files).
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
AREA = HERE.parent
sys.path.insert(0, str(AREA))

import eval_videos as ev  # noqa: E402

# A clone of yifan123/flow_grpo at 879042cf5707f8b90daa98d147d7deac2317c5da (the reference the port is tested against).
FLOW_GRPO = Path(os.environ.get("FLOW_GRPO_DIR") or AREA.parent.parent / "flow_grpo")
AREA5_EVAL = ev.OCR_RUNTIME / "eval_images.py"                     # the SD3.5 OCR recipe's evaluator
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
# Wan2.1-T2V-1.3B-Diffusers scheduler/scheduler_config.json values that set the sigmas (area README "Model facts")
WAN_SCHEDULER = {"num_train_timesteps": 1000, "use_flow_sigmas": True, "flow_shift": 3.0,
                 "final_sigmas_type": "zero", "prediction_type": "flow_prediction"}
TEXT_DIM = 32
SCRATCH = Path(os.environ.get("TEST_SCRATCH") or tempfile.mkdtemp(prefix="eval_videos_test_"))


def load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class StubTokenizer:
    """Stands in for the umT5 tokenizer: one id per character, EOS 1, pad 0, the attention mask."""

    def __call__(self, prompts, padding, max_length, truncation, add_special_tokens, return_attention_mask,
                 return_tensors):
        import torch

        assert padding == "max_length" and truncation and add_special_tokens and return_attention_mask
        ids, mask = [], []
        for p in prompts:
            row = [2 + (ord(c) % 90) for c in p][: max_length - 1] + [1]
            mask.append([1] * len(row) + [0] * (max_length - len(row)))
            ids.append(row + [0] * (max_length - len(row)))

        class Out:
            pass
        out = Out()
        out.input_ids, out.attention_mask = torch.tensor(ids), torch.tensor(mask)
        return out


def tiny_encoder():
    import torch
    from transformers import UMT5Config, UMT5EncoderModel

    torch.manual_seed(0)
    cfg = UMT5Config(vocab_size=96, d_model=TEXT_DIM, d_kv=8, d_ff=64, num_layers=2, num_heads=4,
                     relative_attention_num_buckets=8, relative_attention_max_distance=16)
    return UMT5EncoderModel(cfg).eval()


def tiny_parts(seed=0):
    import torch
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanTransformer3DModel

    torch.manual_seed(seed)
    transformer = WanTransformer3DModel(patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=8,
                                        in_channels=16, out_channels=16, text_dim=TEXT_DIM, freq_dim=16, ffn_dim=32,
                                        num_layers=2, rope_max_seq_len=32).eval()
    vae = AutoencoderKLWan(base_dim=8, z_dim=16, dim_mult=[1, 1, 1, 1], num_res_blocks=1,
                           temperal_downsample=[False, True, True]).eval()
    scheduler = UniPCMultistepScheduler(**WAN_SCHEDULER)
    return transformer, vae, scheduler


def make_tiny_base(root: Path) -> Path:
    transformer, vae, scheduler = tiny_parts()
    transformer.save_pretrained(root / "transformer")
    vae.save_pretrained(root / "vae")
    scheduler.save_pretrained(root / "scheduler")
    files = {p.relative_to(root).as_posix(): {"sha256": ev.sha256_file(p), "bytes": p.stat().st_size}
             for p in sorted(root.rglob("*")) if p.is_file()}
    models = {"schema": "ptc_mm.video_models", "schema_version": "1.0",
              "models": {ev.BASE_NAME: {"repo": "test/tiny-wan", "revision": "0" * 40, "files": files}}}
    (root.parent / "MODELS.json").write_text(json.dumps(models), encoding="utf-8")
    return root.parent / "MODELS.json"


def fake_encode_all_prompts(base_dir, prompts, device, dtype, max_sequence_length, batch_size=8):
    enc = tiny_encoder().to(device=device, dtype=dtype)
    return ev.encode_prompts(StubTokenizer(), enc, prompts, max_sequence_length, batch_size)


class TestArgsAndItems(unittest.TestCase):
    def test_parse_args(self):
        a = ev.parse_args(["--base-only", "--out", "/tmp/x", "--limit", "20"])
        self.assertEqual((a.batch_size, a.dtype, a.limit, a.eval_seeds), (4, "bf16", 20, "11"))
        self.assertIsNone(a.num_frames)
        for bad in (["--out", "/x"], ["--base-only"], ["--base-only", "--checkpoint", "/c", "--out", "/x"],
                    ["--base-only", "--out", "/x", "--limit", "0"]):
            with self.assertRaises(SystemExit):
                ev.parse_args(bad)

    def test_items_are_area5_bytes(self):
        src = json.loads(ev.ITEMS_SOURCE_JSON.read_text(encoding="utf-8"))
        # The training shard is not in the repository (.gitignore): `python eval_videos.py --copy-items` copies it
        # from the SD3.5 OCR recipe's runtime. Without it only the committed files are checked and the test skips.
        absent = [f"{d}/{n}" for d, (_, names) in ev.ITEM_SOURCES.items() for n in names
                  if not (AREA / d / n).exists()]
        self.assertLessEqual(set(absent), {"prompts/train.jsonl"})
        for dst_name, (src_dir, names) in ev.ITEM_SOURCES.items():
            present = [n for n in names if f"{dst_name}/{n}" not in absent]
            self.assertEqual(sorted(p.name for p in (AREA / dst_name).iterdir()), sorted(present))
            for n in present:
                self.assertEqual((AREA / dst_name / n).read_bytes(), (src_dir / n).read_bytes())
                self.assertEqual(src["copies"][dst_name]["files"][n]["sha256"], ev.sha256_file(src_dir / n))
        self.assertEqual(ev.sha256_file(AREA / "items" / "items.jsonl"),
                         "sha256:3884dcff4c2e44d489ef02f6bbcf11f45cc82489d38527e1b8199be35220f3e8")
        if not absent:
            self.assertEqual(ev.sha256_file(AREA / "prompts" / "train.jsonl"),
                             "sha256:75361afde0224ac65b1ef03ad96d8d00cb91793c9aa0f52ce50d860801585fb8")
        items = ev.read_items(AREA / "items")
        self.assertEqual(len(items), 400)
        self.assertTrue(all(ev.target_text(it["prompt"]) == it["target"] for it in items))
        self.assertTrue(all(it["target"].replace(" ", "") for it in items))
        if absent:
            self.skipTest(f"{', '.join(absent)} absent (not committed); run `python eval_videos.py --copy-items`")

    def test_derive_seed_equals_area5(self):
        a5 = load_by_path("a5_eval_images", AREA5_EVAL)
        for parts in (("eval", 11, "ocr-test-0001"), ("eval", 12, "x"), (1, 2, 3)):
            self.assertEqual(ev.derive_seed(*parts), a5.derive_seed(*parts))
        self.assertLess(ev.derive_seed("eval", 11, "a"), 2 ** 63)


class TestMetrics(unittest.TestCase):
    def test_frame_accuracy_hand_cases(self):
        f = ev.video_frame_accuracy
        self.assertEqual(f("Hello World", "hello world"), 1.0)
        self.assertEqual(f("", "abcd"), 0.0)
        self.assertAlmostEqual(f("abxd", "abcd"), 0.75)
        # no substring shortcut (unlike area 5's image formula): extra text lowers the score
        self.assertAlmostEqual(f("xxabcd", "abcd"), 0.5)
        self.assertEqual(f("completely different text", "abcd"), 0.0)        # clipped at len(target)

    def test_video_reward_hand_cases(self):
        self.assertEqual(ev.video_ocr_reward([0, 0, 0]), 0.0)
        self.assertEqual(ev.video_ocr_reward([]), 0.0)
        self.assertAlmostEqual(ev.video_ocr_reward([0.5, 0, 1.0, 0, 0]), 0.75)  # zeros left out

    def test_substring_accuracy_hand_cases(self):
        f = ev.ocr_substring_accuracy
        self.assertEqual(f("xxABCDyy", "abcd"), 1.0)
        self.assertAlmostEqual(f("xxabxdyy", "abcd"), 0.75)
        self.assertEqual(f("", "abcd"), 0.0)
        with self.assertRaises(ValueError):
            f("abc", "  ")
        # mean over ALL frames, zero frames included
        self.assertAlmostEqual(ev.video_substring_accuracy(["abcd", "", "", "abxd"], "abcd"), (1 + 0.75) / 4)
        with self.assertRaises(ValueError):
            ev.video_substring_accuracy([], "abcd")

    def test_substring_accuracy_equals_area5(self):
        a5 = load_by_path("a5_eval_images", AREA5_EVAL)
        for rec, tgt in (("xxabxdyy", "abcd"), ("Hello", "hello world"), ("", "a"), ("aaaa", "ab"), ("q w e", "we")):
            self.assertEqual(ev.ocr_substring_accuracy(rec, tgt), a5.ocr_substring_accuracy(rec, tgt))

    def test_video_ocr_port_matches_flow_grpo(self):
        import types

        import numpy as np

        # Upstream ocr.py runs `from Levenshtein import distance`. That package (GPL) is no longer installed in the
        # runtime venv, which uses rapidfuzz instead; this stand-in exposes rapidfuzz's Levenshtein distance under the
        # old name so that the upstream module loads unchanged.
        if "Levenshtein" not in sys.modules and importlib.util.find_spec("Levenshtein") is None:
            from rapidfuzz.distance import Levenshtein as rf_levenshtein

            stand_in = types.ModuleType("Levenshtein")
            stand_in.distance = rf_levenshtein.distance
            sys.modules["Levenshtein"] = stand_in
            self.addCleanup(sys.modules.pop, "Levenshtein", None)
        ocr_mod = load_by_path("flow_grpo_ocr", FLOW_GRPO / "flow_grpo" / "ocr.py")
        rng = np.random.default_rng(0)
        targets = ["Hello World", "Open 24 Hours", "SALE", "abc def"]
        vocab = ["hello", "world", "HelloWorld", "Hel1o Wor1d", "", "open", "24hours", "Open 24 Hours", "SALE",
                 "SA LE", "xyz", "abcdef", "abc def!!", "#"]
        # script[v][f] = list of (text, score) lines, or "raise"
        script = {}
        videos = []
        for v in range(40):
            frames = np.zeros((17, 4, 4, 3), dtype=np.uint8)
            frames[:, 0, 0, 0] = v
            frames[:, 0, 0, 1] = np.arange(17)
            videos.append(frames)
            for f in range(17):
                if rng.random() < 0.05:
                    script[(v, f)] = "raise"
                else:
                    n = int(rng.integers(0, 3))
                    script[(v, f)] = [(str(rng.choice(vocab)), float(rng.choice([0.0, 0.3, 0.9]))) for _ in range(n)]

        class FakePaddle:                                   # PaddleOCR.ocr(frame, cls=False) result format
            def ocr(self, frame, cls=False):
                s = script[(int(frame[0, 0, 0]), int(frame[0, 0, 1]))]
                if s == "raise":
                    raise RuntimeError("synthetic OCR failure")
                return [[[[[0, 0]] * 4, line] for line in s] if s else None]

        class FakeA(ev.PaddleOcrA):                         # same lines through this port's text() join
            def __init__(self):
                pass

            def text(self, arr):
                s = script[(int(arr[0, 0, 0]), int(arr[0, 0, 1]))]
                if s == "raise":
                    raise RuntimeError("synthetic OCR failure")
                return "".join(t if sc > 0 else "" for t, sc in s)

        upstream = object.__new__(ocr_mod.OcrScorer_video_or_image)
        upstream.ocr, upstream.frame_interval = FakePaddle(), 4
        prompts = [f'A sign that says "{targets[v % len(targets)]}" in the street' for v in range(40)]
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            want = upstream(videos, prompts)
        port = FakeA()
        got = [port.video_reward(v, p) for v, p in zip(videos, prompts)]
        self.assertEqual([g["reward"] for g in got], want)
        self.assertEqual([f["frame"] for f in got[0]["frames"]], [0, 4, 8, 12, 16])
        self.assertTrue(any(f["error"] for g in got for f in g["frames"]), "the script exercises OCR exceptions")
        self.assertTrue(any(0 < w < 1 for w in want) and 0.0 in want)


class TestEncoderAndSampler(unittest.TestCase):
    def test_encode_matches_flow_grpo(self):
        import torch

        emb_mod = load_by_path("flow_grpo_wan_emb", FLOW_GRPO / "flow_grpo" / "diffusers_patch" /
                               "wan_prompt_embedding.py")
        enc, tok = tiny_encoder(), StubTokenizer()
        prompts = ['A poster reading "Hello"', "", 'Neon "Open 24 Hours" sign at night, rain', "x" * 600]
        with torch.no_grad():                  # as flow_grpo's caller compute_text_embeddings (train_wan2_1.py:126)
            want = emb_mod._get_t5_prompt_embeds(enc, tok, prompts, max_sequence_length=512, device="cpu",
                                                 dtype=torch.float32)
        got = ev.encode_prompts(tok, enc, prompts, 512, batch_size=8)
        mine = ev.pad_embeds([got[p] for p in prompts], 512)
        self.assertTrue(torch.equal(mine, want))
        self.assertEqual(got[""].shape[0], 1)                                    # EOS only
        self.assertEqual(got["x" * 600].shape[0], 512)                           # truncated at max length
        # batch of one gives the same embeddings (cached once, trimmed, re-padded)
        one = ev.encode_prompts(tok, enc, prompts[2:3], 512, batch_size=1)
        self.assertTrue(torch.allclose(one[prompts[2]], got[prompts[2]], atol=1e-6))

    def test_schedule_wan_20_steps(self):
        from diffusers import UniPCMultistepScheduler

        ts, sig, idx = ev.schedule(UniPCMultistepScheduler(**WAN_SCHEDULER), 20)
        self.assertEqual(idx, list(range(20)))
        self.assertEqual(len(ts), 20)
        self.assertEqual(len(sig), 21)
        self.assertEqual(float(sig[-1]), 0.0)
        self.assertAlmostEqual(float(sig[0]), 1.0, places=5)
        self.assertEqual(len(set(ts.tolist())), 20)
        self.assertEqual(ts.dtype.__repr__(), "torch.int64")
        print("\n20-step timesteps", ts.tolist(), "\nsigmas", [round(float(s), 4) for s in sig])

    def test_sampler_matches_flow_grpo_deterministic(self):
        import torch
        from diffusers import WanPipeline

        sys.path.insert(0, str(FLOW_GRPO))
        try:
            pipe_mod = load_by_path("flow_grpo_wan_pipe", FLOW_GRPO / "flow_grpo" / "diffusers_patch" /
                                    "wan_pipeline_with_logprob.py")
        finally:
            sys.path.remove(str(FLOW_GRPO))
        transformer, vae, scheduler = tiny_parts()
        torch.manual_seed(1)
        embeds = torch.randn(2, 512, TEXT_DIM) * (torch.arange(512) < 7).view(1, -1, 1)
        neg = torch.randn(1, 512, TEXT_DIM) * (torch.arange(512) < 1).view(1, -1, 1)
        seeds = [ev.derive_seed("eval", 11, "a"), ev.derive_seed("eval", 11, "b")]
        h, w, f, steps = 32, 48, 5, 4
        videos, chain, _, _ = ev.sample_batch(transformer, vae, scheduler, embeds, neg.expand(2, -1, -1), seeds,
                                              steps, 4.5, h, w, f, "cpu", torch.float32, None, deterministic=True,
                                              keep_chain=True)
        shape = ev.latent_shape(transformer, vae, h, w, f)
        self.assertEqual(tuple(shape), (16, 2, 4, 6))
        init = torch.stack([torch.randn(shape, generator=torch.Generator("cpu").manual_seed(s)) for s in seeds])
        self.assertTrue(torch.equal(chain[:, 0], init))
        pipe = WanPipeline(tokenizer=None, text_encoder=None, transformer=transformer, vae=vae, scheduler=scheduler)
        pipe.set_progress_bar_config(disable=True)
        with torch.no_grad():
            want, latents, _, _ = pipe_mod.wan_pipeline_with_logprob(
                pipe, prompt_embeds=embeds, negative_prompt_embeds=neg.expand(2, -1, -1), num_inference_steps=steps,
                guidance_scale=4.5, output_type="pt", return_dict=False, num_frames=f, height=h, width=w,
                determistic=True, latents=init.clone())
        self.assertEqual(tuple(want.shape), (2, f, 3, h, w))
        for k in range(2):
            self.assertEqual(tuple(videos[k].shape), (f, 3, h, w))
            self.assertTrue(torch.allclose(chain[k, -1], latents[-1][k], atol=1e-5, rtol=0),
                            float((chain[k, -1] - latents[-1][k]).abs().max()))
            self.assertTrue(torch.allclose(videos[k], want[k].float(), atol=1e-5, rtol=0),
                            float((videos[k] - want[k]).abs().max()))
        u8 = ev.video_to_uint8(videos[0])
        self.assertEqual((u8.shape, str(u8.dtype)), ((f, h, w, 3), "uint8"))

    def test_sde_step_shapes_and_logprob(self):
        import torch

        _, _, scheduler = tiny_parts()
        _, sig, _ = ev.schedule(scheduler, 20)
        x = torch.randn(2, 16, 2, 4, 6)
        v = torch.randn_like(x)
        idx = torch.tensor([0, 5])
        det, lp, _, _ = ev.wan_sde_step_with_logprob(sig, idx, v, x, True)
        self.assertIsNone(lp)
        dt = (sig[idx + 1] - sig[idx]).view(-1, 1, 1, 1, 1)
        self.assertTrue(torch.equal(det, x + dt * v))
        prev, lp, mean, std = ev.wan_sde_step_with_logprob(sig, idx, v, x, False, noise=torch.randn_like(x))
        self.assertEqual(tuple(lp.shape), (2,))
        # the log-prob of the mean itself is the maximum of the Gaussian
        _, lp_mean, _, _ = ev.wan_sde_step_with_logprob(sig, idx, v, x, False, prev_sample=mean)
        self.assertTrue(bool((lp_mean > lp).all()))


class TestRealOcrOnCpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a = ev.PaddleOcrA(ev.OCR_MODELS, cpu_threads=8)
        cls.b = ev.DoctrB("cpu")

    def test_rendered_text_video(self):
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont

        font = ImageFont.truetype(FONT, 44)
        frames = []
        for i in range(17):
            img = Image.new("RGB", (416, 240), (235, 230, 220))
            if i <= 8:                                     # text on frames 0-8, blank 9-16
                ImageDraw.Draw(img).text((40, 90), "OPEN LATE", fill=(20, 20, 20), font=font)
            frames.append(np.asarray(img))
        frames = np.stack(frames)
        prompt = 'A shop window with a sign that says "Open Late"'
        res = self.a.video_reward(frames, prompt)
        print("\nengine A frames", [(f["frame"], f["text"], round(f["acc"], 3)) for f in res["frames"]])
        self.assertEqual([f["frame"] for f in res["frames"]], [0, 4, 8, 12, 16])
        self.assertEqual([f["acc"] for f in res["frames"]][3:], [0.0, 0.0])
        self.assertGreaterEqual(res["reward"], 0.85)        # mean of the positive frames only
        texts = self.b.texts(list(frames))
        accs = [ev.ocr_substring_accuracy(t, "Open Late") for t in texts]
        print("engine B texts", texts[:2], texts[-1:], "accs", [round(a, 3) for a in accs])
        self.assertEqual(len(texts), 17)
        self.assertTrue(all(a >= 0.85 for a in accs[:9]))
        self.assertTrue(all(a == 0 for a in accs[9:]))
        self.assertAlmostEqual(ev.video_substring_accuracy(texts, "Open Late"), sum(accs) / 17)


class TestEvaluateEndToEnd(unittest.TestCase):
    """evaluate() on a tiny random base, the real OCR engines on the CPU, 2 items."""

    @classmethod
    def setUpClass(cls):
        cls.root = SCRATCH / "e2e"
        if cls.root.exists():
            shutil.rmtree(cls.root)
        cls.base = cls.root / "base" / "tiny"
        cls.base.mkdir(parents=True)
        cls.models_json = make_tiny_base(cls.base)
        cls.saved = (ev.MODELS_JSON, ev.encode_all_prompts)
        ev.MODELS_JSON = cls.models_json
        ev.encode_all_prompts = fake_encode_all_prompts
        cls.common = ["--base", str(cls.base), "--eval-seeds", "11", "--limit", "2", "--batch-size", "2",
                      "--device", "cpu", "--dtype", "fp32", "--height", "32", "--width", "48", "--num-frames", "5",
                      "--num-steps", "3", "--ocr-cpu-threads", "8"]

    @classmethod
    def tearDownClass(cls):
        ev.MODELS_JSON, ev.encode_all_prompts = cls.saved

    def run_eval(self, name, extra):
        out = self.root / name
        self.assertEqual(ev.main(extra + ["--out", str(out)] + self.common), 0)
        return out

    def frame_hashes(self, out):
        return {r["item"]: [f["sha256"] for f in json.loads((out / r["output_ref"]["path"]).read_text())["frames"]]
                for r in map(json.loads, (out / "scores.jsonl").read_text().splitlines())}

    def make_cut(self, sampler):
        import torch
        from diffusers import WanTransformer3DModel
        from peft import LoraConfig

        cut = self.root / "run" / "checkpoint-1"
        cut.mkdir(parents=True)
        t = WanTransformer3DModel.from_pretrained(str(self.base), subfolder="transformer")
        t.add_adapter(LoraConfig(r=4, lora_alpha=8, init_lora_weights="gaussian",
                                 target_modules=list(ev.WAN_LORA_TARGETS)))
        with torch.no_grad():
            for n, p in t.named_parameters():
                if "lora_B" in n:
                    p.normal_(0, 0.5, generator=torch.Generator().manual_seed(3))
        t.save_lora_adapter(str(cut), adapter_name="default", safe_serialization=True,
                            weight_name=ev.LORA_WEIGHTS)
        record = json.loads(self.models_json.read_text())["models"][ev.BASE_NAME]
        ev.write_json(cut / "adapter_config.json", {"r": 4, "lora_alpha": 8, "init_lora_weights": "gaussian",
                                                    "target_modules": sorted(ev.WAN_LORA_TARGETS)})
        ev.write_json(cut / "sampler.json", sampler)
        ev.write_json(cut / "base_ref.json", {"repo": record["repo"], "revision": record["revision"],
                                              "files": {k: v["sha256"] for k, v in record["files"].items()}})
        ev.write_json(cut / "trainer_state.json", {"epoch": 1})
        names = [ev.LORA_WEIGHTS, "adapter_config.json", "sampler.json", "base_ref.json", "trainer_state.json"]
        ev.write_json(cut / "cut.json", {"epoch": 1, "files": {n: ev.sha256_file(cut / n) for n in names},
                                         "sampler_code_sha256": ev.sampler_code_sha256()})
        return cut

    def test_base_then_cut(self):
        a = self.run_eval("base-1", ["--base-only"])
        b = self.run_eval("base-2", ["--base-only"])
        # common random numbers: identical frames run to run
        self.assertEqual(self.frame_hashes(a), self.frame_hashes(b))
        rows = [json.loads(ln) for ln in (a / "scores.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 2)
        for r in rows:
            self.assertEqual(r["errors"], {})
            self.assertEqual(r["num_frames"], 5)
            self.assertEqual([f["frame"] for f in r["ocr_a_frames"]], [0, 4])
            self.assertEqual(len(r["ocr_b_frames"]), 5)
            self.assertEqual(r["ocr_b_acc"], sum(f["acc"] for f in r["ocr_b_frames"]) / 5)
            self.assertEqual(r["ocr_a_acc"], ev.video_ocr_reward([f["acc"] for f in r["ocr_a_frames"]]))
            man = json.loads((a / r["output_ref"]["path"]).read_text())
            self.assertEqual(ev.sha256_file(a / r["output_ref"]["path"]), r["output_ref"]["sha256"])
            vdir = (a / r["output_ref"]["path"]).parent
            for f in man["frames"]:
                self.assertEqual(ev.sha256_file(vdir / f["path"]), f["sha256"])
            self.assertEqual(r["noise_seed"], ev.derive_seed("eval", 11, r["item"]))
        mm2 = {"schema", "schema_version", "evaluator_id", "adapter", "metric", "direction", "score_range",
               "checkpoint", "normalizer_sha256", "env_version_sha256", "item_set_hash", "eval_seeds", "sampler",
               "sampler_hash", "scorer", "framework", "created_at"}
        metas = {}
        for metric, (eid, *_rest) in ev.METRICS.items():
            arm = a / f"arm-{metric}"
            meta = json.loads((arm / "eval_meta.json").read_text())
            metas[metric] = meta
            self.assertEqual(set(meta), mm2)
            self.assertEqual((meta["evaluator_id"], meta["adapter"], meta["metric"]), (eid, "flow_grpo_wan", metric))
            self.assertEqual(meta["framework"]["name"], "diffusers")
            self.assertEqual(meta["sampler_hash"], ev.sha256_bytes(ev.canonical(meta["sampler"])))
            for line in map(json.loads, (arm / "items.jsonl").read_text().splitlines()):
                self.assertEqual(set(line), {"item", "repeat", "seed", "score", "error", "sub_metrics",
                                             "output_ref"})
                self.assertTrue(all(isinstance(v, (int, float)) for v in line["sub_metrics"].values()))
                p = arm / line["output_ref"]["path"]
                self.assertEqual(ev.sha256_file(p), line["output_ref"]["sha256"])
                self.assertEqual(os.stat(p).st_ino, os.stat(a / line["output_ref"]["path"]).st_ino)  # hard link
                self.assertEqual(len(list(p.parent.glob("frame_*.png"))), 5)
        s = metas["ocr_b_acc"]["sampler"]
        self.assertEqual((s["num_steps"], s["guidance_scale"], s["num_frames"], s["deterministic"],
                          s["negative_prompt"], s["max_sequence_length"]), (3, 4.5, 5, True, "", 512))
        summary = json.loads((a / "summary.json").read_text())
        self.assertEqual(set(summary["mm38"]), {"ocr_b_acc", "ocr_a_acc"})
        self.assertIn("checks", summary["mm38"]["ocr_b_acc"])

        cut = self.make_cut(metas["ocr_b_acc"]["sampler"])
        c = self.run_eval("cut-1", ["--checkpoint", str(cut)])
        cmeta = json.loads((c / "arm-ocr_b_acc" / "eval_meta.json").read_text())
        self.assertEqual(cmeta["sampler_hash"], metas["ocr_b_acc"]["sampler_hash"])
        self.assertNotEqual(self.frame_hashes(c), self.frame_hashes(a))
        self.assertEqual(json.loads((c / "summary.json").read_text())["lora"]["r"], 4)
        # binding refusals: a changed file, another sampler, other LoRA targets
        bad = self.root / "run" / "checkpoint-bad"
        shutil.copytree(cut, bad)
        (bad / "trainer_state.json").write_text("{}")
        with self.assertRaises(SystemExit):
            ev.check_cut(bad)
        shutil.rmtree(bad)
        shutil.copytree(cut, bad)
        marker = json.loads((bad / "cut.json").read_text())
        marker["sampler_code_sha256"] = "sha256:" + "0" * 64
        ev.write_json(bad / "cut.json", marker)
        with self.assertRaises(SystemExit):
            ev.check_cut(bad)
        shutil.rmtree(bad)
        shutil.copytree(cut, bad)
        ev.write_json(bad / "adapter_config.json", {"r": 4, "target_modules": ["to_q", "to_v"]})
        marker = json.loads((bad / "cut.json").read_text())
        marker["files"]["adapter_config.json"] = ev.sha256_file(bad / "adapter_config.json")
        ev.write_json(bad / "cut.json", marker)
        with self.assertRaises(SystemExit):
            ev.check_cut(bad)
        with self.assertRaises(SystemExit):                       # --num-frames differs from the cut's sampler
            ev.main(["--checkpoint", str(cut), "--out", str(self.root / "cut-x")]
                    + [x if x != "5" else "9" for x in self.common])
        with self.assertRaises(SystemExit):                       # the output directory must be empty
            self.run_eval("cut-1", ["--checkpoint", str(cut)])


class TestMm38(unittest.TestCase):
    def test_rules(self):
        ok = ev.mm38_checks([0.1, 0.4, 0.6, 0.0, 0.3], [0, 1], 0)
        self.assertTrue(ok["pass"])
        self.assertFalse(ev.mm38_checks([0.0, 0.0, 0.0, 0.5], [0, 1], 0)["checks"]["mode_share_below_0.5"])
        self.assertFalse(ev.mm38_checks([0.01, 0.02, 0.03], [0, 1], 0)["checks"]["mean_inside_by_5pct"])
        self.assertFalse(ev.mm38_checks([0.3, 0.3], [0, 1], 0)["checks"]["sd_positive"])
        self.assertFalse(ev.mm38_checks([0.1, 0.4, 0.6], [0, 1], 1)["pass"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
