#!/usr/bin/env python
"""CPU tests of make_short_split.py and its outputs items-short/ and prompts-short/ (README "Attempt 2 (amendment)").

Run with area 5's venv, from anywhere:
  <sd35-ocr-runtime>/.venv/bin/python -I -B tests/test_make_short_split.py

What is checked: the files on disk are exactly what the script derives from the sources (row for row and byte for byte);
every target is short (at most 2 words) and non-empty; items in area 5's permutation order and the first attempt's 400
items are its first-400 prefix of that permutation; no exact prompt and no normalized-target overlap between items and
shard; the shard is the short unique train prompts in file order minus the dropped ones; the metas' counts and hashes;
the first attempt's directories are untouched.
"""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
AREA = HERE.parent
sys.path.insert(0, str(AREA))

import make_short_split as mss  # noqa: E402

FIRST_ATTEMPT = {"items/items.jsonl": "3884dcff4c2e44d489ef02f6bbcf11f45cc82489d38527e1b8199be35220f3e8",
                 "prompts/train.jsonl": "75361afde0224ac65b1ef03ad96d8d00cb91793c9aa0f52ce50d860801585fb8"}


def jl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


class TestShortSplit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train_lines, cls.train_first, cls.test, cls.cand, cls.perm = mss.read_sources()
        cls.items, cls.shard, cls.n_short, cls.d_prompt, cls.d_target = mss.split(mss.is_short, cls.train_first,
                                                                                  cls.cand, cls.perm)
        cls.disk_items = jl(AREA / "items-short" / "items.jsonl")
        cls.disk_shard = jl(AREA / "prompts-short" / "train.jsonl")

    def test_files_are_the_derivation(self):
        self.assertEqual(self.disk_items, self.items)
        self.assertEqual(self.disk_shard, self.shard)
        self.assertEqual((AREA / "items-short" / "items.jsonl").read_text(encoding="utf-8"),
                         "".join(json.dumps(r, sort_keys=True) + "\n" for r in self.items))
        self.assertEqual((len(self.items), len(self.shard)), (288, 5024))

    def test_rule_order_and_disjointness(self):
        for r in self.items + self.shard:
            self.assertEqual(r["target"], mss.target_text(r["prompt"]))
            self.assertLessEqual(len(r["target"].split()), 2)
            self.assertTrue(mss.norm_target(r["prompt"]))
        pos = {self.cand[k][0]: n for n, k in enumerate(self.perm)}
        order = [pos[r["source_line"]] for r in self.items]
        self.assertEqual(order, sorted(order))                                # permutation order
        self.assertEqual(sum(1 for c in self.cand if mss.is_short(mss.target_text(c[1]))), len(self.items))  # all
        old = {r["item"]: r for r in jl(AREA / "items" / "items.jsonl")}
        first400 = {f"ocr-test-{self.cand[k][0]:04d}" for k in self.perm[:400]}
        self.assertEqual(set(old), first400)
        both = [r for r in self.items if r["item"] in old]
        self.assertEqual(len(both), 106)
        self.assertTrue(all(old[r["item"]] == r for r in both))
        self.assertEqual({r["prompt"] for r in self.items} & {r["prompt"] for r in self.shard}, set())
        self.assertEqual({mss.norm_target(r["prompt"]) for r in self.items}
                         & {mss.norm_target(r["prompt"]) for r in self.shard}, set())
        lines = [r["source_line"] for r in self.shard]
        self.assertEqual(lines, sorted(lines))
        self.assertEqual(self.n_short - self.d_prompt - self.d_target, len(self.shard))
        self.assertEqual((self.n_short, self.d_prompt, self.d_target), (5677, 0, 653))

    def test_metas(self):
        meta = json.loads((AREA / "items-short" / "item_set_meta.json").read_text(encoding="utf-8"))
        pmeta = json.loads((AREA / "prompts-short" / "prompts_meta.json").read_text(encoding="utf-8"))
        old = json.loads((AREA / "items" / "item_set_meta.json").read_text(encoding="utf-8"))
        self.assertEqual(set(meta), set(old))
        self.assertEqual((meta["schema"], meta["schema_version"]), (old["schema"], old["schema_version"]))
        self.assertEqual(meta["source"]["files"], old["source"]["files"])
        self.assertEqual(meta["items_sha256"], sha(AREA / "items-short" / "items.jsonl"))
        self.assertEqual(meta["train_sha256"], sha(AREA / "prompts-short" / "train.jsonl"))
        self.assertEqual(pmeta["train_sha256"], meta["train_sha256"])
        self.assertEqual(pmeta["split"], meta["split"])
        s = meta["split"]
        self.assertEqual((s["n_items"], s["n_train_shard"], s["n_train_unique_short"], s["n_items_in_first_attempt_items"],
                          s["exact_overlap"], s["target_overlap"], s["max_words"], s["split_seed"]),
                         (288, 5024, 5677, 106, 0, 0, 2, 20261113))
        self.assertEqual(sorted(p.name for p in (AREA / "items-short").iterdir()), ["item_set_meta.json", "items.jsonl"])
        self.assertEqual(sorted(p.name for p in (AREA / "prompts-short").iterdir()), ["prompts_meta.json", "train.jsonl"])

    def test_first_attempt_untouched(self):
        for rel, want in FIRST_ATTEMPT.items():
            self.assertEqual(hashlib.sha256((AREA / rel).read_bytes()).hexdigest(), want)

    def test_refuses_to_overwrite(self):
        with self.assertRaises(SystemExit):
            mss.main([])


if __name__ == "__main__":
    unittest.main(verbosity=2)
