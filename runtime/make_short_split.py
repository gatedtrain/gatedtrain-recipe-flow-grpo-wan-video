#!/usr/bin/env python
"""Short-target item set and training shard for attempt 2 (amendment) of area "video generation" (MULTIMODAL-SPEC
row 3; prototype, unpromotable).

The first attempt used area 5's item set (400 items) and training shard (18,359 prompts), byte copies in items/ and
prompts/. The amendment (decided 2026-11-15; the first pre-registered MM-38 fallback, README "Decisions")
restricts both to short target texts. This script re-derives the split from the same
sources with area 5's rules (the SD3.5 OCR recipe's eval_images.py make_items) and one added filter:

  source     flow_grpo dataset/ocr/{train,test}.txt @ 879042cf (MIT), <sd35-ocr-runtime>/data-source/ocr/ (hashes in the meta)
  target     the first double-quoted span of the prompt (flow_grpo ocr.py:32)
  short      the target has at most MAX_WORDS whitespace-separated words (`len(target.split()) <= 2`)
  items      every short-target prompt of the 1,017 test prompts that do not occur in train.txt, in the order of area
             5's permutation numpy default_rng(SeedSequence([0x5d35a5, split_seed])).permutation (split_seed 20261113),
             so a prefix (`--limit N`) is a random sample; ids, fields and values as items/items.jsonl
  shard      the unique short-target train.txt prompts (first occurrence, file order) without any item prompt and
             without any prompt whose target (spaces removed, lower case) equals an item's target

Outputs (refused when a target directory exists and is not empty):
  items-short/items.jsonl, items-short/item_set_meta.json
  prompts-short/train.jsonl, prompts-short/prompts_meta.json

Usage (area 5's venv, CPU, a few seconds):
  python make_short_split.py            write the four files and print the counts
  python make_short_split.py --dry-run  print the counts of this rule and of the alternatives, write nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# The SD3.5 OCR recipe's runtime (the OCR data source): the override
# GATEDTRAIN_RECIPE_FLOW_GRPO_WAN_VIDEO_OCR_RUNTIME, else the sibling checkout ../gatedtrain-recipe-flow-grpo-sd35-ocr/runtime.
OCR_RUNTIME = Path(os.environ.get("GATEDTRAIN_RECIPE_FLOW_GRPO_WAN_VIDEO_OCR_RUNTIME")
                   or HERE.parent.parent / "gatedtrain-recipe-flow-grpo-sd35-ocr" / "runtime")
DATA_SOURCE = OCR_RUNTIME / "data-source" / "ocr"
OLD_ITEMS = HERE / "items" / "items.jsonl"
ITEMS_DIR = HERE / "items-short"
PROMPTS_DIR = HERE / "prompts-short"
ITEM_NAMESPACE = 0x5D35A5        # the SD3.5 OCR recipe's eval_images.py: SeedSequence([ITEM_NAMESPACE, split_seed])
SPLIT_SEED = 20261113            # items/item_set_meta.json split.split_seed
MAX_WORDS = 2
MIN_ITEMS = 250
ORIGIN = "yifan123/flow_grpo@879042cf5707f8b90daa98d147d7deac2317c5da dataset/ocr/{train,test}.txt (MIT)"
TARGET_RULE = "flow_grpo ocr.py:32: the first double-quoted span of the prompt"
SHORT_RULE = (f"short target: the target has at most {MAX_WORDS} whitespace-separated words "
              f"(len(target.split()) <= {MAX_WORDS})")
# Alternatives reported by --dry-run (README "Attempt 2 (amendment)"): name -> predicate on the target.
ALTERNATIVES = {"words<=1": lambda t: len(t.split()) <= 1, "words<=2": lambda t: len(t.split()) <= 2,
                "words<=3": lambda t: len(t.split()) <= 3, "chars<=8": lambda t: len(t) <= 8,
                "chars<=12": lambda t: len(t) <= 12, "chars<=15": lambda t: len(t) <= 15}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def write_json(path: Path, doc) -> None:
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


def target_text(prompt: str) -> str:
    """flow_grpo ocr.py:32: the text between the first pair of double quotes."""
    return prompt.split('"')[1]


def norm_target(prompt: str) -> str:
    return target_text(prompt).replace(" ", "").lower()


def is_short(target: str) -> bool:
    return len(target.split()) <= MAX_WORDS


def read_sources() -> tuple[list[str], dict[str, int], list[tuple[int, str]], list[tuple[int, str]], list[int]]:
    """The SD3.5 OCR recipe's runtime/eval_images.py make_items, unchanged: train lines, first occurrence of each
    train prompt, the test prompts, the candidates (test prompts not in train.txt) and the permutation over the
    candidates."""
    import numpy as np

    train_lines = (DATA_SOURCE / "train.txt").read_text(encoding="utf-8").splitlines()
    test_lines = (DATA_SOURCE / "test.txt").read_text(encoding="utf-8").splitlines()
    train_first: dict[str, int] = {}
    for i, ln in enumerate(train_lines):
        if ln.strip():
            train_first.setdefault(ln.strip(), i)
    test = [(i, ln.strip()) for i, ln in enumerate(test_lines) if ln.strip()]
    candidates = [(i, p) for i, p in test if p not in train_first]
    rng = np.random.default_rng(np.random.SeedSequence([ITEM_NAMESPACE, SPLIT_SEED]))
    perm = [int(x) for x in rng.permutation(len(candidates))]
    return train_lines, train_first, test, candidates, perm


def split(predicate, train_first, candidates, perm):
    """(items in permutation order, shard rows in file order, short unique train prompts, dropped by target)."""
    items = [{"item": f"ocr-test-{i:04d}", "prompt": p, "target": target_text(p), "source_line": i}
             for i, p in (candidates[k] for k in perm) if predicate(target_text(p))]
    item_prompts = {r["prompt"] for r in items}
    item_targets = {norm_target(r["prompt"]) for r in items}
    short = [(p, i) for p, i in sorted(train_first.items(), key=lambda kv: kv[1]) if predicate(target_text(p))]
    shard, dropped_prompt, dropped_target = [], 0, 0
    for p, i in short:
        if p in item_prompts:
            dropped_prompt += 1
            continue
        if norm_target(p) in item_targets:
            dropped_target += 1
            continue
        shard.append({"prompt_id": f"ocr-train-{i:05d}", "prompt": p, "target": target_text(p), "source_line": i})
    return items, shard, len(short), dropped_prompt, dropped_target


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dry-run", action="store_true", help="print the counts (rule and alternatives), write nothing")
    args = p.parse_args(argv)
    import numpy as np

    train_lines, train_first, test, candidates, perm = read_sources()
    if args.dry_run:
        for name, pred in ALTERNATIVES.items():
            items, shard, n_short, _, _ = split(pred, train_first, candidates, perm)
            print(json.dumps({"rule": name, "n_items": len(items), "n_train_unique_short": n_short,
                              "n_train_shard": len(shard)}))
        return 0
    items, shard, n_short, dropped_prompt, dropped_target = split(is_short, train_first, candidates, perm)
    if len(items) < MIN_ITEMS:
        raise SystemExit(f"{len(items)} short items < {MIN_ITEMS}: choose another rule (--dry-run)")
    # self-checks: disjoint by prompt and by normalized target; every target non-empty; ids unique; an item that is
    # also one of the first attempt's 400 is the same row as there
    item_prompts = {r["prompt"] for r in items}
    item_targets = {norm_target(r["prompt"]) for r in items}
    exact = len({r["prompt"] for r in shard} & item_prompts)
    same_target = len({norm_target(r["prompt"]) for r in shard} & item_targets)
    if exact or same_target:
        raise SystemExit(f"items and training shard overlap: {exact} prompts, {same_target} targets")
    if any(not norm_target(r["prompt"]) for r in items + shard):
        raise SystemExit("a target is empty without spaces")
    if len({r["item"] for r in items}) != len(items) or len({r["prompt_id"] for r in shard}) != len(shard):
        raise SystemExit("ids are not unique")
    old = {r["item"]: r for r in (json.loads(ln) for ln in OLD_ITEMS.read_text(encoding="utf-8").splitlines()
                                  if ln.strip())}
    in_old = [r for r in items if r["item"] in old]
    if any(old[r["item"]] != r for r in in_old):
        raise SystemExit("a short item differs from the first attempt's row of the same id")
    for t in (ITEMS_DIR, PROMPTS_DIR):
        if t.exists() and any(t.iterdir()):
            raise SystemExit(f"{t} exists and is not empty")
        t.mkdir(parents=True, exist_ok=True)
    write_jsonl(ITEMS_DIR / "items.jsonl", items)
    write_jsonl(PROMPTS_DIR / "train.jsonl", shard)
    source = {"files": {f"data-source/ocr/{n}": sha256_file(DATA_SOURCE / n) for n in ("train.txt", "test.txt")},
              "source_dir": str(DATA_SOURCE), "origin": ORIGIN, "target_rule": TARGET_RULE}
    split_doc = {
        "n_test": len(test), "n_test_after_exact_overlap": len(candidates), "n_items": len(items),
        "n_items_in_first_attempt_items": len(in_old),
        "n_train_lines": len([ln for ln in train_lines if ln.strip()]), "n_train_unique": len(train_first),
        "n_train_unique_short": n_short, "n_train_dropped_item_prompt": dropped_prompt,
        "n_train_dropped_same_target": dropped_target, "n_train_shard": len(shard), "split_seed": SPLIT_SEED,
        "short_rule": SHORT_RULE, "max_words": MAX_WORDS,
        "generator": (f"numpy default_rng(SeedSequence([{ITEM_NAMESPACE:#x}, split_seed])).permutation over the test "
                      "prompts that do not occur in train.txt (area 5's permutation); the items are every candidate "
                      "with a short target, in permutation order (not sorted). Training shard: the unique train.txt "
                      "prompts (first occurrence, file order) with a short target, without any item prompt and without "
                      "any prompt whose target (spaces removed, lower case) equals an item's target"),
        "order": "items: permutation order (a prefix is a random sample); shard: train.txt file order",
        "numpy": np.__version__, "exact_overlap": exact, "target_overlap": same_target,
        "first_attempt_items_sha256": sha256_file(OLD_ITEMS)}
    tags = ["prototype", "unpromotable"]
    write_json(ITEMS_DIR / "item_set_meta.json", {
        "schema": "ptc_mm.sd35_ocr_item_set", "schema_version": "1.0", "tags": tags, "source": source,
        "split": split_doc, "items_sha256": sha256_file(ITEMS_DIR / "items.jsonl"),
        "train_sha256": sha256_file(PROMPTS_DIR / "train.jsonl")})
    write_json(PROMPTS_DIR / "prompts_meta.json", {
        "schema": "ptc_mm.sd35_ocr_train_prompts", "schema_version": "1.0", "tags": tags, "source": source,
        "split": split_doc, "train_sha256": sha256_file(PROMPTS_DIR / "train.jsonl")})
    print(json.dumps(split_doc, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
