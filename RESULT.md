Prototype result of a single registered run; not a benchmark claim; not affiliated with or endorsed by the owners of the models, datasets, or frameworks used.

# Result (2026-11-15): attempt 2 gate PASS (attempt 1 FAIL, amended)

Flow-GRPO + LoRA on Wan2.1-T2V-1.3B (`Wan-AI/Wan2.1-T2V-1.3B-Diffusers` at `0fad780a`), the minimal port of flow_grpo
`879042cf` `scripts/train_wan2_1.py` on diffusers + peft, run in the SD3.5 OCR recipe's runtime image
(`ptc-a5-runtime:1`; ptc, ptc-skel and ptc-hub are earlier names of gatedtrain, see "Rename to gatedtrain 0.3.0")
and venv. Videos of 17 frames at 416 x 240, 20 sampling steps, CFG 4.5 (training and deployment
sampler alike). Training reward: flow_grpo's video OCR reward with PaddleOCR PP-OCRv4 (every 4th frame, mean of the
positive frame accuracies). Run seed 5000, evaluation seed 11, one video per item.

Deciding metric in both attempts: `ocr_b_acc`, docTR (db_resnet50 + crnn_vgg16_bn) accuracy of the best-matching
substring of the recognized text, the mean over all 17 frames (T3, construct-aligned with the reward: disclosed in the
registration and the report). Supporting (not gating): `ocr_a_acc` (the reward itself, T0).

## Attempt 1: FAIL

`results/2026-11-15/gate/spec.yaml`, `registration.yaml`; flow_grpo's OCR test split as in the SD3.5 recipe (400
items); `train_flow_grpo_wan.py --profile full`: 60 epochs x 4 prompts x 4 videos (120 optimizer steps), cuts every
15 epochs.

| | |
|---|---|
| Baseline (Wan2.1-T2V-1.3B) | 0.1554 |
| Candidate (cut 60) | 0.1469 |
| Delta, 95% cluster-bootstrap interval | -0.0086 [-0.0201, 0.0022], p 0.94 |
| Items | 400 |
| Consumption precision | `row` (observed): 908 consumed and 52 filtered (`zero_advantage`) records |

Plumbing complete (exact overlap 0; the impact query on `prompts/train.jsonl` returns cuts 15/30/45/60 at steps
30/60/90/120). Cause: too little training signal (120 small steps, approx_kl about 0). Decision (2026-11-15):
retry as an amendment with short target texts only (the pre-registered first fallback) and a larger batch.

## Attempt 2 (amendment): PASS

`results/2026-11-15/gate/spec-v2.yaml`, `registration-v2.yaml` (`amends` attempt 1); `make_short_split.py`: targets of
at most 2 words, 288 held-out items (`items-short/`) and a 5,024-prompt training shard (`prompts-short/`), exact and
same-target overlap 0; `train_flow_grpo_wan_v2.py --profile full2`: 100 epochs x 8 prompts x 4 videos (200 optimizer
steps), cuts every 25 epochs, 11.22 GPU-hours. Mean training reward 0.117 over the first 10 epochs, 0.163 over the
last 10.

| | |
|---|---|
| Baseline (Wan2.1-T2V-1.3B) | 0.1456 |
| Candidate (cut 100) | 0.1774 |
| Delta, 95% cluster-bootstrap interval | +0.0318 [0.0152, 0.0494], p <= 7.3e-05 |
| Items, samples per item | 288, 1 (eval seed 11) |
| Sd of paired differences | 0.148 |
| Achieved power | 0.999 |
| Exact overlap with training | 0 |
| Consumption precision | `row` (observed): 2,756 consumed and 444 filtered (`zero_advantage`) records |

The impact query on `prompts-short/train.jsonl` returns cuts 25/50/75/100 at steps 50/100/150/200, none orphaned or
superseded.

## Re-collect with this repository's adapter

Re-collected on 2026-11-16 with this repository's adapter (installed as a plugin of ptc; runtime and OCR runtime
overrides pointing at the original directories) on both attempts' original outputs: every field of the gate report
equals the pre-migration re-collect with ptc's built-in adapter except the run-id-derived `lockfile_hash` (attempt 1:
status FAIL, delta -0.00857 [-0.0201, 0.00217]; attempt 2: status PASS, delta 0.031791 [0.015200, 0.049372]); the
ledgers (960 and 3,200 records) and the lineage are equal except their run-id-derived columns; the impact queries
return the same cuts and steps. The lockfile differs only in the run id, the ledger hash that follows from it, and
the `checkpoint_completeness.sources` text, which now names repository-relative files.

## Published code

The result above was produced, and re-collected on 2026-11-16 with unchanged outcome, with the runtime scripts as
they were before the publication cleanup. The cleanup changed the four runtime scripts that named the SD3.5 OCR
recipe's runtime by an absolute host path: they now read `PTC_RECIPE_FLOW_GRPO_WAN_VIDEO_OCR_RUNTIME`, else the sibling
checkout `../ptc-recipe-flow-grpo-sd35-ocr/runtime`, and their comments, docstrings and two messages no longer cite
unpublished notes. Nothing else in them changed; the shared sampler block (`sampler_code_sha256`
`sha256:dad91e41...d90300`) is byte-identical. Their SHA-256s therefore differ from the ones the runs recorded (the
scripts that ran the result are not published):

| Script | Published |
|---|---|
| `runtime/eval_videos.py` (the scorer's `code_sha256`) | `36034a328e8b9da5568f5164cde5d19d7085f1cd4c059a8eaa55f2c8273bc0c3` |
| `runtime/train_flow_grpo_wan.py` (attempt 1) | `df869750cfe382e25a209e2aeeff8f699ec7793b1bd36b3f3067cd169e2d8d13` |
| `runtime/train_flow_grpo_wan_v2.py` (attempt 2) | `b4686e3fa7f14eace09480fb3d6da30dea71cda00d31678c771305f4a5576de5` |
| `runtime/make_short_split.py` | `c6b63572e5b99be33db07aa97d38ac110184d60019c644a7d9afb24fac6fb029` |

The registry row's `code_sha256` (`rows.py`) is the published `eval_videos.py`'s, so the published adapter accepts arms
produced by the published scorer; the original arms carry the earlier hash.

## Redacted copies

The files under `results/2026-11-15/` are redacted copies (host paths replaced by `<data-root>/flow-grpo-wan-video/runtime/...`,
references to unpublished notes removed from comments). So are three committed metadata files of the runtime, whose
`source`/`source_dir` fields named host paths (now `<sd35-ocr-runtime>/...`): `runtime/ITEMS-SOURCE.json`,
`runtime/items-short/item_set_meta.json` and `runtime/prompts-short/prompts_meta.json`. `items-short/item_set_meta.json`
is part of the registered item set and `prompts-short/prompts_meta.json` part of the registered training shard; the
manifest hashes the registration and `spec-v2.yaml` name were computed with the original files
(`items-short/items.jsonl` is unchanged). The gate-time originals are not published.

The registered item-set hash `da3c385b...` (`registration-v2.yaml` `eval_distribution.item_set_hash`, `spec-v2.yaml`
`items`) and the training-shard hash `e8a940b5...` (`spec-v2.yaml` `prompts`) therefore cover gate-time files that
held host paths. Those files are not published, and the published copies have the paths replaced, so neither hash can
be recomputed from this repository. What can still be checked: the 288 item rows (`runtime/items-short/items.jsonl`,
unchanged since the gate) and the published per-item records of the probe evaluation with their summary statistics
(`tests/fixtures/flow_grpo_wan/probe-base-1/`). The gate's own per-item records are not part of this repository.

`results/2026-11-15/gate/trainer.yaml` is unchanged.

## Rename to ptc-hub 0.2.0 (2026-11-17)

ptc-hub 0.2.0 renamed the persisted identifiers (schema ids, ledger writer id, analysis code version). The files under
results/2026-11-15/ and the byte-copied fixtures under tests/fixtures/flow_grpo_wan/probe-base-1/ keep the names
ptc-skel 0.1.0 wrote; they are read by ptc `78eb913` (this and the other framework commits cited here, `e20736e` and
`8da5706` among them, are unpublished pre-release commits of the framework, earlier named ptc-skel and ptc-hub; the
published framework is `gatedtrain` `v0.3.0`). The gate definitions are carried forward by
results/2026-11-15/gate/registration-ptc-hub.yaml (amends
sha256:4c39674d7e1da922bc90f4db273ce6edeae7038bb8247920459f614770870897, post_hoc: true) with spec-ptc-hub.yaml, and
results/2026-11-15/gate/registration-v2-ptc-hub.yaml (amends
sha256:b259473513f5d529f6034f54a7c6bb4c418d98bff01f6d4ec017bfc821e24451, post_hoc: true) with
spec-v2-ptc-hub.yaml. Each amends the published copy next to it; the gates ran with the originals
(not published, "Redacted copies"), and the published copies differ from the gate-time
files only in redacted comments (parsed, they are equal); `registration.yaml`, `registration-v2.yaml`, `spec.yaml`
and `spec-v2.yaml` stay as they are. `spec.yaml` and `spec-v2.yaml` therefore pin the gate-time hashes, not
the published copies' (`ptc-hub validate` reports REG-HASH on them); the `-ptc-hub` specs pin the published copies.

Why post hoc: the deciding evaluator's identity covers `eval_videos.py`'s SHA-256, and the scorer changed after both
gates (the publication cleanup above, and now the schema id `ptc_hub.item_eval` and the name of the OCR runtime
variable). ptc's rename-only proof (design R17-3) starts from the gate-time scorer, and the gate-time file with the
name map applied is not today's file, so the amendments record the current scorer's `manifest_hash`
(sha256:843d424dde74dccd211e5b30a3eb6e6a6cba29be0a9b31cb748c3a3be1783330, an evaluator change, not only a rename) and
`post_hoc: true`. Under ptc-hub 0.2.0 a post-hoc gate cannot PASS: attempt 2 reports INCONCLUSIVE with reason
`post_hoc`; attempt 1 stays FAIL (`post_hoc` changes only a PASS). The verdicts above (attempt 1 FAIL, attempt 2 PASS)
are the gates' verdicts under their original registrations; they reproduce with ptc `78eb913`, the original gate
files and the original scripts, as described in "Re-collect with this repository's adapter" and "Published code".

Re-pinned for the rename and the wording cleanup (only the new pin is accepted; the historical runs used the versions
in the "Ran the result" column of "Published code"). The four runtime scripts name the SD3.5 OCR recipe's runtime
`OCR_RUNTIME` instead of `AREA5` and cite its files by recipe instead of by the host directory `area5-sd35/`;
`eval_videos.py` also writes `ptc_hub.item_eval`. The shared sampler block is unchanged (`sha256:dad91e41…`), and
apart from that name, the schema id and docstrings the code is unchanged (equal syntax trees).

| Script | Before (published) | Now |
|---|---|---|
| `runtime/eval_videos.py` (the scorer's `code_sha256`) | `sha256:36034a328e8b9da5568f5164cde5d19d7085f1cd4c059a8eaa55f2c8273bc0c3` | `sha256:f628fb1abcd84c0bba4715802ee67d1eb644cfb6b3aca9dbcf92b389667c3a54` |
| `runtime/train_flow_grpo_wan.py` | `sha256:df869750cfe382e25a209e2aeeff8f699ec7793b1bd36b3f3067cd169e2d8d13` | `sha256:4ca6fbedf03e0aeba0aa7ac591970cddf87886dc115e9329468b4e8a68ffbba1` |
| `runtime/train_flow_grpo_wan_v2.py` | `sha256:b4686e3fa7f14eace09480fb3d6da30dea71cda00d31678c771305f4a5576de5` | `sha256:21b11cfd84635c30aa86e5bb1489b75d03e3ac2c3b88006f0eb74e551b2ef473` |
| `runtime/make_short_split.py` | `sha256:c6b63572e5b99be33db07aa97d38ac110184d60019c644a7d9afb24fac6fb029` | `sha256:1d9122c5835c0110d6ca9d80e046326ce8962dc831eb062b06c58637a55d32d0` |

### Re-judgement under ptc-hub 0.2.0 (scratch replay)

The historical records carry the old ids, which ptc-hub 0.2.0 refuses, so each attempt was re-judged on a scratch
replay with ptc `e20736e`, this repository installed, the runtime and OCR runtime overrides naming the original
directories (which hold the scripts the runs executed), against the amended spec (`<data-root>` substituted) and
registration:

- copies of the launch record, the smoke record and both evaluations' `env-runtime.json` and `arm-ocr_b_acc/`; in the
  copies only the schema ids (`ptc_skel.launch_record`, `ptc_skel.smoke_record`, `ptc_skel.item_eval` ->
  `ptc_hub.*`) and the scorer pin, from the gate-time pin to `f628fb1a…` (the new `eval_videos.py` pin), in the arms'
  `scorer.code_sha256` and the evaluations' `env-runtime.json` `script_sha256` were replaced; the training outputs
  were read in place;
- the launch record's label checksum covers the record and the GPU log accounts time by label, so (as in the TRL
  recipe's replay) the replay used a scratch host-work directory holding a copy of the host's `gpu-step.sh` (its
  `WORK=` line naming that directory) and the run's own two `gpu-guard.log` lines under the re-hashed label (the
  host's own log was only read);
  `host_work`, `gpu_step_sha256`, `host_config_sha256` and `label` were recomputed with ptc's own functions.

Compared with the canonical 2026-11-15 reports, every statistic is equal; the differences are exactly the expected
ones: `post_hoc` true (attempt 2: `status` PASS -> INCONCLUSIVE with `status_reasons` [`post_hoc`]),
`registration_hash` (the amendment), `registration_amends`, `registration_amendment` null (new field),
`evaluator_hash` and `evaluator_hash_registered` (the current scorer), `analysis_code_version`, the report `schema`
id, and the run-id-derived `run_id`, `created_at` and `lockfile_hash`. The ledgers are equal except their `writer_id`
(`ptc-hub.ledger@0.2.0`) and run-id columns, the lineage equal.

| Attempt | Canonical (ptc-skel 0.1.0) | Replay (ptc-hub 0.2.0) | delta_hat, 95% interval, p | Ledger rows |
|---|---|---|---|---|
| 1 | FAIL | FAIL (unchanged by `post_hoc`) | -0.008573 [-0.020086, 0.002172], p 0.941 (equal) | 960 (equal) |
| 2 | PASS | INCONCLUSIVE (`post_hoc`) | 0.031791 [0.015200, 0.049372], p <= 7.31e-05 (equal) | 3,200 (equal) |

## Rename to gatedtrain 0.3.0 (2026-11-18)

The framework was renamed again, from ptc-hub to gatedtrain (distribution and import `gatedtrain`, version 0.3.0),
and this recipe with it: distribution `gatedtrain-recipe-flow-grpo-wan-video`, import package
`gatedtrain_recipe_flow_grpo_wan_video`, entry-point group `gatedtrain.adapters`, runtime overrides
`GATEDTRAIN_RECIPE_FLOW_GRPO_WAN_VIDEO_RUNTIME` and `GATEDTRAIN_RECIPE_FLOW_GRPO_WAN_VIDEO_OCR_RUNTIME` (default: the
sibling checkout `../gatedtrain-recipe-flow-grpo-sd35-ocr/runtime`), and the documented image tag
`gatedtrain-a5-runtime:1` (`container/Containerfile`, still byte-identical to the SD3.5 OCR recipe's: only its comment
changed; the RUN instruction, including the package list path `/etc/ptc-a5-packages.txt` it writes, is unchanged, and
the published runs record the image by ID). gatedtrain 0.3.0 renamed the persisted identifiers once more (schema ids
`gatedtrain.*`, ledger writer id `gatedtrain.ledger@0.3.0`, analysis code version `gatedtrain 0.3.0`). The files
under results/2026-11-15/ and the byte-copied fixtures keep their bytes; the recipe-runtime schema ids `ptc_mm.*`
(cut markers, run arguments, item set and prompt metadata, whose hashes the runs and the registrations record) are
data-format identifiers of the published runs and are kept.

### python-Levenshtein replaced by rapidfuzz

The published runs computed the video OCR reward and the OCR accuracy with `Levenshtein.distance` from
python-Levenshtein 0.27.5 (GPL-2.0-or-later), installed in the SD3.5 OCR recipe's runtime venv that these runs use.
The three scripts now import the same function from rapidfuzz 3.14.6 (MIT), which python-Levenshtein wraps; the SD3.5
OCR recipe's `runtime/pyproject.toml` lists `rapidfuzz==3.14.6` instead of `levenshtein`, and its RESULT.md and
`runtime/check_edit_distance.py` give the equivalence check (20,006 string pairs, 0 mismatches of the distance and of
the OCR accuracy). The import lies inside the SHARED SAMPLER block of `train_flow_grpo_wan.py`,
`train_flow_grpo_wan_v2.py` and `eval_videos.py`; the whole difference of the block, in all three files:

```diff
-    from Levenshtein import distance
+    from rapidfuzz.distance.Levenshtein import distance
```

The scripts' docstrings also carry the MIT (flow_grpo) and Apache-2.0 (diffusers `pipeline_wan.py`) attributions of
the ported code (`THIRD_PARTY_NOTICES.md`).

### Pins (old -> new; only the new ones are accepted)

- `sampler_code_sha256` (`adapter.SAMPLER_CODE_SHA256`):
  `sha256:dad91e418fe717bf2ad0b3fe2727dfdea36a316b87c2240efc0aa14955d90300` ->
  `sha256:41606b0b53ad637da4614aaa38fd52b5e9bf0b2147da7ed71c3eaad92dcb1f17`.
- The SD3.5 OCR recipe's `runtime/uv.lock` (`adapter.VENV_LOCK_SHA256`):
  `sha256:8243b02f8287944026511dff3489f2cb23c91b88a0fd0d7a1b8f22678ece20d7` ->
  `sha256:502c7ce70e0950cf36cbd108cd9566e0193cbcabe8914c8d0c5bbd231caecda1` (levenshtein removed, project renamed;
  the other packages unchanged; see that recipe's RESULT.md).

| Script | Before (ptc-hub 0.2.0) | Now |
|---|---|---|
| `runtime/eval_videos.py` (the scorer's `code_sha256`) | `sha256:f628fb1abcd84c0bba4715802ee67d1eb644cfb6b3aca9dbcf92b389667c3a54` | `sha256:61873167b855e84df94edf99492ee52ccb6f3147127502cbfaf157b2bee59b4f` |
| `runtime/train_flow_grpo_wan.py` | `sha256:4ca6fbedf03e0aeba0aa7ac591970cddf87886dc115e9329468b4e8a68ffbba1` | `sha256:bac3ea77d62a1beedcb9b2c305e2a26f9490cc5cc0847dbfd3c68564fc27f0b8` |
| `runtime/train_flow_grpo_wan_v2.py` | `sha256:21b11cfd84635c30aa86e5bb1489b75d03e3ac2c3b88006f0eb74e551b2ef473` | `sha256:406e9de5b687f1d54c82b771f0ca49cd11005b246c3f0eea7acb746a74efd94f` |
| `runtime/make_short_split.py` | `sha256:1d9122c5835c0110d6ca9d80e046326ce8962dc831eb062b06c58637a55d32d0` | `sha256:ba183329fa7dd2f02c812f4e1a1969f39b5f9d89e8d5a53e18c3c7427ca00e16` |

### Amendments

The gate definitions are carried forward by results/2026-11-15/gate/registration-gatedtrain.yaml (amends
`sha256:572927edb524557b64550da5790a7ff2745f7d9516cbcfa45bfb7c7940d88b7b`, `registration-ptc-hub.yaml`) with
spec-gatedtrain.yaml, and registration-v2-gatedtrain.yaml (amends
`sha256:fba1a2c92800b760fe54788c5aab44d16824b39eb41b88f1ead541fdd1a561f4`, `registration-v2-ptc-hub.yaml`) with
spec-v2-gatedtrain.yaml; both keep `post_hoc: true`, as the amendments they carry forward. Besides
`analysis_code_version`, `amends` and `created_at`, only the deciding evaluator's identity differs:
`sha256:843d424d...` -> `sha256:329abb1fba6d417d38e1b44151b04a63f5759802c3ef81967c5d9f3b5e8de843` (the scorer above).
`gatedtrain validate` (`<data-root>` substituted) exits 0 on both new specs; `spec-ptc-hub.yaml` and
`spec-v2-ptc-hub.yaml` are refused (REG-FIELD, old analysis code version, with the amendment fix). The original
verdicts (attempt 1 FAIL, attempt 2 PASS) remain reproducible with ptc `78eb913` as described above.

### No re-collect under gatedtrain 0.3.0; a numeric cross-check instead

As for the SD3.5 OCR recipe, the published runs cannot be re-collected under the new pins without changing what
identifies their checkpoints: every cut's `cut.json` records `sampler_code_sha256` (as do `run_args.json` and
`env-runtime.json`, which the adapter requires to agree with it and with the pin), and `cut.json` is one of the files
whose manifest hash is the cut's checkpoint hash. Rewriting it in a scratch copy would change those hashes, so it is
not done. The runs' per-item scores do not depend on the import change.

Cross-check on 2026-11-18 with gatedtrain `8da5706`: `gatedtrain.gates.evaluate_items`, the function the gate report
is computed with, applied to scratch copies of each published run's two arms (`eval/baseline` and `eval/candidate`
`items.jsonl`, read with `gatedtrain.ext.arm.read_lines`), the registered item set, the clause of the amended spec,
the registration's bootstrap seed and 10,000 replicates, `min_paired`, `max_unpaired`, the evaluator's score range,
the run's plan and the arms' model hashes. All 27 result fields each report also has are equal to the published
report:

| Attempt | Pass rule (published verdict) | delta_hat, 95% interval, p | Power, MDE | n |
|---|---|---|---|---|
| 1 | FAIL (FAIL) | -0.008573 [-0.020086, 0.002172], p 0.941 | 0.99947, 0.01606 | 400 |
| 2 | PASS (PASS) | 0.031791 [0.015200, 0.049372], p 7.31e-05 | 0.99935, 0.02436 | 288 |

Under the amendments' `post_hoc: true` attempt 2 is reported INCONCLUSIVE with reason `post_hoc`, as in the
2026-11-17 re-judgement; attempt 1 stays FAIL. This is a cross-check of the statistics, not a gate report; the last
full re-judgement is the 2026-11-17 one above (ptc-hub 0.2.0).
