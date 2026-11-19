# Third-party notices

This repository is licensed under the Apache License 2.0 (`LICENSE`). It contains code ported from, and data taken
from, the following third-party projects, each distributed under its own license: yifan123/flow_grpo (MIT) and,
through it, huggingface/diffusers (Apache-2.0).

## flow_grpo

- Source: <https://github.com/yifan123/flow_grpo>
- Commit: `879042cf5707f8b90daa98d147d7deac2317c5da`
- License: MIT (the repository's `LICENSE` at that commit, reproduced below)
- What this repository uses:
  - Code: `runtime/train_flow_grpo_wan.py`, `runtime/train_flow_grpo_wan_v2.py` and `runtime/eval_videos.py` port
    `scripts/train_wan2_1.py`, `flow_grpo/diffusers_patch/wan_pipeline_with_logprob.py`,
    `flow_grpo/diffusers_patch/wan_prompt_embedding.py`, `flow_grpo/ocr.py`, `flow_grpo/rewards.py` and
    `flow_grpo/stat_tracking.py` (adapted; the files cite the upstream lines they follow).
  - Data: the prompts in `runtime/items/items.jsonl`, `runtime/items-short/items.jsonl`, `tests/fixtures/flow_grpo/items.jsonl`
    and `tests/fixtures/flow_grpo/train.jsonl` are lines of `dataset/ocr/test.txt` and `dataset/ocr/train.txt` of that
    commit (selected and annotated with a target text; see `runtime/make_short_split.py` and the SD3.5 OCR recipe). The
    upstream repository states no separate license for `dataset/ocr`; it is distributed under the repository's MIT
    license.

```text
MIT License

Copyright (c) 2025 Jie Liu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## huggingface/diffusers (Apache-2.0)

- Source: https://github.com/huggingface/diffusers. flow_grpo names no diffusers source or commit for the file below;
  the commit given is the last one that changed the diffusers file before flow_grpo first committed its file
  (2025-08-14), and the code is used as copied by flow_grpo (commit `879042cf5707f8b90daa98d147d7deac2317c5da`).
- Files: flow_grpo's `flow_grpo/diffusers_patch/wan_pipeline_with_logprob.py` is derived, with modifications, from
  `src/diffusers/pipelines/wan/pipeline_wan.py` (diffusers commit `cb8e61ed2f792db8bcb9606c12e9ae7e400e4b49`; 111 of
  its 184 distinct lines longer than 30 characters, leading spaces stripped, occur verbatim in that file). The Wan sampling loop with
  classifier-free guidance and the SDE step in `runtime/train_flow_grpo_wan.py`, `runtime/train_flow_grpo_wan_v2.py`
  and `runtime/eval_videos.py` (the shared sampler block) are ported from it; each script's docstring carries this
  attribution.
- Copyright line of `pipeline_wan.py` at that commit: "Copyright 2025 The Wan Team and The HuggingFace Team. All
  rights reserved."
- License: Licensed under the Apache License, Version 2.0 (the same text as `LICENSE`). diffusers has no NOTICE file.
