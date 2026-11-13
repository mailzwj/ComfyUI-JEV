# JEV-27B-VL ComfyUI Plugin — Usage Guide

Run **JEV-27B-VL** (Qwen3.8-27B-VL backbone + decision LoRA + decision head) as ComfyUI custom
nodes. This is a faithful port of the `jev27-bare-v1` decision protocol from the model
package's `serve_decide.py`:

- **System 1**: a single forward pass readout — decisions come out in one step (the decision
  LoRA is injected, read, and ejected within the call, so it never contaminates later work);
- **System 2**: the base model thinks first, then the answer distribution is read (no LoRA);
  can be triggered adaptively or forced;
- A plain **JEVChat** node (official chat template, thinking channel on its own output).

Inputs: text (state / question / options) + images (any number, optional).
Outputs: aligned probability distribution, final choice, full protocol JSON (usage and
thinking details included).

## 1. Installation

1. The plugin lives in `ComfyUI/custom_nodes/ComfyUI-JEV/` (no pip install, no new
   third-party dependencies).
2. Put the model in `ComfyUI/models/LLM/JEV-27B-VL/`:

   ```
   models/LLM/JEV-27B-VL/
       JEV-27B-VL.safetensors            merged single file (backbone + vision + lm_head + MTP, int8, 26.5 GB)
       character_lora_bf16.safetensors  decision LoRA (401 modules)
       decision_head.json               S1 verbalizer ids + per-kind bias
       calibration.json                 per-kind softmax temperatures
       adapter_config.json
       tokenizer.json / tokenizer_config.json
       config.json / generation_config.json
       preprocessor_config.json / video_preprocessor_config.json
       chat_template.jinja
   ```

   At import the plugin registers `models/LLM` as a model folder named `llm`, so any merged
   qwen35-style `.safetensors` dropped there becomes selectable.

3. **Restart ComfyUI** (needed on first install, after plugin code changes, and after adding a
   new model file to `models/LLM`). Node category is `JEV`; display names: `JEV Model Loader`,
   `JEV Decide`, `JEV Chat`.

Requirements: ComfyUI ≥ 0.45 (native `qwen35_27b` text encoder); the standard ComfyUI Python
env (torch / transformers / safetensors). No new third-party dependencies.

## 2. Node overview

| Node | Purpose | Link type |
|---|---|---|
| `JEVModelLoader` | Loads the model: merged weights + JEV tokenizer + 401 LoRA hooks (resident, not injected) | outputs `model` (JEVMODEL) |
| `JEVDecide` | Decision: S1 readout + optional adaptive S2 thinking. **Output node** | input `model`; outputs `probabilities` (JEVPROB), `choice`, `info` |
| `JEVChat` | Plain chat through the base model; thinking channel on a separate output. **Output node** | input `model`; outputs `thinking`, `text` |

`JEVMODEL` is a custom link type: wire the loader's `model` output to any number of
`JEVDecide` / `JEVChat` `model` inputs — the model loads once and is shared.

## 3. JEVModelLoader

| Parameter | Default | Description |
|---|---|---|
| `model_name` | `JEV-27B-VL/JEV-27B-VL.safetensors` | Merged single file under `models/LLM` |

Done automatically at load: qwen35_27b / int8 quantization / MTP head detection, JEV tokenizer
atomicity assertions, decision LoRA construction (401 bypass hooks, not injected yet), and
dequantization of the vision tower `pos_embed` (~5 MB, keeps the image path working in
every VRAM mode). Nothing touches the GPU yet — weights stream in on the first execution.

## 4. JEVDecide (decision node)

### Three task kinds `kind`

| kind | Options | Notes |
|---|---|---|
| `noul` | fixed `false` / `true` | yes/no. Leave `options` empty. S2 displays `Yes (true)` / `No (false)` and flips back to `[P(false), P(true)]` |
| `score` | fixed `0`–`5` | six-level rating. Leave `options` empty. `thinking=auto/on` is rejected (protocol) |
| `choice` | **2–256** options in `options`, **one per line** | multi-choice. With >16 options `strategy` decides the procedure |

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `model` | — | Wire from `JEVModelLoader.model` |
| `kind` | `noul` | Task kind (table above) |
| `state` | empty | Decision state / context text (multiline). Images enter the prompt right after this text |
| `images` | none | Optional, from a `Load Image` node; multiple images follow `state` in order |
| `question` | empty | The question to decide |
| `options` | empty | `choice` only, one option per line (labels are assigned A, B, C… including Aa… up to 256) |
| `strategy` | `auto` | Only for choice with >16 options: `auto` = single (one forward); `permute` (average of 4 shuffles); `tournament` (bracketed elimination) |
| `thinking` | `default` | `default`/`off`: System 1 only; `auto`: add S2 when the leading S1 option is below `threshold`, then fold; `on`: always S1 then fold with S2 |
| `threshold` | `0.0` | **0 means the protocol default 0.8** (not zero!). S2 triggers when the max S1 probability is below it |
| `mix` | `0.0` | **0 means the protocol default 0.5** (not zero!). Fold weight `p = (1-mix)·p1 + mix·p2` |
| `think_budget` | `0` | S2 thinking token budget; **0 = protocol default 32768** |
| `reasoning_effort` | `default` | `xhigh` / `medium` / `low`, the official template's thinking-effort instruction |
| `seed` | `-1` | S2 sampling seed; **-1 = the protocol seed** `crc32(question+options)` (same as serve); any other value is used directly |
| `use_mtp` | `False` | MTP speculative decoding. serve runs plain decode, so **keep it off when reproducing/comparing serve**; turn on for pure local speed |
| `system2_only` | `False` | Skip S1 and return the S2 distribution (rejected for score) |
| `want_reasoning` | `True` | Include the S2 reasoning text in the `info` JSON |

### Outputs

- `probabilities`: probability list aligned with the options (`[P(false), P(true)]` for noul, 6
  entries for score, N for choice). Feed it to any downstream node that takes a list/numbers,
  or just use it for inspection.
- `choice`: the winning option text (e.g. `true`, `5`, `Paris`).
- `info`: the full protocol JSON string (`jev27-bare-v1`, same shape as the hosted API response):

  ```json
  {
    "kind": "choice",
    "effective_kind": "choice",
    "options": ["London", "Paris", "Berlin"],
    "probabilities": [0.99982, 0.000101, 0.000083],
    "choice_index": 0,
    "choice": "London",
    "adaptation": "native",
    "protocol": "jev27-bare-v1",
    "model": "JEV-27B-VL\\JEV-27B-VL.safetensors",
    "usage": { "prompt_tokens": 46, "completion_tokens": 0, "total_tokens": 46 },
    "elapsed_seconds": 1.86,
    "num_model_requests": 1
  }
  ```

  When `thinking` is not `off`/`default` it also carries (`used: false` when `auto` did not
  trigger; triggered runs include the details):

  ```json
  "thinking": {
    "mode": "on", "threshold": 0.8, "budget": 512, "used": true,
    "think_tokens": 63, "think_seconds": 4.2, "finished_within_budget": true,
    "reasoning": "...thinking text (omitted when want_reasoning=False)..."
  }
  ```

  Field notes:

- `adaptation`: the S1 readout path — `native` (noul/score and choice with ≤16 options, one
  forward), or the strategy name for choice with >16 options (`permute` / `tournament`).
- A `system2_only=True` response has no `adaptation`/`effective_kind`; it carries `"system": 2`
  instead and `probabilities` is the S2 distribution (not folded).

### Typical wiring (GUI)

```
Load Image ──images──▶ JEV Decide ◀──model── JEV Model Loader
                            │ choice / probabilities / info (terminal outputs; preview/log them)
```

- Text-only decisions: leave `images` unwired; `state` may be empty.
- Vision questions (e.g. "does the image show X"): `kind=noul`, put the question in `question`,
  wire the image to `images`.
- Context / multi-turn: put the context in `state` (frame:
  `[kind]…[state]…[question]…[options]…[decision]:`).

## 5. JEVChat (chat node)

| Parameter | Default | Description |
|---|---|---|
| `model` | — | Wire from `JEVModelLoader.model` |
| `system` | none | Optional system message |
| `prompt` | empty | User message |
| `images` | none | Optional, placed after the prompt text |
| `enable_thinking` | `True` | Thinking channel switch (template `enable_thinking`) |
| `reasoning_effort` | `default` | Thinking-effort instruction (xhigh/medium/low) |
| `max_tokens` | `2048` | Generation cap (thinking included) |
| `temperature` / `top_k` / `top_p` | `1.0` / `20` / `0.95` | Sampler (matches `generation_config` defaults) |
| `seed` | `-1` | **-1 = `crc32(prompt)` derived seed**; any other value is used directly |
| `use_mtp` | `True` | Free-form chat defaults to MTP on for speed |

Outputs: `thinking` (thinking-channel content, empty string when none), `text` (the answer body).
Split rule: the output id stream is cut at the last think-close token; everything after is `text`.
The trailing stop token (`im_end` / pad) is stripped from `text` before decoding — matching vLLM,
which keeps the token in `token_ids` but removes it from the decoded string.

## 6. Calling from the API (optional)

Both `JEVDecide` and `JEVChat` are **output nodes**, so they can be submitted directly through
`POST /prompt` with no extra preview node. Results come back through the WebSocket and
`GET /history/{prompt_id}` under `outputs`.

ComfyUI V3 API conventions (as verified for this project):

1. Body shape is `{"prompt": {"1": {"class_type": …, "inputs": {…}}, "2": …}}`;
2. **V3 prompts get no automatic defaults**: every input must be present in `inputs` (query
   `GET /object_info` for the defaults and combo options first);
3. Combo values must match `/object_info` exactly (on Windows model paths are backslash form,
   `JEV-27B-VL\JEV-27B-VL.safetensors`);
4. Links are written `["source_node_id", output_index]`;
5. Images: first `POST /upload/image` (multipart, field name `image`), then pass the returned
   `name` as the `images` input; a tensor from a GUI `Load Image` node is also accepted.

Minimal example (text choice, 3 options):

```json
{
  "prompt": {
    "1": {
      "class_type": "JEVModelLoader",
      "inputs": { "model_name": "JEV-27B-VL\\JEV-27B-VL.safetensors" }
    },
    "2": {
      "class_type": "JEVDecide",
      "inputs": {
        "model": ["1", 0],
        "kind": "choice",
        "state": "",
        "question": "The capital of France is?",
        "options": "London\nParis\nBerlin",
        "strategy": "auto",
        "thinking": "default",
        "threshold": 0.0,
        "mix": 0.0,
        "think_budget": 0,
        "reasoning_effort": "default",
        "seed": -1,
        "use_mtp": false,
        "system2_only": false,
        "want_reasoning": true
      }
    }
  }
}
```

Bundled examples: `tools/api_run.py` (7 cases — noul / score / choice / s2 / adaptive / chat /
image — fills defaults, uploads images, polls history), and `tools/compare_with_serve.py`
(compares per-option probabilities against the vLLM `serve.sh` `/v1/decide`, run it user-side).

## 7. Performance and memory (measured on RTX 5090 32 GB)

- The int8 weights are ~26.5 GB and are hosted by ComfyUI's MinVram / async offload: weights stream
  from system RAM into VRAM per forward. Expect ~28 GB system RAM resident and ~15–25 GB VRAM peaks
  (comfortable on a 32 GB card; smaller cards still run but fault in more slowly).
- LoRA adapter weights (bf16, ~0.4 GB) are on the GPU only while the S1 hooks are injected; the
  `finally` eject always runs.
- Measured times (same session): first noul 7.7 s (includes the first weight stream); warm S1 about
  1.9–2 s; single-image noul about 10.5 s (includes first image-path setup: one vision-tower
  forward over 999 prompt tokens); S2 adds thinking generation on top (seconds within a 512 budget).
- Numerics: the ComfyUI backbone is int8 convrot, so S1 readouts differ from a bf16 vLLM reference
  service by a small floating-point margin (expected); the hook inject/eject itself is
  bit-exact (measured residual 0.0).

## 8. FAQ

- **Node list did not change after code/model changes?** Restart ComfyUI (node combo lists are
  built at startup).
- **What does `threshold`/`mix` = 0 do?** It does *not* mean "no thinking / pure S1"; 0 means
  "use the protocol default" (0.8 / 0.5, matching serve). To disable S2, set `thinking` to
  `default` or `off`.
- **`seed=-1` vs an explicit number?** -1 derives the seed from the question/options (or
  chat prompt) via `crc32`, so results reproduce the reference service; a number controls it
  fully.
- **Why do the two nodes default `use_mtp` differently?** `JEVDecide` defaults off (serve runs plain
  decode — keeps decisions comparable to the reference); `JEVChat` defaults on (free-form chat
  only wants speed; its sampling is not compared against serve).
- **`NoCapableBackendError` on the image path?** Older plugin builds did not dequantize
  `pos_embed`; in service mode (`NORMAL_VRAM`) ComfyUI hands the int8 table over as bf16 and the
  dequant kernel has no such backend. Current builds handle this at load time — if you still see
  it, confirm you run the latest code and restarted.
- **Tight VRAM / RAM?** The 27B text encoder + vision tower is inherently large; close other
  processes and keep ComfyUI's default MinVram behavior.
- **Compare against the hosted API / vLLM serve?** Use `tools/compare_with_serve.py`: start the
  model package's `serve.sh` (vLLM, port 8000) first, then run it; it prints the per-option
  deltas. S1 should agree to float precision; S2 is intentionally skipped — the two runtimes'
  samplers diverge token-for-token even with equal seeds.

## 9. Verification performed (all green before delivery)

- `tools/verify_model_dir.py`: dir completeness; `load_clip` auto-detection (qwen35_27b / int8 /
  MTP); 16 atomic special tokens; image triple = ids [248053, 248056, 248054]; think-close =
  248069.
- `tools/smoke_load.py`: 401 hooks / 256 choice labels / A–P decision head asserts; real S1
  forwards for all three kinds; S1 bit-deterministic.
- `tools/validate.py`: LoRA bypass — readout shifts 0.848→0.986 with hooks, restores bit-exactly
  after eject (residual 0.0); S2 deterministic under the protocol seed; system2_only; image
  expansion (36→998 prompt tokens); chat thinking/text split.
- `tools/api_run.py` (running ComfyUI, `POST /prompt`, 7/7 passed): noul `[0.015, 0.985]→true`,
  score `→5`, choice `[0.9998,…]→Paris`, S2 forced thinking correct, adaptive (threshold 0.9:
  S1 0.856 triggers S2 fold), chat channel split, single image 999 prompt tokens.
