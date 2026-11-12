"""In-process validation battery: S2 determinism, LoRA bypass effect/cleanness, image path, chat path.

Note: with the MinVram/VBAR offload stack, raw prefill logits are not bit-reproducible across
calls (weight fault-in state drifts).  Determinism assertions therefore use the seeded S2 sampler
(seed makes sampling reproducible) and the S1 readout behavior (LoRA on/off comparison).
"""
import importlib.util
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

COMFYUI = r"D:\Workspace\ComfyUI"
sys.path.insert(0, COMFYUI)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import folder_paths
from folder_paths import add_model_folder_path

llm = os.path.join(COMFYUI, "models", "LLM")
add_model_folder_path("llm", llm)
folder_paths.folder_names_and_paths["llm"][1].add(".safetensors")

spec = importlib.util.spec_from_file_location("jev", os.path.join(ROOT, "jev.py"))
jev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jev)

model = jev.load_jev_model(os.path.join("JEV-27B-VL", "JEV-27B-VL.safetensors"))
images0 = torch.zeros((0, 1, 1, 3), dtype=torch.float32)

# S1: LoRA on vs off on the same frame; eject must restore the off-LoRA readout
device, cond, inner = jev._context(model)
text = "[kind] noul\n[state] The light in the room is on.\n[question] Is the light on?\n[options]\nfalse\ntrue\n[decision]:"
def s1_readout():
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "requests": 0}
    embeds, pos = jev._embed(model, cond, inner, text, images0, usage)
    return jev.s1_pass(model, cond, inner, "noul", "The light in the room is on.", images0, "Is the light on?",
                       ["false", "true"], usage)
p_off = s1_readout()
model.inject_hooks()
p_on = s1_readout()
model.eject_hooks()
p_off2 = s1_readout()
print("s1 off:  ", json.dumps(p_off))
print("s1 on:   ", json.dumps(p_on))
print("s1 off2: ", json.dumps(p_off2))
shift = max(abs(a - b) for a, b in zip(p_off, p_on))
restored = max(abs(a - b) for a, b in zip(p_off, p_off2))
assert shift > 0.05, "decision LoRA has no measurable effect on the readout"
assert restored < shift / 2, "eject did not restore the base readout"
print("ok: bypass effect {:.4f}, eject residual {:.4f}".format(shift, restored))

# S2 determinism: same protocol seed twice -> identical distribution and reasoning
kw = dict(state="The light in the room is on.", images=images0, question="Is the light on?",
          options=None, strategy="auto", thinking="on", threshold=None, mix=None,
          think_budget=512, effort="low", seed=-1, system2_only=False, want_reasoning=True, use_mtp=False)
a = jev.decide(model, "noul", **kw)
b = jev.decide(model, "noul", **kw)
print("s2 noul:", json.dumps({k: a[k] for k in ("probabilities", "choice", "thinking")}, ensure_ascii=False))
assert a["probabilities"] == b["probabilities"], "S2 not deterministic"
assert a["thinking"]["reasoning"] == b["thinking"]["reasoning"]
assert a["thinking"]["used"] is True and a["thinking"]["think_tokens"] > 0
assert a["choice"] == "true" and a["probabilities"][1] > 0.9, "expected 'true' with high confidence"
print("ok: S2 deterministic")

# system2_only returns the raw System 2 distribution
c = jev.decide(model, "noul", state="2 + 2 = 5.", images=images0, question="Is 2 + 2 = 5?",
               options=None, strategy="auto", thinking="on", threshold=None, mix=None,
               think_budget=512, effort="low", seed=-1, system2_only=True, want_reasoning=False, use_mtp=False)
print("s2only:", json.dumps({k: c[k] for k in ("options", "probabilities", "choice", "system")}, ensure_ascii=False))
assert c["system"] == 2 and c["choice"] == "false" and c["probabilities"][0] > 0.9, "expected 'false' with high confidence"

# image path: prompt tokens grow by the smart-resized grid; decision runs
img = Image.open(r"F:\Download\JEV-27B-VL\agent_judge.png").convert("RGB")
tensor = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float().div(255.0).permute(1, 2, 0).unsqueeze(0)
t0 = jev.decide(model, "noul", state="Look at the image.", images=images0, question="Does the image show a judge?",
                options=None, strategy="auto", thinking="off", threshold=None, mix=None,
                think_budget=0, effort="default", seed=-1, system2_only=False, want_reasoning=True, use_mtp=False)
t1 = jev.decide(model, "noul", state="Look at the image.", images=tensor, question="Does the image show a judge?",
                options=None, strategy="auto", thinking="off", threshold=None, mix=None,
                think_budget=0, effort="default", seed=-1, system2_only=False, want_reasoning=True, use_mtp=False)
print("image noul (no img):", json.dumps(t0["probabilities"]))
print("image noul (1 img):", json.dumps(t1["probabilities"]), "prompt_tokens", t0["usage"]["prompt_tokens"], "->", t1["usage"]["prompt_tokens"])
assert t1["usage"]["prompt_tokens"] > t0["usage"]["prompt_tokens"] + 20, "image not expanded"
print("ok: image path")

# chat path: thinking splits the channel, plain prompt stays off
ch1 = jev.chat(model, "", "Reply with exactly: pong", images0, True, "default", 512, 1.0, 20, 0.95, 1234, False)
ch2 = jev.chat(model, "", "Reply with exactly: pong", images0, False, "default", 512, 1.0, 20, 0.95, 1234, False)
print("chat thinking-on:  thinking={!r} text={!r}".format(ch1["thinking"][:80], ch1["text"]))
print("chat thinking-off: thinking={!r} text={!r}".format(ch2["thinking"][:80], ch2["text"]))
assert ch1["text"].strip() != "" and ch2["text"].strip() != ""
ch3 = jev.chat(model, "", "What is 17 * 23? Show the work.", images0, True, "low", 512, 1.0, 20, 0.95, 1234, False)
print("chat low-effort math: thinking={!r} text={!r}".format(ch3["thinking"][:120], ch3["text"]))
assert "391" in ch3["text"], "expected 391 in the answer"
print("ok: chat path")

print("ok: validate")
