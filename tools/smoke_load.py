"""In-process smoke test: load the plugin engine, verify hooks/labels, run real S1 forwards."""
import importlib.util
import json
import os
import sys

COMFYUI = r"D:\Workspace\ComfyUI"
sys.path.insert(0, COMFYUI)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import torch

import folder_paths
from folder_paths import add_model_folder_path

llm = os.path.join(COMFYUI, "models", "LLM")
if os.path.isdir(llm):
    add_model_folder_path("llm", llm)
    folder_paths.folder_names_and_paths["llm"][1].add(".safetensors")

spec = importlib.util.spec_from_file_location("jev", os.path.join(ROOT, "jev.py"))
jev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jev)

model_name = os.path.join("JEV-27B-VL", "JEV-27B-VL.safetensors")
model = jev.load_jev_model(model_name)
print("ok: hooks", len(model.hooks), "labels", len(model.labels), "ans", len(model.ans_labels))

images = torch.zeros((0, 1, 1, 3), dtype=torch.float32)
kw = dict(state="The light in the room is on.", images=images, question="Is the light on?",
          options=None, strategy="auto", thinking="off", threshold=None, mix=None,
          think_budget=0, effort="default", seed=-1, system2_only=False, want_reasoning=True, use_mtp=False)

r1 = jev.decide(model, "noul", **kw)
r2 = jev.decide(model, "noul", **kw)
r3 = jev.decide(model, "noul", **kw)
print("noul:", json.dumps({k: r1[k] for k in ("probabilities", "choice", "usage", "num_model_requests", "elapsed_seconds")}, ensure_ascii=False))
assert r2["probabilities"] == r1["probabilities"] and r3["probabilities"] == r2["probabilities"], "S1 not deterministic"
assert sum(r1["probabilities"]) > 0.9999
print("ok: S1 deterministic")

sc = jev.decide(model, "score", state="The answer is clearly 5 out of 5.", images=images, question="How correct is the statement?",
                options=None, strategy="auto", thinking="off", threshold=None, mix=None,
                think_budget=0, effort="default", seed=-1, system2_only=False, want_reasoning=True, use_mtp=False)
print("score:", json.dumps({k: sc[k] for k in ("probabilities", "choice")}, ensure_ascii=False))

ch = jev.decide(model, "choice", state="Paris is the capital of France.", images=images, question="Which city is the capital of France?",
                options=["Paris", "Berlin", "Madrid"], strategy="auto", thinking="off", threshold=None, mix=None,
                think_budget=0, effort="default", seed=-1, system2_only=False, want_reasoning=True, use_mtp=False)
print("choice:", json.dumps({k: ch[k] for k in ("probabilities", "choice", "adaptation")}, ensure_ascii=False))
print("ok: smoke_load")
