"""Cross-parity check against a running serve_decide (vLLM) instance.

Run the reference server per the model package (serve.sh), then:
    python compare_with_serve.py
It POSTs the same noul/score/choice cases to the vLLM /v1/decide endpoint and to the
in-process plugin decide(), and prints the per-option probability deltas.

S1 is a single forward with the same prompt tokens, temperatures and biases, so the
distributions should agree to float precision (small deviations come from int8
quantization of the backbone on the ComfyUI side and BF16 vs FP8 attention).
System 2 (thinking) is NOT compared here: the two runtimes sample with different kernels,
so generated thoughts differ token-for-token even with equal seeds.
"""
import json
import os
import sys
import time
import urllib.request

COMFYUI = r"D:\Workspace\ComfyUI"
sys.path.insert(0, COMFYUI)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SERVE = "http://127.0.0.1:8000"
MODEL = "jev-decision"

import folder_paths
from folder_paths import add_model_folder_path

llm = os.path.join(COMFYUI, "models", "LLM")
add_model_folder_path("llm", llm)
folder_paths.folder_names_and_paths["llm"][1].add(".safetensors")

import importlib.util

spec = importlib.util.spec_from_file_location("jev", os.path.join(ROOT, "jev.py"))
jev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jev)

import torch

CASES = [
    ("noul", "The light in the room is on.", "Is the light on?", None),
    ("score", "The answer is clearly 5 out of 5.", "How correct is the statement?", None),
    ("choice", "Paris is the capital of France.", "Which city is the capital of France?", ["Paris", "Berlin", "Madrid"]),
]


def serve_decide(kind, state, question, options):
    payload = {"kind": kind, "state": state, "question": question}
    if options:
        payload["options"] = options
    req = urllib.request.Request(SERVE + "/v1/decide", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def plugin_decide(model, kind, state, question, options):
    return jev.decide(model, kind, state, torch.zeros((0, 1, 1, 3), dtype=torch.float32), question, options,
                      "auto", "off", None, None, 0, "default", -1, False, False, False)


def main():
    model = jev.load_jev_model(os.path.join("JEV-27B-VL", "JEV-27B-VL.safetensors"))
    worst = 0.0
    for kind, state, question, options in CASES:
        s = serve_decide(kind, state, question, options)
        p = plugin_decide(model, kind, state, question, options)
        delta = max(abs(a - b) for a, b in zip(s["probabilities"], p["probabilities"]))
        worst = max(worst, delta)
        print("{}: serve={} plugin={} max_delta={:.6f} serve_choice={} plugin_choice={}"
              .format(kind, [round(x, 5) for x in s["probabilities"]],
                      [round(x, 5) for x in p["probabilities"]], delta, s["choice"], p["choice"]))
        t0 = s.get("usage", {})
        print("   serve usage:", t0, " plugin usage:", p["usage"])
    print("worst per-option delta: {:.6f}".format(worst))
    print("ok: compare_with_serve")


if __name__ == "__main__":
    main()
