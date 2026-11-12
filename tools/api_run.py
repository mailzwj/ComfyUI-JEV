"""Live ComfyUI API tests: post V3 prompts for the JEV nodes and print their outputs.

Usage: JEV_API_BASE=http://127.0.0.1:8188 python api_run.py [noul|score|choice|s2|adaptive|chat|image]
Requires a running ComfyUI with the plugin loaded. Follows the standard ComfyUI API
convention: fetch /object_info for exact option values, send every input explicitly
(V3 prompts do not server-side-apply widget defaults).
"""
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = os.environ.get("JEV_API_BASE", "http://127.0.0.1:8188")
IMG = r"F:\Download\JEV-27B-VL\agent_judge.png"


def post_json(url, payload):
    req = urllib.request.Request(BASE + url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def get_json(url):
    with urllib.request.urlopen(BASE + url) as r:
        return json.loads(r.read())


def object_info():
    return get_json("/object_info")


def _entry_info(spec, name):
    """object_info input entry is [type, info_dict] (or [type] for link-only sockets)."""
    entry = spec["input"].get("required", {}).get(name) or spec["input"].get("optional", {}).get(name)
    if isinstance(entry, (list, tuple)) and len(entry) > 1 and isinstance(entry[1], dict):
        return entry[1]
    if isinstance(entry, dict):
        return entry
    return {}


def input_defaults(info, class_type):
    """All required+optional input names with their widget defaults, per /object_info."""
    spec = info[class_type]["input"]
    out = {}
    for section in ("required", "optional"):
        for name, specv in spec.get(section, {}).items():
            einfo = _entry_info({"input": {section: {name: specv}}}, name)
            out[name] = einfo.get("default")
    return out


def option_value(info, class_type, field, want_prefix=None):
    opts = _entry_info(info[class_type], field).get("options", [])
    if want_prefix:
        hits = [o for o in opts if o.startswith(want_prefix)]
        if hits:
            return hits[0]
    return opts[0]


def upload_image(path):
    boundary = "----jevboundary"
    with open(path, "rb") as f:
        body = f.read()
    fname = os.path.basename(path)
    data = ("--{}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{}\"\r\n"
            "Content-Type: image/png\r\n\r\n".format(boundary, fname).encode() + body +
            "\r\n--{}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{}--\r\n"
            .format(boundary, boundary).encode())
    req = urllib.request.Request(BASE + "/upload/image", data=data,
                                 headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())["name"]


def submit_and_wait(prompt):
    payload = json.dumps({"prompt": prompt}).encode()
    if os.environ.get("JEV_DEBUG"):
        print("DEBUG payload head:", payload[:200])
    req = urllib.request.Request(BASE + "/prompt", data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            res = json.loads(r.read())
    except urllib.error.HTTPError as e:
        print("PROMPT REJECTED:", e.read().decode()[:800])
        raise SystemExit(2)
    pid = res["prompt_id"]
    while True:
        h = get_json("/history/" + pid)
        if pid in h:
            if h[pid].get("status", {}).get("status_str") == "error":
                print("EXECUTION ERROR:", json.dumps(h[pid]["status"], ensure_ascii=False))
                raise SystemExit(3)
            return h[pid]
        time.sleep(1.0)


def build_decide(kind, **overrides):
    info = object_info()
    inputs = input_defaults(info, "JEVDecide")
    inputs["kind"] = kind
    inputs["model"] = ["1", 0]
    inputs.update(overrides)
    return {
        "1": {"class_type": "JEVModelLoader", "inputs": {"model_name": option_value(info, "JEVModelLoader", "model_name", "JEV-27B-VL")}},
        "2": {"class_type": "JEVDecide", "inputs": inputs},
    }


def build_chat(prompt_text, **overrides):
    info = object_info()
    inputs = input_defaults(info, "JEVChat")
    inputs["prompt"] = prompt_text
    inputs["model"] = ["1", 0]
    inputs.update(overrides)
    return {
        "1": {"class_type": "JEVModelLoader", "inputs": {"model_name": option_value(info, "JEVModelLoader", "model_name", "JEV-27B-VL")}},
        "2": {"class_type": "JEVChat", "inputs": inputs},
    }


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "noul"
    if what == "noul":
        p = build_decide("noul", state="The light in the room is on.", question="Is the light on?")
    elif what == "score":
        p = build_decide("score", state="The answer is clearly 5 out of 5.", question="How correct is the statement?")
    elif what == "choice":
        p = build_decide("choice", state="Paris is the capital of France.",
                         question="Which city is the capital of France?", options="Paris\nBerlin\nMadrid")
    elif what == "s2":
        p = build_decide("noul", state="The light in the room is on.", question="Is the light on?",
                         thinking="on", think_budget=512, seed=0)
    elif what == "adaptive":
        p = build_decide("choice", state="I am not sure, could go either way.", question="Which is more likely?",
                         options="A\nB", thinking="auto", threshold=0.9, think_budget=512, seed=0)
    elif what == "chat":
        p = build_chat("What is 17 * 23? Answer in one short sentence.", max_tokens=256, seed=0)
    elif what == "image":
        name = upload_image(IMG)
        p = build_decide("noul", state="Look at the attached image.", question="Does the image show a judge?", images=name)
    else:
        print("unknown target", what)
        return
    h = submit_and_wait(p)
    print(json.dumps(h["outputs"].get("2", {}), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
