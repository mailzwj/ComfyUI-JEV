"""Live ComfyUI API tests: post V3 prompts for the JEV nodes and print their outputs.

Usage: python api_run.py [noul|score|choice|s2|adaptive|chat|image]
Requires a running ComfyUI (http://127.0.0.1:8188) with the plugin loaded.
"""
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8188"
MODEL = "JEV-27B-VL/JEV-27B-VL.safetensors"
IMG = r"F:\Download\JEV-27B-VL\agent_judge.png"


def post_json(url, payload):
    req = urllib.request.Request(BASE + url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def decide_prompt(kind, state, question, options, **extra):
    inputs = {"kind": kind, "state": state, "question": question, "model": ["1", 0]}
    if options:
        inputs["options"] = options
    inputs.update(extra)
    return {
        "1": {"class_type": "JEVModelLoader", "inputs": {"model_name": MODEL}},
        "2": {"class_type": "JEVDecide", "inputs": inputs},
    }


def chat_prompt(prompt, **extra):
    return {
        "1": {"class_type": "JEVModelLoader", "inputs": {"model_name": MODEL}},
        "2": {"class_type": "JEVChat", "inputs": {"model": ["1", 0], "prompt": prompt, **extra}},
    }


def submit_and_wait(prompt):
    res = post_json("/prompt", {"prompt": prompt})
    pid = res["prompt_id"]
    while True:
        h = post_json("/history/" + pid, {})
        if pid in h:
            return h[pid]
        time.sleep(1.0)


def upload_image(path):
    boundary = "----jevboundary"
    with open(path, "rb") as f:
        body = f.read()
    data = ("--{}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{}\"\r\n"
            "Content-Type: image/png\r\n\r\n".format(boundary, path.split("\\")[-1]).encode() + body +
            "\r\n--{}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{}--\r\n"
            .format(boundary, boundary).encode())
    req = urllib.request.Request(BASE + "/upload/image", data=data,
                                 headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())["name"]


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "noul"
    if what == "noul":
        p = decide_prompt("noul", "The light in the room is on.", "Is the light on?", None)
    elif what == "score":
        p = decide_prompt("score", "The answer is clearly 5 out of 5.", "How correct is the statement?", None)
    elif what == "choice":
        p = decide_prompt("choice", "Paris is the capital of France.", "Which city is the capital of France?", "Paris\nBerlin\nMadrid")
    elif what == "s2":
        p = decide_prompt("noul", "The light in the room is on.", "Is the light on?", None, thinking="on", think_budget=512, seed=0)
    elif what == "adaptive":
        p = decide_prompt("choice", "I am not sure, could go either way.", "Which is more likely?", "A\nB",
                          thinking="auto", threshold=0.9, think_budget=512, seed=0)
    elif what == "chat":
        p = chat_prompt("What is 17 * 23? Answer in one short sentence.", max_tokens=256, seed=0)
    elif what == "image":
        name = upload_image(IMG)
        p = decide_prompt("noul", "Look at the attached image.", "Does the image show a judge?", None, images=name)
    else:
        print("unknown target", what)
        return
    h = submit_and_wait(p)
    print(json.dumps(h["outputs"].get("2", {}), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
