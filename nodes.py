import json
import os

import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence

import folder_paths
import comfy.model_management
from comfy_api.latest import io

from . import jev

JEVMODEL = io.Custom("JEVMODEL")
JEVPROB = io.Custom("JEVPROB")

_LLM_LIST = folder_paths.get_filename_list("llm")
DEFAULT_MODEL = next((n for n in _LLM_LIST if n.replace(os.sep, "/").endswith("JEV-27B-VL/JEV-27B-VL.safetensors")),
                     "JEV-27B-VL/JEV-27B-VL.safetensors")

_EMPTY = torch.zeros((0, 1, 1, 3), dtype=torch.float32)


def _load_images(images):
    """images is None, a Tensor (HWC/CHW, 0..1), or a str/list of filenames from the ComfyUI input dir."""
    if images is None or (isinstance(images, torch.Tensor) and images.numel() == 0):
        return _EMPTY
    if isinstance(images, torch.Tensor):
        if images.dim() == 3:
            images = images.unsqueeze(0)
        if images.shape[1] == 3 and images.shape[-1] != 3:
            images = images.permute(0, 2, 3, 1)
        return images.to(dtype=torch.float32)
    names = images if isinstance(images, (list, tuple)) else [images]
    out = []
    for name in names:
        path = folder_paths.get_annotated_filepath(name)
        img = Image.open(path)
        for frame in ImageSequence.Iterator(img):
            frame = ImageOps.exif_transpose(frame.convert("RGB"))
            arr = torch.from_numpy(np.array(frame).astype(np.float32) / 255.0)
            out.append(arr)
    if not out:
        return _EMPTY
    return torch.stack(out, dim=0).to(dtype=torch.float32)


class JEVModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="JEVModelLoader",
            display_name="JEV Model Loader",
            category="JEV",
            inputs=[
                io.Combo.Input("model_name", options=_LLM_LIST, default=DEFAULT_MODEL,
                               tooltip="Merged JEV-27B-VL .safetensors under models/LLM"),
            ],
            outputs=[JEVMODEL.Output("model")],
            description="Loads the JEV-27B-VL text encoder with its decision LoRA hooks.",
        )

    @classmethod
    def execute(cls, model_name):
        return io.NodeOutput(jev.load_jev_model(model_name))


class JEVDecide(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="JEVDecide",
            display_name="JEV Decide",
            category="JEV",
            is_output_node=True,
            inputs=[
                JEVMODEL.Input("model"),
                io.Combo.Input("kind", options=["noul", "score", "choice"], default="noul",
                               tooltip="noul: false/true; score: 0-5; choice: 2-256 options, one per line"),
                io.String.Input("state", multiline=True, default="", tooltip="The decision state text"),
                io.Image.Input("images", optional=True, tooltip="Images in order; each image follows the state text"),
                io.String.Input("question", multiline=True, default=""),
                io.String.Input("options", multiline=True, default="", tooltip="choice only, one option per line"),
                io.Combo.Input("strategy", options=["auto", "single", "permute", "tournament"], default="auto",
                               tooltip="choices > 16 options only; auto = single"),
                io.Combo.Input("thinking", options=["default", "off", "auto", "on"], default="default",
                               tooltip="default/off: System 1 only; auto: think when the leading option is below threshold; on: always"),
                io.Float.Input("threshold", default=0.0, min=0.0, max=1.0, step=0.05, tooltip="0 = protocol default 0.8"),
                io.Float.Input("mix", default=0.0, min=0.0, max=1.0, step=0.05,
                               tooltip="System 2 weight in the fold; 0 = protocol default 0.5"),
                io.Int.Input("think_budget", default=0, min=0, max=262144, step=256, tooltip="0 = 32768"),
                io.Combo.Input("reasoning_effort", options=["default", "xhigh", "medium", "low"], default="default"),
                io.Int.Input("seed", default=-1, min=-1, max=2147483647, tooltip="-1 = protocol seed (crc32 of question+options)"),
                io.Boolean.Input("use_mtp", default=False, tooltip="serve_decide runs plain decode; keep off for parity"),
                io.Boolean.Input("system2_only", default=False, tooltip="skip System 1, return the System 2 distribution"),
                io.Boolean.Input("want_reasoning", default=True, tooltip="include the System 2 reasoning text in info"),
            ],
            outputs=[
                JEVPROB.Output("probabilities", tooltip="Probabilities aligned with the options"),
                io.String.Output("choice"),
                io.String.Output("info", tooltip="Full protocol response (jev27-bare-v1) as JSON"),
            ],
            description="JEV decision protocol: System 1 readout with optional adaptive System 2 thinking.",
        )

    @classmethod
    def execute(cls, model, kind, state, images=None, question="", options="", strategy="auto", thinking="default",
                threshold=0.0, mix=0.0, think_budget=0, reasoning_effort="default", seed=-1, use_mtp=False,
                system2_only=False, want_reasoning=True):
        images = _load_images(images)
        opts = [l for l in options.splitlines() if l.strip() != ""] if kind == "choice" else None
        resp = jev.decide(model, kind, state, images, question, opts, strategy, thinking,
                          None if threshold == 0.0 else threshold, None if mix == 0.0 else mix,
                          think_budget, reasoning_effort, seed, system2_only, want_reasoning, use_mtp)
        info_str = json.dumps(resp, ensure_ascii=False)
        # ui: results surfaced in /history and the websocket (the data outputs only flow through graph links)
        return io.NodeOutput(resp["probabilities"], resp["choice"], info_str,
                             ui={"probabilities": [resp["probabilities"]], "choice": [resp["choice"]], "info": [info_str]})


class JEVChat(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="JEVChat",
            display_name="JEV Chat",
            category="JEV",
            is_output_node=True,
            inputs=[
                JEVMODEL.Input("model"),
                io.String.Input("system", multiline=True, optional=True, tooltip="Optional system message"),
                io.String.Input("prompt", multiline=True, default=""),
                io.Image.Input("images", optional=True, tooltip="Images in order, after the prompt text"),
                io.Boolean.Input("enable_thinking", default=True),
                io.Combo.Input("reasoning_effort", options=["default", "xhigh", "medium", "low"], default="default"),
                io.Int.Input("max_tokens", default=2048, min=1, max=262144, step=128),
                io.Float.Input("temperature", default=1.0, min=0.0, max=5.0, step=0.05),
                io.Int.Input("top_k", default=20, min=0, max=1000),
                io.Float.Input("top_p", default=0.95, min=0.0, max=1.0, step=0.01),
                io.Int.Input("seed", default=-1, min=-1, max=2147483647, tooltip="-1 = crc32 of the prompt"),
                io.Boolean.Input("use_mtp", default=True),
            ],
            outputs=[
                io.String.Output("thinking", tooltip="Content of the thinking channel (empty when absent)"),
                io.String.Output("text"),
            ],
            description="Plain chat through the JEV-27B-VL base model with the Qwen 3.8 chat template.",
        )

    @classmethod
    def execute(cls, model, system, prompt, images=None, enable_thinking=True, reasoning_effort="default",
                max_tokens=2048, temperature=1.0, top_k=20, top_p=0.95, seed=-1, use_mtp=True):
        images = _load_images(images)
        out = jev.chat(model, system or "", prompt, images, enable_thinking, reasoning_effort,
                       max_tokens, temperature, top_k, top_p, seed, use_mtp)
        return io.NodeOutput(out["thinking"], out["text"],
                             ui={"thinking": [out["thinking"]], "text": [out["text"]]})
