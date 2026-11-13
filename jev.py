"""JEV-27B-VL decision protocol (jev27-bare-v1) for ComfyUI.

System 1: one forward pass over the bare prompt frame; the answer distribution is read from the
last-position logits of the LoRA-adapted lm_head (bypass-injected decision LoRA, manual toggle).
System 2: the unmodified base model thinks over the same input; the answer-letter distribution is
read in one step after the thinking channel closes.  Ports serve_decide.py of the model package.
"""
import json
import math
import os
import random
import string
import time
import zlib

import torch

import folder_paths
import comfy.lora
import comfy.model_management
import comfy.ops
import comfy.sd
import comfy.text_encoders.qwen35 as qwen35
from comfy.text_encoders.qwen_vl import qwen2vl_mrope_position_ids
from comfy.weight_adapter.bypass import BypassInjectionManager
from safetensors.torch import load_file
from transformers import Qwen2TokenizerFast

PROTOCOL = "jev27-bare-v1"
PLACEHOLDER = "XQXCONTENTXQX"
MAX_OPTIONS = 256
DEFAULT_BUDGET = 32768

PAD_ID = 248044
IM_END_ID = 248046
IMG_TRIPLE_ID = [248053, 248056, 248054]
THINK_OPEN_ID = 248068
THINK_CLOSE_ID = 248069

CLASS_LABELS = list(string.ascii_uppercase) + [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]

S1_RANGE = "choice"
VERBALIZER = "verbalizer_ids"


def _softmax(z):
    m = max(z)
    e = [math.exp(x - m) for x in z]
    s = sum(e)
    return [x / s for x in e]


def _fold(p1, p2, w):
    return [(1 - w) * x + w * y for x, y in zip(p1, p2)]


def _groups(n, k=16):
    g = math.ceil(n / k)
    base, extra = divmod(n, g)
    out, i = [], 0
    for j in range(g):
        size = base + (1 if j < extra else 0)
        out.append(list(range(i, i + size)))
        i += size
    return out


def _usage(usage):
    return {"prompt_tokens": usage["prompt_tokens"], "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["prompt_tokens"] + usage["completion_tokens"]}


class JEVTokenizer(qwen35.Qwen35ImageTokenizer):
    """qwen35_27b tokenizer with the JEV fast tokenizer swapped in (no BOS; atomic specials verified)."""

    def __init__(self, model_dir, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, model_type="qwen35_27b")
        inner = getattr(self, self.clip)
        inner.tokenizer = Qwen2TokenizerFast.from_pretrained(model_dir)
        inner.inv_vocab = {v: k for k, v in inner.tokenizer.get_vocab().items()}
        dec = lambda ids: inner.tokenizer.decode(ids, skip_special_tokens=False)
        for tid in (PAD_ID, 248045, IM_END_ID, 248047, 248048, 248049, 248050, 248051, 248052,
                    248053, 248054, 248055, 248056, 248057, THINK_OPEN_ID, THINK_CLOSE_ID):
            assert inner.tokenizer.encode(dec([tid]), add_special_tokens=False) == [tid], "token not atomic: {}".format(tid)
        self.img_triple = dec(IMG_TRIPLE_ID)
        self.end_think = dec([THINK_CLOSE_ID])
        self.open_think = dec([THINK_OPEN_ID])
        assert inner.tokenizer.encode(self.img_triple, add_special_tokens=False) == IMG_TRIPLE_ID
        assert inner.tokenizer.encode(self.end_think, add_special_tokens=False) == [THINK_CLOSE_ID]
        self.raw = inner.tokenizer


def _label_ids(raw, context):
    out = []
    for lab in CLASS_LABELS:
        t = raw.encode(lab, add_special_tokens=False)
        if len(t) == 1 and t[0] in raw.encode(context.format(lab), add_special_tokens=False):
            out.append((lab, t[0]))
        if len(out) == MAX_OPTIONS:
            break
    return out


def _ans_labels(raw):
    base = raw.encode("Answer: (", add_special_tokens=False)
    out = []
    for lab in CLASS_LABELS:
        ids = raw.encode("Answer: ({})".format(lab), add_special_tokens=False)
        if ids[:len(base)] == base and len(ids) == len(base) + 2:
            out.append((lab, ids[len(base)]))
        if len(out) == MAX_OPTIONS:
            break
    return out


def _load_lora_hooks(model_dir, cond):
    lora_path = os.path.join(model_dir, "character_lora_bf16.safetensors")
    lora_alpha = float(json.load(open(os.path.join(model_dir, "adapter_config.json")))["lora_alpha"])
    weights = load_file(lora_path, device="cpu")
    stems = sorted({k[:-len(".lora_A.weight")] for k in weights.keys()})
    cond_keys = [k[:-len(".weight")] for k in cond.state_dict().keys() if k.endswith(".weight") and ".mtp." not in k]
    remap = {}
    for stem in stems:
        tail = stem.split("layers.", 1)[-1]
        if "layers." in stem:
            match = [k for k in cond_keys if k.endswith("layers.{}".format(tail))]
        else:
            match = [k for k in cond_keys if k.endswith("lm_head")]
        assert len(match) == 1, "lora stem {} -> {}".format(stem, match)
        key = match[0]
        remap[key] = stem
    data = {}
    for key, stem in remap.items():
        data["{}.lora_A.weight".format(key)] = weights["{}.lora_A.weight".format(stem)]
        data["{}.lora_B.weight".format(key)] = weights["{}.lora_B.weight".format(stem)]
        data["{}.alpha".format(key)] = torch.tensor(lora_alpha)
    adapters = comfy.lora.load_lora(data, {k: k for k in remap})
    manager = BypassInjectionManager()
    for key, adapter in adapters.items():
        manager.add_adapter(key, adapter)
    manager.create_injections(cond)
    assert len(manager.hooks) == len(stems), "hooks {} != lora stems {}".format(len(manager.hooks), len(stems))
    return manager.hooks


class JEVModel:
    def __init__(self, clip, model_dir, model_name, head, temps):
        self.clip = clip
        self.model_name = model_name
        self.tokenizer = JEVTokenizer(model_dir)
        raw = self.tokenizer.raw
        dec = _label_ids(raw, "x\n{}) y")
        assert [t for _, t in dec[:head["slots"]["ranges"]["choice"][1] - head["slots"]["ranges"]["choice"][0]]] == head[VERBALIZER][head["slots"]["ranges"][S1_RANGE][0]:head["slots"]["ranges"][S1_RANGE][1]], "first labels must be the trained A-P head"
        self.labels = [l for l, _ in dec]
        self.label_ids = [t for _, t in dec]
        self.ans_labels = _ans_labels(raw)
        self.head = head
        self.temps = temps
        cond = clip.cond_stage_model
        inner = getattr(cond, cond.clip)
        _dequantize_visual_pos_embed(inner)
        self.hooks = _load_lora_hooks(model_dir, inner)

    def inject_hooks(self):
        for hook in self.hooks:
            hook.inject()

    def eject_hooks(self):
        for hook in self.hooks:
            hook.eject()


def load_jev_model(model_name):
    full = folder_paths.get_full_path_or_raise("llm", model_name)
    model_dir = os.path.dirname(full)
    clip = comfy.sd.load_clip([full])
    head = json.load(open(os.path.join(model_dir, "decision_head.json")))
    calib = json.load(open(os.path.join(model_dir, "calibration.json")))
    temps = calib["per_kind"]
    temps.setdefault("score", 1.0)
    return JEVModel(clip, model_dir, model_name, head, temps)


def _dequantize_visual_pos_embed(inner):
    """Materialize the int8 pos-embedding table (~5 MB) at load time.

    In NORMAL_VRAM service mode the quantized-embedding dequant kernel finds the qdata as bf16 and
    has no capable backend (NoCapableBackendError) on the image path; a dense table takes the plain
    embedding fallback in every vram mode.
    """
    visual = getattr(inner.transformer, "visual", None)
    pe = getattr(visual, "pos_embed", None)
    weight = getattr(pe, "weight", None)
    if weight is None or not isinstance(weight, comfy.ops.QuantizedTensor):
        return
    layout_cls = comfy.ops.get_layout_class(getattr(pe, "layout_type", None))
    dense = layout_cls.dequantize(weight._qdata, weight._params)
    pe.weight = torch.nn.Parameter(dense.detach(), requires_grad=False)


def _inner(model):
    cond = model.clip.cond_stage_model
    return cond, getattr(cond, cond.clip)


def _context(model):
    clip = model.clip
    cond, inner = _inner(model)
    cond.reset_clip_options()
    clip.load_model()
    device = clip.patcher.load_device
    cond.set_clip_options({"layer": None})
    cond.set_clip_options({"execution_device": device})
    return device, cond, inner


def _embed(model, cond, inner, text, images, usage):
    tokens = model.tokenizer.tokenize_with_weights(text, llama_template="{}", image=images, prevent_empty_text=True)
    row = list(tokens.values())[0]
    tokens_only = [[t[0] for t in b] for b in row]
    embeds, _, _, embeds_info = inner.process_tokens(tokens_only, inner.execution_device)
    usage["prompt_tokens"] += embeds.shape[1]
    usage["requests"] += 1
    return embeds, qwen2vl_mrope_position_ids(embeds_info, embeds.shape[1], embeds.device)


def _last_logits(inner, embeds, position_ids):
    x, _ = inner.transformer.model.forward(None, embeds=embeds, attention_mask=None, position_ids=position_ids)
    return inner.transformer.logits(x)[0, -1]


def s1_pass(model, cond, inner, kind, state, images, question, opts, usage):
    head, temps = model.head, model.temps
    lo, hi = head["slots"]["ranges"][kind]
    if kind == "choice":
        n = len(opts)
        ids = model.label_ids[:n]
        bias = [head["bias"][lo + i] if lo + i < hi else 0.0 for i in range(n)]
        lines = ["{}) {}".format(model.labels[i], o) for i, o in enumerate(opts)]
    else:
        ids = head[VERBALIZER][lo:hi]
        bias = head["bias"][lo:hi]
        lines = opts
    text = ("[kind] {}\n[state] {}{}\n[question] {}\n[options]\n{}\n[decision]:"
            .format(kind, state, model.tokenizer.img_triple * len(images), question, "\n".join(lines)))
    embeds, pos = _embed(model, cond, inner, text, images, usage)
    lg = _last_logits(inner, embeds, pos)
    return _softmax([(max(float(lg[t]), -1e9) + b) / temps[kind] for t, b in zip(ids, bias)])


def s1_dist(model, cond, inner, kind, state, images, question, opts, strategy, usage):
    if kind != "choice" or len(opts) <= 16 or strategy == "single":
        return s1_pass(model, cond, inner, kind, state, images, question, opts, usage)
    n = len(opts)
    if strategy == "permute":
        rng = random.Random(zlib.crc32(question.encode()))
        orders = [list(range(n))] + [rng.sample(range(n), n) for _ in range(3)]
        res = [s1_pass(model, cond, inner, kind, state, images, question, [opts[i] for i in o], usage) for o in orders]
        p = [0.0] * n
        for o, r in zip(orders, res):
            for i, v in zip(o, r):
                p[i] += v / len(orders)
        return p
    groups = _groups(n)
    parts = [s1_pass(model, cond, inner, kind, state, images, question, [opts[i] for i in g], usage) for g in groups]
    in_group = {o: p for g, ps in zip(groups, parts) for o, p in zip(g, ps)}
    chosen = [max(g, key=lambda o: (in_group[o], -o)) for g in groups]
    rest = sorted((o for g in groups for o in g if o not in set(chosen)), key=lambda o: (-in_group[o], o))
    fin = sorted(chosen + rest[:max(0, 16 - len(chosen))])
    final = dict(zip(fin, s1_pass(model, cond, inner, kind, state, images, question, [opts[i] for i in fin], usage)))
    group_of = {o: gi for gi, g in enumerate(groups) for o in g}
    share = [0.0] * len(groups)
    cap = [0.0] * len(groups)
    for f in fin:
        share[group_of[f]] += final[f]
        cap[group_of[f]] += in_group[f]
    among = sum(a * b for a, b in zip(share, cap))
    p = [final[o] * among if o in final else share[group_of[o]] * in_group[o] for o in range(n)]
    s = sum(p)
    return [x / s for x in p]


def s2_pass(model, cond, inner, kind, state, images, question, opts, budget, effort, seed, want_reasoning, use_mtp, usage):
    raw = model.tokenizer.raw
    shown = ["Yes (true)", "No (false)"] if kind == "noul" else opts
    labs = model.ans_labels[:len(shown)]
    body = "\n".join("({}) {}".format(l, o) for (l, _), o in zip(labs, shown))
    tail = ("\n\nQuestion: {}\n\nOptions:\n{}\n\n"
            "Think it through carefully, then give your final answer on the last line in the form: Answer: (X)"
            .format(question, body))
    kw = {}
    if effort != "default":
        kw["reasoning_effort"] = effort
    chat = raw.apply_chat_template([{"role": "user", "content": PLACEHOLDER}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=True, **kw)
    pre, post = chat.split(PLACEHOLDER)
    gen_text = pre + state + model.tokenizer.img_triple * len(images) + tail + post
    embeds, pos = _embed(model, cond, inner, gen_text, images, usage)
    seed_eff = seed if seed is not None and seed >= 0 else zlib.crc32((question + body).encode()) & 0x7FFFFFFF
    t0 = time.time()
    ids = inner.transformer.generate(embeds, True, budget, 1.0, 20, 0.95, 0.0, 1.0, seed_eff,
                                     stop_tokens=[THINK_CLOSE_ID], position_ids=pos, mtp=use_mtp)
    finished = len(ids) > 0 and ids[-1] == THINK_CLOSE_ID
    if finished:
        ids = ids[:-1]
    thought = raw.decode(ids, skip_special_tokens=False)
    read_text = gen_text + thought + model.tokenizer.end_think + "\n\n" + "Answer: ("
    embeds, pos = _embed(model, cond, inner, read_text, images, usage)
    lg = _last_logits(inner, embeds, pos)
    ids_read = [t for _, t in labs]
    p = _softmax([max(float(lg[t]), -1e9) for t in ids_read])
    if kind == "noul":
        p = [p[1], p[0]]  # (A) yes / (B) no -> [P(false), P(true)]
    ntok = len(ids) + (1 if finished else 0)
    usage["completion_tokens"] += ntok
    info = {"think_tokens": ntok, "think_seconds": round(time.time() - t0, 3),
            "finished_within_budget": finished}
    if want_reasoning:
        info["reasoning"] = thought.strip()
    return p, info


def decide(model, kind, state, images, question, options, strategy, thinking, threshold, mix, think_budget,
           effort, seed, system2_only, want_reasoning, use_mtp):
    t0 = time.time()
    if kind == "choice":
        opts = options or []
        assert 2 <= len(opts) <= MAX_OPTIONS, "choice needs 2-{} options, got {}".format(MAX_OPTIONS, len(opts))
    else:
        opts = ["false", "true"] if kind == "noul" else [str(i) for i in range(6)]
    if thinking == "default":
        thinking = "off"
    if kind == "score" and thinking in ("auto", "on"):
        raise ValueError("thinking is supported for noul and choice")
    if kind == "score":
        thinking = "off"
    if system2_only and kind == "score":
        raise ValueError("system2_only is supported for noul and choice")
    tau = 0.8 if threshold is None else threshold
    w = 0.5 if mix is None else mix
    budget = think_budget if think_budget else DEFAULT_BUDGET
    if strategy == "auto":
        strategy = "single"
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "requests": 0}
    device, cond, inner = _context(model)
    with comfy.model_management.cuda_device_context(device), comfy.ops.use_quantized_matmul(cond, device):
        if system2_only:
            p2, think = s2_pass(model, cond, inner, kind, state, images, question, opts, budget, effort, seed,
                                want_reasoning, use_mtp, usage)
            k = max(range(len(p2)), key=p2.__getitem__)
            return {"kind": kind, "options": opts, "probabilities": p2, "choice_index": k, "choice": opts[k],
                    "system": 2, "model": model.model_name,
                    "thinking": {"used": True, "budget": budget, **think},
                    "usage": _usage(usage), "elapsed_seconds": time.time() - t0, "num_model_requests": usage["requests"]}
        model.inject_hooks()
        try:
            p1 = s1_dist(model, cond, inner, kind, state, images, question, opts, strategy, usage)
        finally:
            model.eject_hooks()
        probs, think = p1, None
        if thinking == "on" or (thinking == "auto" and max(p1) < tau):
            p2, think = s2_pass(model, cond, inner, kind, state, images, question, opts, budget, effort, seed,
                                want_reasoning, use_mtp, usage)
            probs = _fold(p1, p2, w)
            think = {"used": True, **think}
        elif thinking != "off":
            think = {"used": False}
    k = max(range(len(probs)), key=probs.__getitem__)
    native = kind != "choice" or len(opts) <= 16
    resp = {"kind": kind, "effective_kind": kind, "options": opts, "probabilities": probs, "choice_index": k,
            "choice": opts[k], "adaptation": "native" if native else strategy, "protocol": PROTOCOL,
            "model": model.model_name, "usage": _usage(usage), "elapsed_seconds": time.time() - t0,
            "num_model_requests": usage["requests"]}
    if think is not None:
        resp["thinking"] = {"mode": thinking, "threshold": tau, "budget": budget, **think}
    return resp


def chat(model, system, prompt, images, thinking, effort, max_tokens, temperature, top_k, top_p, seed, use_mtp):
    t0 = time.time()
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "requests": 0}
    device, cond, inner = _context(model)
    with comfy.model_management.cuda_device_context(device), comfy.ops.use_quantized_matmul(cond, device):
        raw = model.tokenizer.raw
        parts = [{"type": "text", "text": prompt}] + [{"type": "image"} for _ in range(len(images))]
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": parts})
        kw = {}
        if effort != "default":
            kw["reasoning_effort"] = effort
        text = raw.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking, **kw)
        embeds, pos = _embed(model, cond, inner, text, images, usage)
        seed_eff = seed if seed is not None and seed >= 0 else zlib.crc32(prompt.encode()) & 0x7FFFFFFF
        ids = inner.transformer.generate(embeds, True, max_tokens, temperature, top_k, top_p, 0.0, 1.0, seed_eff,
                                         stop_tokens=[PAD_ID, IM_END_ID], position_ids=pos, mtp=use_mtp)
        usage["completion_tokens"] += len(ids)
        # generate keeps the token that triggered the stop (it only truncates after it); vLLM strips
        # it from the decoded text, so drop trailing stop tokens before decoding.
        while ids and ids[-1] in (PAD_ID, IM_END_ID):
            ids.pop()
        close = max((i for i, t in enumerate(ids) if t == THINK_CLOSE_ID), default=-1)
        if close >= 0:
            thinking_text = raw.decode(ids[:close], skip_special_tokens=False).strip()
            answer = raw.decode(ids[close + 1:], skip_special_tokens=False).strip()
        else:
            thinking_text = ""
            answer = raw.decode(ids, skip_special_tokens=False).strip()
    return {"thinking": thinking_text, "text": answer, "usage": _usage(usage), "elapsed_seconds": time.time() - t0}
