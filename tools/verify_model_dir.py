import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, COMFY_ROOT)
os.chdir(COMFY_ROOT)

import folder_paths
import comfy.sd
from transformers import Qwen2TokenizerFast

MODEL_NAME = "JEV-27B-VL/JEV-27B-VL.safetensors"
MODEL_DIR = os.path.join(COMFY_ROOT, "models", "LLM", "JEV-27B-VL")
ATOMIC_IDS = (248044, 248045, 248046, 248047, 248048, 248049, 248050, 248051, 248052,
              248053, 248054, 248055, 248056, 248057, 248068, 248069)


def main():
    llm_dir = os.path.join(COMFY_ROOT, "models", "LLM")
    folder_paths.add_model_folder_path("llm", llm_dir)
    folder_paths.folder_names_and_paths["llm"][1].add(".safetensors")
    assert os.path.normpath(MODEL_NAME) in folder_paths.get_filename_list("llm"), "merged model not listed under llm"

    for n in ("tokenizer.json", "tokenizer_config.json", "config.json", "generation_config.json",
              "preprocessor_config.json", "video_preprocessor_config.json", "chat_template.jinja",
              "adapter_config.json"):
        assert os.path.isfile(os.path.join(MODEL_DIR, n)), f"missing sidecar {n}"

    head = json.load(open(os.path.join(MODEL_DIR, "decision_head.json")))
    calib = json.load(open(os.path.join(MODEL_DIR, "calibration.json")))
    for kind in ("noul", "score", "choice"):
        assert kind in head["slots"]["ranges"] and kind in calib["per_kind"], kind
    print("ok: head/calib", head["slots"]["ranges"],
          {k: round(v, 4) for k, v in calib["per_kind"].items()})

    tok = Qwen2TokenizerFast.from_pretrained(MODEL_DIR)
    for tid in ATOMIC_IDS:
        back = tok.encode(tok.decode([tid], skip_special_tokens=False), add_special_tokens=False)
        assert back == [tid], f"token {tid} not atomic: {back}"
    triple = tok.encode(tok.decode([248053, 248056, 248054], skip_special_tokens=False), add_special_tokens=False)
    assert triple == [248053, 248056, 248054]
    print("ok: 16 atomic tokens, image triple", triple,
          "end_think", tok.encode(tok.decode([248069], skip_special_tokens=False), add_special_tokens=False))

    clip = comfy.sd.load_clip([folder_paths.get_full_path("llm", MODEL_NAME)])
    cond = clip.cond_stage_model
    assert cond.clip_name == "qwen35_27b", cond.clip_name
    model = getattr(cond, cond.clip).transformer
    keys = list(cond.state_dict().keys())
    assert "qwen35_27b.transformer.model.lm_head.weight" in keys
    assert "qwen35_27b.transformer.mtp.fc.weight" in keys
    quant = sum("comfy_quant" in k for k in keys)
    assert quant > 0, "no quant metadata found"
    import comfy.text_encoders.qwen35 as q35
    assert isinstance(model, q35.Qwen35) and model.mtp is not None
    print(f"ok: load_clip {cond.clip_name} {type(model).__name__} layers={model.num_layers} "
          f"mtp={model.mtp is not None} keys={len(keys)} quant_keys={quant}")


if __name__ == "__main__":
    main()
