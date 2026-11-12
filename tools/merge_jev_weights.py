import os

from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "models", "LLM", "JEV-27B-VL"))


def main():
    merged = {}
    for name in ("backbone_int8_convrot.safetensors", "vision_projector_int8_convrot.safetensors"):
        with safe_open(os.path.join(MODEL_DIR, name), framework="pt", device="cpu") as f:
            for key in f.keys():
                assert key not in merged, f"duplicate key {key}"
                merged[key] = f.get_tensor(key)
    assert len(merged) == 2241, f"expected 2241 keys, got {len(merged)}"
    for key in ("lm_head.weight", "mtp.fc.weight", "model.visual.blocks.0.attn.qkv.weight",
                "model.visual.patch_embed.proj.weight", "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"):
        assert key in merged, f"missing {key}"
    out = os.path.join(MODEL_DIR, "JEV-27B-VL.safetensors")
    save_file(merged, out)
    print(f"ok: wrote {os.path.getsize(out) / 1e6:.1f} MB with {len(merged)} keys")


if __name__ == "__main__":
    main()
