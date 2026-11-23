"""Write TinyLlama-1.1B-Chat's pinned safetensors as one GGUF file that FlashProve's tools can read.

Usage: python -I export_gguf.py MODEL_DIR OUT.gguf

MODEL_DIR holds the pinned files of TinyLlama/TinyLlama-1.1B-Chat-v1.0 (config.json and
model.safetensors; the caller checks their SHA-256 first). The file written:

- architecture `llama`, with `llama.attention.key_length 64`, which FlashProve's readers need (they
  take 128 when it is absent);
- every matrix in BF16, the stored bits copied unchanged, so the file holds exactly the published
  values; the three kinds of norm gains in F32, widened exactly, as FlashProve requires;
- the query and key rows permuted as llama.cpp's convert_hf_to_gguf.py does for `llama`
  (LlamaModel.permute), so that each rotary pair is two neighbouring rows, the order that
  FlashProve's integer model and float pass assume and that fold.txt's `qkrows asis` states;
- no tokenizer: FlashProve's programs take token ids.

The package gguf must be version 0.19.0, the version that FlashProve's tools pin. The program
writes nothing over an existing file.
"""

import json
import sys
from importlib.metadata import version
from pathlib import Path

import gguf
import numpy as np
import torch
from safetensors import safe_open

GGUF_VERSION = "0.19.0"
ARCH = "llama"
# The configuration this exporter is written for; any other value stops it.
CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 2048,
    "num_hidden_layers": 22,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "intermediate_size": 5632,
    "vocab_size": 32000,
    "max_position_embeddings": 2048,
    "rope_theta": 10000.0,
    "rms_norm_eps": 1e-05,
    "hidden_act": "silu",
    "rope_scaling": None,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
}


def stop(message):
    sys.exit(f"export_gguf: {message}")


def permute(w, heads):
    """llama.cpp's LlamaModel.permute: each head's rows from (first half, second half) to pairs."""
    rows = w.shape[0]
    return w.reshape(heads, 2, rows // heads // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


def bf16_bytes(t):
    """The stored BF16 bits of a 2-D tensor as uint8 rows, unchanged."""
    if t.dtype != torch.bfloat16 or t.dim() != 2:
        stop(f"expected a 2-D bfloat16 tensor, got {t.dtype} {tuple(t.shape)}")
    bits = t.contiguous().view(torch.int16).numpy().view(np.uint16)
    return bits.view(np.uint8).reshape(t.shape[0], 2 * t.shape[1])


def f32(t):
    """A 1-D BF16 tensor widened exactly to float32."""
    if t.dtype != torch.bfloat16 or t.dim() != 1:
        stop(f"expected a 1-D bfloat16 tensor, got {t.dtype} {tuple(t.shape)}")
    return t.to(torch.float32).numpy()


def main(model_dir, out):
    if version("gguf") != GGUF_VERSION:
        stop(f"the package gguf must be {GGUF_VERSION}, not {version('gguf')}")
    model_dir, out = Path(model_dir), Path(out)
    if out.exists() or out.is_symlink():
        stop(f"{out} exists; nothing is overwritten")
    config = json.loads((model_dir / "config.json").read_text())
    for key, wanted in CONFIG.items():
        if config.get(key) != wanted:
            stop(f"config.json: {key} is {config.get(key)!r}, not {wanted!r}")
    if (
        config.get("head_dim", 64) != 64
        or config.get("attention_bias", False)
        or config.get("mlp_bias", False)
    ):
        stop("config.json: head_dim, attention_bias or mlp_bias is not as expected")
    d, layers = CONFIG["hidden_size"], CONFIG["num_hidden_layers"]
    heads, kv_heads = CONFIG["num_attention_heads"], CONFIG["num_key_value_heads"]
    dh = d // heads

    writer = gguf.GGUFWriter(str(out), ARCH)
    writer.add_string("general.name", "TinyLlama-1.1B-Chat-v1.0")
    writer.add_uint32(f"{ARCH}.context_length", CONFIG["max_position_embeddings"])
    writer.add_uint32(f"{ARCH}.embedding_length", d)
    writer.add_uint32(f"{ARCH}.block_count", layers)
    writer.add_uint32(f"{ARCH}.feed_forward_length", CONFIG["intermediate_size"])
    writer.add_uint32(f"{ARCH}.attention.head_count", heads)
    writer.add_uint32(f"{ARCH}.attention.head_count_kv", kv_heads)
    writer.add_uint32(f"{ARCH}.attention.key_length", dh)
    writer.add_uint32(f"{ARCH}.attention.value_length", dh)
    writer.add_uint32(f"{ARCH}.rope.dimension_count", dh)
    writer.add_float32(f"{ARCH}.rope.freq_base", CONFIG["rope_theta"])
    writer.add_float32(f"{ARCH}.attention.layer_norm_rms_epsilon", CONFIG["rms_norm_eps"])
    writer.add_uint32(f"{ARCH}.vocab_size", CONFIG["vocab_size"])
    writer.add_uint32("general.file_type", int(gguf.LlamaFileType.MOSTLY_BF16))

    names = {
        "model.embed_tokens.weight": "token_embd.weight",
        "model.norm.weight": "output_norm.weight",
        "lm_head.weight": "output.weight",
    }
    parts = {
        "self_attn.q_proj": "attn_q",
        "self_attn.k_proj": "attn_k",
        "self_attn.v_proj": "attn_v",
        "self_attn.o_proj": "attn_output",
        "mlp.gate_proj": "ffn_gate",
        "mlp.up_proj": "ffn_up",
        "mlp.down_proj": "ffn_down",
        "input_layernorm": "attn_norm",
        "post_attention_layernorm": "ffn_norm",
    }
    for layer in range(layers):
        for hf, ours in parts.items():
            names[f"model.layers.{layer}.{hf}.weight"] = f"blk.{layer}.{ours}.weight"

    with safe_open(model_dir / "model.safetensors", framework="pt") as source:
        stored = set(source.keys())
        if stored != set(names):
            stop(f"model.safetensors: unexpected tensors {sorted(stored ^ set(names))[:6]}")
        order = ["model.embed_tokens.weight"]
        for layer in range(layers):
            order += [f"model.layers.{layer}.{hf}.weight" for hf in parts]
        order += ["model.norm.weight", "lm_head.weight"]
        for hf in order:
            t = source.get_tensor(hf)
            name = names[hf]
            if name.endswith("_norm.weight"):
                writer.add_tensor(name, f32(t))
                continue
            if name.endswith("attn_q.weight"):
                t = permute(t, heads)
            elif name.endswith("attn_k.weight"):
                t = permute(t, kv_heads)
            writer.add_tensor(name, bf16_bytes(t), raw_dtype=gguf.GGMLQuantizationType.BF16)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        stop("usage: export_gguf.py MODEL_DIR OUT.gguf")
    main(sys.argv[1], sys.argv[2])
