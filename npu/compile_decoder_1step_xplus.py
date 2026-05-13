"""
1-step Whisper decoder with split cross-attention output for GGML-equivalent
host-driven streaming decode.

QAI Hub-style fixed KV cache (size 199 for max_decode=200).
Single decoder session per bucket — no graph swap, just loop .run().

Pipeline (all local, no QAI Hub cloud dependency):
1. Export 1 ONNX per bucket (single decoder, fixed KV cache)
2. LOCAL compile: qairt-converter (QAIRT 2.37) -> DLC
3. LOCAL link: qnn-context-binary-generator -> context binary (.bin)
4. Generate EPContext ONNX wrapper

NPU I/O per call:
  Inputs:  input_ids[1,1], attention_mask[1,1,1,200],
           cross_attention_mask[1,1,1,emb],
           12 KV_self(fixed 199), 12 KV_cross(bucket-sized), position_ids[1]
  Outputs: logits[1,51865,1,1], 12 KV_self_out(fixed 199),
           peak_attn[1,1,1,emb], cross_attn[1,n_aheads,1,emb]

KV self cache uses SLIDING WINDOW (concat + trim first position → always 199).
Do NOT call set_nstep_mode() — default SHAAttention behavior is sliding window.

Target device : Snapdragon X Plus 8-Core CRD (sc8340xp, Hexagon v73)
On-device SDK : QNN SDK 2.37 + onnxruntime-qnn 1.23.2
Compile SDK   : QAIRT 2.37.1.250807

IMPORTANT: Requires numpy<2.0 (e.g., numpy==1.26.4).
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn

import onnx
from onnx import TensorProto, helper
from qai_hub_models.models._shared.hf_whisper.model import (
    MASK_NEG,
    HfWhisper,
)
from whisper_npu_profile import (
    bucket_audio_emb_len as profile_bucket_audio_emb_len,
    default_build_dir,
    resolve_hf_model_source,
    resolve_target_profile,
    resolve_whisper_model_profile,
)

# -- Model config --------------------------------------------------------------
MODEL_PROFILE = resolve_whisper_model_profile("base")
TARGET_PROFILE = resolve_target_profile("s25_xplus_compat")
HF_VERSION = resolve_hf_model_source(MODEL_PROFILE)
NUM_BLOCKS = MODEL_PROFILE.num_blocks
ATTENTION_DIM = MODEL_PROFILE.attention_dim
NUM_HEADS = MODEL_PROFILE.num_heads
HEAD_DIM = MODEL_PROFILE.head_dim
VOCAB_SIZE = MODEL_PROFILE.vocab_size
MAX_DECODE = 200     # max decode length (fixed KV cache = 199)
CACHE_SIZE = MAX_DECODE - 1  # 199
DEFAULT_ALIGNMENT_HEADS = MODEL_PROFILE.alignment_heads

# -- Paths (all under current directory) ---------------------------------------
WORK = Path(__file__).parent / default_build_dir("decoder_1step", MODEL_PROFILE, TARGET_PROFILE)
ONNX_DIR = WORK / "onnx"
DLC_DIR = WORK / "dlc"
OUTPUT_DIR = WORK / "output"
WRAPPER_DIR = WORK / "wrappers"

# -- Target device -------------------------------------------------------------
DEVICE_NAME = TARGET_PROFILE.device_name

# -- QAIRT 2.37 SDK paths (local compile + link) ------------------------------
_DEFAULT_SDK = str(Path(__file__).parent / "qairt" / "2.37.1.250807")
SDK = Path(os.environ.get("QNN_SDK_ROOT", _DEFAULT_SDK))
CONVERTER = SDK / "bin/x86_64-linux-clang/qairt-converter"
CONTEXT_GEN = SDK / "bin/x86_64-linux-clang/qnn-context-binary-generator"
HTP_LIB = SDK / "lib/x86_64-linux-clang/libQnnHtp.so"
HTP_EXT_LIB = SDK / "lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so"

SOC_MODEL = TARGET_PROFILE.soc_model
DSP_ARCH = TARGET_PROFILE.dsp_arch


# ==============================================================================
# OneStepDecoder: single-step decoder with cross-attn output
# ==============================================================================

def parse_alignment_heads(spec):
    heads = []
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        layer_s, head_s = item.split(",")
        heads.append((int(layer_s), int(head_s)))
    return heads


class OneStepDecoder(nn.Module):
    """
    1-step Whisper decoder with fixed KV cache (sliding window) and split
    attention output.

    Uses DEFAULT SHAAttention behavior (NOT n-step mode):
      concat(cache, current) then trim first position → output cache stays 199.

    Inputs:
      input_ids[1,1], attention_mask[1,1,1,200],
      cross_attention_mask[1,1,1,emb],
      12 KV_self_in (each [8,1,64,199] or [8,1,199,64]),
      12 KV_cross (each [8,1,64,emb] or [8,1,emb,64]),
      position_ids[1]

    Outputs:
      logits[1,51865,1,1], 12 KV_self_out (same 199),
      peak_attn[1,1,1,emb], cross_attn[1,n_aheads,1,emb]
    """

    def __init__(self, qc_decoder, audio_emb_len, alignment_heads=None,
                 output_mode="debug"):
        super().__init__()
        self.embed_tokens = qc_decoder.embed_tokens
        self.embed_positions = qc_decoder.embed_positions
        self.layers = qc_decoder.layers
        self.layer_norm = qc_decoder.layer_norm
        self.proj_out = qc_decoder.proj_out
        self.num_blocks = len(self.layers)
        self.audio_emb_len = audio_emb_len
        self.output_mode = output_mode

        self.alignment_heads = alignment_heads
        self.peak_attn_layer = len(self.layers) - 1
        self.align_attn_layers = set()
        if output_mode != "logits_kv":
            self.layers[self.peak_attn_layer].encoder_attn.return_attn_weights = True

        if output_mode != "logits_kv" and alignment_heads:
            layer_heads = {}
            for layer_idx, head_idx in alignment_heads:
                layer_heads.setdefault(layer_idx, []).append(head_idx)
            self._layer_heads = layer_heads
            for layer_idx, heads in layer_heads.items():
                self.layers[layer_idx].encoder_attn.selected_heads = heads
                self.align_attn_layers.add(layer_idx)
        else:
            self._layer_heads = None

    def forward(self, input_ids, attention_mask, cross_attention_mask, *kv_and_pos):
        """
        Args:
            input_ids: [1, 1] int32
            attention_mask: [1, 1, 1, 200] causal mask
            cross_attention_mask: [1, 1, 1, emb] additive source mask
            *kv_and_pos: flat list of:
                [k_self_0, v_self_0, ..., k_self_5, v_self_5,  (12 tensors)
                 k_cross_0, v_cross_0, ..., k_cross_5, v_cross_5,  (12 tensors)
                 position_ids]  (1 tensor)
        """
        nb = self.num_blocks
        # Unpack KV caches
        kv_self = [(kv_and_pos[i], kv_and_pos[i + 1]) for i in range(0, nb * 2, 2)]
        cs = nb * 2
        kv_cross = [(kv_and_pos[cs + i], kv_and_pos[cs + i + 1]) for i in range(0, nb * 2, 2)]
        position_ids = kv_and_pos[-1]  # last arg

        # Embeddings
        input_embeds = self.embed_tokens(input_ids.to(torch.int64))
        positions = self.embed_positions(input_ids, position_ids=position_ids)
        hidden_states = input_embeds.unsqueeze(0) + positions

        # Decoder layers
        next_cache = []
        peak_attn_weights = None
        layer_attn_weights = []
        for idx, layer in enumerate(self.layers):
            out = layer(
                hidden_states,
                attention_mask=attention_mask,
                cross_attention_mask=cross_attention_mask,
                past_key_value=kv_self[idx],
                cross_attn_past_key_value=kv_cross[idx],
            )
            hidden_states = out[0]
            next_cache.append(out[1])
            if idx == self.peak_attn_layer:
                if len(out) > 3:
                    peak_attn_weights = out[2]
                    layer_attn_weights.append(out[3])
                elif len(out) > 2:
                    peak_attn_weights = out[2]
            elif idx in self.align_attn_layers and len(out) > 2:
                layer_attn_weights.append(out[2])

        hidden_states = self.layer_norm(hidden_states)
        logits = self.proj_out(hidden_states.permute(0, 3, 1, 2))

        if len(layer_attn_weights) > 1:
            align_attn_weights = torch.cat(layer_attn_weights, dim=1)
        elif len(layer_attn_weights) == 1:
            align_attn_weights = layer_attn_weights[0]
        else:
            align_attn_weights = peak_attn_weights

        # Flatten KV self output
        flat_kv = []
        for k_c, v_c in next_cache:
            flat_kv.extend([k_c, v_c])

        if self.output_mode == "logits_kv":
            return tuple([logits] + flat_kv)
        return tuple([logits] + flat_kv + [peak_attn_weights, align_attn_weights])


# ==============================================================================
# Phase 1: ONNX Export
# ==============================================================================

def export_onnx(qc_decoder, audio_emb_len, alignment_heads, sec_tag,
                output_mode="debug"):
    """Export 1-step decoder to ONNX with fixed KV cache."""
    unique_decoder_id = f"decoder_1step_{sec_tag}"
    onnx_path = ONNX_DIR / f"{unique_decoder_id}.onnx"

    if onnx_path.exists():
        print(f"  ONNX exists, skip: {onnx_path}")
        return onnx_path

    # Dummy inputs
    input_ids = torch.tensor([[50258]], dtype=torch.int32)
    attention_mask = torch.full((1, 1, 1, MAX_DECODE), MASK_NEG, dtype=torch.float32)
    attention_mask[:, :, :, -1:] = 0  # unmask last 1 position (step 0, sliding window)
    cross_attention_mask = torch.zeros((1, 1, 1, audio_emb_len), dtype=torch.float32)

    kv_self = []
    for _ in range(NUM_BLOCKS):
        kv_self.append(torch.zeros(NUM_HEADS, 1, HEAD_DIM, CACHE_SIZE))
        kv_self.append(torch.zeros(NUM_HEADS, 1, CACHE_SIZE, HEAD_DIM))

    kv_cross = []
    for _ in range(NUM_BLOCKS):
        kv_cross.append(torch.randn(NUM_HEADS, 1, HEAD_DIM, audio_emb_len))
        kv_cross.append(torch.randn(NUM_HEADS, 1, audio_emb_len, HEAD_DIM))

    position_ids = torch.tensor([0], dtype=torch.int64)

    dummy = (input_ids, attention_mask, cross_attention_mask, *kv_self, *kv_cross, position_ids)

    # Input names
    inames = ["input_ids", "attention_mask", "cross_attention_mask"]
    for i in range(NUM_BLOCKS):
        inames += [f"k_cache_self_{i}_in", f"v_cache_self_{i}_in"]
    for i in range(NUM_BLOCKS):
        inames += [f"k_cache_cross_{i}", f"v_cache_cross_{i}"]
    inames.append("position_ids")

    # Output names: logits + 12 KV_self_out, optionally attention diagnostics.
    onames = ["logits"]
    for i in range(NUM_BLOCKS):
        onames += [f"k_cache_self_{i}_out", f"v_cache_self_{i}_out"]
    if output_mode != "logits_kv":
        onames.append("peak_attn")
        onames.append("cross_attn")

    # Create model (NO set_nstep_mode — default sliding window)
    model = OneStepDecoder(
        qc_decoder, audio_emb_len, alignment_heads=alignment_heads,
        output_mode=output_mode)
    model.eval()

    with torch.no_grad():
        torch.onnx.export(
            model, dummy, str(onnx_path),
            input_names=inames, output_names=onames,
            opset_version=13, do_constant_folding=True,
        )

    # Embed external data
    ext_data = Path(str(onnx_path) + ".data")
    if ext_data.exists():
        onnx_model = onnx.load(str(onnx_path), load_external_data=True)
        onnx_model.graph.name = f"{unique_decoder_id}_graph"
        onnx.save(onnx_model, str(onnx_path))
        ext_data.unlink()
    else:
        onnx_model = onnx.load(str(onnx_path))
        onnx_model.graph.name = f"{unique_decoder_id}_graph"
        onnx.save(onnx_model, str(onnx_path))

    print(f"  Exported: {onnx_path}")
    return onnx_path


# ==============================================================================
# Phase 2: Local Compile (qairt-converter: ONNX -> DLC)
# ==============================================================================

def compile_onnx_to_dlc(onnx_path):
    """Convert ONNX to DLC using local QAIRT 2.37 converter."""
    if not CONVERTER.exists():
        print(f"  ERROR: qairt-converter not found at {CONVERTER}")
        sys.exit(1)

    dlc_path = DLC_DIR / f"{onnx_path.stem}.dlc"
    if dlc_path.exists():
        print(f"  DLC exists, skip: {dlc_path}")
        return dlc_path

    env = os.environ.copy()
    env["PATH"] = str(Path(sys.executable).parent) + ":" + env.get("PATH", "")
    env["PYTHONPATH"] = str(SDK / "lib/python") + ":" + env.get("PYTHONPATH", "")
    ld = str(SDK / "lib/x86_64-linux-clang")
    local_libs = OUTPUT_DIR / "_local_libs"
    if local_libs.exists():
        ld = str(local_libs) + ":" + ld
    env["LD_LIBRARY_PATH"] = ld + ":" + env.get("LD_LIBRARY_PATH", "")

    cmd = [
        str(CONVERTER),
        "--input_network", str(onnx_path),
        "--output_path", str(dlc_path),
        "--float_bitwidth", "16",
    ]

    print(f"  Converting ONNX -> DLC...")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[-2000:]}")
        sys.exit(1)

    print(f"  OK: {dlc_path}")
    return dlc_path


# ==============================================================================
# Phase 3: Local Link (qnn-context-binary-generator)
# ==============================================================================

def generate_context_binary(dlc_path, sec_tag):
    """Link DLC into context binary."""
    if not CONTEXT_GEN.exists():
        print(f"  ERROR: qnn-context-binary-generator not found at {CONTEXT_GEN}")
        sys.exit(1)

    bin_name = f"whisper_decoder_1step_{sec_tag}_{TARGET_PROFILE.artifact_suffix}"
    output_bin = OUTPUT_DIR / bin_name

    htp_config = WORK / "htp_config.json"
    backend_config = WORK / "backend_config.json"

    with open(htp_config, "w") as f:
        json.dump({
            "graphs": [{"graph_names": ["model"], "vtcm_mb": 0, "O": 3}],
            "devices": [{"soc_model": SOC_MODEL, "dsp_arch": DSP_ARCH}],
        }, f, indent=2)

    with open(backend_config, "w") as f:
        json.dump({
            "backend_extensions": {
                "shared_library_path": str(HTP_EXT_LIB),
                "config_file_path": str(htp_config),
            }
        }, f, indent=2)

    env = os.environ.copy()
    ld_paths = [str(SDK / "lib/x86_64-linux-clang")]
    local_libs = OUTPUT_DIR / "_local_libs"
    if local_libs.exists():
        ld_paths.insert(0, str(local_libs))
    env["LD_LIBRARY_PATH"] = ":".join(ld_paths)

    cmd = [
        str(CONTEXT_GEN),
        "--dlc_path", str(dlc_path),
        "--backend", str(HTP_LIB),
        "--binary_file", bin_name,
        "--output_dir", str(OUTPUT_DIR),
        "--config_file", str(backend_config),
        "--input_output_tensor_mem_type", "memhandle",
        "--log_level", "info",
    ]

    print(f"  Linking DLC -> context binary...")
    print(f"  SOC: {SOC_MODEL} ({DSP_ARCH})")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.returncode != 0 and result.stderr.strip():
        print(f"  STDERR: {result.stderr[-2000:]}")

    actual_bin = Path(str(output_bin) + ".bin")
    if actual_bin.exists():
        size_mb = actual_bin.stat().st_size / (1024 * 1024)
        print(f"\n  SUCCESS: {actual_bin} ({size_mb:.1f} MB)")
        return actual_bin

    print(f"  FAILED: {actual_bin} not found")
    return None


# ==============================================================================
# Phase 4: Generate EPContext ONNX wrapper
# ==============================================================================

def _make_info(name, shape, dtype=TensorProto.FLOAT16):
    return helper.make_tensor_value_info(name, dtype, shape)


def generate_wrapper(bin_path, audio_emb_len, sec_tag, n_alignment_heads,
                     output_mode="debug"):
    """Generate EPContext ONNX wrapper for 1-step decoder."""
    out_path = WRAPPER_DIR / f"decoder_1step_{sec_tag}.onnx"

    # Inputs
    inputs = [
        _make_info("input_ids", [1, 1], TensorProto.INT32),
        _make_info("attention_mask", [1, 1, 1, MAX_DECODE]),
        _make_info("cross_attention_mask", [1, 1, 1, audio_emb_len]),
    ]
    for i in range(NUM_BLOCKS):
        inputs.append(_make_info(f"k_cache_self_{i}_in", [NUM_HEADS, 1, HEAD_DIM, CACHE_SIZE]))
        inputs.append(_make_info(f"v_cache_self_{i}_in", [NUM_HEADS, 1, CACHE_SIZE, HEAD_DIM]))
    for i in range(NUM_BLOCKS):
        inputs.append(_make_info(f"k_cache_cross_{i}", [NUM_HEADS, 1, HEAD_DIM, audio_emb_len]))
        inputs.append(_make_info(f"v_cache_cross_{i}", [NUM_HEADS, 1, audio_emb_len, HEAD_DIM]))
    inputs.append(_make_info("position_ids", [1], TensorProto.INT32))

    # Outputs: logits + 12 KV_self_out, optionally attention diagnostics.
    outputs = [
        _make_info("logits", [1, VOCAB_SIZE, 1, 1]),
    ]
    for i in range(NUM_BLOCKS):
        outputs.append(_make_info(f"k_cache_self_{i}_out", [NUM_HEADS, 1, HEAD_DIM, CACHE_SIZE]))
        outputs.append(_make_info(f"v_cache_self_{i}_out", [NUM_HEADS, 1, CACHE_SIZE, HEAD_DIM]))
    if output_mode != "logits_kv":
        outputs.append(_make_info("peak_attn", [1, 1, 1, audio_emb_len]))
        outputs.append(_make_info("cross_attn", [1, n_alignment_heads, 1, audio_emb_len]))

    input_names = [inp.name for inp in inputs]
    output_names = [out.name for out in outputs]

    # Keep EPContext lookup flat-directory friendly. The deploy layout places
    # wrapper and context binary side-by-side, and recent ORT/QNN rejects ".."
    # in ep_cache_context.
    rel_bin = bin_path.name
    unique_decoder_id = f"decoder_1step_{sec_tag}"
    node = helper.make_node(
        "EPContext", name=unique_decoder_id,
        inputs=input_names, outputs=output_names,
        ep_cache_context=rel_bin, embed_mode=0,
        source="Qnn", domain="com.microsoft",
    )
    graph = helper.make_graph([node], f"{unique_decoder_id}_graph", inputs, outputs)
    model = helper.make_model(graph, opset_imports=[
        helper.make_opsetid("", 13), helper.make_opsetid("com.microsoft", 1)
    ])
    onnx.save(model, str(out_path))
    print(f"  {out_path.name}")


# ==============================================================================
# Main
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compile 1-step Whisper decoder with cross-attn output (per bucket)")
    parser.add_argument("--audio-sec", type=float, default=2.0,
                        help="Audio chunk length in seconds (default: 2.0)")
    parser.add_argument("--build-dir", type=str, default=None,
                        help="Build directory (default: auto by model/target)")
    parser.add_argument("--model", type=str, default="base",
                        choices=["tiny.en", "tiny", "base.en", "base", "small.en", "small",
                                 "medium.en", "medium", "large", "large-v1", "large-v2", "large-v3"],
                        help="Whisper model variant")
    parser.add_argument("--target", type=str, default="s25_xplus_compat",
                        choices=["xplus", "s25", "s25_xplus_compat"],
                        help="Target device profile")
    parser.add_argument("--alignment-heads", type=str, default=None,
                        help="Semicolon-separated layer,head pairs for DTW alignment")
    parser.add_argument("--output-mode", type=str, default="debug",
                        choices=["debug", "logits_kv"],
                        help="debug: logits + KV + attention outputs. "
                             "logits_kv: lean logits + final KV outputs.")
    return parser.parse_args()


def main():
    global MODEL_PROFILE, TARGET_PROFILE, HF_VERSION
    global NUM_BLOCKS, ATTENTION_DIM, NUM_HEADS, HEAD_DIM, VOCAB_SIZE
    global DEFAULT_ALIGNMENT_HEADS, WORK, ONNX_DIR, DLC_DIR, OUTPUT_DIR, WRAPPER_DIR
    global DEVICE_NAME, SOC_MODEL, DSP_ARCH

    args = parse_args()
    MODEL_PROFILE = resolve_whisper_model_profile(args.model)
    TARGET_PROFILE = resolve_target_profile(args.target)
    HF_VERSION = resolve_hf_model_source(MODEL_PROFILE)
    NUM_BLOCKS = MODEL_PROFILE.num_blocks
    ATTENTION_DIM = MODEL_PROFILE.attention_dim
    NUM_HEADS = MODEL_PROFILE.num_heads
    HEAD_DIM = MODEL_PROFILE.head_dim
    VOCAB_SIZE = MODEL_PROFILE.vocab_size
    DEFAULT_ALIGNMENT_HEADS = MODEL_PROFILE.alignment_heads
    DEVICE_NAME = TARGET_PROFILE.device_name
    SOC_MODEL = TARGET_PROFILE.soc_model
    DSP_ARCH = TARGET_PROFILE.dsp_arch

    audio_sec = args.audio_sec
    audio_emb_len = profile_bucket_audio_emb_len(audio_sec, TARGET_PROFILE)
    sec_tag = f"{audio_sec:g}s"
    alignment_heads = parse_alignment_heads(args.alignment_heads or DEFAULT_ALIGNMENT_HEADS)

    WORK = Path(args.build_dir or default_build_dir("decoder_1step", MODEL_PROFILE, TARGET_PROFILE))
    ONNX_DIR = WORK / "onnx"
    DLC_DIR = WORK / "dlc"
    OUTPUT_DIR = WORK / "output"
    WRAPPER_DIR = WORK / "wrappers"

    for d in [ONNX_DIR, DLC_DIR, OUTPUT_DIR, WRAPPER_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Verify numpy version
    import numpy as np
    if int(np.__version__.split('.')[0]) >= 2:
        print(f"ERROR: numpy {np.__version__} detected. QAIRT 2.37 requires numpy<2.0.")
        sys.exit(1)

    print("=" * 60)
    print("1-step Whisper Decoder (fixed KV cache, cross-attn output)")
    print(f"  Model             : {MODEL_PROFILE.hf_repo}")
    print(f"  Audio length      : {audio_sec}s (AUDIO_EMB_LEN={audio_emb_len})")
    print(f"  Alignment heads   : {len(alignment_heads)}")
    print(f"  Output mode       : {args.output_mode}")
    print(f"  Max decode        : {MAX_DECODE}")
    print(f"  KV self cache     : {CACHE_SIZE} (fixed, sliding window)")
    print(f"  Device            : {DEVICE_NAME}")
    print(f"  Chipset           : Hexagon {DSP_ARCH} (soc_model={SOC_MODEL})")
    print(f"  QAIRT SDK         : {SDK}")
    print(f"  numpy             : {np.__version__}")
    print(f"  Build dir         : {WORK}")
    print("=" * 60)

    # -- 1. Load model (NO set_nstep_mode — keep sliding window KV cache) --
    print(f"\n1. Loading {MODEL_PROFILE.hf_repo} (default sliding window mode)...")
    model = HfWhisper.load_whisper_model(HF_VERSION)
    decoder_module = model.get_decoder()
    # NOT calling set_nstep_mode() — default SHAAttention uses sliding window

    # -- 2. Export ONNX --
    print(f"\n2. Exporting 1-step decoder ONNX (emb={audio_emb_len})...")
    onnx_path = export_onnx(decoder_module, audio_emb_len, alignment_heads, sec_tag,
                            output_mode=args.output_mode)

    # -- 3. Local compile (ONNX -> DLC) --
    print(f"\n3. Compiling ONNX -> DLC (local QAIRT 2.37)...")
    dlc_path = compile_onnx_to_dlc(onnx_path)

    # -- 4. Local link (DLC -> context binary) --
    print(f"\n4. Linking DLC -> context binary...")
    bin_path = generate_context_binary(dlc_path, sec_tag)

    if bin_path is None:
        print("\nLINK FAILED. Exiting.")
        sys.exit(1)

    # -- 5. Generate EPContext ONNX wrapper --
    print(f"\n5. Generating EPContext ONNX wrapper...")
    generate_wrapper(bin_path, audio_emb_len, sec_tag, len(alignment_heads),
                     output_mode=args.output_mode)

    # -- Summary --
    print("\n" + "=" * 60)
    print(f"DONE: 1-step decoder (audio={audio_sec}s) compiled!")
    print(f"  Audio          : {audio_sec}s (AUDIO_EMB_LEN={audio_emb_len})")
    print(f"  KV self cache  : {CACHE_SIZE} (fixed)")
    print(f"  Max decode     : {MAX_DECODE}")
    print(f"  Context binary : {bin_path}")
    print(f"  ONNX wrapper   : {WRAPPER_DIR}/decoder_1step_{sec_tag}.onnx")
    print(f"\nOn-device inference:")
    print(f"  - Single decoder session (no graph swap)")
    print(f"  - Up to {MAX_DECODE} NPU calls per audio chunk")
    print(f"  - Each call: 1 token + 1 cross-attention weight")
    print(f"  - CPU performs degen detection (sort-based median + backward peak)")
    print("=" * 60)


if __name__ == "__main__":
    main()
