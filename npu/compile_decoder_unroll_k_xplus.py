"""
Unroll-K Whisper decoder with K cross-attention outputs for CPU-side degen detection.

Pipeline (all local, no QAI Hub dependency):
1. Export ceil(max_decode_len/K) ONNX files (one per chunk, K autoregressive steps each)
2. LOCAL compile: qairt-converter (QAIRT 2.37) -> DLC files
3. LOCAL link: qnn-context-binary-generator -> 1 weight-shared context binary
4. Generate EPContext ONNX wrappers

Each chunk fuses K decoder steps into a single NPU graph:
- K autoregressive decoding steps (argmax between steps)
- K peak-attention outputs (last decoder layer, all-head average)
- K alignment-attention outputs (selected heads for DTW carryover)
- Degeneration detection is performed on HOST CPU (not NPU)
  using true sort-based median filter (matching hjlee_degeneration original)
- Reduces host-NPU round-trips by K times

NPU output per chunk:
  [K logits] [NUM_BLOCKS*2 KV caches] [K peak_attn] [K cross_attn]
  - No degen flags, no attn_pos, no prev_cross_attn I/O
  - peak_attn : [1, 1, 1, audio_emb_len] per step (K total)
  - cross_attn: [1, n_aheads, 1, audio_emb_len] per step (K total)
  - Extra data: K × audio_emb_len × fp16 ≈ 1KB (negligible)

Audio length is configurable via --audio-sec:
  2s  -> AUDIO_EMB_LEN=100  (cross-KV dim, matching 2s encoder)
  30s -> AUDIO_EMB_LEN=1500 (cross-KV dim, matching 30s encoder)

Target device : Snapdragon X Plus 8-Core CRD (sc8340xp, Hexagon v73)
On-device SDK : QNN SDK 2.37 + onnxruntime-qnn 1.23.2
Compile SDK   : QAIRT 2.37.1.250807

IMPORTANT: Requires numpy<2.0 (e.g., numpy==1.26.4).
           QAIRT 2.37's C extension (libPyIrGraph.so, pybind11-based) is
           incompatible with numpy 2.x due to ABI changes.
"""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import onnx
from onnx import TensorProto, helper
from qai_hub_models.models._shared.hf_whisper.model import (
    MASK_NEG,
    HfWhisper,
)
from qai_hub_models.models._shared.hf_whisper.model_adaptation import set_nstep_mode
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

# -- Paths (all under current directory) ---------------------------------------
WORK = Path(__file__).parent / default_build_dir("unroll_k", MODEL_PROFILE, TARGET_PROFILE)
ONNX_DIR = WORK / "onnx"
DLC_DIR = WORK / "dlc"
OUTPUT_DIR = WORK / "output"
WRAPPER_DIR = WORK / "wrappers"

# -- Target device -------------------------------------------------------------
DEVICE_NAME = TARGET_PROFILE.device_name

# -- QAIRT 2.37 SDK paths (local compile + link) ------------------------------
# Snapdragon X Plus: chipset=sc8340xp, hexagon=v73, soc_model=60
# Must match on-device QNN SDK version (2.37) to avoid runtime crashes.
_DEFAULT_SDK = str(Path(__file__).parent / "qairt" / "2.37.1.250807")
SDK = Path(os.environ.get("QNN_SDK_ROOT", _DEFAULT_SDK))
CONVERTER = SDK / "bin/x86_64-linux-clang/qairt-converter"
CONTEXT_GEN = SDK / "bin/x86_64-linux-clang/qnn-context-binary-generator"
HTP_LIB = SDK / "lib/x86_64-linux-clang/libQnnHtp.so"
HTP_EXT_LIB = SDK / "lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so"

SOC_MODEL = TARGET_PROFILE.soc_model
DSP_ARCH = TARGET_PROFILE.dsp_arch

# Whisper forced decode prefix (baked into chunk 0's graph as constants)
SOT = 50258
EN = 50259
TRANSCRIBE = 50359
NOTIMESTAMPS = 50363
EOT = 50257
FORCED_PREFIX = [SOT, EN, TRANSCRIBE, NOTIMESTAMPS]
DEFAULT_ALIGNMENT_HEADS = MODEL_PROFILE.alignment_heads


def configure_profiles(model_key="base", target_key="xplus", build_dir=None):
    global MODEL_PROFILE, TARGET_PROFILE, HF_VERSION
    global NUM_BLOCKS, ATTENTION_DIM, NUM_HEADS, HEAD_DIM, VOCAB_SIZE
    global WORK, ONNX_DIR, DLC_DIR, OUTPUT_DIR, WRAPPER_DIR
    global DEVICE_NAME, SOC_MODEL, DSP_ARCH, DEFAULT_ALIGNMENT_HEADS

    MODEL_PROFILE = resolve_whisper_model_profile(model_key)
    TARGET_PROFILE = resolve_target_profile(target_key)
    HF_VERSION = resolve_hf_model_source(MODEL_PROFILE)
    NUM_BLOCKS = MODEL_PROFILE.num_blocks
    ATTENTION_DIM = MODEL_PROFILE.attention_dim
    NUM_HEADS = MODEL_PROFILE.num_heads
    HEAD_DIM = MODEL_PROFILE.head_dim
    VOCAB_SIZE = MODEL_PROFILE.vocab_size
    DEVICE_NAME = TARGET_PROFILE.device_name
    SOC_MODEL = TARGET_PROFILE.soc_model
    DSP_ARCH = TARGET_PROFILE.dsp_arch
    DEFAULT_ALIGNMENT_HEADS = MODEL_PROFILE.alignment_heads

    WORK = Path(build_dir or default_build_dir("unroll_k", MODEL_PROFILE, TARGET_PROFILE))
    ONNX_DIR = WORK / "onnx"
    DLC_DIR = WORK / "dlc"
    OUTPUT_DIR = WORK / "output"
    WRAPPER_DIR = WORK / "wrappers"


# ==============================================================================
# UnrolledKChunkDecoder: K autoregressive steps + K cross-attn outputs
# ==============================================================================

class UnrolledKChunkDecoder(nn.Module):
    """
    K autoregressive decoder steps fused into a single NPU graph.
    Outputs:
      - K peak-attention tensors for backward peak detection
      - K alignment-head tensors for DTW carryover alignment

    No degen logic on NPU — no median filter, no smoothing, no skip masks.
    The host CPU performs true sort-based median + backward peak detection.

    For each chunk at offset s (= chunk_idx * K):
    - Step k uses mask of size s+k+2, position s+k
    - KV cache grows from s+1 to s+K+1
    - Cross-attention weights from last layer output per step

    Two flavours:
    - **Legacy / FORCED_PREFIX** (prefill_len=None): chunk_0 takes input_ids
      shape [1,1] (just SOT). Internal forced-prefix Python branch bakes
      EN/TRANSCRIBE/NOTIMESTAMPS as graph constants for steps 1..3.
      Limited to English transcription.
    - **Prefill chain** (prefill_len=L, only used for chunk_0): chunk_0 takes
      input_ids shape [1, L] from the caller. Steps 0..L-1 consume the L
      prefill tokens directly (positions chunk_offset..chunk_offset+L-1).
      Steps L..K-1 autoregress with internal argmax. K must be >= L. The
      caller can therefore pass any prompt — multilingual SOT seq, prompt
      prefill from previous chunk, etc.
    """

    def __init__(self, qc_decoder, chunk_offset, K, num_blocks, audio_emb_len,
                 mask_neg=MASK_NEG, alignment_heads=None, prefill_len=None,
                 output_mode="debug"):
        super().__init__()
        self.embed_tokens = qc_decoder.embed_tokens
        self.embed_positions = qc_decoder.embed_positions
        self.layers = qc_decoder.layers
        self.layer_norm = qc_decoder.layer_norm
        self.proj_out = qc_decoder.proj_out

        self.K = K
        self.chunk_offset = chunk_offset
        self.num_blocks = num_blocks
        self.audio_emb_len = audio_emb_len
        # When set, this chunk is the prefill chunk_0: input_ids has shape
        # [1, prefill_len] and the first prefill_len internal steps consume
        # caller-provided tokens instead of argmax/forced prefix.
        self.prefill_len = prefill_len
        self.output_mode = output_mode
        if prefill_len is not None:
            assert K >= prefill_len, (
                f"prefill_len={prefill_len} must be <= K={K} so all prefill "
                f"tokens fit in chunk_0's autoregressive loop")

        # Two distinct attention signals are required at runtime:
        #   - peak_attn : last decoder layer, all-head average
        #   - cross_attn: selected alignment heads across multiple layers
        self.alignment_heads = alignment_heads
        self.peak_attn_layer = len(self.layers) - 1
        self.align_attn_layers = set()
        if output_mode == "debug":
            self.layers[self.peak_attn_layer].encoder_attn.return_attn_weights = True

        if output_mode == "debug" and alignment_heads:
            layer_heads = {}
            for layer_idx, head_idx in alignment_heads:
                layer_heads.setdefault(layer_idx, []).append(head_idx)
            self._layer_heads = layer_heads
            for layer_idx, heads in layer_heads.items():
                self.layers[layer_idx].encoder_attn.selected_heads = heads
                self.align_attn_layers.add(layer_idx)
        else:
            self._layer_heads = None

        # Match ours_streaming's host-side policy for N/K mode: when a chunk
        # needs to pick the next internal token, do not let <|endoftext|>
        # short-circuit the remaining unrolled steps. The host suppresses EOT
        # in carry-over modes before choosing the next token; without the same
        # bias here, buffered logits are generated from a different history.
        argmax_bias = torch.zeros(VOCAB_SIZE, dtype=torch.float32)
        argmax_bias[EOT] = -1.0e9
        self.register_buffer("internal_argmax_bias", argmax_bias)

        # Static attention masks and position IDs (graph constants per step)
        for k in range(K):
            step = chunk_offset + k
            mask = torch.full((1, 1, 1, step + 2), mask_neg, dtype=torch.float32)
            mask[:, :, :, 1:] = 0.0
            self.register_buffer(f"mask_{k}", mask)
            self.register_buffer(f"pos_{k}", torch.tensor([step], dtype=torch.int64))

    def _decoder_step(self, input_ids, mask, kv_self, kv_cross, pos, cross_attention_mask):
        """
        Single decoder step.

        Returns:
            logits: [1, VOCAB_SIZE, 1, 1]
            updated_kv_self: list of (k, v) tuples
            peak_attn_weights: [1, 1, 1, audio_emb_len] (last-layer all-heads avg)
            align_attn_weights: [1, n_aheads, 1, audio_emb_len] (alignment heads)
        """
        input_embeds = self.embed_tokens(input_ids.to(torch.int64))
        positions = self.embed_positions(input_ids, position_ids=pos)
        hidden_states = input_embeds.unsqueeze(0) + positions

        next_cache = []
        layer_attn_weights = []  # collect selected alignment heads
        peak_attn_weights = None
        for idx, layer in enumerate(self.layers):
            out = layer(
                hidden_states,
                attention_mask=mask,
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

        # Combine selected alignment heads from all alignment layers.
        if len(layer_attn_weights) > 1:
            align_attn_weights = torch.cat(layer_attn_weights, dim=1)
        elif len(layer_attn_weights) == 1:
            align_attn_weights = layer_attn_weights[0]
        else:
            align_attn_weights = None

        return logits, next_cache, peak_attn_weights, align_attn_weights

    def forward(self, input_ids_0, cross_attention_mask, *kv_args):
        """
        Forward pass: K autoregressive steps, output K cross-attention weights.

        Args:
            input_ids_0: INT32
                - Legacy mode: [1, 1] (single first token)
                - Prefill mode (prefill_len=L): [1, L] (L caller-provided tokens)
            cross_attention_mask: [1, 1, 1, audio_emb_len]
                Additive source mask for decoder cross-attention. Runtime sets
                padded encoder tail to a large negative value so attention
                renormalizes over real audio frames only.
            *kv_args: flat list of KV cache tensors
                [k_self_0, v_self_0, ..., k_self_{N-1}, v_self_{N-1},
                 k_cross_0, v_cross_0, ..., k_cross_{N-1}, v_cross_{N-1}]

        Returns:
            tuple of:
                logits_0..K-1: [1, VOCAB_SIZE, 1, 1] each
                k/v_cache_self_0..{N-1}_out: final KV caches
                peak_attn_0..K-1: [1, 1, 1, audio_emb_len] each
                cross_attn_0..K-1: [1, n_aheads, 1, audio_emb_len] each
        """
        nb = self.num_blocks
        kv_self = [(kv_args[i], kv_args[i + 1]) for i in range(0, nb * 2, 2)]
        cs = nb * 2
        kv_cross = [(kv_args[cs + i], kv_args[cs + i + 1]) for i in range(0, nb * 2, 2)]

        all_logits = []
        all_tokens = []
        all_peak_attn = []
        all_cross_attn = []
        # First step input. Legacy mode: input_ids_0 already shape [1,1], use
        # as-is so the legacy graph is byte-identical to the previously-compiled
        # bins. Prefill mode: slice the first column from the [1, L] tensor.
        if self.prefill_len is not None:
            current_ids = input_ids_0[:, 0:1]
        else:
            current_ids = input_ids_0

        for k in range(self.K):
            mask = getattr(self, f"mask_{k}")
            pos = getattr(self, f"pos_{k}")

            logits, kv_self, peak_attn, align_attn = self._decoder_step(
                current_ids, mask, kv_self, kv_cross, pos, cross_attention_mask)
            all_logits.append(logits)
            all_peak_attn.append(peak_attn)
            all_cross_attn.append(align_attn if align_attn is not None else peak_attn)

            if self.prefill_len is not None:
                # Prefill chain: feed caller-provided tokens for the first
                # prefill_len positions, then autoregress with argmax.
                next_k = k + 1
                if next_k < self.prefill_len:
                    # Trace-time slice — produces a Slice op in the graph.
                    selected_token = input_ids_0[:, next_k:next_k + 1].to(torch.int32)
                else:
                    next_token = (logits[:, :, 0, 0] + self.internal_argmax_bias) \
                        .argmax(dim=1, keepdim=True)
                    selected_token = next_token.to(torch.int32)
            else:
                # Legacy chain: forced English SOT prefix as constants.
                step = self.chunk_offset + k
                if step + 1 < len(FORCED_PREFIX):
                    selected_token = torch.tensor(
                        [[FORCED_PREFIX[step + 1]]],
                        dtype=torch.int32,
                        device=input_ids_0.device)
                else:
                    next_token = (logits[:, :, 0, 0] + self.internal_argmax_bias) \
                        .argmax(dim=1, keepdim=True)
                    selected_token = next_token.to(torch.int32)

            all_tokens.append(selected_token)

            # Intermediate token for next step (except last step)
            if k < self.K - 1:
                current_ids = selected_token

        flat_kv = []
        for k_c, v_c in kv_self:
            flat_kv.extend([k_c, v_c])

        if self.output_mode == "token_only":
            # Paper unroll30exp style: only token_seq leaves the graph.
            # Valid only for single-chunk (K30) since chunk chain needs KV out.
            token_seq = torch.cat(all_tokens, dim=1).to(torch.int32)
            return (token_seq,)
        if self.output_mode == "token_seq_kv":
            token_seq = torch.cat(all_tokens, dim=1).to(torch.int32)
            return tuple([token_seq] + flat_kv)
        if self.output_mode == "logits_kv":
            return tuple(all_logits + flat_kv)

        return tuple(all_logits + flat_kv + all_peak_attn + all_cross_attn)


# ==============================================================================
# Phase 1: ONNX Export
# ==============================================================================

def chunk_layout(chunk_idx, K, prefill_len=None):
    """
    Compute (chunk_offset, K_internal, cache_in, cache_out) for one chunk.

    Legacy chain (prefill_len=None):
        K_chunk_0 = K. chunk_offset = chunk_idx * K. K_internal = K.

    Prefill chain (prefill_len=L):
        K_chunk_0 = L + 1  (L caller-provided positions + 1 autoregress step).
        chunk_0:  K_internal = K_chunk_0,    chunk_offset = 0
        chunk_1+: K_internal = K (CLI arg),  chunk_offset = K_chunk_0 + (i-1)*K

    Side benefit: prefill_len = K - 1 (e.g. prefill_len=4 with K=5) makes
    K_chunk_0 == K, which means chunk_1+ have IDENTICAL cache shapes/positions
    to the legacy chain — those chunk bins can be reused without recompile.
    """
    if prefill_len is None:
        K_chunk_0 = K
    else:
        K_chunk_0 = prefill_len + 1

    if chunk_idx == 0:
        s = 0
        K_int = K_chunk_0
    else:
        s = K_chunk_0 + (chunk_idx - 1) * K
        K_int = K

    cache_in = s + 1
    cache_out = s + K_int + 1
    return s, K_int, cache_in, cache_out


def export_onnx(qc_decoder, chunk_idx, K, audio_emb_len, alignment_heads=None,
                prefill_len=None, output_mode="debug"):
    """Export a single chunk's unrolled decoder to ONNX with fully static shapes."""
    s, K_int, cache_in, cache_out = chunk_layout(chunk_idx, K, prefill_len)
    is_prefill_chunk0 = (prefill_len is not None and chunk_idx == 0)
    onnx_path = ONNX_DIR / f"decoder_chunk_{chunk_idx}.onnx"

    if onnx_path.exists():
        print(f"  Chunk {chunk_idx}: exists, skip")
        return onnx_path

    # Dummy input_ids: prefill chunk_0 takes [1, prefill_len], otherwise [1, 1]
    if is_prefill_chunk0:
        input_ids = torch.zeros((1, prefill_len), dtype=torch.int32)
        input_ids[0, 0] = 50258  # SOT placeholder
    else:
        input_ids = torch.tensor([[50258]], dtype=torch.int32)

    kv_self = []
    for _ in range(NUM_BLOCKS):
        kv_self.append(torch.zeros(NUM_HEADS, 1, HEAD_DIM, cache_in))
        kv_self.append(torch.zeros(NUM_HEADS, 1, cache_in, HEAD_DIM))

    kv_cross = []
    for _ in range(NUM_BLOCKS):
        kv_cross.append(torch.randn(NUM_HEADS, 1, HEAD_DIM, audio_emb_len))
        kv_cross.append(torch.randn(NUM_HEADS, 1, audio_emb_len, HEAD_DIM))

    cross_attention_mask = torch.zeros((1, 1, 1, audio_emb_len), dtype=torch.float32)
    dummy = (input_ids, cross_attention_mask, *kv_self, *kv_cross)

    # Input names (NO prev_cross_attn)
    inames = ["input_ids", "cross_attention_mask"]
    for i in range(NUM_BLOCKS):
        inames += [f"k_cache_self_{i}_in", f"v_cache_self_{i}_in"]
    for i in range(NUM_BLOCKS):
        inames += [f"k_cache_cross_{i}", f"v_cache_cross_{i}"]

    if output_mode == "token_only":
        onames = ["token_seq"]
    elif output_mode == "token_seq_kv":
        onames = ["token_seq"]
    else:
        onames = [f"logits_{k}" for k in range(K_int)]
    if output_mode != "token_only":
        for i in range(NUM_BLOCKS):
            onames += [f"k_cache_self_{i}_out", f"v_cache_self_{i}_out"]
    if output_mode == "debug":
        onames += [f"peak_attn_{k}" for k in range(K_int)]
        onames += [f"cross_attn_{k}" for k in range(K_int)]

    # Create chunk model. prefill_len is only meaningful for chunk_0; chunk_1+
    # use the legacy autoregressive path (single token in, K argmax steps).
    chunk_model = UnrolledKChunkDecoder(
        qc_decoder, chunk_offset=s, K=K_int, num_blocks=NUM_BLOCKS,
        audio_emb_len=audio_emb_len, alignment_heads=alignment_heads,
        prefill_len=(prefill_len if is_prefill_chunk0 else None),
        output_mode=output_mode)
    chunk_model.eval()

    with torch.no_grad():
        torch.onnx.export(
            chunk_model, dummy, str(onnx_path),
            input_names=inames, output_names=onames,
            opset_version=13, do_constant_folding=True,
        )

    # Re-save with weights embedded (no external .data file)
    ext_data = Path(str(onnx_path) + ".data")
    if ext_data.exists():
        onnx_model = onnx.load(str(onnx_path), load_external_data=True)
        onnx.save(onnx_model, str(onnx_path))
        ext_data.unlink()

    tag = f", prefill_len={prefill_len}" if is_prefill_chunk0 else ""
    print(f"  Chunk {chunk_idx}: exported (offset={s}, K={K_int}, "
          f"cache_in={cache_in}, cache_out={cache_out}{tag})")
    return onnx_path


# ==============================================================================
# Phase 2: Local Compile (qairt-converter: ONNX -> DLC)
# ==============================================================================

def compile_onnx_to_dlc(onnx_paths):
    """Convert ONNX files to DLC using local QAIRT 2.37 converter."""
    if not CONVERTER.exists():
        print(f"  ERROR: qairt-converter not found at {CONVERTER}")
        print(f"  Set QNN_SDK_ROOT or check QAIRT SDK installation.")
        sys.exit(1)

    env = os.environ.copy()
    env["PATH"] = str(Path(sys.executable).parent) + ":" + env.get("PATH", "")
    env["PYTHONPATH"] = str(SDK / "lib/python") + ":" + env.get("PYTHONPATH", "")
    local_libs = OUTPUT_DIR / "_local_libs"
    ld = str(SDK / "lib/x86_64-linux-clang")
    if local_libs.exists():
        ld = str(local_libs) + ":" + ld
    env["LD_LIBRARY_PATH"] = ld + ":" + env.get("LD_LIBRARY_PATH", "")

    dlc_paths = {}
    failed = []
    for chunk_idx, onnx_path in sorted(onnx_paths.items()):
        dlc_path = DLC_DIR / f"decoder_chunk_{chunk_idx}.dlc"
        if dlc_path.exists():
            print(f"  Chunk {chunk_idx}: DLC exists, skip")
            dlc_paths[chunk_idx] = dlc_path
            continue

        cmd = [
            str(CONVERTER),
            "--input_network", str(onnx_path),
            "--output_path", str(dlc_path),
            "--float_bitwidth", "16",
        ]
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
        if result.returncode == 0 and dlc_path.exists():
            size_mb = dlc_path.stat().st_size / (1024 * 1024)
            print(f"  Chunk {chunk_idx}: OK ({size_mb:.0f} MB)")
            dlc_paths[chunk_idx] = dlc_path
        else:
            print(f"  Chunk {chunk_idx}: FAILED")
            if result.stderr:
                for line in result.stderr.strip().split('\n')[-5:]:
                    print(f"    {line}")
            failed.append(chunk_idx)

    if failed:
        print(f"\n  FAILED chunks: {failed}")
        sys.exit(1)

    return dlc_paths


# ==============================================================================
# Phase 3: Local Link (qnn-context-binary-generator)
# ==============================================================================

def generate_context_binary(dlc_paths, K, num_chunks, sec_tag):
    """
    Link all chunk DLCs into a single weight-shared context binary.
    No model count limit (unlike QAI Hub's 15-model cap).
    """
    if not CONTEXT_GEN.exists():
        print(f"  ERROR: qnn-context-binary-generator not found at {CONTEXT_GEN}")
        print(f"  Set QNN_SDK_ROOT env var to your QAIRT SDK install path.")
        sys.exit(1)

    dlc_list = ",".join(str(dlc_paths[c]) for c in sorted(dlc_paths.keys()))
    bin_name = f"whisper_decoder_unroll{K}_nchunk{num_chunks}_{sec_tag}_{TARGET_PROFILE.artifact_suffix}"
    output_bin = OUTPUT_DIR / bin_name

    # Backend config for Hexagon v73 (Snapdragon X Plus)
    htp_config = WORK / "htp_config.json"
    backend_config = WORK / "backend_config.json"

    with open(htp_config, "w") as f:
        json.dump({
            "context": {"weight_sharing_enabled": True},
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
        "--dlc_path", dlc_list,
        "--backend", str(HTP_LIB),
        "--binary_file", bin_name,
        "--output_dir", str(OUTPUT_DIR),
        "--config_file", str(backend_config),
        "--input_output_tensor_mem_type", "memhandle",
        "--log_level", "info",
    ]

    print(f"  Command: {' '.join(cmd[:6])} ...")
    print(f"  DLC count: {len(dlc_paths)}")
    print(f"  SOC: {SOC_MODEL} ({DSP_ARCH})")
    print(f"  Output: {output_bin}")

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[-2000:]}")
        return None

    actual_bin = Path(str(output_bin) + ".bin")
    if actual_bin.exists():
        size_mb = actual_bin.stat().st_size / (1024 * 1024)
        print(f"\n  SUCCESS: {actual_bin} ({size_mb:.1f} MB)")
        return actual_bin
    return None


# ==============================================================================
# Phase 4: Generate EPContext ONNX wrappers (precompiled_qnn_onnx)
# ==============================================================================

def _make_info(name, shape, dtype=TensorProto.FLOAT16):
    return helper.make_tensor_value_info(name, dtype, shape)


def gen_chunk_wrapper(chunk_idx, K, bin_path, out_path, audio_emb_len, n_aheads=1,
                      prefill_len=None, output_mode="debug"):
    """Generate EPContext ONNX wrapper for a single chunk."""
    s, K_int, cache_in, cache_out = chunk_layout(chunk_idx, K, prefill_len)
    is_prefill_chunk0 = (prefill_len is not None and chunk_idx == 0)
    ids_seq_len = prefill_len if is_prefill_chunk0 else 1

    # Inputs: NO prev_cross_attn. input_ids shape depends on prefill mode.
    inputs = [
        _make_info("input_ids", [1, ids_seq_len], TensorProto.INT32),
        _make_info("cross_attention_mask", [1, 1, 1, audio_emb_len]),
    ]
    for i in range(NUM_BLOCKS):
        inputs.append(_make_info(f"k_cache_self_{i}_in", [NUM_HEADS, 1, HEAD_DIM, cache_in]))
        inputs.append(_make_info(f"v_cache_self_{i}_in", [NUM_HEADS, 1, cache_in, HEAD_DIM]))
    for i in range(NUM_BLOCKS):
        inputs.append(_make_info(f"k_cache_cross_{i}", [NUM_HEADS, 1, HEAD_DIM, audio_emb_len]))
        inputs.append(_make_info(f"v_cache_cross_{i}", [NUM_HEADS, 1, audio_emb_len, HEAD_DIM]))

    # Outputs:
    #   token_only:   token_seq only (paper unroll30exp style, single-chunk only)
    #   token_seq_kv: lean token_seq + final KV
    #   logits_kv:   K logits + final KV
    #   debug:       K logits + final KV + K peak/cross attention tensors
    outputs = []
    if output_mode == "token_only":
        outputs.append(_make_info("token_seq", [1, K_int], TensorProto.INT32))
    elif output_mode == "token_seq_kv":
        outputs.append(_make_info("token_seq", [1, K_int], TensorProto.INT32))
    else:
        for k in range(K_int):
            outputs.append(_make_info(f"logits_{k}", [1, VOCAB_SIZE, 1, 1]))
    if output_mode != "token_only":
        for i in range(NUM_BLOCKS):
            outputs.append(_make_info(f"k_cache_self_{i}_out", [NUM_HEADS, 1, HEAD_DIM, cache_out]))
            outputs.append(_make_info(f"v_cache_self_{i}_out", [NUM_HEADS, 1, cache_out, HEAD_DIM]))
    if output_mode == "debug":
        for k in range(K_int):
            outputs.append(_make_info(f"peak_attn_{k}", [1, 1, 1, audio_emb_len]))
        for k in range(K_int):
            outputs.append(_make_info(f"cross_attn_{k}", [1, n_aheads, 1, audio_emb_len]))

    input_names = [inp.name for inp in inputs]
    output_names = [out.name for out in outputs]

    node = helper.make_node(
        "EPContext", name=f"decoder_chunk_{chunk_idx}",
        inputs=input_names, outputs=output_names,
        ep_cache_context=bin_path, embed_mode=0,
        source="Qnn", domain="com.microsoft",
    )
    graph = helper.make_graph([node], f"decoder_chunk_{chunk_idx}_graph", inputs, outputs)
    model = helper.make_model(graph, opset_imports=[
        helper.make_opsetid("", 13), helper.make_opsetid("com.microsoft", 1)
    ])
    onnx.save(model, str(out_path))


def generate_wrappers(bin_path, K, num_chunks, audio_emb_len, n_aheads=1,
                      prefill_len=None, output_mode="debug"):
    """Generate EPContext ONNX wrappers for all chunks."""
    # Keep wrapper/context lookup flat-directory friendly. The deploy layout
    # places wrappers and the context binary side-by-side, and recent ORT/QNN
    # rejects ".." in ep_cache_context.
    rel_bin = bin_path.name
    for chunk_idx in range(num_chunks):
        out_path = WRAPPER_DIR / f"decoder_chunk_{chunk_idx}.onnx"
        gen_chunk_wrapper(chunk_idx, K, rel_bin, out_path, audio_emb_len,
                          n_aheads=n_aheads, prefill_len=prefill_len,
                          output_mode=output_mode)
        if chunk_idx % 5 == 0:
            print(f"  decoder_chunk_{chunk_idx}.onnx ...")
    print(f"  ... decoder_chunk_{num_chunks - 1}.onnx")


# ==============================================================================
# Main
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compile unroll-K Whisper decoder with K cross-attn outputs")
    parser.add_argument("--max-decode-len", type=int, default=50,
                        help="Maximum decode length (default: 50)")
    parser.add_argument("--k", type=int, default=3,
                        help="Number of autoregressive steps per chunk (default: 3)")
    parser.add_argument("--audio-sec", type=float, default=2.0,
                        help="Audio chunk length in seconds (default: 2.0). "
                             "Must match the encoder's --audio-sec. "
                             "E.g., 2.0 -> AUDIO_EMB_LEN=100, 30.0 -> 1500")
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
                        help="Alignment heads as 'layer,head;layer,head;...' "
                             "If not set, uses the model preset from whisper.cpp.")
    parser.add_argument("--prefill-len", type=int, default=None,
                        help="Compile a PREFILL chain. chunk_0 takes input_ids of "
                             "shape [1, prefill_len] from caller (no FORCED_PREFIX). "
                             "Caller can pass any prompt sequence — multilingual SOT "
                             "or carryover prefill. Requires --k >= prefill_len. "
                             "Output bin/wrapper names are suffixed with _p<L>.")
    parser.add_argument("--output-mode", type=str, default="debug",
                        choices=["debug", "logits_kv", "token_seq_kv", "token_only"],
                        help="debug: logits + KV + attention outputs. "
                             "logits_kv: logits + final KV outputs. "
                             "token_seq_kv: lean token_seq + final KV outputs. "
                             "token_only: token_seq only (paper unroll30exp style, single-chunk only).")
    return parser.parse_args()


def main():
    args = parse_args()
    configure_profiles(args.model, args.target, args.build_dir)
    K = args.k
    max_decode_len = args.max_decode_len
    audio_sec = args.audio_sec
    prefill_len = args.prefill_len
    output_mode = args.output_mode

    # K_chunk_0 = K (legacy) or prefill_len + 1 (prefill chain).
    K_chunk_0 = K if prefill_len is None else (prefill_len + 1)

    # num_chunks: chunk_0 covers K_chunk_0 positions, chunk_i (i>=1) covers K positions.
    remaining = max(0, max_decode_len - K_chunk_0)
    num_chunks = 1 + math.ceil(remaining / K)

    # Compute audio embedding length from --audio-sec
    # Whisper mel: 100 frames/sec -> encoder conv2 stride=2 -> 50 frames/sec
    audio_emb_len = profile_bucket_audio_emb_len(audio_sec, TARGET_PROFILE)
    sec_tag = f"{audio_sec:g}s"
    if prefill_len is not None:
        sec_tag = f"{sec_tag}_p{prefill_len}"

    for d in [ONNX_DIR, DLC_DIR, OUTPUT_DIR, WRAPPER_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Verify numpy version (must be <2.0 for QAIRT 2.37 compatibility)
    import numpy as np
    if int(np.__version__.split('.')[0]) >= 2:
        print(f"ERROR: numpy {np.__version__} detected. QAIRT 2.37 requires numpy<2.0.")
        print(f"  Fix: pip install 'numpy<2.0'")
        sys.exit(1)

    print("=" * 60)
    print("Unroll-K Whisper Decoder (K cross-attn outputs for CPU-side degen)")
    print(f"  Model             : {MODEL_PROFILE.hf_repo}")
    print(f"  Audio length      : {audio_sec}s (AUDIO_EMB_LEN={audio_emb_len})")
    print(f"  Max decode        : {max_decode_len}")
    print(f"  K (steps/chunk)   : {K} (chunk_0 K = {K_chunk_0})")
    print(f"  Prefill length    : {prefill_len if prefill_len is not None else 'legacy/FORCED_PREFIX'}")
    print(f"  Output mode       : {output_mode}")
    print(f"  Chunks            : {num_chunks}")
    print(f"  Device            : {DEVICE_NAME}")
    print(f"  Chipset           : Hexagon {DSP_ARCH} (soc_model={SOC_MODEL})")
    print(f"  Runtime           : precompiled_qnn_onnx")
    print(f"  QAIRT SDK         : {SDK}")
    print(f"  numpy             : {np.__version__}")
    print(f"  Build dir         : {WORK}")
    print("=" * 60)

    # Parse alignment heads
    alignment_heads = None
    if args.alignment_heads:
        alignment_heads = []
        for pair in args.alignment_heads.split(";"):
            parts = pair.strip().split(",")
            if len(parts) == 2:
                alignment_heads.append((int(parts[0]), int(parts[1])))
        print(f"  Alignment heads   : {alignment_heads} ({len(alignment_heads)} heads)")
    else:
        alignment_heads = []
        for pair in DEFAULT_ALIGNMENT_HEADS.split(";"):
            parts = pair.strip().split(",")
            if len(parts) == 2:
                alignment_heads.append((int(parts[0]), int(parts[1])))
        print(f"  Alignment heads   : {alignment_heads} ({len(alignment_heads)} heads)")

    # -- 1. Load model & enable n-step mode --
    print(f"\n1. Loading {MODEL_PROFILE.hf_repo} + n-step mode...")
    model = HfWhisper.load_whisper_model(HF_VERSION)
    decoder_module = model.get_decoder()
    set_nstep_mode(decoder_module)

    # -- 2. Export ONNX --
    chain_tag = f"prefill_len={prefill_len}" if prefill_len is not None else "legacy/FORCED_PREFIX"
    print(f"\n2. Exporting {num_chunks} ONNX files (K={K}, audio={audio_sec}s, "
          f"emb={audio_emb_len}, chain={chain_tag})...")
    onnx_paths = {}
    for chunk_idx in range(num_chunks):
        onnx_paths[chunk_idx] = export_onnx(
            decoder_module, chunk_idx, K, audio_emb_len,
            alignment_heads=alignment_heads, prefill_len=prefill_len,
            output_mode=output_mode)

    # -- 3. Local compile (ONNX -> DLC) --
    print(f"\n3. Compiling {num_chunks} ONNX -> DLC (local QAIRT 2.37)...")
    dlc_paths = compile_onnx_to_dlc(onnx_paths)

    # -- 4. Local link (DLCs -> context binary) --
    print(f"\n4. Linking {num_chunks} DLCs -> context binary...")
    bin_path = generate_context_binary(dlc_paths, K, num_chunks, sec_tag)

    if bin_path is None:
        print("\nLINK FAILED. Exiting.")
        sys.exit(1)

    # -- 5. Generate EPContext ONNX wrappers --
    print(f"\n5. Generating EPContext ONNX wrappers...")
    n_aheads = len(alignment_heads) if alignment_heads else 1
    generate_wrappers(bin_path, K, num_chunks, audio_emb_len, n_aheads=n_aheads,
                      prefill_len=prefill_len, output_mode=output_mode)

    # -- Summary --
    print("\n" + "=" * 60)
    print(f"DONE: {num_chunks} chunks (K={K}, audio={audio_sec}s) compiled + locally linked!")
    print(f"  Audio          : {audio_sec}s (AUDIO_EMB_LEN={audio_emb_len})")
    print(f"  Context binary : {bin_path}")
    print(f"  ONNX wrappers  : {WRAPPER_DIR}/decoder_chunk_{{0..{num_chunks - 1}}}.onnx")
    print(f"\nDirectory layout:")
    print(f"  {WORK}/")
    print(f"    onnx/     -- {num_chunks} exported ONNX files")
    print(f"    dlc/      -- {num_chunks} compiled DLC files")
    print(f"    output/   -- weight-shared context binary (.bin)")
    print(f"    wrappers/ -- {num_chunks} EPContext ONNX wrappers")
    print(f"\nOn-device inference:")
    print(f"  - {num_chunks} NPU calls (was {max_decode_len} with 1-step pipeline)")
    if output_mode == "token_seq_kv":
        print(f"  - Each call: token_seq + final self-KV")
    elif output_mode == "logits_kv":
        print(f"  - Each call: {K} logits + final self-KV")
    else:
        print(f"  - Each call: {K} tokens + {K} cross-attention weights")
        print(f"  - CPU performs degen detection (sort-based median + backward peak)")
    print("=" * 60)


if __name__ == "__main__":
    main()
