"""
Compile the FULL N/K unrolled Whisper decoder design (N=30, K=5, num_chunks=6).

Two chain variants per bucket:
  normal  (no prev degen, no carryover):  chunk_0 K=4, chunk_1 K=6, chunks 2..5 K=5
  prefill (prev degen + 2-word carryover): chunk_0 K=6, chunk_1 K=4, chunks 2..5 K=5

Both chains converge at chunk_2 with cache_in=11. Chunks 2..5 are byte-identical
between the two chains and weight-shared in the same context binary.

Per-bucket compilation matrix:
  - 2s..10s:  8 + prefill variants
      0_normal, 1_normal,
      0_prefill_p1..p4, 1_prefill_p1..p4,
      2, 3, 4, 5

Normal chain is compiled for every bucket. After a 30s fallback pass, the next
round resumes with no prompt prefill, but the effective audio may still land in
a bucket > 2s. Restricting "normal" to 2s changes the decoder state machine
relative to GGML and forces synthetic prefill fallbacks at runtime.

Output per bucket:
  build/{bucket}s/output/whisper_decoder_nk_full_{bucket}s_xplus.bin
  build/{bucket}s/wrappers/decoder_{bucket}s_chunk_<name>.onnx
    where <name> ∈ {0_normal, 0_prefill, 1_normal, 1_prefill, 2, 3, 4, 5}

Usage:
  python compile_decoder_nk_full.py --bucket 2.0
  python compile_decoder_nk_full.py --bucket 4.0   # only prefill chain
  python compile_decoder_nk_full.py --all-buckets  # 2s..10s

Reuses helpers (UnrolledKChunkDecoder, compile_onnx_to_dlc, etc.) from the
existing compile_decoder_unroll_k_xplus.py module.
"""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import onnx
from onnx import TensorProto, helper

# Reuse the patched UnrolledKChunkDecoder + helpers from the legacy script
import compile_decoder_unroll_k_xplus as legacy
from qairt_paths import qairt_missing_message
from whisper_npu_profile import default_build_dir, resolve_target_profile, resolve_whisper_model_profile

# Forward useful constants
NUM_BLOCKS = legacy.NUM_BLOCKS
NUM_HEADS = legacy.NUM_HEADS
HEAD_DIM = legacy.HEAD_DIM
VOCAB_SIZE = legacy.VOCAB_SIZE
SOC_MODEL = legacy.SOC_MODEL
DSP_ARCH = legacy.DSP_ARCH
SDK = legacy.SDK
CONVERTER = legacy.CONVERTER
CONTEXT_GEN = legacy.CONTEXT_GEN
HTP_LIB = legacy.HTP_LIB
HTP_EXT_LIB = legacy.HTP_EXT_LIB

MODEL_PROFILE = resolve_whisper_model_profile("base")
TARGET_PROFILE = resolve_target_profile("xplus")
DEFAULT_ALIGNMENT_HEADS = MODEL_PROFILE.alignment_heads
MAX_INLINE_ONNX_BYTES = 2 * 1024**3 - 16 * 1024**2
DEFAULT_TOOL_TIMEOUT_SEC = 600
LARGE_TOOL_TIMEOUT_SEC = 3600


def sync_legacy_constants():
    global NUM_BLOCKS, NUM_HEADS, HEAD_DIM, VOCAB_SIZE, SOC_MODEL, DSP_ARCH
    NUM_BLOCKS = legacy.NUM_BLOCKS
    NUM_HEADS = legacy.NUM_HEADS
    HEAD_DIM = legacy.HEAD_DIM
    VOCAB_SIZE = legacy.VOCAB_SIZE
    SOC_MODEL = legacy.SOC_MODEL
    DSP_ARCH = legacy.DSP_ARCH


def current_tool_timeout_sec():
    return (
        LARGE_TOOL_TIMEOUT_SEC
        if legacy.MODEL_PROFILE.key in {"large", "large-v1", "large-v2", "large-v3"}
        else DEFAULT_TOOL_TIMEOUT_SEC
    )


def bucket_audio_emb_len(bucket_sec: float) -> int:
    raw = int(bucket_sec * 50)
    emb = ((raw + 7) // 8) * 8
    # 2s must match the unpadded encoder build (100) used on-device.
    if abs(bucket_sec - 2.0) < 1e-6:
        return 100
    return emb

# ─────────────────────────────────────────────────────────────────────────────
# Per-chain K tables (chunks 0..5)
# ─────────────────────────────────────────────────────────────────────────────
# normal:  chunk_0 takes [1, 4] (4 SOT only); chunk_1 K=6 autoregress
# prefill: chunk_0 takes [1, 4 + prompt_len]; chunk_1 K=10-(4+prompt_len)
# All chains converge: cache after chunk_1 = 11 for every supported prompt_len.
NUM_CHUNKS = 6
K_TABLE_NORMAL  = [4, 6, 5, 5, 5, 5]
SOT_PREFIX_LEN = K_TABLE_NORMAL[0]
SHARED_PREFIX_STEPS = K_TABLE_NORMAL[0] + K_TABLE_NORMAL[1]
PREFILL_PROMPT_TOKEN_LENS = [1, 2, 3, 4]
K_TABLE_PREFILL = [SOT_PREFIX_LEN + 2, SHARED_PREFIX_STEPS - (SOT_PREFIX_LEN + 2), 5, 5, 5, 5]


# ─────────────────────────────────────────────────────────────────────────────
# Per-chunk layout helpers
# ─────────────────────────────────────────────────────────────────────────────

def chunk_offset(K_table, chunk_idx):
    """Cumulative position offset before chunk_idx (= sum of preceding K_internals)."""
    return sum(K_table[:chunk_idx])


def cache_shapes(K_table, chunk_idx):
    """(K_internal, cache_in, cache_out) for one chunk under the given K table."""
    s = chunk_offset(K_table, chunk_idx)
    K = K_table[chunk_idx]
    return K, s + 1, s + K + 1


# Sanity-check that all supported prefill variants converge at chunk_2 with the same cache_in.
def _verify_chains_converge():
    for prompt_len in PREFILL_PROMPT_TOKEN_LENS:
        k0 = SOT_PREFIX_LEN + prompt_len
        k1 = SHARED_PREFIX_STEPS - k0
        assert k1 >= 1, f"prefill prompt_len={prompt_len} leaves invalid chunk_1 K={k1}"
        k_table_prefill = [k0, k1, *K_TABLE_NORMAL[2:]]
        for chunk_idx in range(2, NUM_CHUNKS):
            _, ci_n, co_n = cache_shapes(K_TABLE_NORMAL, chunk_idx)
            _, ci_p, co_p = cache_shapes(k_table_prefill, chunk_idx)
            assert (ci_n, co_n) == (ci_p, co_p), (
                f"Normal vs prefill(prompt_len={prompt_len}) cache mismatch at chunk_{chunk_idx}: "
                f"normal=({ci_n},{co_n}) prefill=({ci_p},{co_p})")
_verify_chains_converge()


# ─────────────────────────────────────────────────────────────────────────────
# Chunk descriptor
# ─────────────────────────────────────────────────────────────────────────────

class ChunkSpec:
    """One distinct decoder chunk graph to compile."""
    def __init__(self, name, K_table, chunk_idx, prefill_len=None, prompt_len=0):
        self.name = name              # e.g., "0_normal", "0_prefill", "2", ...
        self.K_table = K_table        # K table for the chain this chunk belongs to
        self.chunk_idx = chunk_idx
        self.prefill_len = prefill_len  # set on chunk_0 of each chain
        self.prompt_len = prompt_len
        K, ci, co = cache_shapes(K_table, chunk_idx)
        self.K_internal = K
        self.cache_in = ci
        self.cache_out = co
        self.chunk_offset = chunk_offset(K_table, chunk_idx)


def chunks_for_bucket(bucket_sec):
    """
    Return the list of ChunkSpec to compile for this bucket.
    Every bucket gets both:
      - normal chain chunk_0/1
      - prefill chain chunk_0/1 variants
      - shared chunks 2..5
    """
    specs = []

    # Prefill chain variants (always present).
    for prompt_len in PREFILL_PROMPT_TOKEN_LENS:
        k0 = SOT_PREFIX_LEN + prompt_len
        k1 = SHARED_PREFIX_STEPS - k0
        k_table_prefill = [k0, k1, *K_TABLE_NORMAL[2:]]
        specs.append(ChunkSpec(
            f"0_prefill_p{prompt_len}",
            k_table_prefill,
            0,
            prefill_len=k0,
            prompt_len=prompt_len,
        ))
        specs.append(ChunkSpec(
            f"1_prefill_p{prompt_len}",
            k_table_prefill,
            1,
            prompt_len=prompt_len,
        ))

    # Normal chain chunk_0 + chunk_1 (all buckets).
    specs.append(ChunkSpec("0_normal", K_TABLE_NORMAL, 0, prefill_len=SOT_PREFIX_LEN))
    specs.append(ChunkSpec("1_normal", K_TABLE_NORMAL, 1))

    # Shared chunks 2..5 (use normal K table, but they're identical to prefill)
    for c in range(2, NUM_CHUNKS):
        specs.append(ChunkSpec(str(c), K_TABLE_NORMAL, c))

    return specs


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: ONNX export per chunk
# ─────────────────────────────────────────────────────────────────────────────

def export_chunk_onnx(qc_decoder, spec, bucket_tag, audio_emb_len,
                      alignment_heads, onnx_dir):
    """Export one chunk to ONNX with fully-static shapes."""
    import torch
    unique_chunk_id = f"decoder_{bucket_tag}_chunk_{spec.name}"
    onnx_path = onnx_dir / f"{unique_chunk_id}.onnx"

    if onnx_path.exists():
        print(f"  Chunk {spec.name}: ONNX exists, skip")
        return onnx_path

    # Dummy input_ids: chunk_0 of each chain takes [1, prefill_len]; chunks 1+ take [1, 1]
    if spec.prefill_len is not None and spec.chunk_idx == 0:
        input_ids = torch.zeros((1, spec.prefill_len), dtype=torch.int32)
        input_ids[0, 0] = 50258  # SOT placeholder
    else:
        input_ids = torch.tensor([[50258]], dtype=torch.int32)

    kv_self = []
    for _ in range(NUM_BLOCKS):
        kv_self.append(torch.zeros(NUM_HEADS, 1, HEAD_DIM, spec.cache_in))
        kv_self.append(torch.zeros(NUM_HEADS, 1, spec.cache_in, HEAD_DIM))

    kv_cross = []
    for _ in range(NUM_BLOCKS):
        kv_cross.append(torch.randn(NUM_HEADS, 1, HEAD_DIM, audio_emb_len))
        kv_cross.append(torch.randn(NUM_HEADS, 1, audio_emb_len, HEAD_DIM))

    cross_attention_mask = torch.zeros((1, 1, 1, audio_emb_len), dtype=torch.float32)
    dummy = (input_ids, cross_attention_mask, *kv_self, *kv_cross)

    inames = ["input_ids", "cross_attention_mask"]
    for i in range(NUM_BLOCKS):
        inames += [f"k_cache_self_{i}_in", f"v_cache_self_{i}_in"]
    for i in range(NUM_BLOCKS):
        inames += [f"k_cache_cross_{i}", f"v_cache_cross_{i}"]

    onames = [f"logits_{k}" for k in range(spec.K_internal)]
    for i in range(NUM_BLOCKS):
        onames += [f"k_cache_self_{i}_out", f"v_cache_self_{i}_out"]
    onames += [f"peak_attn_{k}" for k in range(spec.K_internal)]
    onames += [f"cross_attn_{k}" for k in range(spec.K_internal)]

    chunk_model = legacy.UnrolledKChunkDecoder(
        qc_decoder,
        chunk_offset=spec.chunk_offset,
        K=spec.K_internal,
        num_blocks=NUM_BLOCKS,
        audio_emb_len=audio_emb_len,
        alignment_heads=alignment_heads,
        prefill_len=(spec.prefill_len if spec.chunk_idx == 0 else None),
    )
    chunk_model.eval()

    with torch.no_grad():
        # opset 18 directly (not 13). torch 2.9 always exports at >=18 then
        # tries to downgrade via the onnx C version_converter, which has a
        # thread-unsafe LayerNormalization adapter that crashes when several
        # compile_decoder_nk_full.py invocations run in parallel. Asking for 18
        # up front skips the converter call entirely → safe for parallel batch.
        # QAIRT 2.37's qairt-converter accepts opset 18 ONNX without complaint.
        torch.onnx.export(
            chunk_model, dummy, str(onnx_path),
            input_names=inames, output_names=onames,
            opset_version=18, do_constant_folding=True,
            external_data=True,
        )

    # Keep large exports in external-data form. Re-embedding >2GB models into
    # a single protobuf breaks large-v3 conversion.
    ext_data = Path(str(onnx_path) + ".data")
    use_external = ext_data.exists() or (
        onnx_path.exists() and onnx_path.stat().st_size >= MAX_INLINE_ONNX_BYTES
    )
    m = onnx.load(str(onnx_path), load_external_data=use_external)
    m.graph.name = f"{unique_chunk_id}_graph"
    if use_external:
        onnx.save_model(
            m,
            str(onnx_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=ext_data.name,
            size_threshold=1024,
        )
    else:
        onnx.save(m, str(onnx_path))

    print(f"  Chunk {spec.name}: exported "
          f"(K={spec.K_internal}, offset={spec.chunk_offset}, "
          f"cache_in={spec.cache_in}, cache_out={spec.cache_out}, "
          f"prefill_len={spec.prefill_len})")
    return onnx_path


def verify_onnx_outputs(onnx_path, expected_K, expected_n_aheads):
    """
    Mandatory check before linking. Asserts the exported ONNX has all expected
    outputs, especially peak_attn_* and cross_attn_*, before starting QAIRT conversion.
    """
    m = onnx.load(str(onnx_path))
    out_names = [o.name for o in m.graph.output]

    missing = []
    for k in range(expected_K):
        if f"logits_{k}" not in out_names:
            missing.append(f"logits_{k}")
    for k in range(expected_K):
        if f"peak_attn_{k}" not in out_names:
            missing.append(f"peak_attn_{k}")
    for k in range(expected_K):
        if f"cross_attn_{k}" not in out_names:
            missing.append(f"cross_attn_{k}")
    for i in range(NUM_BLOCKS):
        for tag in [f"k_cache_self_{i}_out", f"v_cache_self_{i}_out"]:
            if tag not in out_names:
                missing.append(tag)

    if missing:
        raise RuntimeError(
            f"verify_onnx_outputs FAILED for {onnx_path.name}: missing {missing}\n"
            f"  All outputs in ONNX: {out_names}\n"
            f"  Likely cause: model_adaptation.py patch missing in whisper_env "
            f"site-packages → no peak/cross attention outputs. See feedback_model_adaptation_sync.md.")

    # Verify peak_attn/cross_attn shapes match runtime expectations.
    for o in m.graph.output:
        if o.name.startswith("peak_attn_"):
            sh = [d.dim_value for d in o.type.tensor_type.shape.dim]
            if len(sh) != 4 or sh[1] != 1:
                raise RuntimeError(
                    f"verify_onnx_outputs FAILED for {onnx_path.name}: "
                    f"{o.name} shape={sh} expected [1,1,1,audio_emb_len]")
        if o.name.startswith("cross_attn_"):
            sh = [d.dim_value for d in o.type.tensor_type.shape.dim]
            # Expected shape: [1, n_aheads, 1, audio_emb_len]
            if len(sh) != 4 or sh[1] != expected_n_aheads:
                raise RuntimeError(
                    f"verify_onnx_outputs FAILED for {onnx_path.name}: "
                    f"{o.name} shape={sh} expected n_aheads={expected_n_aheads}")
            break


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2/3: ONNX → DLC → bin
# ─────────────────────────────────────────────────────────────────────────────

def compile_onnx_to_dlc(onnx_paths, dlc_dir):
    """Convert each ONNX to DLC. Returns dict {name: Path}."""
    dlc_dir.mkdir(parents=True, exist_ok=True)
    if not CONVERTER.exists():
        sys.exit(qairt_missing_message(SDK))

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SDK / "lib/python") + ":" + env.get("PYTHONPATH", "")
    env["LD_LIBRARY_PATH"] = (str(SDK / "lib/x86_64-linux-clang") + ":"
                              + env.get("LD_LIBRARY_PATH", ""))

    dlc_paths = {}
    failed = []
    for name, onnx_path in onnx_paths.items():
        dlc_path = dlc_dir / f"{onnx_path.stem}.dlc"
        if dlc_path.exists():
            print(f"  Chunk {name}: DLC exists, skip")
            dlc_paths[name] = dlc_path
            continue

        cmd = [str(CONVERTER), "--input_network", str(onnx_path),
               "--output_path", str(dlc_path), "--float_bitwidth", "16"]
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=current_tool_timeout_sec()
        )
        if result.returncode == 0 and dlc_path.exists():
            size_mb = dlc_path.stat().st_size / (1024 * 1024)
            print(f"  Chunk {name}: DLC OK ({size_mb:.0f} MB)")
            dlc_paths[name] = dlc_path
        else:
            print(f"  Chunk {name}: DLC FAILED")
            for line in (result.stderr or "").strip().splitlines()[-5:]:
                print(f"    {line}")
            failed.append(name)

    if failed:
        sys.exit(f"DLC compile failed for: {failed}")
    return dlc_paths


def link_dlcs_to_bin(dlc_paths, bin_name, output_dir, label=None):
    """Link a DLC group into one weight-shared context binary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if not CONTEXT_GEN.exists():
        sys.exit(qairt_missing_message(SDK))

    # Order matters only for stable layout — sort by name.
    dlc_list = ",".join(str(dlc_paths[n]) for n in sorted(dlc_paths.keys()))

    htp_config = output_dir / "htp_config.json"
    backend_config = output_dir / "backend_config.json"
    with open(htp_config, "w") as f:
        json.dump({
            "context": {"weight_sharing_enabled": True},
            "graphs": [{"graph_names": ["model"], "vtcm_mb": 0, "O": 3}],
            "devices": [legacy.TARGET_PROFILE.backend_device_config],
        }, f, indent=2)
    with open(backend_config, "w") as f:
        json.dump({
            "backend_extensions": {
                "shared_library_path": str(HTP_EXT_LIB),
                "config_file_path": str(htp_config),
            }
        }, f, indent=2)

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = (str(SDK / "lib/x86_64-linux-clang") + ":"
                              + env.get("LD_LIBRARY_PATH", ""))

    cmd = [
        str(CONTEXT_GEN),
        "--dlc_path", dlc_list,
        "--backend", str(HTP_LIB),
        "--binary_file", bin_name,
        "--output_dir", str(output_dir),
        "--config_file", str(backend_config),
        "--input_output_tensor_mem_type", "memhandle",
        "--log_level", "info",
    ]
    label = label or bin_name
    print(f"  Linking {len(dlc_paths)} DLCs → {label} ({bin_name}.bin) ...")

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        return None

    actual_bin = output_dir / f"{bin_name}.bin"
    if not actual_bin.exists():
        print(result.stdout[-2000:])
        return None

    size_mb = actual_bin.stat().st_size / (1024 * 1024)
    print(f"  SUCCESS: {actual_bin.name} ({size_mb:.1f} MB)")
    return actual_bin


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: EPContext wrappers
# ─────────────────────────────────────────────────────────────────────────────

def _make_info(name, shape, dtype=TensorProto.FLOAT16):
    return helper.make_tensor_value_info(name, dtype, shape)


def gen_chunk_wrapper(spec, bucket_tag, bin_basename, audio_emb_len, n_aheads,
                      out_path):
    """Generate one EPContext ONNX wrapper for a chunk in the bucket bin."""
    K = spec.K_internal
    cache_in, cache_out = spec.cache_in, spec.cache_out
    ids_seq_len = spec.prefill_len if (spec.prefill_len is not None and spec.chunk_idx == 0) else 1

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

    outputs = []
    for k in range(K):
        outputs.append(_make_info(f"logits_{k}", [1, VOCAB_SIZE, 1, 1]))
    for i in range(NUM_BLOCKS):
        outputs.append(_make_info(f"k_cache_self_{i}_out", [NUM_HEADS, 1, HEAD_DIM, cache_out]))
        outputs.append(_make_info(f"v_cache_self_{i}_out", [NUM_HEADS, 1, cache_out, HEAD_DIM]))
    for k in range(K):
        outputs.append(_make_info(f"peak_attn_{k}", [1, 1, 1, audio_emb_len]))
    for k in range(K):
        outputs.append(_make_info(f"cross_attn_{k}", [1, n_aheads, 1, audio_emb_len]))

    input_names = [i.name for i in inputs]
    output_names = [o.name for o in outputs]
    unique_chunk_id = f"decoder_{bucket_tag}_chunk_{spec.name}"

    node = helper.make_node(
        "EPContext", name=unique_chunk_id,
        inputs=input_names, outputs=output_names,
        ep_cache_context=bin_basename, embed_mode=0,
        source="Qnn", domain="com.microsoft",
    )
    graph = helper.make_graph([node], f"{unique_chunk_id}_graph", inputs, outputs)
    model = helper.make_model(graph, opset_imports=[
        helper.make_opsetid("", 13), helper.make_opsetid("com.microsoft", 1)
    ])
    onnx.save(model, str(out_path))


def chunk_bin_group(spec):
    """Return the split-bin group name for one chunk wrapper."""
    if spec.chunk_idx >= 2:
        return "shared"
    if spec.prompt_len > 0:
        return f"prefill_p{spec.prompt_len}"
    return "normal"


# ─────────────────────────────────────────────────────────────────────────────
# Per-bucket pipeline
# ─────────────────────────────────────────────────────────────────────────────

def compile_bucket(bucket_sec, build_root, alignment_heads, n_aheads,
                   only_export=False, split_family_bins=False):
    """End-to-end compile for one bucket."""
    # Pad to multiple of 8, matching whisper.cpp/ONNX encoder convention.
    # Without this, cross-KV dim differs from what the standard decoder sees
    # and every token prediction downstream diverges.
    audio_emb_len = bucket_audio_emb_len(bucket_sec)
    bsec_int = int(bucket_sec)
    bucket_tag = f"{bsec_int}s"
    bdir = build_root / f"{bsec_int}s"
    onnx_dir = bdir / "onnx"
    dlc_dir = bdir / "dlc"
    out_dir = bdir / "output"
    wrap_dir = bdir / "wrappers"
    for d in [onnx_dir, dlc_dir, out_dir, wrap_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n========== Bucket {bsec_int}s (audio_emb_len={audio_emb_len}) ==========")

    specs = chunks_for_bucket(bucket_sec)
    print(f"  Chunks to compile: {[s.name for s in specs]}")

    # Phase 1: load model + export ONNX
    print(f"\n[1/4] Loading {legacy.HF_VERSION} + exporting {len(specs)} chunks...")
    model = legacy.HfWhisper.load_whisper_model(legacy.HF_VERSION)
    decoder_module = model.get_decoder()
    legacy.set_nstep_mode(decoder_module)

    onnx_paths = {}
    for spec in specs:
        path = export_chunk_onnx(decoder_module, spec, bucket_tag, audio_emb_len,
                                 alignment_heads, onnx_dir)
        onnx_paths[spec.name] = path

    # Verify the FIRST exported chunk has peak/cross attention outputs (mandatory check
    # before burning a full compile cycle — feedback_no_assumptions_on_compile.md)
    first_name = specs[0].name
    print(f"\n[verify] Checking {first_name} for peak_attn/cross_attn + logits outputs...")
    verify_onnx_outputs(onnx_paths[first_name], specs[0].K_internal, n_aheads)
    print(f"  ✓ verify_onnx_outputs passed for {first_name}")

    if only_export:
        print("\n--only-export specified, stopping after ONNX export.")
        return None

    # Phase 2: ONNX → DLC
    print(f"\n[2/4] Converting ONNX → DLC ({len(specs)} chunks)...")
    dlc_paths = compile_onnx_to_dlc(onnx_paths, dlc_dir)

    # Phase 3: link DLCs → context binary / binaries
    print(f"\n[3/4] Linking DLCs → context binary{' groups' if split_family_bins else ''}...")
    wrapper_bins = {}
    bin_paths = []
    if split_family_bins:
        grouped = {}
        for spec in specs:
            group = chunk_bin_group(spec)
            grouped.setdefault(group, {})
            grouped[group][spec.name] = dlc_paths[spec.name]

        for group_name, group_dlcs in grouped.items():
            bin_name = (
                f"whisper_decoder_nk_full_{bucket_tag}_{group_name}_"
                f"{legacy.TARGET_PROFILE.artifact_suffix}"
            )
            bin_path = link_dlcs_to_bin(
                group_dlcs,
                bin_name,
                out_dir,
                label=f"{bucket_tag}/{group_name}",
            )
            if bin_path is None:
                sys.exit(f"Linking failed for bucket {bsec_int}s group {group_name}")
            bin_paths.append(bin_path)
            for spec_name in group_dlcs:
                wrapper_bins[spec_name] = bin_path.name
    else:
        bin_name = f"whisper_decoder_nk_full_{bucket_tag}_{legacy.TARGET_PROFILE.artifact_suffix}"
        bin_path = link_dlcs_to_bin(
            dlc_paths,
            bin_name,
            out_dir,
            label=f"{bucket_tag}/monolithic",
        )
        if bin_path is None:
            sys.exit(f"Linking failed for bucket {bsec_int}s")
        bin_paths.append(bin_path)
        for spec_name in dlc_paths:
            wrapper_bins[spec_name] = bin_path.name

    # Phase 4: EPContext wrappers
    print(f"\n[4/4] Generating EPContext wrappers ({len(specs)} files)...")
    for spec in specs:
        wrap_path = wrap_dir / f"decoder_{bsec_int}s_chunk_{spec.name}.onnx"
        gen_chunk_wrapper(spec, bucket_tag, wrapper_bins[spec.name], audio_emb_len,
                          n_aheads, wrap_path)
    print(f"  Wrote {len(specs)} wrapper(s) to {wrap_dir}")

    print(f"\n========== Bucket {bsec_int}s DONE ==========")
    if len(bin_paths) == 1:
        print(f"  bin: {bin_paths[0]}")
    else:
        print("  bins:")
        for path in bin_paths:
            print(f"    - {path}")
    print(f"  wrappers: {wrap_dir}/decoder_{bsec_int}s_chunk_*.onnx")
    return bin_paths


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bucket", type=float, default=None,
                   help="Compile only this bucket (e.g., 2.0)")
    p.add_argument("--all-buckets", action="store_true",
                   help="Compile all 9 buckets (2..10s)")
    p.add_argument("--build-dir", type=str, default=None,
                   help="Build directory root (default: auto by model/target)")
    p.add_argument("--model", type=str, default="base",
                   choices=["tiny.en", "tiny", "base.en", "base", "small.en", "small",
                            "medium.en", "medium", "large", "large-v1", "large-v2", "large-v3"],
                   help="Whisper model variant")
    p.add_argument("--target", type=str, default="xplus",
                   choices=["xplus", "s25", "rubikpi"],
                   help="Target device profile")
    p.add_argument("--alignment-heads", type=str, default=None,
                   help="Alignment heads as 'layer,head;...' (default: model preset)")
    p.add_argument("--only-export", action="store_true",
                   help="Stop after exporting + verifying ONNX (no DLC, no link)")
    p.add_argument("--split-family-bins", action="store_true",
                   help="Link chunk groups into separate context binaries: shared, normal, and each prefill family")
    return p.parse_args()


def main():
    global MODEL_PROFILE, TARGET_PROFILE, DEFAULT_ALIGNMENT_HEADS
    args = parse_args()
    MODEL_PROFILE = resolve_whisper_model_profile(args.model)
    TARGET_PROFILE = resolve_target_profile(args.target)
    DEFAULT_ALIGNMENT_HEADS = MODEL_PROFILE.alignment_heads
    legacy.configure_profiles(args.model, args.target)
    sync_legacy_constants()

    # numpy < 2.0 check
    import numpy as np
    if int(np.__version__.split('.')[0]) >= 2:
        sys.exit(f"numpy {np.__version__} detected. QAIRT 2.37 requires numpy<2.0.")

    if not args.bucket and not args.all_buckets:
        sys.exit("Specify --bucket <sec> or --all-buckets")

    if args.bucket and args.all_buckets:
        sys.exit("--bucket and --all-buckets are mutually exclusive")

    buckets = [float(b) for b in range(2, 11)] if args.all_buckets else [args.bucket]

    # Parse alignment heads
    aheads = []
    alignment_spec = args.alignment_heads or DEFAULT_ALIGNMENT_HEADS
    for pair in alignment_spec.split(";"):
        parts = pair.strip().split(",")
        if len(parts) == 2:
            aheads.append((int(parts[0]), int(parts[1])))
    n_aheads = len(aheads)

    print(f"Compile target: {len(buckets)} bucket(s) {[int(b) for b in buckets]}s")
    print(f"  Chains:       normal + prefill (all buckets)")
    print(f"  K table normal:  {K_TABLE_NORMAL}")
    print(f"  K table prefill: {K_TABLE_PREFILL}")
    print(f"  Alignment heads: {aheads} ({n_aheads} heads)")
    print(f"  Model:        {MODEL_PROFILE.hf_repo}")
    print(f"  Target:       {TARGET_PROFILE.target_device} (Hexagon {TARGET_PROFILE.dsp_arch}, soc_model={TARGET_PROFILE.soc_model})")
    build_dir = args.build_dir or default_build_dir("nk_full", MODEL_PROFILE, TARGET_PROFILE)
    print(f"  Build dir:    {build_dir}")
    print(f"  numpy:        {np.__version__}")

    build_root = Path(build_dir)
    build_root.mkdir(parents=True, exist_ok=True)

    for b in buckets:
        compile_bucket(
            b,
            build_root,
            aheads,
            n_aheads,
            only_export=args.only_export,
            split_family_bins=args.split_family_bins,
        )

    print(f"\nAll buckets compiled. Output under {build_root}/")


if __name__ == "__main__":
    main()
