#!/usr/bin/env python3
"""
Stage bucketed decoder artifacts for the X Plus NPU runtime directory.

Flow:
  1. For each compiled bucket (xplus_build_nk_full/{N}s/), copy
       output/whisper_decoder_nk_full_{N}s_xplus.bin
       wrappers/decoder_{N}s_chunk_*.onnx
     into a flat staging directory.
  2. Patch each wrapper's ep_cache_context attr to be a basename only (so the
     wrapper finds the bin sitting in the same flat directory).
  3. Write config.json with decoder metadata.
  4. Print a one-liner scp command for pushing to the target X Plus laptop.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import onnx

from whisper_npu_profile import default_build_dir, resolve_target_profile, resolve_whisper_model_profile

MODEL_PROFILE = resolve_whisper_model_profile("base")
TARGET_PROFILE = resolve_target_profile("xplus")
BUILD_ROOT = Path(default_build_dir("nk_full", MODEL_PROFILE, TARGET_PROFILE))
BUILD_ROOT_1STEP_PATTERN = default_build_dir("decoder_1step_bucket", MODEL_PROFILE, TARGET_PROFILE)
ENCODER_BUILD_ROOT = Path(default_build_dir("encoder", MODEL_PROFILE, TARGET_PROFILE))
STAGE_DIR = Path("/tmp/npusper_npu")

# Must match the K table in compile_decoder_nk_full.py
K_NORMAL  = [4, 6, 5, 5, 5, 5]
K_PREFILL = [6, 4, 5, 5, 5, 5]
PREFILL_PROMPT_TOKEN_LENS = [1, 2, 3, 4]
NUM_CHUNKS = 6
N_ALIGN_HEADS = MODEL_PROFILE.n_alignment_heads

ALL_BUCKETS = [2, 3, 4, 5, 6, 7, 8, 9, 10]
DECODER_MODE_NK = "nk"
DECODER_MODE_1STEP = "1step"
DECODER_1STEP_KV_CACHE_SIZE = 199
REINFER_30S_KV_CACHE_SIZE = 199
SCRIPT_DIR = Path(__file__).resolve().parent


def support_file_candidates(model_profile):
    keys = [model_profile.key, model_profile.key.replace(".", "_")]
    if model_profile.key.endswith(".en"):
        keys.append(model_profile.key[:-3])
    candidates = []
    for key in keys:
        candidates.append(
            SCRIPT_DIR / "whisper-onnx" / "models" / key
        )
    return candidates


def copy_support_files(stage_dir):
    copied = []
    for fname in ["vocab.txt", "mel_filters.bin"]:
        found = None
        for candidate in support_file_candidates(MODEL_PROFILE):
            path = candidate / fname
            if path.exists():
                found = path
                break
        if found is None:
            raise FileNotFoundError(
                f"missing required support file '{fname}' for model {MODEL_PROFILE.key}; "
                f"searched under {support_file_candidates(MODEL_PROFILE)}"
            )
        dst = stage_dir / fname
        shutil.copy2(found, dst)
        copied.append(dst.name)
    dims_path = stage_dir / "dims.txt"
    is_multilingual = 0 if MODEL_PROFILE.key.endswith(".en") else 1
    dims_text = "\n".join([
        f"n_mels={MODEL_PROFILE.num_mel_bins}",
        "n_audio_ctx=1500",
        f"n_audio_state={MODEL_PROFILE.attention_dim}",
        f"n_audio_head={MODEL_PROFILE.num_heads}",
        f"n_audio_layer={MODEL_PROFILE.num_blocks}",
        f"n_vocab={MODEL_PROFILE.vocab_size}",
        "n_text_ctx=448",
        f"n_text_state={MODEL_PROFILE.attention_dim}",
        f"n_text_head={MODEL_PROFILE.num_heads}",
        f"n_text_layer={MODEL_PROFILE.num_blocks}",
        f"head_dim={MODEL_PROFILE.head_dim}",
        f"is_multilingual={is_multilingual}",
        "",
    ])
    dims_path.write_text(dims_text, encoding="utf-8")
    copied.append("dims.txt")
    print(f"  copied/generated support files: {', '.join(copied)}")


def bucket_audio_emb_len(bucket_int):
    raw = int(bucket_int * 50)
    emb = ((raw + 7) // 8) * 8
    if bucket_int == 2:
        return 100
    return emb


def patch_wrapper_ep_context(wrapper_path):
    """Rewrite wrapper metadata for flat deploy and older ORT compatibility."""
    m = onnx.load(str(wrapper_path))
    changed = False
    for node in m.graph.node:
        for attr in node.attribute:
            if attr.name == "ep_cache_context":
                base = os.path.basename(attr.s.decode())
                if attr.s.decode() != base:
                    attr.s = base.encode()
                    changed = True
    if m.ir_version > 10:
        m.ir_version = 10
        changed = True
    if changed:
        onnx.save(m, str(wrapper_path))


def stage_bucket(bucket_int, stage_dir, build_root):
    """Copy one bucket's bin + wrappers into the staging dir."""
    bdir = build_root / f"{bucket_int}s"
    bin_path = bdir / "output" / f"whisper_decoder_nk_full_{bucket_int}s_{TARGET_PROFILE.artifact_suffix}.bin"
    if not bin_path.exists():
        raise FileNotFoundError(f"missing {bin_path}")

    shutil.copy2(bin_path, stage_dir / bin_path.name)

    wrap_dir = bdir / "wrappers"
    n_wrappers = 0
    for wpath in sorted(wrap_dir.glob(f"decoder_{bucket_int}s_chunk_*.onnx")):
        dst = stage_dir / wpath.name
        shutil.copy2(wpath, dst)
        patch_wrapper_ep_context(dst)
        n_wrappers += 1
    print(f"  bucket {bucket_int}s: 1 bin + {n_wrappers} wrappers")
    return n_wrappers


def stage_encoder_bucket(bucket_int, stage_dir, build_root):
    sec_tag = f"{bucket_int}s"
    bin_path = build_root / "output" / f"whisper_encoder_{sec_tag}_{TARGET_PROFILE.artifact_suffix}.bin"
    wrapper_path = build_root / "wrappers" / f"encoder_{sec_tag}.onnx"
    if not bin_path.exists():
        raise FileNotFoundError(f"missing {bin_path}")
    if not wrapper_path.exists():
        raise FileNotFoundError(f"missing {wrapper_path}")

    shutil.copy2(bin_path, stage_dir / bin_path.name)
    dst = stage_dir / wrapper_path.name
    shutil.copy2(wrapper_path, dst)
    patch_wrapper_ep_context(dst)
    print(f"  encoder {bucket_int}s: 1 bin + 1 wrapper")


def stage_bucket_1step(bucket_int, stage_dir, build_root_pattern):
    sec_tag = f"{bucket_int}s"
    bdir = Path(str(build_root_pattern).format(sec=bucket_int))
    bin_path = bdir / "output" / f"whisper_decoder_1step_{sec_tag}_{TARGET_PROFILE.artifact_suffix}.bin"
    if not bin_path.exists():
        raise FileNotFoundError(f"missing {bin_path}")

    shutil.copy2(bin_path, stage_dir / bin_path.name)

    wrapper_src = bdir / "wrappers" / f"decoder_1step_{sec_tag}.onnx"
    if not wrapper_src.exists():
        raise FileNotFoundError(f"missing {wrapper_src}")
    dst = stage_dir / wrapper_src.name
    shutil.copy2(wrapper_src, dst)
    patch_wrapper_ep_context(dst)
    print(f"  bucket {bucket_int}s: 1 bin + 1 wrapper (1step)")
    return 1


def write_config(stage_dir, buckets, decoder_mode, with_30s=False,
                 reinfer_kv_cache_size=0):
    cfg = {
        "model": MODEL_PROFILE.deploy_model_name,
        "num_chunks": NUM_CHUNKS,
        "K_normal": K_NORMAL,
        "K_prefill": K_PREFILL,
        "prefill_prompt_token_lens": PREFILL_PROMPT_TOKEN_LENS,
        "max_decode_len": sum(K_NORMAL),
        "buckets_sec": [float(b) for b in buckets],
        "bucket_audio_emb_len": [bucket_audio_emb_len(b) for b in buckets],
        "num_blocks": MODEL_PROFILE.num_blocks,
        "num_heads": MODEL_PROFILE.num_heads,
        "head_dim": MODEL_PROFILE.head_dim,
        "vocab_size": MODEL_PROFILE.vocab_size,
        "has_30s_reinfer": with_30s,
        "n_alignment_heads": N_ALIGN_HEADS,
        "decoder_mode": decoder_mode,
        "decoder_1step_kv_cache_size": DECODER_1STEP_KV_CACHE_SIZE,
        "reinfer_kv_cache_size": reinfer_kv_cache_size if with_30s else 0,
        "alignment_heads_preset": MODEL_PROFILE.aheads_preset,
        "alignment_heads": MODEL_PROFILE.alignment_heads,
        "target_device": TARGET_PROFILE.target_device,
    }
    cfg_path = stage_dir / "config.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)
    print(f"  wrote {cfg_path}")


def main():
    global MODEL_PROFILE, TARGET_PROFILE, BUILD_ROOT, BUILD_ROOT_1STEP_PATTERN, ENCODER_BUILD_ROOT, N_ALIGN_HEADS
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="base",
                   choices=["tiny.en", "tiny", "base.en", "base", "small.en", "small",
                            "medium.en", "medium", "large", "large-v1", "large-v2", "large-v3"])
    p.add_argument("--target", default="xplus", choices=["xplus", "s25", "rubikpi"])
    p.add_argument("--build-root", default=None)
    p.add_argument("--encoder-build-root", default=None)
    p.add_argument("--decoder-1step-build-root-pattern", default=None)
    p.add_argument("--stage-dir", default=str(STAGE_DIR))
    p.add_argument("--buckets", default="all", help="comma-separated bucket secs or 'all'")
    p.add_argument("--decoder-mode", choices=[DECODER_MODE_NK, DECODER_MODE_1STEP],
                   default=DECODER_MODE_NK)
    p.add_argument("--with-30s", action="store_true",
                   help="Also stage encoder_30s + decoder_1step_30s for the fallback path")
    p.add_argument("--reinfer-kv-cache-size", type=int, default=REINFER_30S_KV_CACHE_SIZE,
                   help="KV cache size for 30s reinference decoder (default: 199)")
    args = p.parse_args()

    MODEL_PROFILE = resolve_whisper_model_profile(args.model)
    TARGET_PROFILE = resolve_target_profile(args.target)
    BUILD_ROOT = Path(args.build_root or default_build_dir("nk_full", MODEL_PROFILE, TARGET_PROFILE))
    ENCODER_BUILD_ROOT = Path(args.encoder_build_root or default_build_dir("encoder", MODEL_PROFILE, TARGET_PROFILE))
    BUILD_ROOT_1STEP_PATTERN = args.decoder_1step_build_root_pattern or default_build_dir("decoder_1step_bucket", MODEL_PROFILE, TARGET_PROFILE)
    N_ALIGN_HEADS = MODEL_PROFILE.n_alignment_heads

    if args.buckets == "all":
        buckets = ALL_BUCKETS
    else:
        buckets = [int(b) for b in args.buckets.split(",")]

    build_root = BUILD_ROOT
    encoder_build_root = ENCODER_BUILD_ROOT
    stage_dir = Path(args.stage_dir)
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)

    print(f"Staging to {stage_dir}")
    total_wrappers = 0
    for b in buckets:
        stage_encoder_bucket(b, stage_dir, encoder_build_root)
        if args.decoder_mode == DECODER_MODE_1STEP:
            total_wrappers += stage_bucket_1step(
                b, stage_dir, args.decoder_1step_build_root_pattern)
        else:
            total_wrappers += stage_bucket(b, stage_dir, build_root)

    if args.with_30s:
        stage_encoder_bucket(30, stage_dir, encoder_build_root)
        total_wrappers += stage_bucket_1step(
            30, stage_dir, args.decoder_1step_build_root_pattern)

    copy_support_files(stage_dir)

    write_config(stage_dir, buckets, args.decoder_mode,
                 with_30s=args.with_30s,
                 reinfer_kv_cache_size=args.reinfer_kv_cache_size)

    files = sorted(stage_dir.iterdir())
    n_bin = sum(1 for f in files if f.suffix == ".bin")
    n_onnx = sum(1 for f in files if f.suffix == ".onnx")
    total_size_mb = sum(f.stat().st_size for f in files) / (1024 * 1024)

    print()
    print(f"Staged: {n_bin} bin + {n_onnx} onnx + 1 config.json = {total_size_mb:.0f} MB")
    print()
    print("To push to the X Plus runtime directory:")
    print("  ssh <xplus-host> 'powershell -NoProfile -Command \"New-Item -ItemType Directory -Force $HOME\\Desktop\\npusper_npu | Out-Null\"'")
    print(f"  scp {stage_dir}/* <xplus-host>:Desktop/npusper_npu/")


if __name__ == "__main__":
    main()
