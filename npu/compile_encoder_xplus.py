"""
Compile Whisper encoder for Snapdragon X Plus 8-Core CRD.

Encoder input shape is determined by --audio-sec:
  2s  -> (1, 80, 200)  -> cross-KV with AUDIO_EMB_LEN=100
  30s -> (1, 80, 3000) -> cross-KV with AUDIO_EMB_LEN=1500

Pipeline (all local, no QAI Hub dependency):
1. ONNX export
2. Local qairt-converter (QAIRT 2.37) -> DLC
3. Local qnn-context-binary-generator -> context binary (.bin)
4. Generate EPContext ONNX wrapper

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
from typing import cast

import onnx
import torch
from onnx import TensorProto, helper
from qai_hub_models.models._shared.hf_whisper.model import (
    HfWhisper,
    HfWhisperEncoder,
)
from transformers import WhisperConfig

from whisper_npu_profile import (
    bucket_audio_emb_len,
    default_build_dir,
    resolve_hf_model_source,
    resolve_target_profile,
    resolve_whisper_model_profile,
)
from qairt_paths import (
    qairt_context_binary_generator_path,
    qairt_converter_path,
    qairt_htp_ext_lib_path,
    qairt_htp_lib_path,
    qairt_missing_message,
    resolve_qnn_sdk_root,
)

MODEL_PROFILE = resolve_whisper_model_profile("base")
TARGET_PROFILE = resolve_target_profile("xplus")
HF_VERSION = MODEL_PROFILE.hf_repo
NUM_BLOCKS = MODEL_PROFILE.num_blocks
NUM_HEADS = MODEL_PROFILE.num_heads
ATTENTION_DIM = MODEL_PROFILE.attention_dim
HEAD_DIM = MODEL_PROFILE.head_dim
NUM_MEL_BINS = MODEL_PROFILE.num_mel_bins

WORK = Path(__file__).parent / default_build_dir("encoder", MODEL_PROFILE, TARGET_PROFILE)
ONNX_DIR = WORK / "onnx"
DLC_DIR = WORK / "dlc"
OUTPUT_DIR = WORK / "output"

# QAIRT 2.37 SDK (must match on-device QNN SDK version)
SDK = resolve_qnn_sdk_root()
CONVERTER = qairt_converter_path(SDK)
CONTEXT_GEN = qairt_context_binary_generator_path(SDK)
HTP_LIB = qairt_htp_lib_path(SDK)
HTP_EXT_LIB = qairt_htp_ext_lib_path(SDK)

SOC_MODEL = TARGET_PROFILE.soc_model
DSP_ARCH = TARGET_PROFILE.dsp_arch
MAX_INLINE_ONNX_BYTES = 2 * 1024**3 - 16 * 1024**2
DEFAULT_TOOL_TIMEOUT_SEC = 300
LARGE_TOOL_TIMEOUT_SEC = 3600


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compile Whisper encoder for Snapdragon X Plus")
    parser.add_argument("--audio-sec", type=float, default=2.0,
                        help="Audio chunk length in seconds (default: 2.0). "
                             "Determines encoder input mel shape and cross-KV output size. "
                             "E.g., 2.0 -> mel[1,80,200], AUDIO_EMB_LEN=100")
    parser.add_argument("--build-dir", type=str, default=None,
                        help="Build directory (default: auto by model/target)")
    parser.add_argument("--model", type=str, default="base",
                        choices=["tiny.en", "tiny", "base.en", "base", "small.en", "small",
                                 "medium.en", "medium", "large", "large-v1", "large-v2", "large-v3"],
                        help="Whisper model variant")
    parser.add_argument("--target", type=str, default="xplus",
                        choices=["xplus", "s25", "rubikpi"],
                        help="Target device profile")
    return parser.parse_args()


def main():
    global MODEL_PROFILE, TARGET_PROFILE, HF_VERSION
    global NUM_BLOCKS, NUM_HEADS, ATTENTION_DIM, HEAD_DIM, NUM_MEL_BINS
    global WORK, ONNX_DIR, DLC_DIR, OUTPUT_DIR, SOC_MODEL, DSP_ARCH

    args = parse_args()
    MODEL_PROFILE = resolve_whisper_model_profile(args.model)
    TARGET_PROFILE = resolve_target_profile(args.target)
    HF_VERSION = resolve_hf_model_source(MODEL_PROFILE)
    NUM_BLOCKS = MODEL_PROFILE.num_blocks
    NUM_HEADS = MODEL_PROFILE.num_heads
    ATTENTION_DIM = MODEL_PROFILE.attention_dim
    HEAD_DIM = MODEL_PROFILE.head_dim
    NUM_MEL_BINS = MODEL_PROFILE.num_mel_bins
    SOC_MODEL = TARGET_PROFILE.soc_model
    DSP_ARCH = TARGET_PROFILE.dsp_arch
    tool_timeout_sec = (
        LARGE_TOOL_TIMEOUT_SEC
        if args.model in {"large", "large-v1", "large-v2", "large-v3"}
        else DEFAULT_TOOL_TIMEOUT_SEC
    )

    audio_sec = args.audio_sec
    build_dir = args.build_dir or default_build_dir("encoder", MODEL_PROFILE, TARGET_PROFILE)
    WORK = Path(build_dir)
    ONNX_DIR = WORK / "onnx"
    DLC_DIR = WORK / "dlc"
    OUTPUT_DIR = WORK / "output"

    # Compute audio dimensions from --audio-sec
    # Whisper mel spectrogram: 100 frames/second (16kHz, hop=160)
    # Encoder conv2 stride=2 halves the sequence length
    # IMPORTANT: pad audio_emb_len to a multiple of 8 to match the standard
    # whisper.cpp / ONNX RT encoder padding convention (see whisper_ort.cpp
    # whisper_encode ONNX path: target_ctx = ((n_frames/2 + 7) / 8) * 8).
    # Without this padding, the NPU encoder output sequence length differs from
    # the standard path, which changes the cross-attention distribution enough
    # to produce different first-token predictions and cascade through all
    # subsequent tokens.
    audio_emb_len = bucket_audio_emb_len(audio_sec, TARGET_PROFILE)
    mels_audio_len = audio_emb_len * 2

    import numpy as np
    if int(np.__version__.split('.')[0]) >= 2:
        print(f"ERROR: numpy {np.__version__} detected. QAIRT 2.37 requires numpy<2.0.")
        print(f"  Fix: pip install 'numpy<2.0'")
        sys.exit(1)

    for d in [ONNX_DIR, DLC_DIR, OUTPUT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Encode audio length in filenames for clarity
    sec_tag = f"{audio_sec:g}s"
    onnx_path = ONNX_DIR / f"encoder_{sec_tag}.onnx"
    dlc_path = DLC_DIR / f"encoder_{sec_tag}.dlc"
    bin_name = f"whisper_encoder_{sec_tag}_{TARGET_PROFILE.artifact_suffix}"
    bin_path = OUTPUT_DIR / f"{bin_name}.bin"

    print("=" * 60)
    print("Whisper Encoder -> QNN context binary (fully local)")
    print(f"  Model        : {MODEL_PROFILE.hf_repo}")
    print(f"  Target       : {TARGET_PROFILE.target_device}")
    print(f"  Audio length : {audio_sec}s")
    print(f"  Mel frames   : {mels_audio_len} (input)")
    print(f"  AUDIO_EMB_LEN: {audio_emb_len} (cross-KV dim)")
    print(f"  QAIRT SDK    : {SDK}")
    print(f"  SOC          : {SOC_MODEL} (Hexagon {DSP_ARCH})")
    print(f"  numpy        : {np.__version__}")
    print("=" * 60)

    # 1. Load encoder
    print(f"\n1. Loading {MODEL_PROFILE.hf_repo} encoder...")
    model = HfWhisper.load_whisper_model(HF_VERSION)
    encoder_module = model.get_encoder()
    config = cast(WhisperConfig, model.config)
    encoder = HfWhisperEncoder(config, encoder_module)
    encoder.eval()

    # 2. Export ONNX
    if onnx_path.exists():
        print(f"\n2. ONNX exists, skipping: {onnx_path}")
    else:
        print(f"\n2. Exporting encoder ONNX (mel input: [1, {NUM_MEL_BINS}, {mels_audio_len}])...")
        mel_input = torch.randn(1, NUM_MEL_BINS, mels_audio_len, dtype=torch.float32)
        input_names = ["input_features"]
        output_names = HfWhisperEncoder.get_output_names(num_blocks=NUM_BLOCKS)

        with torch.no_grad():
            torch.onnx.export(
                encoder, mel_input, str(onnx_path),
                input_names=input_names,
                output_names=output_names,
                opset_version=18,
                do_constant_folding=True,
                external_data=True,
            )
        print(f"  Exported: {onnx_path}")

        # Keep large exports in external-data form. Re-embedding >2GB models
        # into a single protobuf breaks large-v3 conversion.
        ext_data = Path(str(onnx_path) + ".data")
        use_external = ext_data.exists() or (
            onnx_path.exists() and onnx_path.stat().st_size >= MAX_INLINE_ONNX_BYTES
        )
        if use_external:
            onnx_model = onnx.load(str(onnx_path), load_external_data=True)
            onnx.save_model(
                onnx_model,
                str(onnx_path),
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=ext_data.name,
                size_threshold=1024,
            )
            print("  Kept external tensor data for large ONNX")

    # 3. Local compile: ONNX -> DLC
    if dlc_path.exists():
        print(f"\n3. DLC exists, skipping: {dlc_path}")
    else:
        print("\n3. Converting ONNX -> DLC (local QAIRT 2.37)...")
        if not CONVERTER.exists():
            print(qairt_missing_message(SDK))
            sys.exit(1)

        env = os.environ.copy()
        env["PYTHONPATH"] = str(SDK / "lib/python") + ":" + env.get("PYTHONPATH", "")
        local_libs = OUTPUT_DIR / "_local_libs"
        ld = str(SDK / "lib/x86_64-linux-clang")
        if local_libs.exists():
            ld = str(local_libs) + ":" + ld
        env["LD_LIBRARY_PATH"] = ld + ":" + env.get("LD_LIBRARY_PATH", "")

        cmd = [
            str(CONVERTER),
            "--input_network", str(onnx_path),
            "--output_path", str(dlc_path),
            "--float_bitwidth", "16",
        ]
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=tool_timeout_sec
        )
        if result.returncode == 0 and dlc_path.exists():
            size_mb = dlc_path.stat().st_size / (1024 * 1024)
            print(f"  DLC OK: {dlc_path} ({size_mb:.1f} MB)")
        else:
            print(f"  DLC FAILED!")
            if result.stderr:
                print(result.stderr[-1000:])
            sys.exit(1)

    # 4. Local link: DLC -> context binary
    if bin_path.exists():
        print(f"\n4. Context binary exists, skipping: {bin_path}")
    else:
        print("\n4. Generating context binary...")
        if not CONTEXT_GEN.exists():
            print(qairt_missing_message(SDK))
            sys.exit(1)

        htp_config = WORK / "htp_config_encoder.json"
        backend_config = WORK / "backend_config_encoder.json"

        with open(htp_config, "w") as f:
            json.dump({
                "context": {"weight_sharing_enabled": True},
                "graphs": [{"graph_names": ["model"], "vtcm_mb": 0, "O": 3}],
                "devices": [TARGET_PROFILE.backend_device_config],
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
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=tool_timeout_sec
        )
        if result.stdout:
            print(result.stdout[-1000:])
        if result.returncode != 0 or not bin_path.exists():
            print("  LINK FAILED!")
            if result.stderr:
                print(result.stderr[-1000:])
            sys.exit(1)

        size_mb = bin_path.stat().st_size / (1024 * 1024)
        print(f"  Context binary OK: {bin_path} ({size_mb:.1f} MB)")

    # 5. Generate EPContext ONNX wrapper
    wrapper_dir = WORK / "wrappers"
    wrapper_dir.mkdir(parents=True, exist_ok=True)
    wrapper_path = wrapper_dir / f"encoder_{sec_tag}.onnx"

    print(f"\n5. Generating EPContext ONNX wrapper...")
    bin_name = bin_path.name

    def _info(name, shape, dtype=TensorProto.FLOAT16):
        return helper.make_tensor_value_info(name, dtype, shape)

    inputs = [_info("input_features", [1, NUM_MEL_BINS, mels_audio_len])]
    outputs = []
    for i in range(NUM_BLOCKS):
        outputs.append(_info(f"k_cache_cross_{i}", [NUM_HEADS, 1, HEAD_DIM, audio_emb_len]))
        outputs.append(_info(f"v_cache_cross_{i}", [NUM_HEADS, 1, audio_emb_len, HEAD_DIM]))

    input_names = [inp.name for inp in inputs]
    output_names = [out.name for out in outputs]

    unique_encoder_id = f"encoder_{sec_tag}"

    node = helper.make_node(
        "EPContext", name=unique_encoder_id,
        inputs=input_names, outputs=output_names,
        ep_cache_context=bin_name, embed_mode=0,
        source="Qnn", domain="com.microsoft",
    )
    graph = helper.make_graph([node], f"{unique_encoder_id}_graph", inputs, outputs)
    onnx_model = helper.make_model(graph, opset_imports=[
        helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)
    ])
    onnx.save(onnx_model, str(wrapper_path))
    print(f"  Wrapper: {wrapper_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Encoder compilation done!")
    print(f"  Audio length     : {audio_sec}s ({mels_audio_len} mel -> {audio_emb_len} emb)")
    print(f"  Context binary   : {bin_path}")
    print(f"  EPContext wrapper : {wrapper_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
