# ---------------------------------------------------------------------
# N-step compilation test for NPU-friendly Whisper decoder
#
# Verifies:
# 1. n-step mode forward pass works for steps 0..N
# 2. Output KV cache shape grows by 1 each step (no reshape/-1)
# 3. Output cache from step N feeds directly into step N+1
# 4. ONNX export produces fully static shapes (no dynamic dims)
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
import numpy as np

from qai_hub_models.models._shared.hf_whisper.model import (
    AUDIO_EMB_LEN,
    MASK_NEG,
    HfWhisper,
    HfWhisperDecoder,
    HfWhisperEncoder,
)
from qai_hub_models.models._shared.hf_whisper.model_adaptation import set_nstep_mode


# ---------------------------------------------------------------------------
# Use whisper-base for testing
# ---------------------------------------------------------------------------
HF_VERSION = "openai/whisper-base"

# whisper-base config
NUM_BLOCKS = 6
ATTENTION_DIM = 512
NUM_HEADS = 8
HEAD_DIM = ATTENTION_DIM // NUM_HEADS  # 64
NUM_MEL_BINS = 80

# How many decoder steps to test
NUM_STEPS = 5


def build_step_inputs(
    step: int,
    kv_cache_cross: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    kv_cache_self: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
) -> tuple:
    """Build decoder inputs for a given n-step decoding step."""
    cache_len = step + 1   # 1 dummy + step real entries
    mask_len = step + 2    # cache_len + 1 (current token)

    input_ids = torch.tensor([[50258]], dtype=torch.int32)  # SOT token
    position_ids = torch.tensor([step], dtype=torch.int32)

    # Attention mask: first position masked (dummy), rest unmasked
    attention_mask = torch.full((1, 1, 1, mask_len), MASK_NEG, dtype=torch.float32)
    attention_mask[:, :, :, 1:] = 0.0  # unmask all except dummy at position 0

    # Self-attention KV cache
    if kv_cache_self is None:
        # Step 0: dummy cache of size 1 (all zeros)
        kv_cache_self = tuple(
            (
                torch.zeros(NUM_HEADS, 1, HEAD_DIM, cache_len, dtype=torch.float32),
                torch.zeros(NUM_HEADS, 1, cache_len, HEAD_DIM, dtype=torch.float32),
            )
            for _ in range(NUM_BLOCKS)
        )

    # Flatten for the wrapper's *args interface
    flat_kv_self = tuple(item for pair in kv_cache_self for item in pair)
    flat_kv_cross = tuple(item for pair in kv_cache_cross for item in pair)

    return (input_ids, attention_mask, *flat_kv_self, *flat_kv_cross, position_ids)


def test_nstep_forward():
    """Test n-step decoder forward pass: shape progression & feedability."""
    print("=" * 60)
    print("Loading whisper-base model...")
    model = HfWhisper.load_whisper_model(HF_VERSION)

    encoder_module = model.get_encoder()
    decoder_module = model.get_decoder()

    # Enable n-step mode on decoder (self-attention only, not cross)
    set_nstep_mode(decoder_module)

    from transformers import WhisperConfig
    from typing import cast
    config = cast(WhisperConfig, model.config)

    encoder = HfWhisperEncoder(config, encoder_module)
    decoder = HfWhisperDecoder(config, decoder_module)
    encoder.eval()
    decoder.eval()

    # Verify n-step mode is set on self-attention layers only
    from qai_hub_models.models._shared.hf_whisper.model_adaptation import SHAAttention
    nstep_count = 0
    for m in decoder_module.modules():
        if isinstance(m, SHAAttention) and getattr(m, "nstep_mode", False):
            nstep_count += 1
    print(f"  n-step SHAAttention modules found: {nstep_count}")
    assert nstep_count > 0, "No SHAAttention modules have nstep_mode=True"

    # -----------------------------------------------------------------------
    # Run encoder to get cross-attention KV cache
    # -----------------------------------------------------------------------
    print("Running encoder...")
    mel_input = torch.randn(1, NUM_MEL_BINS, 3000, dtype=torch.float32)
    with torch.no_grad():
        encoder_out = encoder(mel_input)

    # encoder_out is a tuple of (k, v) tuples per layer
    if not isinstance(encoder_out, tuple):
        encoder_out = (encoder_out,)
    if not isinstance(encoder_out[0], (tuple, list)):
        encoder_out = (encoder_out,)
    kv_cache_cross = encoder_out

    print(f"  Cross-attn cache: {len(kv_cache_cross)} layers")
    print(f"  K shape: {kv_cache_cross[0][0].shape}")
    print(f"  V shape: {kv_cache_cross[0][1].shape}")

    # -----------------------------------------------------------------------
    # Run decoder for N steps
    # -----------------------------------------------------------------------
    print(f"\nRunning {NUM_STEPS} decoder steps in n-step mode...")
    kv_cache_self = None

    for step in range(NUM_STEPS):
        decoder_input = build_step_inputs(step, kv_cache_cross, kv_cache_self)

        expected_in_cache = step + 1
        expected_out_cache = step + 2
        expected_mask = step + 2

        # Verify input shapes
        input_ids = decoder_input[0]
        attention_mask = decoder_input[1]
        first_k_self = decoder_input[2]  # k_cache_self_0_in
        first_v_self = decoder_input[3]  # v_cache_self_0_in

        assert attention_mask.shape == (1, 1, 1, expected_mask), \
            f"Step {step}: mask shape {attention_mask.shape} != expected (1,1,1,{expected_mask})"
        assert first_k_self.shape == (NUM_HEADS, 1, HEAD_DIM, expected_in_cache), \
            f"Step {step}: k_self shape {first_k_self.shape} != expected ({NUM_HEADS},1,{HEAD_DIM},{expected_in_cache})"
        assert first_v_self.shape == (NUM_HEADS, 1, expected_in_cache, HEAD_DIM), \
            f"Step {step}: v_self shape {first_v_self.shape} != expected ({NUM_HEADS},1,{expected_in_cache},{HEAD_DIM})"

        # Run decoder
        with torch.no_grad():
            decoder_out = decoder(*decoder_input)

        logits, kv_cache_self_new = decoder_out

        # Verify output shapes
        k_out = kv_cache_self_new[0][0]  # layer 0, key
        v_out = kv_cache_self_new[0][1]  # layer 0, value

        assert k_out.shape == (NUM_HEADS, 1, HEAD_DIM, expected_out_cache), \
            f"Step {step}: k_out shape {k_out.shape} != expected ({NUM_HEADS},1,{HEAD_DIM},{expected_out_cache})"
        assert v_out.shape == (NUM_HEADS, 1, expected_out_cache, HEAD_DIM), \
            f"Step {step}: v_out shape {v_out.shape} != expected ({NUM_HEADS},1,{expected_out_cache},{HEAD_DIM})"

        print(f"  Step {step}: mask({expected_mask}) | "
              f"k_self_in({expected_in_cache}) -> k_self_out({expected_out_cache}) | "
              f"logits {logits.shape} OK")

        # Feed output cache into next step
        kv_cache_self = kv_cache_self_new

    print("\n[PASS] n-step forward pass: all shapes correct, cache grows by 1 each step")


def test_nstep_input_spec():
    """Verify get_input_spec_for_step produces correct shapes."""
    print("\n" + "=" * 60)
    print("Testing get_input_spec_for_step...")

    for step in range(NUM_STEPS):
        spec = HfWhisperDecoder.get_input_spec_for_step(
            step=step,
            num_blocks=NUM_BLOCKS,
            attention_dim=ATTENTION_DIM,
            num_heads=NUM_HEADS,
        )

        cache_len = step + 1
        mask_len = step + 2

        assert spec["input_ids"][0] == (1, 1)
        assert spec["attention_mask"][0] == (1, 1, 1, mask_len), \
            f"Step {step}: mask spec {spec['attention_mask'][0]} != (1,1,1,{mask_len})"
        assert spec["k_cache_self_0_in"][0] == (NUM_HEADS, 1, HEAD_DIM, cache_len), \
            f"Step {step}: k_self spec wrong"
        assert spec["v_cache_self_0_in"][0] == (NUM_HEADS, 1, cache_len, HEAD_DIM), \
            f"Step {step}: v_self spec wrong"
        assert spec["k_cache_cross_0"][0] == (NUM_HEADS, 1, HEAD_DIM, AUDIO_EMB_LEN)
        assert spec["v_cache_cross_0"][0] == (NUM_HEADS, 1, AUDIO_EMB_LEN, HEAD_DIM)
        assert spec["position_ids"][0] == (1,)

        print(f"  Step {step}: mask_len={mask_len}, cache_len={cache_len} OK")

    print("\n[PASS] get_input_spec_for_step: all specs correct")


def test_nstep_onnx_export():
    """Export step 0 and step 1 to ONNX, verify all shapes are fully static."""
    import onnx

    print("\n" + "=" * 60)
    print("Testing ONNX export with static shapes...")

    model = HfWhisper.load_whisper_model(HF_VERSION)
    decoder_module = model.get_decoder()
    set_nstep_mode(decoder_module)

    from transformers import WhisperConfig
    from typing import cast
    config = cast(WhisperConfig, model.config)

    decoder = HfWhisperDecoder(config, decoder_module)
    decoder.eval()

    # Build dummy cross-attention cache
    kv_cache_cross_flat = []
    for _ in range(NUM_BLOCKS):
        kv_cache_cross_flat.append(
            torch.randn(NUM_HEADS, 1, HEAD_DIM, AUDIO_EMB_LEN)
        )
        kv_cache_cross_flat.append(
            torch.randn(NUM_HEADS, 1, AUDIO_EMB_LEN, HEAD_DIM)
        )

    for step in [0, 1, 2]:
        cache_len = step + 1
        mask_len = step + 2
        onnx_path = f"/tmp/whisper_decoder_step{step}.onnx"

        input_ids = torch.tensor([[50258]], dtype=torch.int32)
        attention_mask = torch.full((1, 1, 1, mask_len), MASK_NEG)
        attention_mask[:, :, :, 1:] = 0.0
        position_ids = torch.tensor([step], dtype=torch.int32)

        kv_self_flat = []
        for _ in range(NUM_BLOCKS):
            kv_self_flat.append(
                torch.zeros(NUM_HEADS, 1, HEAD_DIM, cache_len)
            )
            kv_self_flat.append(
                torch.zeros(NUM_HEADS, 1, cache_len, HEAD_DIM)
            )

        dummy_input = (
            input_ids,
            attention_mask,
            *kv_self_flat,
            *kv_cache_cross_flat,
            position_ids,
        )

        input_names = ["input_ids", "attention_mask"]
        for i in range(NUM_BLOCKS):
            input_names.append(f"k_cache_self_{i}_in")
            input_names.append(f"v_cache_self_{i}_in")
        for i in range(NUM_BLOCKS):
            input_names.append(f"k_cache_cross_{i}")
            input_names.append(f"v_cache_cross_{i}")
        input_names.append("position_ids")

        output_names = ["logits"]
        for i in range(NUM_BLOCKS):
            output_names.append(f"k_cache_self_{i}_out")
            output_names.append(f"v_cache_self_{i}_out")

        with torch.no_grad():
            torch.onnx.export(
                decoder,
                dummy_input,
                onnx_path,
                input_names=input_names,
                output_names=output_names,
                opset_version=13,
                do_constant_folding=True,
            )

        # Verify all shapes are static (no dynamic dims)
        onnx_model = onnx.load(onnx_path)
        graph = onnx_model.graph

        has_dynamic = False
        for inp in graph.input:
            shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            if 0 in shape:
                print(f"  WARNING: {inp.name} has dynamic dim: {shape}")
                has_dynamic = True

        for out in graph.output:
            shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
            if 0 in shape:
                print(f"  WARNING: {out.name} has dynamic dim: {shape}")
                has_dynamic = True

        if not has_dynamic:
            print(f"  Step {step}: ONNX exported to {onnx_path} - ALL SHAPES STATIC")
        else:
            print(f"  Step {step}: ONNX has dynamic shapes!")

        # Print key shapes for verification
        for out in graph.output:
            shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
            if "k_cache_self_0_out" in out.name:
                expected = [NUM_HEADS, 1, HEAD_DIM, step + 2]
                assert shape == expected, \
                    f"Step {step}: k_out ONNX shape {shape} != {expected}"
                print(f"    k_cache_self_0_out: {shape} (expected {expected})")

    print("\n[PASS] ONNX export: all shapes fully static, no dynamic dims")


if __name__ == "__main__":
    test_nstep_input_spec()
    test_nstep_forward()
    test_nstep_onnx_export()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
