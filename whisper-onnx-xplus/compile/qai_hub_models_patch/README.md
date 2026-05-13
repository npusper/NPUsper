# qai_hub_models patch for N-Step Whisper Decoder

## Base Package
- **Package**: `qai_hub_models` (PyPI: `qai-hub-models`)
- **Version**: `0.46.1`
- **Source**: https://github.com/qualcomm/ai-hub-models/tree/v0.46.1
- **Target path**: `qai_hub_models/models/_shared/hf_whisper/`

## Why

Original qai_hub_models uses a fixed sliding-window KV cache with dynamic reshape (`-1`).
QNN HTP (NPU) requires fully static shapes.
This patch adds "n-step mode" where the KV cache grows by exactly 1 each step,
enabling per-step compilation with fully static shapes.

## Files

```
original/                      <-- qai_hub_models v0.46.1 originals
  model_adaptation.py
  model.py

modified/                      <-- our patched versions
  model_adaptation.py
  model.py
  test_nstep.py                <-- new file (n-step test)

model_adaptation.diff          <-- diff: original -> modified
model.diff                     <-- diff: original -> modified
```

## Changes Summary

### model_adaptation.py (2 changes)
1. Added `nstep_mode` branch in `SHAAttention.forward()` -- skips KV trim/reshape
2. Added `set_nstep_mode()` function at module level

### model.py (1 change)
1. Added `HfWhisperDecoder.get_input_spec_for_step()` static method for step-specific input specs

### test_nstep.py (new)
- Tests n-step forward pass, KV cache growth, ONNX static shape export

## Apply

```bash
pip install qai-hub-models==0.46.1
SITE=$(python -c "import qai_hub_models; print(qai_hub_models.__path__[0])")
cp modified/model_adaptation.py "$SITE/models/_shared/hf_whisper/"
cp modified/model.py "$SITE/models/_shared/hf_whisper/"
cp modified/test_nstep.py "$SITE/models/_shared/hf_whisper/"
```
