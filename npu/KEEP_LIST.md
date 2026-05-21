# Keep List

This directory keeps only the files needed to rebuild and run the final X Plus NPU path.

## Included

- `compile_*.py`, `package_nk_full_deploy.py`, `qairt_paths.py`, `whisper_npu_profile.py`: Linux-side ONNX export, QAIRT conversion/linking, and runtime staging.
- `qai_hub_models_patch/`: required Whisper adaptation patch applied before ONNX export.
- `whisper-onnx/`: Windows ARM64 ONNX Runtime QNN runtime for `ours_streaming.exe`.
- `whisper-onnx/models/base/{dims.txt,vocab.txt,mel_filters.bin}`: small support files used by the staging script.
- `reference_config/config.json`: reference final NPU runtime configuration.
- `requirements-compile.txt`: Linux compile environment pins.
- `requirements-xplus-runtime.txt`: Windows ARM64 runtime Python pins.
- `docs/NPU_DEPLOYMENT_GUIDE.md`: first-time NPU build and run guide.
- `docs/VERSION_CONTRACT.md`: pinned compile/runtime version contract.

## Excluded

- Generated QNN context binaries: `*.bin`.
- Intermediate model files: large `*.onnx`, `*.onnx.data`, `*.dlc`.
- Runtime DLLs copied from ONNX Runtime/QNN packages.
- Datasets, WAV files, profiler outputs, raw logs, and result dumps.
- S25, RubikPi, infmasking, and ablation-only research files.
