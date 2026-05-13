# KEEP_LIST

This repo keeps only the X Plus NPU deploy-mode3 reproduction path.

## Keep

- `compile/*.py`: Linux-side ONNX export, QAIRT conversion/linking, and packaging.
- `compile/qai_hub_models_patch/`: required Whisper adaptation patch.
- `runtime/whisper-onnx/`: minimal C++ ONNX Runtime QNN runtime for `ours_streaming.exe`.
- `runtime/whisper-onnx/models/base/{dims.txt,vocab.txt,mel_filters.bin}`: small support files used by packaging.
- `deploy_templates/mode3/config.json`: reference final deploy configuration.
- `requirements-compile.txt`: Linux compile environment.
- `requirements-xplus-runtime.txt`: Windows ARM64 runtime Python package.
- `docs/NPU_DEPLOY_MODE3_GUIDE.md`: first-time deploy mode3 build and run guide.
- `docs/VERSION_CONTRACT.md`: pinned compile/runtime version contract.

## Exclude

- Generated QNN context binaries: `*.bin`.
- Intermediate model files: large `*.onnx`, `*.onnx.data`, `*.dlc`.
- Runtime DLLs copied from ORT/QNN packages.
- Datasets, WAV files, profiler outputs, raw logs, and result dumps.
- S25, RubikPi, infmasking, and ablation-only research files.
