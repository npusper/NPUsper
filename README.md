# Live Transcription on Snapdragon X Plus

This branch is a compact reproduction package for the final NPUsper X Plus laptop paths. It keeps only the source code and instructions needed for another researcher or engineer to rebuild and run the final systems.

The branch has two independent tracks:

- [`whisper-onnx-xplus`](./whisper-onnx-xplus): ONNX Runtime + QNN NPU deployment path for deploy mode3.
- [`whisper-ggml-xplus`](./whisper-ggml-xplus): GGML / whisper.cpp path for NPUsper with the Adreno OpenCL backend.

Large generated artifacts are not stored in this branch. Build outputs, QNN context binaries, ONNX exports, GGML model weights, logs, and temporary run outputs should be regenerated from the subdirectory READMEs.

## Intended Hosts

- Model build host: Linux workstation, used for ONNX export, QAIRT conversion/linking, and deploy packaging.
- Runtime host: Snapdragon X Plus Windows ARM64 laptop, reachable over SSH from the build host.

## Start Here

1. Follow `whisper-onnx-xplus/README.md` for the NPU deploy mode3 path.
2. Follow `whisper-ggml-xplus/README.md` for the Adreno OpenCL GGML path.
3. If you do not have a WAV input, generate a smoke-test file:

```bash
python3 scripts/make_smoke_wav.py test_speech.wav
```

The generated WAV is only for runtime plumbing checks. Use real speech audio when validating transcription quality.

If Git Credential Manager blocks HTTPS clone on Windows, use SSH clone or download the branch archive instead:

```powershell
git clone -b laptop_x_plus <repo-url>
```
