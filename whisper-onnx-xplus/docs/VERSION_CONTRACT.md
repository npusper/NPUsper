# Version Contract

Use these versions when reproducing the X Plus ONNX/QNN path.

## Linux Compile Host

- Python: `3.10` or `3.11`
- NumPy: `1.26.4`
- PyTorch: `2.8.0`
- ONNX: `1.18.0`
- ONNXScript: `0.4.0`
- Transformers: `4.53.3`
- QAI Hub: `0.45.0`
- QAI Hub Models: `0.46.1`
- QAIRT/QNN SDK: `2.37.1.250807`

The exact Python package pins are in `../requirements-compile.txt`.

## X Plus Runtime Host

- Windows: ARM64
- Python: `3.11` ARM64, invoked as `py -3.11-arm64`
- ONNX Runtime QNN Python package: `onnxruntime-qnn==1.23.2`
- ONNX Runtime QNN C++ package: `Microsoft.ML.OnnxRuntime.QNN` NuGet `1.23.2`
- QNN runtime DLLs: from the ONNX Runtime QNN NuGet package, matching QNN `2.37.1`

Use the NuGet package for C++ headers, `onnxruntime.lib`, and deploy-local runtime DLLs. The Python package is kept for ARM64 Python-side ONNX Runtime sanity checks.
