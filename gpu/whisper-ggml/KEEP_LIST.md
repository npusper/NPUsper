# Keep List

This directory keeps only the files needed to build and run the final GGML + OpenCL path on the X Plus laptop.

## Included

- `CMakeLists.txt`, `cmake/`, `vendor/ggml/`: build system and vendored GGML source required for OpenCL.
- `whisper.cpp`, `whisper.h`, `adapter/`: runtime source used by the final executable.
- `examples/ours_streaming/`: final NPUsper streaming executable.
- `models/`: GGML model download and conversion helpers.

## Excluded

- Comparison baselines and intermediate research scripts.
- Internal run notes.
- Downloaded model weights and generated build outputs.
- Audio samples, evaluation outputs, logs, and temporary files.
