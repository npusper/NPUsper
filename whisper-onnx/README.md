# whisper-onnx

Streaming speech recognition based on ONNX Runtime, ported from whisper-ggml.

## Streaming Systems

| Binary | System |
|--------|--------|
| `ours_streaming` | Ours |

---

## Prerequisites

- CMake >= 3.17
- C++17 compiler
- ONNX Runtime (GPU or CPU)
- Python 3 with `numpy`, `soundfile`, `jiwer` (for evaluation)

---

## Models

ONNX models are stored as directories (not single files like ggml).
Place models under `models/` (e.g., `models/base/`, `models/small/`).

---

## Build

Build output goes to `build/linux/bin/`.

### First build

```bash
cd whisper-onnx
cmake -S . -B build/linux -DCMAKE_BUILD_TYPE=Release \
      -DORT_ROOT=~/onnxruntime-linux-x64-gpu-1.23.0
cmake --build build/linux -j$(nproc)
```

### Rebuild after code change

```bash
cmake --build build/linux --target ours_streaming -j$(nproc)
```

---

## Evaluation

Run from the `eval/` directory.

### Example

```bash
cd eval

python run_streaming_comparison.py \
  --model base \
  --step 2000 \
  --no-realtime \
  --num_samples 100 \
  --ours-streaming \
  --ours-carryover-mode 3 \
  --ours-debug \
  --bin-dir build/linux/bin \
  --no_gpu \
  --simul-whisper
```

### Key flags

| Flag | Description |
|------|-------------|
| `--model` | `tiny`, `base`, `small`, `medium` |
| `--num_samples` | Number of samples to evaluate |
| `--step` | Chunk size in ms (e.g. `2000`) |
| `--no-realtime` | Skip real-time sleep (faster simulation) |
| `--dataset` | `tedlium` (default: auto) |
| `--bin-dir` | Path to built binaries (relative to `whisper-onnx/`) |
| `--no_gpu` | Disable GPU (CPU only) |
| `--ours-streaming` | Enable Ours |
| `--ours-carryover-mode` | default: `3` |
| `--ours-word-end-offset` | Carryover offset in sec (default `-0.2`) |
| `--ours-debug` | Enable debug output (per-round details to stderr) |
| `--simul-whisper` | Enable SimulWhisper baseline |

### Results

Saved to `eval/comparison_results/comparison_<datetime>_<model>_<N>samples_step<step>/`.

---

## Project Structure

```
whisper-onnx/
  whisper_ort.h / whisper_ort.cpp   # Core whisper ONNX Runtime wrapper
  examples/
    ours_streaming/                 # Ours
  models/                           # ONNX model directories
  build/
    linux/                          # Linux build output
  eval/
    run_streaming_comparison.py     # N-way comparison script
    comparison_results/             # Evaluation results
```
