# whisper-ggml

Streaming speech recognition based on whisper.cpp, with multi-device build support.

## Streaming Systems

| Binary | System |
|--------|--------|
| `whisper_streaming_cpp` | Whisper-Streaming |
| `whisper_streaming_cpp_optimized` | WhisperFlow |
| `simul_streaming` | SimulStreaming (AlignAtt) |
| `simul_whisper` | Simul-Whisper (AlignAtt + CIF) |
| `ours_streaming` | Ours |

---

## Prerequisites

- CMake >= 3.14
- C++17 compiler
- CUDA toolkit (for NVIDIA GPU build)
- Python 3 with `numpy`, `soundfile`, `jiwer` (for evaluation)

---

## Clone

```bash
cd live-transcription/whisper-ggml

# Initialize vendor/ggml submodule
git submodule update --init vendor/ggml
```

---

## Models

### Whisper ggml model

```bash
bash models/download-ggml-model.sh base
# Available: tiny, base, small, medium
```

### CIF model (SimulWhisper only)

Already included in the repo at `models/cif_base.bin`.

---

## Build

Build output goes to `build/<target>/bin/`. Currently supported targets:

### Linux (CUDA)

```bash
cmake -S . -B build/linux -DCMAKE_BUILD_TYPE=Release -DWHISPER_CUDA=ON
cmake --build build/linux -j$(nproc)
```

> **Note**: Use `-DWHISPER_CUDA=ON`, not `-DGGML_CUDA=ON`.
> `WHISPER_CUDA=ON` internally sets `GGML_CUDA=ON`.

### CPU-only

```bash
cmake -S . -B build/linux -DCMAKE_BUILD_TYPE=Release
cmake --build build/linux -j$(nproc)
```

### Rebuild a single binary (after code change)

```bash
cmake --build build/linux --target ours_streaming -j$(nproc)
```

### Verify GPU is enabled

After build, check `build/linux/CMakeCache.txt`:

```
WHISPER_CUDA:BOOL=ON
GGML_CUDA:BOOL=ON
```

Or run any binary and confirm the output contains `use gpu    = 1`.

---

## Evaluation (N-way comparison)

Run from the `eval/` directory. Specify `--bin-dir` to point to the built binaries.

### Basic run

```bash
cd eval

python run_streaming_comparison.py \
  --model base \
  --num_samples 10 \
  --step 2000 \
  --no-realtime \
  --bin-dir build/linux/bin \
  --whisper-streaming \
  --whisperflow \
  --simul-streaming \
  --simul-whisper \
  --ours-streaming \
  --ours-carryover-mode 3
```

### Key flags

| Flag | Description |
|------|-------------|
| `--model` | `tiny`, `base`, `small`, `medium` |
| `--num_samples` | Number of LibriSpeech samples to evaluate |
| `--step` | Chunk size in ms (e.g. `2000`) |
| `--no-realtime` | Skip real-time sleep (faster simulation) |
| `--bin-dir` | Path to built binaries (relative to `whisper-ggml/`) |
| `--whisper-streaming` | Enable Whisper-Streaming |
| `--whisperflow` | Enable WhisperFlow |
| `--simul-streaming` | Enable SimulStreaming |
| `--simul-whisper` | Enable SimulWhisper |
| `--ours-streaming` | Enable Ours |
| `--ours-carryover-mode` | default: `3` |
| `--ours-word-end-offset` | Carryover offset in sec (default `-0.2`) |

### Results

Saved to `eval/comparison_results/comparison_<datetime>_<model>_<N>samples_step<step>/`.

---

## Project Structure

```
whisper-ggml/
  whisper.h / whisper.cpp             # Core whisper implementation
  vendor/ggml/                        # Vendored ggml (git submodule)
  adapter/ggml-adapter.h              # API compatibility shim
  examples/
    whisper_streaming_cpp/            # Whisper-Streaming
    whisper_streaming_cpp_optimized/  # WhisperFlow
    simul_streaming/                  # SimulStreaming
    simul_whisper/                    # SimulWhisper
    ours_streaming/                   # Ours
  models/                             # ggml model files + CIF weights
  build/
    linux/                            # Linux (CUDA) build output
    sm8650p/                          # Qualcomm SM8650P (cross-compile)
    rubikpi/                          # Rubik Pi (cross-compile)
    galaxy_s25/                       # Galaxy S25 (cross-compile)
    qlaptop/                          # Qualcomm laptop (cross-compile)
  cmake/toolchains/                   # Cross-compilation toolchain files
  eval/
    run_streaming_comparison.py       # N-way comparison script
    comparison_results/               # Evaluation results (gitignored)
```
