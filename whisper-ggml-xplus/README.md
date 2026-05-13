# whisper-ggml-xplus

GGML / whisper.cpp runtime for running NPUsper with the Adreno OpenCL backend on a Snapdragon X Plus laptop.

This directory is the GPU-side reproduction path. It is separate from `whisper-onnx-xplus`, which targets the QNN NPU deploy flow.

## What This Contains

```text
CMakeLists.txt              Build entry point for whisper.cpp + GGML OpenCL
whisper.cpp / whisper.h     Modified Whisper runtime
vendor/ggml/                Vendored GGML source with OpenCL backend support
examples/ours_streaming/    Final NPUsper streaming executable
models/                     GGML model download and conversion helpers
```

Generated model weights, build directories, logs, and WAV files are intentionally excluded.

## Target Machine

Run these steps on the Snapdragon X Plus laptop.

Required tools:

- Windows 11 ARM64
- CMake
- Ninja
- A Windows ARM64 C++ compiler, for example `clang-cl`
- Python 3.11 ARM64
- Qualcomm/Windows OpenCL runtime from the device GPU driver
- OpenCL build-time headers and import library, for example from vcpkg `opencl`

## OpenCL Build Dependency

Install OpenCL headers and the import library with vcpkg:

```powershell
git clone https://github.com/microsoft/vcpkg C:\src\vcpkg
C:\src\vcpkg\bootstrap-vcpkg.bat
C:\src\vcpkg\vcpkg install opencl:arm64-windows
```

The Qualcomm GPU driver provides the runtime implementation on the X Plus laptop. vcpkg provides the files CMake needs at configure time.

## Build

Open a Windows ARM64 developer shell in this directory.

```powershell
cmake -S . -B build\win-arm-opencl -G Ninja `
  -DCMAKE_TOOLCHAIN_FILE=C:\src\vcpkg\scripts\buildsystems\vcpkg.cmake `
  -DVCPKG_TARGET_TRIPLET=arm64-windows `
  -DCMAKE_C_COMPILER=clang-cl `
  -DCMAKE_CXX_COMPILER=clang-cl `
  -DWHISPER_OPENCL=ON `
  -DWHISPER_OPENCL_USE_ADRENO_KERNELS=ON `
  -DCMAKE_BUILD_TYPE=Release

cmake --build build\win-arm-opencl --config Release --target ours_streaming
```

Expected binary:

```text
build\win-arm-opencl\bin\ours_streaming.exe
```

If you install OpenCL manually instead of vcpkg, pass the paths explicitly:

```powershell
-DOpenCL_INCLUDE_DIR=C:\path\to\OpenCL\include `
-DOpenCL_LIBRARY=C:\path\to\OpenCL.lib
```

## OpenMP Runtime DLL

If `ours_streaming.exe` exits with loader error `0xC0000135`, make `libomp140.aarch64.dll` discoverable before running. The DLL is installed by Visual Studio Build Tools with the ARM64 LLVM/OpenMP components.

```powershell
$LibOmp = Get-ChildItem `
  "${env:ProgramFiles}\Microsoft Visual Studio\2022", `
  "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022" `
  -Recurse -Filter libomp140.aarch64.dll -ErrorAction SilentlyContinue |
  Select-Object -First 1
if (-not $LibOmp) { throw "libomp140.aarch64.dll not found. Install VS Build Tools ARM64 LLVM/OpenMP components." }
Copy-Item $LibOmp.FullName .\build\win-arm-opencl\bin\ -Force
```

## Download Model

```powershell
cd models
.\download-ggml-model.cmd base
cd ..
```

Expected model path:

```text
models\ggml-base.bin
```

## Run NPUsper Streaming

Generate a smoke-test WAV if needed:

```powershell
py -3.11-arm64 ..\scripts\make_smoke_wav.py .\test_speech.wav
```

Run the streaming executable. WAV files are positional arguments; this executable does not accept `-f`.

```powershell
.\build\win-arm-opencl\bin\ours_streaming.exe `
  -m .\models\ggml-base.bin `
  --step 2000 `
  --no-realtime `
  .\test_speech.wav
```

GPU execution is enabled by default. Use `-ng` only when you intentionally want to compare against CPU execution. Use real speech audio instead of the generated smoke WAV when validating transcription quality.

## Artifact Policy

Do not commit:

- `build*` directories
- downloaded `models\ggml-*.bin`
- WAV/audio files
- logs, CSV files, or temporary run outputs
