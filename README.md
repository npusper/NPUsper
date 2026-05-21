# NPUsper X Plus

Snapdragon X Plus laptop artifact for the final NPUsper GPU and NPU paths.

- `gpu/whisper-ggml`: GGML / whisper.cpp runtime using the Adreno OpenCL backend.
- `npu/whisper-onnx`: ONNX Runtime QNN runtime using QNN context binaries compiled with QAIRT.

Generated artifacts are not stored here. Rebuild model weights, ONNX exports, QNN context binaries, DLL copies, WAV files, logs, and result files locally.

## Layout

```text
gpu/whisper-ggml/        GGML + OpenCL runtime source for Adreno GPU
npu/                     Linux-side ONNX export, QAIRT conversion, and packaging scripts
npu/whisper-onnx/        Windows ARM64 ONNX Runtime QNN runtime source
npu/reference_config/    Reference final NPU runtime configuration
npu/docs/                Detailed NPU setup and pinned version notes
scripts/                 Small shared utilities, including smoke-test WAV generation
```

## Target Machines

- Linux build host: exports Whisper ONNX, converts models with QAIRT, links QNN context binaries, and stages the NPU runtime package.
- Snapdragon X Plus Windows ARM64 laptop: builds and runs both final runtimes.

Use ARM64 Python on the laptop. On Windows, check available interpreters with:

```powershell
py -0p
```

The commands below use `py -3.11-arm64` when Python runs on the X Plus laptop.

## Common Smoke WAV

The repository includes a deterministic WAV generator for checking runtime plumbing:

```bash
python3 scripts/make_smoke_wav.py test_speech.wav
```

Use real speech audio for transcription quality checks. The generated WAV only verifies that the binary, model files, and backend libraries load correctly.

## GPU Runtime: GGML + OpenCL

Run these steps on the X Plus laptop.

Required tools:

- Windows 11 ARM64
- CMake and Ninja
- ARM64 C++ compiler, for example `clang-cl`
- Python 3.11 ARM64
- Qualcomm/Windows OpenCL runtime from the device GPU driver
- OpenCL headers and import library for CMake configure time

Install the OpenCL development files with vcpkg:

```powershell
git clone https://github.com/microsoft/vcpkg C:\src\vcpkg
C:\src\vcpkg\bootstrap-vcpkg.bat
C:\src\vcpkg\vcpkg install opencl:arm64-windows
```

Build:

```powershell
cd <repo>\gpu\whisper-ggml

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

Download the GGML base model and run:

```powershell
cd models
.\download-ggml-model.cmd base
cd ..

py -3.11-arm64 ..\..\scripts\make_smoke_wav.py .\test_speech.wav

.\build\win-arm-opencl\bin\ours_streaming.exe `
  -m .\models\ggml-base.bin `
  --step 2000 `
  --no-realtime `
  .\test_speech.wav
```

If the executable fails with loader error `0xC0000135`, copy `libomp140.aarch64.dll` from the ARM64 Visual Studio Build Tools installation into `build\win-arm-opencl\bin\`. The GPU README has the exact PowerShell helper.

See `gpu/whisper-ggml/README.md` for the full GPU notes.

## NPU Model Build: ONNX + QAIRT

Run these steps on the Linux build host.

Required tools:

- Python 3.10 or 3.11
- Qualcomm AI Runtime SDK / QNN SDK `2.37.1.250807`

Create the build environment:

```bash
cd npu

python3.10 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements-compile.txt
.venv/bin/python patch_qai_hub_models.py
```

Point the scripts to QAIRT:

```bash
export QNN_SDK_ROOT=/path/to/qairt/2.37.1.250807
```

or place the SDK under:

```text
npu/qairt/2.37.1.250807/
```

Build the X Plus NPU artifacts:

```bash
for b in 2 3 4 5 6 7 8 9 10 30; do
  .venv/bin/python compile_encoder_xplus.py \
    --model base --target xplus --audio-sec ${b}.0
done

.venv/bin/python compile_decoder_1step_xplus.py \
  --model base --target xplus --audio-sec 30.0

.venv/bin/python compile_decoder_nk_full.py \
  --model base --target xplus --all-buckets

.venv/bin/python package_nk_full_deploy.py \
  --model base --target xplus --buckets all --with-30s \
  --stage-dir ./artifacts/xplus_npu_runtime
```

The staged directory contains encoder context binaries, decoder context binaries, EPContext ONNX wrappers, `config.json`, `dims.txt`, `vocab.txt`, and `mel_filters.bin`.

Copy the staged files to the X Plus laptop:

```bash
ssh <xplus-host> 'powershell -NoProfile -Command "New-Item -ItemType Directory -Force $HOME\Desktop\npusper_npu | Out-Null"'
scp artifacts/xplus_npu_runtime/* <xplus-host>:Desktop/npusper_npu/
```

See `npu/README.md` and `npu/docs/NPU_DEPLOYMENT_GUIDE.md` for a step-by-step NPU walkthrough.

## NPU Runtime: ONNX Runtime + QNN

Run these steps on the X Plus laptop after copying the staged NPU files.

Install runtime Python dependencies:

```powershell
py -3.11-arm64 -m pip install -r <repo>\npu\requirements-xplus-runtime.txt
```

Download the ONNX Runtime QNN C++ NuGet package. The Python package is not enough for building the C++ executable because it does not provide the C++ headers or `onnxruntime.lib`.

```powershell
$Deps = "$env:USERPROFILE\deps"
$OrtRoot = "$Deps\Microsoft.ML.OnnxRuntime.QNN.1.23.2"
$OrtZip = "$Deps\Microsoft.ML.OnnxRuntime.QNN.1.23.2.zip"
New-Item -ItemType Directory -Force $Deps | Out-Null
Invoke-WebRequest `
  -Uri "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.QNN/1.23.2" `
  -OutFile $OrtZip
Expand-Archive -Force $OrtZip $OrtRoot
```

Build the runtime. The `/utf-8` compiler flag avoids source encoding failures on non-English Windows locales.

```powershell
cd <repo>\npu\whisper-onnx
$env:CL = "/utf-8 $env:CL"

powershell -ExecutionPolicy Bypass -File .\build_ours_streaming_xplus.ps1 `
  -OrtRoot $OrtRoot `
  -DeployDir "$env:USERPROFILE\Desktop\npusper_npu"
```

Run:

```powershell
py -3.11-arm64 <repo>\scripts\make_smoke_wav.py "$env:USERPROFILE\Desktop\npusper_npu\test_speech.wav"

cd "$env:USERPROFILE\Desktop\npusper_npu"
.\ours_streaming.exe `
  -m . `
  --use-npu `
  --qnn-htp-path .\QnnHtp.dll `
  --no-realtime `
  --step 2000 `
  --carryover-mode 3 `
  --aheads-preset base `
  test_speech.wav
```