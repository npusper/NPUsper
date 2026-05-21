# NPU Deployment Guide

This guide rebuilds the final ONNX Runtime QNN path for NPUsper on a Snapdragon X Plus laptop. The Linux host exports ONNX, converts and links QNN artifacts with QAIRT, then stages a runtime directory for the Windows ARM64 laptop.

## 1. Linux Compile Host

Run from `npu/`:

```bash
python3.10 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements-compile.txt
.venv/bin/python patch_qai_hub_models.py
```

Install Qualcomm AI Runtime SDK / QNN SDK `2.37.1.250807`. The SDK is an external licensed dependency and is not committed to this branch.

Use one of these layouts:

```bash
export QNN_SDK_ROOT=/path/to/qairt/2.37.1.250807
```

or:

```text
npu/qairt/2.37.1.250807/
```

Verify the two required tools:

```bash
test -x "$QNN_SDK_ROOT/bin/x86_64-linux-clang/qairt-converter"
test -x "$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-context-binary-generator"
```

## 2. Compile Encoder Buckets

The final runtime uses encoder buckets from 2s to 10s, plus a 30s encoder for the fallback path.

```bash
for b in 2 3 4 5 6 7 8 9 10 30; do
  .venv/bin/python compile_encoder_xplus.py \
    --model base \
    --target xplus \
    --audio-sec ${b}.0
done
```

## 3. Compile the 30s Decoder

```bash
.venv/bin/python compile_decoder_1step_xplus.py \
  --model base \
  --target xplus \
  --audio-sec 30.0
```

## 4. Compile and Link the Bucketed N/K Decoder

```bash
.venv/bin/python compile_decoder_nk_full.py \
  --model base \
  --target xplus \
  --all-buckets
```

The final schedule is:

```text
K_normal  = [4, 6, 5, 5, 5, 5]
K_prefill = [6, 4, 5, 5, 5, 5]
buckets   = 2s, 3s, 4s, 5s, 6s, 7s, 8s, 9s, 10s
```

## 5. Stage the NPU Runtime Directory

```bash
.venv/bin/python package_nk_full_deploy.py \
  --model base \
  --target xplus \
  --buckets all \
  --with-30s \
  --stage-dir ./artifacts/xplus_npu_runtime
```

The staged directory should contain encoder QNN context binaries, N/K decoder context binaries and EPContext ONNX wrappers, 30s fallback artifacts, `config.json`, `dims.txt`, `vocab.txt`, and `mel_filters.bin`.

## 6. Copy to the X Plus Laptop

Use the SSH alias or hostname for your target X Plus laptop.

```bash
ssh <xplus-host> 'powershell -NoProfile -Command "New-Item -ItemType Directory -Force $HOME\Desktop\npusper_npu | Out-Null"'
scp artifacts/xplus_npu_runtime/* <xplus-host>:Desktop/npusper_npu/
```

## 7. Prepare ONNX Runtime on Windows ARM64

Run these commands on the X Plus laptop. Use ARM64 Python explicitly:

```powershell
py -0p
py -3.11-arm64 -m pip install -r <repo>\npu\requirements-xplus-runtime.txt
```

Download and extract the ONNX Runtime QNN C++ NuGet package. This package provides the C++ headers and `onnxruntime.lib`; the Python package alone is not enough to build `ours_streaming.exe`.

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

## 8. Build the Windows ARM64 Runtime

```powershell
cd <repo>\npu\whisper-onnx
$env:CL = "/utf-8 $env:CL"

powershell -ExecutionPolicy Bypass -File .\build_ours_streaming_xplus.ps1 `
  -OrtRoot $OrtRoot `
  -DeployDir "$env:USERPROFILE\Desktop\npusper_npu"
```

The build script also copies ONNX Runtime and QNN DLLs from the NuGet package into the runtime directory.

## 9. Run

Generate a smoke-test WAV if you do not already have a speech sample:

```powershell
py -3.11-arm64 <repo>\scripts\make_smoke_wav.py "$env:USERPROFILE\Desktop\npusper_npu\test_speech.wav"
```

Run with the deploy-local QNN HTP DLL:

```powershell
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

Use real speech audio instead of the generated smoke WAV when validating transcription quality.

## Version Contract

Use matching compile-time and runtime QNN versions:

- QAIRT/QNN SDK: `2.37.1.250807`
- ONNX Runtime QNN Python package: `onnxruntime-qnn==1.23.2`
- ONNX Runtime QNN C++ package: `Microsoft.ML.OnnxRuntime.QNN` NuGet `1.23.2`

See `VERSION_CONTRACT.md` for the full pinned stack. Do not commit generated `.bin`, `.dlc`, large `.onnx`, DLL, WAV, log, or result files.
