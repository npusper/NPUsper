# NPUsper NPU

ONNX Runtime QNN path for running NPUsper on the Snapdragon X Plus NPU.

This directory contains the Linux build scripts, pinned Python dependencies, QAIRT helper code, final runtime configuration, and the Windows ARM64 C++ runtime source.

## Scope

- Model: `openai/whisper-base`
- Target device: Snapdragon X Plus
- QNN profile: Hexagon `v73`, `soc_model=60`
- Offline compiler: QAIRT/QNN `2.37.1.250807`
- Runtime: ONNX Runtime QNN `1.23.2`
- Decoder design: bucketed N/K decoder with 30s fallback ( = Controlled Unrolling )
- N/K schedule: `K_normal=[4,6,5,5,5,5]`, `K_prefill=[6,4,5,5,5,5]`

## Layout

```text
compile_encoder_xplus.py          Export encoder ONNX and build QNN context binaries
compile_decoder_1step_xplus.py    Build the 30s fallback decoder
compile_decoder_nk_full.py        Build the bucketed N/K decoder family
package_nk_full_deploy.py         Stage files for the laptop runtime directory
qai_hub_models_patch/             Patch applied before ONNX export
qairt_paths.py                    QAIRT path discovery helper
reference_config/config.json      Reference final runtime configuration
requirements-compile.txt          Linux compile environment pins
requirements-xplus-runtime.txt    Windows ARM64 runtime Python pins
whisper-onnx/                     Windows ARM64 ONNX Runtime QNN runtime
docs/                             Detailed setup and version notes
```

## Linux Build Host

Create the compile environment from this directory:

```bash
python3.10 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements-compile.txt
.venv/bin/python patch_qai_hub_models.py
```

Install Qualcomm AI Runtime SDK / QNN SDK `2.37.1.250807`. This SDK is an external dependency and is not included in the repository.

Use either `QNN_SDK_ROOT`:

```bash
export QNN_SDK_ROOT=/path/to/qairt/2.37.1.250807
```

or this repo-local layout:

```text
npu/qairt/2.37.1.250807/
```

Verify the two required Linux tools:

```bash
test -x "$QNN_SDK_ROOT/bin/x86_64-linux-clang/qairt-converter"
test -x "$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-context-binary-generator"
```

## Build NPU Artifacts

Compile encoder buckets from 2s to 10s and the 30s fallback encoder:

```bash
for b in 2 3 4 5 6 7 8 9 10 30; do
  .venv/bin/python compile_encoder_xplus.py \
    --model base \
    --target xplus \
    --audio-sec ${b}.0
done
```

Compile the 30s 1-step decoder:

```bash
.venv/bin/python compile_decoder_1step_xplus.py \
  --model base \
  --target xplus \
  --audio-sec 30.0
```

Compile and link the bucketed N/K decoder:

```bash
.venv/bin/python compile_decoder_nk_full.py \
  --model base \
  --target xplus \
  --all-buckets
```

Stage the runtime files:

```bash
.venv/bin/python package_nk_full_deploy.py \
  --model base \
  --target xplus \
  --buckets all \
  --with-30s \
  --stage-dir ./artifacts/xplus_npu_runtime
```

Expected staged files:

- Encoder QNN context binaries and EPContext ONNX wrappers for `2s..10s` and `30s`.
- Bucketed decoder QNN context binaries and EPContext ONNX wrappers for `2s..10s`.
- 30s fallback decoder context binary and wrapper.
- `config.json`, `dims.txt`, `vocab.txt`, and `mel_filters.bin`.

Copy the staged directory to the laptop:

```bash
ssh <xplus-host> 'powershell -NoProfile -Command "New-Item -ItemType Directory -Force $HOME\Desktop\npusper_npu | Out-Null"'
scp artifacts/xplus_npu_runtime/* <xplus-host>:Desktop/npusper_npu/
```

## Windows ARM64 Runtime

Run these commands on the X Plus laptop.

Install Python runtime dependencies with ARM64 Python:

```powershell
py -0p
py -3.11-arm64 -m pip install -r <repo>\npu\requirements-xplus-runtime.txt
```

Download ONNX Runtime QNN `1.23.2` from NuGet:

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

Build `ours_streaming.exe` and copy it into the runtime directory:

```powershell
cd <repo>\npu\whisper-onnx
$env:CL = "/utf-8 $env:CL"

powershell -ExecutionPolicy Bypass -File .\build_ours_streaming_xplus.ps1 `
  -OrtRoot $OrtRoot `
  -DeployDir "$env:USERPROFILE\Desktop\npusper_npu"
```

The build script also copies ONNX Runtime and QNN DLLs from the NuGet package into the runtime directory.

## Run

Generate a smoke-test WAV if needed:

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

Use real speech audio for transcription quality checks. The generated WAV only verifies that the runtime package loads and executes.

## Notes

- Compile-time QAIRT and runtime QNN DLL versions must match.
- Do not commit generated `.bin`, `.dlc`, large `.onnx`, DLL, WAV, log, or result files.
- See `docs/VERSION_CONTRACT.md` for the pinned package stack.
