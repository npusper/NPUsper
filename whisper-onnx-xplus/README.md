# whisper-onnx-xplus

Reproducible Snapdragon X Plus NPU path for the NPUsper deploy-mode3 runtime.

This directory contains only the Linux-side model compile/link/package scripts and the Windows ARM64 ONNX Runtime QNN runtime needed to reproduce a deploy-mode3 directory on the X Plus laptop.

## Scope

- Model: `openai/whisper-base`
- Target: Snapdragon X Plus / Hexagon `v73`, `soc_model=60`
- Runtime: ONNX Runtime QNN `1.23.2`
- Offline compiler: QAIRT/QNN `2.37.1.250807`
- Decoder mode: bucketed N/K, mode3 carryover + 30s reinference fallback
- N/K schedule: `K_normal=[4,6,5,5,5,5]`, `K_prefill=[6,4,5,5,5,5]`

## Layout

```text
compile/                 Linux export, QAIRT convert/link, deploy packaging
runtime/whisper-onnx/    Windows ARM64 C++ runtime for ours_streaming.exe
deploy_templates/mode3/  Reference final config for deploy_mode3
docs/                    Step-by-step deploy mode3 guide and version contract
artifacts/               Generated outputs, ignored by git
```

For the full walkthrough, see `docs/NPU_DEPLOY_MODE3_GUIDE.md`.

## Linux Compile Host

Install Python dependencies and patch `qai_hub_models` in the same interpreter that will run the compile scripts:

```bash
python3.10 -m venv whisper_env
whisper_env/bin/pip install --upgrade pip
whisper_env/bin/pip install -r requirements-compile.txt
whisper_env/bin/python compile/patch_qai_hub_models.py
```

Install QAIRT/QNN SDK `2.37.1.250807`. The SDK is not committed to this branch.

Use either `$QNN_SDK_ROOT`:

```bash
export QNN_SDK_ROOT=/path/to/qairt/2.37.1.250807
```

or place it at:

```text
whisper-onnx-xplus/qairt/2.37.1.250807/
```

Build the deploy-mode3 artifacts:

```bash
for b in 2 3 4 5 6 7 8 9 10 30; do
  whisper_env/bin/python compile/compile_encoder_xplus.py \
    --model base --target xplus --audio-sec ${b}.0
done

whisper_env/bin/python compile/compile_decoder_1step_xplus.py \
  --model base --target xplus --audio-sec 30.0

whisper_env/bin/python compile/compile_decoder_nk_full.py \
  --model base --target xplus --all-buckets

whisper_env/bin/python compile/package_nk_full_deploy.py \
  --model base --target xplus --buckets all --with-30s \
  --stage-dir ./artifacts/deploy_mode3
```

The staged directory should contain encoder bins, N/K decoder bins, the 30s 1-step decoder, EPContext wrappers, `config.json`, `dims.txt`, `vocab.txt`, and `mel_filters.bin`.

## Push to X Plus

```bash
scp artifacts/deploy_mode3/* <xplus-host>:Desktop/deploy_mode3/
```

## Build Runtime on X Plus

On the Windows ARM64 X Plus laptop, install the runtime Python package with ARM64 Python:

```powershell
py -3.11-arm64 -m pip install -r <repo>\whisper-onnx-xplus\requirements-xplus-runtime.txt
```

Download the ONNX Runtime QNN C++ NuGet package. This is required for C++ headers and `onnxruntime.lib`; the pip package does not provide them.

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

Build and copy `ours_streaming.exe` into the deploy directory:

```powershell
cd <repo>\whisper-onnx-xplus\runtime\whisper-onnx
powershell -ExecutionPolicy Bypass -File .\build_ours_streaming_xplus.ps1 `
  -OrtRoot $OrtRoot `
  -DeployDir "$env:USERPROFILE\Desktop\deploy_mode3"
```

The build script also copies ONNX Runtime and QNN DLLs from the NuGet package into the deploy directory.

## Run

```powershell
py -3.11-arm64 <repo>\scripts\make_smoke_wav.py "$env:USERPROFILE\Desktop\deploy_mode3\test_speech.wav"

cd "$env:USERPROFILE\Desktop\deploy_mode3"
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

The generated WAV is only for smoke testing. Use real speech audio when validating transcription quality.

## Notes

- Do not commit generated `.bin`, `.dlc`, large `.onnx`, DLL, WAV, or log files.
- Compile-time QAIRT and runtime QNN DLL versions must match.
- `deploy_templates/mode3/config.json` is a reference snapshot of the working X Plus deploy-mode3 configuration.
