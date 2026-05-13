# NPUsper S25

S25 GPU and NPU runtime source with build and run instructions.

## Layout

```
gpu/
  whisper-ggml/          GGML/whisper.cpp OpenCL runtime source

npu/
  whisper-onnx/          ONNX Runtime + QNN EP C++ runtime source
  compile_encoder_xplus.py
  compile_decoder_1step_xplus.py
  compile_decoder_unroll_k_xplus.py
  compile_decoder_nk_full.py
  package_nk_full_deploy.py
  patch_qai_hub_models.py
  whisper_npu_profile.py
  qai_hub_models_patch/  Local patch for qai-hub-models Whisper export
  requirements*.txt
```

## Common Dependencies

Install these on a Linux x86_64 host:

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake ninja-build git curl unzip \
  python3.10 python3.10-venv python3.10-dev \
  android-tools-adb libsndfile1
```

Set the Android NDK path for both GPU and NPU Android builds:

```bash
export ANDROID_NDK_ROOT=/path/to/android-ndk-r26d
export PATH="$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin:$PATH"
```

## GPU Runtime: GGML + OpenCL

The GPU runtime lives under `gpu/whisper-ggml/`.  Model files are not checked
in; place a local `ggml-base.bin` or another compatible GGML Whisper model in a
local deploy directory.

Build for Android arm64 OpenCL:

```bash
cmake -S gpu/whisper-ggml -B build/gpu-opencl-android \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-29 \
  -DCMAKE_BUILD_TYPE=Release \
  -DWHISPER_OPENCL=ON \
  -DCMAKE_CXX_FLAGS=-DGGML_OPENCL_DISABLE_ADRENO_KQ_KQV

cmake --build build/gpu-opencl-android -j"$(nproc)"
```

If CMake cannot find an Android OpenCL library, copy the device vendor library
to the host and pass it to CMake:

```bash
mkdir -p /tmp/android-opencl
adb pull /vendor/lib64/libOpenCL.so /tmp/android-opencl/libOpenCL.so

cmake -S gpu/whisper-ggml -B build/gpu-opencl-android \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-29 \
  -DCMAKE_BUILD_TYPE=Release \
  -DWHISPER_OPENCL=ON \
  -DOpenCL_LIBRARY=/tmp/android-opencl/libOpenCL.so \
  -DCMAKE_CXX_FLAGS=-DGGML_OPENCL_DISABLE_ADRENO_KQ_KQV
```

Deploy and run a transcription example:

```bash
REMOTE=/data/local/tmp/npusper_gpu
adb shell "rm -rf $REMOTE && mkdir -p $REMOTE/models"
adb push build/gpu-opencl-android/bin/ours_streaming "$REMOTE/"
adb push /path/to/ggml-base.bin "$REMOTE/models/"
adb push /path/to/input.wav "$REMOTE/input.wav"

adb shell "cd $REMOTE && ./ours_streaming \
  -m models/ggml-base.bin \
  --step 2000 \
  --dtw base \
  --carryover-mode 3 \
  input.wav"
```

Use `./ours_streaming --help` on the device for the full option list.

## NPU Model Build: ONNX + QAIRT

The NPU build path uses QAIRT/QNN 2.37 and ONNX Runtime QNN on the target
device. Install the SDKs and runtime shared libraries separately.

```bash
export QNN_SDK_ROOT=/opt/qcom/aistack/qairt/2.37.1.250807
export PATH="$QNN_SDK_ROOT/bin/x86_64-linux-clang:$PATH"

python3.10 -m venv .venv-npu
source .venv-npu/bin/activate
pip install --upgrade pip
pip install -r npu/requirements.txt
python npu/patch_qai_hub_models.py
```

Build encoder and decoder files for the S25-compatible target profile:

```bash
# Bucketed encoders used by the streaming runtime.
for sec in 2 3 4 5 6 7 8 9 10; do
  python npu/compile_encoder_xplus.py \
    --model base \
    --target s25_xplus_compat \
    --audio-sec "$sec"
done

# N/K controlled-unrolling decoder buckets.
python npu/compile_decoder_nk_full.py \
  --model base \
  --target s25_xplus_compat \
  --all-buckets

# Optional 30-second reinference path.
python npu/compile_encoder_xplus.py \
  --model base \
  --target s25_xplus_compat \
  --audio-sec 30

python npu/compile_decoder_1step_xplus.py \
  --model base \
  --target s25_xplus_compat \
  --audio-sec 30
```

Stage the generated context binaries and ONNX wrappers into a local deploy
directory:

```bash
python npu/package_nk_full_deploy.py \
  --model base \
  --target s25_xplus_compat \
  --stage-dir /tmp/npusper_npu_deploy \
  --with-30s
```

Copy the stage directory and the required ONNX Runtime/QNN shared libraries to
the device deploy directory.  The exact shared library paths depend on your
local ONNX Runtime Android package and QAIRT SDK install:

```bash
REMOTE=/data/local/tmp/npusper_npu
adb shell "rm -rf $REMOTE && mkdir -p $REMOTE"
adb push /tmp/npusper_npu_deploy/. "$REMOTE/"
adb push /path/to/libonnxruntime.so "$REMOTE/"
adb push "$QNN_SDK_ROOT/lib/aarch64-android/libQnnHtp.so" "$REMOTE/"
adb push "$QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpPrepare.so" "$REMOTE/"
adb push "$QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpV79Stub.so" "$REMOTE/"
adb push "$QNN_SDK_ROOT/lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so" "$REMOTE/"
adb push "$QNN_SDK_ROOT/lib/aarch64-android/libQnnSystem.so" "$REMOTE/"
```

## NPU Runtime: ONNX Runtime + QNN EP

Build the C++ runtime:

```bash
cmake -S npu/whisper-onnx -B build/npu-android \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-29 \
  -DCMAKE_BUILD_TYPE=Release \
  -DORT_ROOT=/path/to/onnxruntime-android-arm64

cmake --build build/npu-android -j"$(nproc)"
```

Deploy and run a transcription example:

```bash
REMOTE=/data/local/tmp/npusper_npu
adb push build/npu-android/bin/ours_streaming "$REMOTE/"
adb push /path/to/input.wav "$REMOTE/input.wav"

adb shell "cd $REMOTE && LD_LIBRARY_PATH=$REMOTE ADSP_LIBRARY_PATH=$REMOTE \
  ./ours_streaming \
    -m . \
    --use-npu \
    --qnn-htp-path $REMOTE/libQnnHtp.so \
    --step 2000 \
    --carryover-mode 3 \
    --aheads-preset base \
    input.wav"
```

Use `./ours_streaming --help` on the device for the full option list.
