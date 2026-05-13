# whisper_streaming_cpp

This is an implementation of whisper_streaming on top of whisper.cpp.
It follows the approaches of whisper_streaming, holds and actively maintains audio_buffer, transcript_buffer and prompt.

params related:
-kc for prompt context
-ng for no gpu execution
-dtw medium for token-level timestamp, which is required for our method
--step 3000 for setting step length to 3000, which is the default step size. We can also set other step size in milisecond.

```bash
# with prompt on GPU
./whisper_streaming_cpp -m models/ggml-medium.bin samples/bernie4min.wav -kc -dtw medium --step 3000 > streaming_log.log 2>&1
# with prompt on CPU
./whisper_streaming_cpp -m models/ggml-medium.bin samples/bernie4min.wav -kc -ng -dtw medium --step 3000 > streaming_log.log 2>&1
# without prompt on GPU
./whisper_streaming_cpp -m models/ggml-medium.bin samples/bernie4min.wav -dtw medium --step 3000 > streaming_log.log 2>&1
```

## Building

```bash
make clean
make
make whisper_streaming_cpp
```

