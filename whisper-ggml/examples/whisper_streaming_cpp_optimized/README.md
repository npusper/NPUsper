# whisper_streaming_cpp

This is our optimized version of whisper_streaming on top of whisper.cpp
It follows the approaches of whisper_streaming, holds and actively maintains audio_buffer, transcript_buffer and prompt.
Beyond that, it also implement the encoding context reduction technique, decoding beam reduce technique, and encoding / prompting / decoding pipelining on CPU / GPU (ongoing).

params related:
-kc for prompt context.
-ng for no gpu execution.
-dtw medium for token-level timestamp, which is required for our method.
--step 3000 for setting step length to 3000, which is the default step size. We can also set other step size in milisecond.
-ac -1 for no zero padding for audio.
-at "audio_tag_file_path" for adding our audio tag at the end of the audio.

```bash
# execution with all the optimization
./whisper_streaming_cpp_optimized -m models/ggml-medium.bin samples/bernie4min.wav -ac -1 -at "audio_tag_file_path" -kc -dtw medium --step 3000 > streaming_log.log 2>&1
```

## Building

```bash
make clean
make
make whisper_streaming_cpp_optimized
```

