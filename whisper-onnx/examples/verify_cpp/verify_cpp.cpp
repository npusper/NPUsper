// verify_cpp: C++ equivalent of verify_onnx.py
//
// Loads a WAV file, computes mel (same C++ path as streaming binaries),
// encodes, and greedy-decodes to text.  Prints the decoded transcript
// so you can compare it against the expected ground truth.
//
// Usage:
//   ./verify_cpp -m ../models/base -a ../eval/comparison_results/cached_audio/test_audio_2samples_1spad.wav
//
#include "common.h"
#include "whisper_ort.h"

#include <cstdio>
#include <string>
#include <vector>
#include <thread>

struct verify_params {
    int32_t n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());
    std::string model   = "models/base";
    std::string audio   = "";
    std::string language = "en";
    bool use_gpu = true;
};

static bool params_parse(int argc, char ** argv, verify_params & p) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if      (arg == "-m" || arg == "--model")    { p.model    = argv[++i]; }
        else if (arg == "-a" || arg == "--audio")    { p.audio    = argv[++i]; }
        else if (arg == "-l" || arg == "--language") { p.language = argv[++i]; }
        else if (arg == "-ng" || arg == "--no-gpu")  { p.use_gpu  = false; }
        else if (arg == "-t" || arg == "--threads")  { p.n_threads = std::stoi(argv[++i]); }
        else {
            fprintf(stderr, "unknown arg: %s\n", arg.c_str());
            return false;
        }
    }
    return !p.audio.empty();
}

int main(int argc, char ** argv) {
    verify_params params;
    if (!params_parse(argc, argv, params)) {
        fprintf(stderr, "Usage: %s -m <model_dir> -a <audio.wav> [-l en] [-ng] [-t N]\n", argv[0]);
        return 1;
    }

    // ── Load model ────────────────────────────────────────────────────────────
    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu    = params.use_gpu;
    cparams.flash_attn = false;

    whisper_context * ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);
    if (!ctx) {
        fprintf(stderr, "error: failed to init whisper context from '%s'\n", params.model.c_str());
        return 2;
    }

    // ── Load audio ────────────────────────────────────────────────────────────
    std::vector<float> pcmf32;
    std::vector<std::vector<float>> pcmf32s;
    if (!read_wav(params.audio, pcmf32, pcmf32s, false)) {
        fprintf(stderr, "error: failed to read WAV '%s'\n", params.audio.c_str());
        return 3;
    }

    float audio_sec = (float)pcmf32.size() / WHISPER_SAMPLE_RATE;
    fprintf(stderr, "Audio: %.2f sec  (%zu samples)\n", audio_sec, pcmf32.size());

    // ── Pad to 30 s (same as simul_streaming) ────────────────────────────────
    const size_t n_samples_30s = WHISPER_SAMPLE_RATE * 30;
    if (pcmf32.size() < n_samples_30s) pcmf32.resize(n_samples_30s, 0.0f);

    // ── Compute mel ───────────────────────────────────────────────────────────
    if (whisper_pcm_to_mel(ctx, pcmf32.data(), (int)pcmf32.size(), params.n_threads) != 0) {
        fprintf(stderr, "error: whisper_pcm_to_mel failed\n");
        return 4;
    }
    fprintf(stderr, "Mel: computed\n");

    // ── Encode ────────────────────────────────────────────────────────────────
    if (whisper_encode(ctx, 0, params.n_threads) != 0) {
        fprintf(stderr, "error: whisper_encode failed\n");
        return 5;
    }
    fprintf(stderr, "Encode: done\n");

    // ── Build SOT sequence ────────────────────────────────────────────────────
    std::vector<whisper_token> sot_seq;
    sot_seq.push_back(whisper_token_sot(ctx));
    if (whisper_is_multilingual(ctx)) {
        int lang_id = whisper_lang_id(params.language.c_str());
        sot_seq.push_back(whisper_token_lang(ctx, lang_id));
        sot_seq.push_back(whisper_token_transcribe(ctx));
    }
    sot_seq.push_back(whisper_token_not(ctx));

    fprintf(stderr, "SOT tokens:");
    for (auto t : sot_seq) fprintf(stderr, " %d", t);
    fprintf(stderr, "\n");

    // ── Prefill ───────────────────────────────────────────────────────────────
    if (whisper_decode(ctx, sot_seq.data(), (int)sot_seq.size(), 0, params.n_threads) != 0) {
        fprintf(stderr, "error: prefill failed\n");
        return 6;
    }

    // ── Greedy decode ─────────────────────────────────────────────────────────
    const int n_vocab       = whisper_n_vocab(ctx);
    const int eot_token     = whisper_token_eot(ctx);
    const int max_tokens    = 200;
    int n_past              = (int)sot_seq.size();
    int last_decode_n_tokens = (int)sot_seq.size();

    std::string result;
    std::vector<whisper_token> decoded_tokens;

    for (int step = 0; step < max_tokens; step++) {
        float * logits_all = whisper_get_logits(ctx);
        float * logits = logits_all + (last_decode_n_tokens - 1) * n_vocab;

        whisper_token next_token = 0;
        float max_logit = logits[0];
        for (int v = 1; v < n_vocab; v++) {
            if (logits[v] > max_logit) { max_logit = logits[v]; next_token = v; }
        }

        if (next_token >= eot_token) {
            fprintf(stderr, "Stopped at step %d: token %d >= eot %d\n", step, next_token, eot_token);
            break;
        }

        decoded_tokens.push_back(next_token);
        const char * str = whisper_token_to_str(ctx, next_token);
        if (str) result += str;

        if (whisper_decode(ctx, &next_token, 1, n_past, params.n_threads) != 0) {
            fprintf(stderr, "error: decode step %d failed\n", step);
            break;
        }
        n_past++;
        last_decode_n_tokens = 1;
    }

    // ── Print results ─────────────────────────────────────────────────────────
    fprintf(stderr, "\n");
    printf("Transcript: '%s'\n", result.c_str());
    fprintf(stderr, "Tokens (%zu):", decoded_tokens.size());
    for (auto t : decoded_tokens) fprintf(stderr, " %d", t);
    fprintf(stderr, "\n");

    whisper_free(ctx);
    return 0;
}
