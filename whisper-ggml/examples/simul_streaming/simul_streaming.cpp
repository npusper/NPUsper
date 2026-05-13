// SimulStreaming C++ implementation
// Based on whisper_streaming_cpp_optimized, with AlignAtt policy instead of Local Agreement.
// Reference: SimulStreaming/simulstreaming/whisper/simul_whisper/simul_whisper.py
//
#include "common.h"
#include "whisper.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <string>
#include <thread>
#include <vector>
#include <fstream>
#include <iostream>
#include <chrono>
#include <numeric>


// command-line parameters
struct simul_streaming_params {
    int32_t n_threads  = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t step_ms    = 1000;  // 1 second chunks (SimulStreaming default segment_length=1.0)
    int32_t max_tokens = 0;
    int32_t audio_ctx  = 0;

    // AlignAtt parameters
    int32_t frame_threshold  = 4;    // stop when content_mel_len - most_attended <= this
    int32_t rewind_threshold = 200;  // rewind if last_attend - most_attended > this

    bool translate     = false;
    bool print_special = false;
    bool no_timestamps = false;
    bool use_gpu       = true;   // GPU by default
    bool flash_attn    = false;
    bool debug         = false;  // print debug info per chunk
    bool no_realtime   = false;  // skip sleep, process chunks as fast as possible

    std::string language  = "en";
    std::string model     = "models/ggml-base.bin";
    std::string dtw       = "base";  // required for cross-attention extraction

    std::vector<std::string> fname_inp = {};
    std::string fname_out = "";
};

void print_usage(int argc, char ** argv, const simul_streaming_params & params);

bool params_parse(int argc, char ** argv, simul_streaming_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg[0] != '-') {
            params.fname_inp.push_back(arg);
            continue;
        }

        if (arg == "-h" || arg == "--help") {
            print_usage(argc, argv, params);
            exit(0);
        }
        else if (arg == "-t"   || arg == "--threads")          { params.n_threads        = std::stoi(argv[++i]); }
        else if (               arg == "--step")               { params.step_ms          = std::stoi(argv[++i]); }
        else if (arg == "-mt"  || arg == "--max-tokens")       { params.max_tokens       = std::stoi(argv[++i]); }
        else if (arg == "-ac"  || arg == "--audio-ctx")        { params.audio_ctx        = std::stoi(argv[++i]); }
        else if (arg == "-ft"  || arg == "--frame-threshold")  { params.frame_threshold  = std::stoi(argv[++i]); }
        else if (arg == "-rt"  || arg == "--rewind-threshold") { params.rewind_threshold = std::stoi(argv[++i]); }
        else if (arg == "-tr"  || arg == "--translate")        { params.translate        = true; }
        else if (arg == "-ps"  || arg == "--print-special")    { params.print_special    = true; }
        else if (arg == "-dtw" || arg == "--dtw")              { params.dtw              = argv[++i]; }
        else if (arg == "-l"   || arg == "--language")         { params.language         = argv[++i]; }
        else if (arg == "-m"   || arg == "--model")            { params.model            = argv[++i]; }
        else if (arg == "-of"  || arg == "--output-file")      { params.fname_out        = argv[++i]; }
        else if (arg == "-ng"  || arg == "--no-gpu")           { params.use_gpu          = false; }
        else if (arg == "-fa"  || arg == "--flash-attn")       { params.flash_attn       = true; }
        else if (               arg == "--debug")              { params.debug            = true; }
        else if (               arg == "--no-realtime")        { params.no_realtime      = true; }
        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            print_usage(argc, argv, params);
            exit(1);
        }
    }
    return true;
}

void print_usage(int /*argc*/, char ** argv, const simul_streaming_params & params) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options] input.wav\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "SimulStreaming: AlignAtt-based streaming ASR\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,       --help              show this help message\n");
    fprintf(stderr, "  -t N,     --threads N         [%-7d] number of threads\n",                params.n_threads);
    fprintf(stderr, "            --step N            [%-7d] audio step size in ms\n",             params.step_ms);
    fprintf(stderr, "  -mt N,    --max-tokens N      [%-7d] max tokens per chunk (0=auto)\n",    params.max_tokens);
    fprintf(stderr, "  -ac N,    --audio-ctx N       [%-7d] audio context size\n",               params.audio_ctx);
    fprintf(stderr, "  -ft N,    --frame-threshold N [%-7d] AlignAtt frame threshold\n",         params.frame_threshold);
    fprintf(stderr, "  -rt N,    --rewind-threshold N[%-7d] AlignAtt rewind threshold\n",        params.rewind_threshold);
    fprintf(stderr, "  -tr,      --translate         [%-7s] translate to English\n",             params.translate ? "true" : "false");
    fprintf(stderr, "  -dtw M,   --dtw M             [%-7s] DTW model preset\n",                 params.dtw.c_str());
    fprintf(stderr, "  -l LANG,  --language LANG     [%-7s] spoken language\n",                  params.language.c_str());
    fprintf(stderr, "  -m FNAME, --model FNAME       [%-7s] model path\n",                       params.model.c_str());
    fprintf(stderr, "  -of FNAME,--output-file FNAME           output file path\n");
    fprintf(stderr, "  -ng,      --no-gpu            [%-7s] disable GPU\n",                      params.use_gpu ? "false" : "true");
    fprintf(stderr, "            --no-realtime       [%-7s] skip sleep, process as fast as possible\n", params.no_realtime ? "true" : "false");
    fprintf(stderr, "            --debug             [%-7s] print debug info\n",                 params.debug ? "true" : "false");
    fprintf(stderr, "\n");
}

// =====================================================================
// Audio chunk helper
// =====================================================================
bool get_audio_chunk(const std::vector<float> &pcmf32_all, std::vector<float> &pcmf32_new,
                     int64_t pcmf32_index, int step_ms, int sample_rate) {
    int64_t pcmf32_index_sample = (pcmf32_index * sample_rate) / 1000;
    int num_samples = (step_ms * sample_rate) / 1000;
    pcmf32_new.clear();

    bool has_more = true;
    if (pcmf32_index_sample + num_samples >= (int64_t)pcmf32_all.size()) {
        has_more = false;
        pcmf32_new.insert(pcmf32_new.end(), pcmf32_all.begin() + pcmf32_index_sample, pcmf32_all.end());
    } else {
        pcmf32_new.insert(pcmf32_new.end(), pcmf32_all.begin() + pcmf32_index_sample,
                          pcmf32_all.begin() + pcmf32_index_sample + num_samples);
    }
    return has_more;
}

// =====================================================================
// AlignAtt attention processing functions
// =====================================================================

// Z-score normalize across token dimension (dim=-2)
// data layout: [n_heads][n_tokens][n_audio_ctx] (row-major)
static void zscore_normalize(std::vector<float> & data, int n_heads, int n_tokens, int n_audio_ctx) {
    for (int h = 0; h < n_heads; h++) {
        for (int a = 0; a < n_audio_ctx; a++) {
            // Compute mean and std across token dimension for this (head, audio_frame)
            float sum = 0.0f, sum2 = 0.0f;
            for (int t = 0; t < n_tokens; t++) {
                float v = data[h * n_tokens * n_audio_ctx + t * n_audio_ctx + a];
                sum += v;
                sum2 += v * v;
            }
            float mean = sum / n_tokens;
            float var = sum2 / n_tokens - mean * mean;
            float std_val = sqrtf(var + 1e-9f);
            for (int t = 0; t < n_tokens; t++) {
                float & v = data[h * n_tokens * n_audio_ctx + t * n_audio_ctx + a];
                v = (v - mean) / std_val;
            }
        }
    }
}

// Median filter (width=7) along audio_ctx dimension with reflect padding
static void median_filter(std::vector<float> & data, int n_heads, int n_tokens, int n_audio_ctx, int width = 7) {
    int half = width / 2;
    std::vector<float> buf(n_audio_ctx);
    std::vector<float> window(width);

    for (int h = 0; h < n_heads; h++) {
        for (int t = 0; t < n_tokens; t++) {
            float * row = data.data() + h * n_tokens * n_audio_ctx + t * n_audio_ctx;
            for (int a = 0; a < n_audio_ctx; a++) {
                for (int j = -half; j <= half; j++) {
                    int idx = a + j;
                    // Reflect padding
                    if (idx < 0) idx = -idx;
                    if (idx >= n_audio_ctx) idx = 2 * (n_audio_ctx - 1) - idx;
                    window[j + half] = row[idx];
                }
                std::sort(window.begin(), window.end());
                buf[a] = window[half];
            }
            memcpy(row, buf.data(), n_audio_ctx * sizeof(float));
        }
    }
}

// Average across heads: [n_heads][n_tokens][n_audio_ctx] -> [n_tokens][n_audio_ctx]
static std::vector<float> average_heads(const std::vector<float> & data, int n_heads, int n_tokens, int n_audio_ctx) {
    std::vector<float> result(n_tokens * n_audio_ctx, 0.0f);
    for (int h = 0; h < n_heads; h++) {
        for (int t = 0; t < n_tokens; t++) {
            for (int a = 0; a < n_audio_ctx; a++) {
                result[t * n_audio_ctx + a] += data[h * n_tokens * n_audio_ctx + t * n_audio_ctx + a];
            }
        }
    }
    float inv = 1.0f / n_heads;
    for (auto & v : result) v *= inv;
    return result;
}

// Find argmax in range [0, len) of an array
static int argmax(const float * arr, int len) {
    int best = 0;
    for (int i = 1; i < len; i++) {
        if (arr[i] > arr[best]) best = i;
    }
    return best;
}

// =====================================================================
// Conservative word emission: emit all words except the last one
// (unless is_last, then emit everything)
// =====================================================================

// Split token IDs into words by detecting leading space in token text
static std::vector<std::vector<whisper_token>> split_to_words(
    struct whisper_context * ctx,
    const std::vector<whisper_token> & tokens)
{
    std::vector<std::vector<whisper_token>> words;
    std::vector<whisper_token> current_word;

    for (auto tok : tokens) {
        const char * str = whisper_token_to_str(ctx, tok);
        if (str && str[0] == ' ' && !current_word.empty()) {
            words.push_back(current_word);
            current_word.clear();
        }
        current_word.push_back(tok);
    }
    if (!current_word.empty()) {
        words.push_back(current_word);
    }
    return words;
}

// Conservative emission: all words except last (unless is_last)
static std::vector<whisper_token> emit_conservative(
    struct whisper_context * ctx,
    const std::vector<whisper_token> & decoded_tokens,
    bool is_last)
{
    if (is_last) return decoded_tokens;

    auto words = split_to_words(ctx, decoded_tokens);
    if (words.size() <= 1) return {};  // nothing to emit yet

    std::vector<whisper_token> emitted;
    for (size_t i = 0; i < words.size() - 1; i++) {
        for (auto t : words[i]) emitted.push_back(t);
    }
    return emitted;
}

// =====================================================================
// Prompt construction (reused from whisper_streaming)
// =====================================================================
std::vector<whisper_token> build_prompt(struct whisper_context * ctx,
    const std::vector<std::string> & committed_text)
{
    // Take last N characters of committed text as prompt
    std::vector<std::string> p;
    int l = 0;
    for (int i = (int)committed_text.size() - 1; i >= 0 && l < 200; i--) {
        l += committed_text[i].length() + 1;
        p.push_back(committed_text[i]);
    }
    std::reverse(p.begin(), p.end());

    std::vector<whisper_token> token_ids;
    for (const auto & word : p) {
        auto id = whisper_token_to_id(ctx, word);
        if (id != -1) {
            token_ids.push_back(id);
        }
    }
    return token_ids;
}

// =====================================================================
// Helper: print timestamped words (matching whisper_streaming format)
// =====================================================================
void print_committed(const std::vector<std::string> & committed) {
    fprintf(stderr, "\n=== Final transcript ===\n");
    for (const auto & w : committed) {
        fprintf(stderr, "%s", w.c_str());
    }
    fprintf(stderr, "\n========================\n");
}

// Print committed tokens with timestamp and latency
// Format: Start Time: X, End Time: Y, Transcript: Z, Latency: W, Emission Time: E
void print_tsw_with_latency(const std::vector<std::tuple<double, double, std::string, double, double>> & records) {
    double latency_sum = 0.0;
    int count = 0;
    for (const auto & entry : records) {
        double start_time = std::get<0>(entry);
        double end_time = std::get<1>(entry);
        const std::string & transcript = std::get<2>(entry);
        double latency = std::get<3>(entry);
        double emission_time = std::get<4>(entry);
        printf("Start Time: %g, End Time: %g, Transcript: %s, Latency: %g, Emission Time: %g\n",
               start_time, end_time, transcript.c_str(), latency, emission_time);
        latency_sum += latency;
        count++;
    }
    if (count > 0) {
        printf("Average latency: %g\n", latency_sum / count);
    }
}

// =====================================================================
// Main
// =====================================================================
int main(int argc, char ** argv) {
    simul_streaming_params params;

    if (!params_parse(argc, argv, params)) {
        return 1;
    }

    if (params.fname_inp.empty()) {
        fprintf(stderr, "error: no input file specified\n");
        print_usage(argc, argv, params);
        return 1;
    }

    if (params.dtw.empty()) {
        fprintf(stderr, "error: --dtw is required for SimulStreaming (cross-attention extraction)\n");
        return 1;
    }

    // Init whisper context with DTW enabled
    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu    = params.use_gpu;
    cparams.flash_attn = params.flash_attn;
    cparams.dtw_token_timestamps = true;
    cparams.dtw_aheads_preset = WHISPER_AHEADS_NONE;

    if (params.dtw == "tiny")      cparams.dtw_aheads_preset = WHISPER_AHEADS_TINY;
    if (params.dtw == "tiny.en")   cparams.dtw_aheads_preset = WHISPER_AHEADS_TINY_EN;
    if (params.dtw == "base")      cparams.dtw_aheads_preset = WHISPER_AHEADS_BASE;
    if (params.dtw == "base.en")   cparams.dtw_aheads_preset = WHISPER_AHEADS_BASE_EN;
    if (params.dtw == "small")     cparams.dtw_aheads_preset = WHISPER_AHEADS_SMALL;
    if (params.dtw == "small.en")  cparams.dtw_aheads_preset = WHISPER_AHEADS_SMALL_EN;
    if (params.dtw == "medium")    cparams.dtw_aheads_preset = WHISPER_AHEADS_MEDIUM;
    if (params.dtw == "medium.en") cparams.dtw_aheads_preset = WHISPER_AHEADS_MEDIUM_EN;
    if (params.dtw == "large.v1")  cparams.dtw_aheads_preset = WHISPER_AHEADS_LARGE_V1;
    if (params.dtw == "large.v2")  cparams.dtw_aheads_preset = WHISPER_AHEADS_LARGE_V2;
    if (params.dtw == "large.v3")  cparams.dtw_aheads_preset = WHISPER_AHEADS_LARGE_V3;

    if (cparams.dtw_aheads_preset == WHISPER_AHEADS_NONE) {
        fprintf(stderr, "error: unknown DTW preset '%s'\n", params.dtw.c_str());
        return 3;
    }

    struct whisper_context * ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "error: failed to initialize whisper context\n");
        return 2;
    }

    // Read audio file
    std::vector<float> pcmf32_all;
    std::vector<std::vector<float>> pcmf32s;
    if (!::read_wav(params.fname_inp[0], pcmf32_all, pcmf32s, false)) {
        fprintf(stderr, "error: failed to read WAV file '%s'\n", params.fname_inp[0].c_str());
        return 4;
    }

    fprintf(stderr, "\n");
    fprintf(stderr, "%s: SimulStreaming (AlignAtt) mode\n", __func__);
    fprintf(stderr, "%s: model = %s, dtw = %s, language = %s\n", __func__, params.model.c_str(), params.dtw.c_str(), params.language.c_str());
    fprintf(stderr, "%s: audio = %.1f sec, step = %d ms, frame_threshold = %d, rewind_threshold = %d\n",
            __func__, (float)pcmf32_all.size() / WHISPER_SAMPLE_RATE, params.step_ms,
            params.frame_threshold, params.rewind_threshold);
    fprintf(stderr, "\n");

    // State
    std::vector<float> audio_buffer;               // accumulated audio (up to 30s)
    std::vector<std::string> committed_words;      // committed word strings
    std::vector<whisper_token> committed_tokens;    // flat view (for latency/output bookkeeping)
    std::vector<std::vector<whisper_token>> token_segments;  // per-segment token groups (current buffer)
    std::vector<whisper_token> context_buffer;      // tokens promoted from removed segments (Python: self.context)
    int n_segments_in_buffer = 0;                  // audio segments in buffer
    int last_attend_frame = -params.rewind_threshold;
    int64_t buffer_start_sample = 0;               // absolute sample offset (tracks buffer trimming)

    // Latency tracking: (start_time, end_time, transcript, latency) per committed token
    std::vector<std::tuple<double, double, std::string, double, double>> latency_record;

    const int n_samples_step = (params.step_ms * WHISPER_SAMPLE_RATE) / 1000;
    int64_t pcmf32_index = 0;  // current position in samples
    int n_iter = 0;
    const auto t_start = std::chrono::high_resolution_clock::now();

    while (pcmf32_index < (int64_t)pcmf32_all.size()) {
        bool is_last = false;
        auto t_now = std::chrono::high_resolution_clock::now();

        int64_t chunk_size;
        if (params.no_realtime) {
            // No-realtime mode: take exactly one step worth of audio, no sleep
            chunk_size = std::min((int64_t)n_samples_step, (int64_t)pcmf32_all.size() - pcmf32_index);
        } else {
            // Simulate real-time audio arrival: wait until the next chunk's audio is "available"
            double target_time = (double)(pcmf32_index + n_samples_step) / WHISPER_SAMPLE_RATE;
            t_now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(t_now - t_start).count();
            if (elapsed < target_time) {
                double sleep_sec = target_time - elapsed;
                std::this_thread::sleep_for(std::chrono::microseconds((int64_t)(sleep_sec * 1e6)));
            }

            // Get next audio chunk (elapsed-time-based: consume all audio that has "arrived")
            t_now = std::chrono::high_resolution_clock::now();
            auto t_after_sleep = t_now;
            double elapsed_after = std::chrono::duration<double>(t_after_sleep - t_start).count();
            int64_t available_samples = std::min((int64_t)(elapsed_after * WHISPER_SAMPLE_RATE), (int64_t)pcmf32_all.size());
            chunk_size = available_samples - pcmf32_index;
            if (chunk_size <= 0) chunk_size = std::min((int64_t)n_samples_step, (int64_t)pcmf32_all.size() - pcmf32_index);
        }
        std::vector<float> pcmf32_new;
        pcmf32_new.insert(pcmf32_new.end(),
                          pcmf32_all.begin() + pcmf32_index,
                          pcmf32_all.begin() + pcmf32_index + chunk_size);
        pcmf32_index += chunk_size;
        is_last = (pcmf32_index >= (int64_t)pcmf32_all.size());

        // Accumulate into audio buffer
        audio_buffer.insert(audio_buffer.end(), pcmf32_new.begin(), pcmf32_new.end());
        n_segments_in_buffer++;

        // Sliding window: trim buffer to 30s max
        // Python: when audio overflows, promote oldest token segment to context buffer
        //   self.context.append_token_ids(self.tokens[1][0,:])
        //   self.tokens = [initial_tokens] + self.tokens[2:]
        const float buffer_len_sec = 30.0f;
        const float step_sec = (float)params.step_ms / 1000.0f;
        while ((float)n_segments_in_buffer * step_sec >= buffer_len_sec && n_segments_in_buffer > 1) {
            int remove_samples = std::min((int)(n_samples_step), (int)audio_buffer.size());
            audio_buffer.erase(audio_buffer.begin(), audio_buffer.begin() + remove_samples);
            buffer_start_sample += remove_samples;
            int removed_frames = remove_samples / (WHISPER_HOP_LENGTH * 2);
            last_attend_frame -= removed_frames;
            n_segments_in_buffer--;

            // Promote oldest token segment to context buffer (not discard)
            if (!token_segments.empty()) {
                for (auto t : token_segments[0]) {
                    context_buffer.push_back(t);
                }
                token_segments.erase(token_segments.begin());
            }
        }

        float buffer_seconds = (float)audio_buffer.size() / WHISPER_SAMPLE_RATE;

        // Buffer stats (matching whisper_streaming format)
        printf("[BUFFER_STATS] iter=%d buffer_sec=%.3f buffer_samples=%ld\n",
               n_iter, buffer_seconds, (long)audio_buffer.size());

        if (params.debug) {
            fprintf(stderr, "[iter=%d] buffer=%.2fs, is_last=%d\n", n_iter, buffer_seconds, is_last);
        }

        // =====================================================
        // Compute content_mel_len from audio sample count
        // =====================================================
        // content_mel_len = actual audio content in encoder space (after stride-2 conv)
        // hop_length=160, conv stride=2, so each encoder frame = 320 samples
        int content_mel_len = (int)audio_buffer.size() / 320;
        if (content_mel_len < 1) content_mel_len = 1;

        // =====================================================
        // Zero-pad to 30s total + full decode
        // =====================================================
        // Python SimulStreaming: log_mel_spectrogram(audio, padding=N_SAMPLES) + pad_or_trim(mel, 3000)
        // → mel은 항상 3000 프레임 (30s). 우리도 오디오를 30s로 맞춰서 mel이 3000 프레임이 되게 함.
        // buffer < 30s → 30s까지 zero-pad, buffer == 30s → 패딩 불필요
        std::vector<float> pcmf32_padded = audio_buffer;
        const size_t n_samples_30s = WHISPER_SAMPLE_RATE * 30;
        if (pcmf32_padded.size() < n_samples_30s) {
            pcmf32_padded.resize(n_samples_30s, 0.0f);
        }

        // Input stats
        printf("[INPUT_STATS] iter=%d input_sec=%.3f input_samples=%ld content_mel_len=%d\n",
               n_iter, (float)pcmf32_padded.size() / WHISPER_SAMPLE_RATE,
               (long)pcmf32_padded.size(), content_mel_len);

        // =====================================================
        // Encode: mel + encode (single pass, no whisper_full)
        // =====================================================

        auto t_mel_s = std::chrono::high_resolution_clock::now();
        if (whisper_pcm_to_mel(ctx, pcmf32_padded.data(), pcmf32_padded.size(), params.n_threads) != 0) {
            fprintf(stderr, "%s: failed to compute mel spectrogram\n", __func__);
            return 7;
        }
        auto t_mel_e = std::chrono::high_resolution_clock::now();
        double iter_mel_us = std::chrono::duration<double, std::micro>(t_mel_e - t_mel_s).count();

        auto t_enc_s = std::chrono::high_resolution_clock::now();
        if (whisper_encode(ctx, 0, params.n_threads) != 0) {
            fprintf(stderr, "%s: failed to encode\n", __func__);
            return 7;
        }
        auto t_enc_e = std::chrono::high_resolution_clock::now();
        double iter_encode_us = std::chrono::duration<double, std::micro>(t_enc_e - t_enc_s).count();

        // =====================================================
        // Build decoder input sequence (SimulStreaming paper):
        //   [sot_prev, context_buffer...] + [sot, lang, task, no_timestamps, seg_tokens...]
        //
        // context_buffer = tokens promoted from removed audio segments
        // seg_tokens     = tokens from segments still in audio buffer
        // =====================================================
        const int n_text_ctx = whisper_n_text_ctx(ctx);  // 448
        // n_sot_special: sot + lang + task + no_timestamps = 4 (multilingual) or 2
        const int n_sot_special = whisper_is_multilingual(ctx) ? 4 : 2;

        // Collect seg_tokens (from current buffer segments)
        std::vector<whisper_token> seg_tokens;
        for (const auto & seg : token_segments) {
            for (auto t : seg) {
                seg_tokens.push_back(t);
            }
        }

        // Trim to fit within n_text_ctx (448)
        // Total = (sot_prev + context_buffer) + n_sot_special + seg_tokens + new_decode_room
        {
            int ctx_overhead = context_buffer.empty() ? 0 : (1 + (int)context_buffer.size()); // sot_prev + context
            int total = ctx_overhead + n_sot_special + (int)seg_tokens.size();
            // First, trim context_buffer from front
            if (total > n_text_ctx && !context_buffer.empty()) {
                int excess = total - n_text_ctx;
                int trim = std::min(excess, (int)context_buffer.size());
                context_buffer.erase(context_buffer.begin(), context_buffer.begin() + trim);
                if (context_buffer.empty()) {
                    ctx_overhead = 0;  // no sot_prev needed anymore
                } else {
                    ctx_overhead = 1 + (int)context_buffer.size();
                }
                total = ctx_overhead + n_sot_special + (int)seg_tokens.size();
            }
            // Then, trim seg_tokens from front if still exceeds
            if (total > n_text_ctx && !seg_tokens.empty()) {
                int excess = total - n_text_ctx;
                int trim = std::min(excess, (int)seg_tokens.size());
                seg_tokens.erase(seg_tokens.begin(), seg_tokens.begin() + trim);
            }
        }

        // Build full sequence: [sot_prev, context...] + [sot, lang, task, no_timestamps, seg_tokens...]
        std::vector<whisper_token> sot_seq;

        // Part 1: context with sot_prev prefix (only if context exists)
        if (!context_buffer.empty()) {
            sot_seq.push_back(whisper_token_prev(ctx));  // sot_prev
            for (auto t : context_buffer) {
                sot_seq.push_back(t);
            }
        }

        // Part 2: SOT special tokens
        sot_seq.push_back(whisper_token_sot(ctx));
        if (whisper_is_multilingual(ctx)) {
            sot_seq.push_back(whisper_token_lang(ctx, whisper_lang_id(params.language.c_str())));
            if (params.translate) {
                sot_seq.push_back(whisper_token_translate(ctx));
            } else {
                sot_seq.push_back(whisper_token_transcribe(ctx));
            }
        }
        sot_seq.push_back(whisper_token_not(ctx));  // no_timestamps

        // Part 3: segment tokens (forced decoding target from current buffer)
        for (auto t : seg_tokens) {
            sot_seq.push_back(t);
        }
        int sot_sequence_len = (int)sot_seq.size();

        // Enable alignment heads cross-attention capture
        whisper_set_aheads_cross_attention_capture(ctx, true);

        // Decode SOT + prompt tokens (prefill)
        auto t_prefill_s = std::chrono::high_resolution_clock::now();
        if (whisper_decode(ctx, sot_seq.data(), sot_seq.size(), 0, params.n_threads) != 0) {
            fprintf(stderr, "%s: failed to decode SOT sequence\n", __func__);
            return 7;
        }
        auto t_prefill_e = std::chrono::high_resolution_clock::now();
        double iter_prefill_us = std::chrono::duration<double, std::micro>(t_prefill_e - t_prefill_s).count();

        // =====================================================
        // Token-by-token decode with inline cross-attention extraction
        // =====================================================
        int n_past = sot_sequence_len;
        int last_decode_n_tokens = sot_sequence_len;
        // Dynamic max_tokens: remaining budget after sot_seq
        int max_tokens = params.max_tokens > 0 ? params.max_tokens : (n_text_ctx - sot_sequence_len);
        int n_vocab = whisper_n_vocab(ctx);

        std::vector<whisper_token> decoded_tokens;
        // Accumulate per-token head-averaged attention: [n_decoded][n_audio_ctx]
        std::vector<std::vector<float>> per_token_avg_attn;
        int attn_n_audio_ctx_saved = 0;

        // Cross-attention extraction timing
        double cross_attn_copy_us = 0.0;
        double cross_attn_calc_us = 0.0;
        double iter_decode_us = 0.0;
        int n_decoded_tokens = 0;

        // AlignAtt state (inline per-token check, matching Python behavior)
        int stop_token_idx = -1;
        int final_most_attended = last_attend_frame;
        bool rewind_detected = false;
        std::vector<int> token_attend_frames;

        for (int step = 0; step < max_tokens; step++) {
            // Greedy argmax (equivalent to best_of=1)
            float * logits_all = whisper_get_logits(ctx);
            float * logits = logits_all + (last_decode_n_tokens - 1) * n_vocab;

            whisper_token next_token = 0;
            float max_logit = logits[0];
            for (int v = 1; v < n_vocab; v++) {
                if (logits[v] > max_logit) {
                    max_logit = logits[v];
                    next_token = v;
                }
            }

            // EOT or special token → stop
            if (next_token >= whisper_token_eot(ctx)) break;

            decoded_tokens.push_back(next_token);

            // Decode this token
            auto t_dec_s = std::chrono::high_resolution_clock::now();
            if (whisper_decode(ctx, &next_token, 1, n_past, params.n_threads) != 0) {
                fprintf(stderr, "%s: failed to decode token at step %d\n", __func__, step);
                break;
            }
            auto t_dec_e = std::chrono::high_resolution_clock::now();
            iter_decode_us += std::chrono::duration<double, std::micro>(t_dec_e - t_dec_s).count();
            n_decoded_tokens++;
            n_past++;
            last_decode_n_tokens = 1;

            // Extract alignment heads cross-attention for this token
            bool attn_ok = false;
            {
                auto t_copy_s = std::chrono::high_resolution_clock::now();
                int copy_ret = whisper_copy_aheads_cross_attention_from_ctx(ctx);
                auto t_copy_e = std::chrono::high_resolution_clock::now();
                cross_attn_copy_us += std::chrono::duration<double, std::micro>(t_copy_e - t_copy_s).count();

                if (copy_ret != 0) goto next_step;

                auto t_calc_s = std::chrono::high_resolution_clock::now();

                int attn_n_tokens_cur, attn_n_audio_ctx, attn_n_aheads;
                const float * attn_data = whisper_get_aheads_cross_attention_data_from_ctx(
                    ctx, &attn_n_tokens_cur, &attn_n_audio_ctx, &attn_n_aheads);

                if (attn_data != nullptr && attn_n_aheads > 0) {
                    attn_n_audio_ctx_saved = attn_n_audio_ctx;

                    // Average across alignment heads
                    // Layout: data[t + a * n_tokens + h * n_tokens * n_audio_ctx]
                    // For single-token decode: n_tokens=1, t=0 → data[a + h * n_audio_ctx]
                    std::vector<float> avg(attn_n_audio_ctx, 0.0f);
                    for (int h = 0; h < attn_n_aheads; h++) {
                        for (int a = 0; a < attn_n_audio_ctx; a++) {
                            avg[a] += attn_data[a + h * attn_n_audio_ctx];
                        }
                    }
                    float inv = 1.0f / attn_n_aheads;
                    for (auto & v : avg) v *= inv;
                    per_token_avg_attn.push_back(std::move(avg));
                    attn_ok = true;
                }

                auto t_calc_e = std::chrono::high_resolution_clock::now();
                cross_attn_calc_us += std::chrono::duration<double, std::micro>(t_calc_e - t_calc_s).count();
            }

            // =====================================================
            // Inline AlignAtt check (matches Python per-token behavior)
            // =====================================================
            if (attn_ok) {
                int n_tok = (int)per_token_avg_attn.size();
                int n_ctx = attn_n_audio_ctx_saved;

                // Build attention matrix from accumulated tokens
                std::vector<float> attn_mat(n_tok * n_ctx, 0.0f);
                for (int t = 0; t < n_tok; t++) {
                    for (int a = 0; a < n_ctx && a < (int)per_token_avg_attn[t].size(); a++) {
                        attn_mat[t * n_ctx + a] = per_token_avg_attn[t][a];
                    }
                }

                // Z-score normalize across token dimension (need >= 2 tokens)
                if (n_tok >= 2) {
                    zscore_normalize(attn_mat, 1, n_tok, n_ctx);
                }

                // Median filter (width=7)
                median_filter(attn_mat, 1, n_tok, n_ctx, 7);

                // Compute attend frames for all tokens
                token_attend_frames.resize(n_tok);
                for (int t = 0; t < n_tok; t++) {
                    const float * row = attn_mat.data() + t * n_ctx;
                    token_attend_frames[t] = argmax(row, std::min(content_mel_len, n_ctx));
                }

                int most_attended = token_attend_frames[n_tok - 1];

                if (params.debug) {
                    fprintf(stderr, "  [token %d] most_attended=%d, content_mel_len=%d, diff=%d\n",
                            step, most_attended, content_mel_len, content_mel_len - most_attended);
                }

                // Rewind check (skip on last chunk)
                if (!is_last && last_attend_frame - most_attended > params.rewind_threshold) {
                    // Special token guard (Python: if current_tokens[0, -2] >= DEC_PAD)
                    // Previous token in full sequence: if >=2 decoded, use previous decoded;
                    // otherwise use last SOT token (which is always special)
                    whisper_token prev_token = (decoded_tokens.size() >= 2)
                        ? decoded_tokens[decoded_tokens.size() - 2]
                        : sot_seq.back();
                    if (prev_token >= whisper_token_eot(ctx)) {
                        if (params.debug) {
                            fprintf(stderr, "  [token %d] omit rewinding from special tokens\n", step);
                        }
                        // Don't rewind — just continue decoding
                    } else {
                        fprintf(stderr, "[iter=%d] rewind detected: last_attend=%d, most_attended=%d\n",
                                n_iter, last_attend_frame, most_attended);
                        rewind_detected = true;
                        last_attend_frame = -params.rewind_threshold;
                        break;
                    }
                }

                final_most_attended = most_attended;

                // AlignAtt stop condition (Python: content_mel_len - most_attended <= threshold)
                int threshold = is_last ? 4 : params.frame_threshold;
                if (content_mel_len - most_attended <= threshold) {
                    if (params.debug) {
                        fprintf(stderr, "  [token %d] STOP: %d - %d = %d <= %d\n",
                                step, content_mel_len, most_attended,
                                content_mel_len - most_attended, threshold);
                    }
                    stop_token_idx = n_tok - 1;  // strip this token
                    break;
                }
            }
            next_step:;
        }

        // Print timings
        whisper_print_timings(ctx);

        fprintf(stderr, "[ITER_TIMING] iter=%d mel=%.2f ms encode=%.2f ms prefill=%.2f ms decode=%.2f ms n_tokens=%d\n",
                n_iter, iter_mel_us / 1000.0, iter_encode_us / 1000.0,
                iter_prefill_us / 1000.0, iter_decode_us / 1000.0, n_decoded_tokens);

        fprintf(stderr, "[CROSS_ATTN_TIME] iter=%d copy=%.2f ms calc=%.2f ms\n",
                n_iter, cross_attn_copy_us / 1000.0, cross_attn_calc_us / 1000.0);

        if (decoded_tokens.empty()) {
            if (params.debug) {
                fprintf(stderr, "[iter=%d] no tokens decoded\n", n_iter);
            }
            token_segments.push_back({});  // empty segment to keep 1:1 with audio
            n_iter++;
            continue;
        }

        if (rewind_detected) {
            token_segments.push_back({});  // empty segment to keep 1:1 with audio
            n_iter++;
            continue;
        }

        // Determine tokens to consider for emission
        std::vector<whisper_token> tokens_for_emission;
        if (stop_token_idx >= 0) {
            // Stop triggered at token index stop_token_idx — strip it and everything after
            tokens_for_emission.insert(tokens_for_emission.end(),
                                        decoded_tokens.begin(),
                                        decoded_tokens.begin() + stop_token_idx);
        } else {
            // No stop triggered — use all decoded tokens
            tokens_for_emission = decoded_tokens;
        }

        // Conservative emission: all words except last (unless is_last)
        std::vector<whisper_token> emitted = emit_conservative(ctx, tokens_for_emission, is_last);

        // Compute emission wall time (seconds from start)
        t_now = std::chrono::high_resolution_clock::now();
        double emission_wall_time = std::chrono::duration<double>(t_now - t_start).count();

        // Map emitted tokens back to their attend frames from decoded_tokens
        // emitted is a prefix of tokens_for_emission which is a prefix of decoded_tokens
        // Store emitted tokens per segment (always push, even if empty)
        token_segments.push_back(emitted);

        // Update committed state and record latency
        if (!emitted.empty()) {
            for (int ei = 0; ei < (int)emitted.size(); ei++) {
                committed_tokens.push_back(emitted[ei]);
                const char * str = whisper_token_to_str(ctx, emitted[ei]);
                if (str) {
                    committed_words.push_back(str);

                    // Compute token's audio position from its attend frame
                    // attend_frame is relative to buffer; convert to absolute time
                    int attend_frame = (ei < (int)token_attend_frames.size()) ? token_attend_frames[ei] : -1;
                    double token_start_time = 0.0;
                    double token_end_time = 0.0;
                    if (attend_frame >= 0) {
                        // Each encoder frame = 320 samples (hop=160, stride-2 conv)
                        int64_t abs_sample = buffer_start_sample + (int64_t)attend_frame * 320;
                        token_start_time = (double)abs_sample / WHISPER_SAMPLE_RATE;
                        // end_time: next frame boundary
                        token_end_time = (double)(abs_sample + 320) / WHISPER_SAMPLE_RATE;
                    }

                    double latency = std::max(0.0, emission_wall_time - token_end_time);
                    latency_record.push_back(std::make_tuple(token_start_time, token_end_time, std::string(str), latency, emission_wall_time));

                    printf("%s", str);
                    fflush(stdout);
                }
            }
        }

        // Update last_attend_frame
        if (!rewind_detected && final_most_attended >= 0) {
            last_attend_frame = final_most_attended;
        }

        fprintf(stderr, "[iter=%d] emitted %zu tokens, last_attend=%d, content_mel_len=%d\n",
                n_iter, emitted.size(), last_attend_frame, content_mel_len);

        n_iter++;
    }

    printf("\n");
    // Print committed tokens with latency (Start Time format for comparison script parsing)
    print_tsw_with_latency(latency_record);
    print_committed(committed_words);

    whisper_free(ctx);

    return 0;
}
