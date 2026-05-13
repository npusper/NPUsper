// "Ours" Streaming Whisper C++ implementation
// No 30s padding + backward peak detection for hallucination prevention.
// Reference: attention_analysis.py, chunk_experiment.py
//
// Key differences from SimulStreaming:
//   - No 30s padding: audio_ctx computed from actual audio length
//   - Backward peak detection: cross-attention delta (max_peak < min_peak) → hallucination stop
//   - Peak margin: stop when attention peak enters margin zone near audio end → carryover
//   - Carryover: carry audio from peak position to next chunk
//   - Prompt prefill: use previous chunk tokens as prompt context
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
#include <cctype>


// command-line parameters
struct ours_streaming_params {
    int32_t n_threads  = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t step_ms    = 1000;  // chunk size in ms
    int32_t max_tokens = 0;

    // Backward peak detection
    int32_t smoothing_window     = 10;   // moving average window
    int32_t median_filter_window = 7;    // median filter window
    int32_t cross_attn_layer     = -1;   // decoder layer (-1 = last)
    float   exclude_time         = 0.3f; // seconds excluded from max peak search
    bool    skip_punctuation_delta = true;
    bool    skip_subword_delta     = true;

    // DTW alignment heads (for carryover modes 1, 2, 3)
    std::string dtw_model = "";  // alignment head preset (e.g., "base", "small"); empty = auto-detect from model path

    // Peak margin & carryover
    float   peak_margin_sec        = 0.0f; // margin zone near chunk end (0 = disabled)
    float   carryover_overlap_sec  = 0.0f; // overlap before peak for carryover
    float   min_chunk_sec          = 0.0f; // min effective audio duration
    int32_t carryover_mode         = 0;    // 0=peak, 1=pfill, 2=olap, 3=olap+reinfer
    float   word_end_offset_sec    = 0.0f; // offset after DTW word end for carryover

    // Prompt & EOT control
    int32_t prompt_prefill_n = 0;     // N tokens from prev chunk as prompt (0 = disabled)
    bool    skip_eot          = false; // suppress EOT token generation

    bool translate     = false;
    bool print_special = false;
    bool no_timestamps = false;
    bool use_gpu       = true;
    bool flash_attn    = false;
    bool no_realtime   = false;  // skip sleep, process chunks as fast as possible
    bool no_postprocess = false; // disable dedup and prompt prefill
    bool debug         = false;

    std::string language  = "en";
    std::string model     = "models/ggml-base.bin";

    std::vector<std::string> fname_inp = {};
    std::string fname_out = "";
};

void print_usage(int argc, char ** argv, const ours_streaming_params & params);

bool params_parse(int argc, char ** argv, ours_streaming_params & params) {
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
        else if (arg == "-t"   || arg == "--threads")              { params.n_threads            = std::stoi(argv[++i]); }
        else if (               arg == "--step")                   { params.step_ms              = std::stoi(argv[++i]); }
        else if (arg == "-mt"  || arg == "--max-tokens")           { params.max_tokens           = std::stoi(argv[++i]); }
        else if (               arg == "--smoothing")              { params.smoothing_window     = std::stoi(argv[++i]); }
        else if (               arg == "--median-filter")          { params.median_filter_window = std::stoi(argv[++i]); }
        else if (               arg == "--cross-attn-layer")       { params.cross_attn_layer     = std::stoi(argv[++i]); }
        else if (               arg == "--exclude-time")           { params.exclude_time         = std::stof(argv[++i]); }
        else if (               arg == "--peak-margin")            { params.peak_margin_sec      = std::stof(argv[++i]); }
        else if (               arg == "--carryover-overlap")      { params.carryover_overlap_sec = std::stof(argv[++i]); }
        else if (               arg == "--min-chunk")              { params.min_chunk_sec        = std::stof(argv[++i]); }
        else if (               arg == "--carryover-mode")        { params.carryover_mode       = std::stoi(argv[++i]); }
        else if (               arg == "--word-end-offset")       { params.word_end_offset_sec  = std::stof(argv[++i]); }
        else if (               arg == "--prompt-prefill")         { params.prompt_prefill_n     = std::stoi(argv[++i]); }
        else if (               arg == "--skip-eot")               { params.skip_eot             = true; }
        else if (               arg == "--no-skip-punct")          { params.skip_punctuation_delta = false; }
        else if (               arg == "--no-skip-subword")        { params.skip_subword_delta   = false; }
        else if (arg == "-tr"  || arg == "--translate")            { params.translate            = true; }
        else if (arg == "-ps"  || arg == "--print-special")        { params.print_special        = true; }
        else if (arg == "-l"   || arg == "--language")             { params.language              = argv[++i]; }
        else if (arg == "-m"   || arg == "--model")                { params.model                 = argv[++i]; }
        else if (arg == "-of"  || arg == "--output-file")          { params.fname_out             = argv[++i]; }
        else if (arg == "-ng"  || arg == "--no-gpu")               { params.use_gpu               = false; }
        else if (               arg == "--no-realtime")            { params.no_realtime           = true; }
        else if (               arg == "--no-postprocess")         { params.no_postprocess        = true; }
        else if (               arg == "--debug")                  { params.debug                 = true; }
        else if (arg == "-dtw" || arg == "--dtw")                  { params.dtw_model             = argv[++i]; }
        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            print_usage(argc, argv, params);
            return false;
        }
    }
    return true;
}

void print_usage(int argc, char ** argv, const ours_streaming_params & params) {
    fprintf(stderr, "\nUsage: %s [options] file0.wav [file1.wav ...]\n\n", argv[0]);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -h,       --help                         show this help message and exit\n");
    fprintf(stderr, "  -t N,     --threads N         [%-7d] number of threads\n",                  params.n_threads);
    fprintf(stderr, "            --step N            [%-7d] chunk step in ms\n",                    params.step_ms);
    fprintf(stderr, "  -mt N,    --max-tokens N      [%-7d] max tokens per decode\n",              params.max_tokens);
    fprintf(stderr, "            --smoothing N       [%-7d] smoothing window\n",                   params.smoothing_window);
    fprintf(stderr, "            --median-filter N   [%-7d] median filter window\n",               params.median_filter_window);
    fprintf(stderr, "            --cross-attn-layer N[%-7d] decoder layer (-1=last)\n",            params.cross_attn_layer);
    fprintf(stderr, "            --exclude-time F    [%-7.2f] exclude seconds from max peak\n",    params.exclude_time);
    fprintf(stderr, "            --peak-margin F     [%-7.2f] peak margin seconds\n",              params.peak_margin_sec);
    fprintf(stderr, "            --carryover-overlap F[%-6.2f] overlap before peak\n",             params.carryover_overlap_sec);
    fprintf(stderr, "            --min-chunk F       [%-7.2f] min effective audio seconds\n",      params.min_chunk_sec);
    fprintf(stderr, "            --carryover-mode N  [%-7d] carryover mode (0=peak,1=pfill,2=olap,3=olap+reinfer)\n", params.carryover_mode);
    fprintf(stderr, "            --word-end-offset F [%-7.2f] offset for DTW carryover\n", params.word_end_offset_sec);
    fprintf(stderr, "  -dtw S,   --dtw S             [%-7s] alignment heads preset for DTW (base,small,etc)\n", params.dtw_model.empty() ? "auto" : params.dtw_model.c_str());
    fprintf(stderr, "            --prompt-prefill N  [%-7d] tokens from prev chunk\n",             params.prompt_prefill_n);
    fprintf(stderr, "            --skip-eot          [%-7s] suppress EOT generation\n",            params.skip_eot ? "true" : "false");
    fprintf(stderr, "            --no-skip-punct     [%-7s] don't skip punctuation delta\n",       params.skip_punctuation_delta ? "false" : "true");
    fprintf(stderr, "            --no-skip-subword   [%-7s] don't skip subword delta\n",           params.skip_subword_delta ? "false" : "true");
    fprintf(stderr, "  -tr,      --translate         [%-7s] translate to English\n",               params.translate ? "true" : "false");
    fprintf(stderr, "  -l LANG,  --language LANG     [%-7s] spoken language\n",                    params.language.c_str());
    fprintf(stderr, "  -m FNAME, --model FNAME       [%-7s] model path\n",                         params.model.c_str());
    fprintf(stderr, "  -of FNAME,--output-file FNAME           output file path\n");
    fprintf(stderr, "  -ng,      --no-gpu            [%-7s] disable GPU\n",                        params.use_gpu ? "false" : "true");
    fprintf(stderr, "            --no-realtime       [%-7s] skip sleep, process as fast as possible\n", params.no_realtime ? "true" : "false");
    fprintf(stderr, "            --no-postprocess    [%-7s] disable dedup and prompt prefill\n",    params.no_postprocess ? "true" : "false");
    fprintf(stderr, "            --debug             [%-7s] print debug info\n",                   params.debug ? "true" : "false");
    fprintf(stderr, "\n");
}

// =====================================================================
// Signal processing helpers
// =====================================================================

// 1D median filter with reflect padding (matches scipy.ndimage.median_filter)
static void median_filter_1d(std::vector<float> & data, int size) {
    if (size <= 1 || data.empty()) return;
    int half = size / 2;
    int n = (int)data.size();
    std::vector<float> buf(n);
    std::vector<float> window(size);

    for (int i = 0; i < n; i++) {
        for (int j = -half; j <= half; j++) {
            int idx = i + j;
            // Reflect padding
            if (idx < 0) idx = -idx;
            if (idx >= n) idx = 2 * (n - 1) - idx;
            if (idx < 0) idx = 0;
            if (idx >= n) idx = n - 1;
            window[j + half] = data[idx];
        }
        std::sort(window.begin(), window.end());
        buf[i] = window[half];
    }
    data = buf;
}

// Moving average smoothing (matches np.convolve mode='same' with zero-padding)
static void moving_average(std::vector<float> & data, int window) {
    if (window <= 1 || data.empty()) return;
    int n = (int)data.size();
    int half = window / 2;
    float inv_w = 1.0f / window;
    std::vector<float> result(n, 0.0f);

    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int j = 0; j < window; j++) {
            int idx = i - half + j;
            if (idx >= 0 && idx < n) {
                sum += data[idx];
            }
            // else: zero-padding (implicit)
        }
        result[i] = sum * inv_w;
    }
    data = result;
}

// =====================================================================
// Cross-attention helpers
// =====================================================================

// Extract attention for a single token, averaged across all heads.
// Input data layout (ggml column-major): data[a + t * n_audio_ctx + h * n_audio_ctx * n_tokens]
// Returns vector of length n_audio_ctx.
static std::vector<float> get_avg_attn_for_token(
    const float * data, int n_audio_ctx, int n_tokens, int n_heads, int token_idx) {
    std::vector<float> avg(n_audio_ctx, 0.0f);
    for (int h = 0; h < n_heads; h++) {
        for (int a = 0; a < n_audio_ctx; a++) {
            avg[a] += data[a + token_idx * n_audio_ctx + h * n_audio_ctx * n_tokens];
        }
    }
    float inv = 1.0f / n_heads;
    for (auto & v : avg) v *= inv;
    return avg;
}

// =====================================================================
// DTW (Dynamic Time Warping) for word-level timestamp estimation
// Based on whisper/timing.py — used for DTW-based carryover modes
// =====================================================================

// Standard DTW with backtrace.  cost[i][j] = cost for text token i at audio frame j.
// Returns monotonic alignment path in (text_indices, time_indices).
static void dtw_compute(
    const std::vector<std::vector<float>> & cost,
    int N, int M,
    std::vector<int> & text_indices,
    std::vector<int> & time_indices)
{
    // Accumulated cost matrix (flattened)
    std::vector<float> D(N * M, INFINITY);
    auto idx = [M](int i, int j) { return i * M + j; };

    D[idx(0, 0)] = cost[0][0];
    for (int i = 1; i < N; i++) D[idx(i, 0)] = D[idx(i-1, 0)] + cost[i][0];
    for (int j = 1; j < M; j++) D[idx(0, j)] = D[idx(0, j-1)] + cost[0][j];

    for (int i = 1; i < N; i++) {
        for (int j = 1; j < M; j++) {
            D[idx(i, j)] = cost[i][j] + std::min({D[idx(i-1, j)], D[idx(i-1, j-1)], D[idx(i, j-1)]});
        }
    }

    // Backtrace
    text_indices.clear();
    time_indices.clear();
    int i = N - 1, j = M - 1;
    text_indices.push_back(i);
    time_indices.push_back(j);

    while (i > 0 || j > 0) {
        if      (i == 0) { j--; }
        else if (j == 0) { i--; }
        else {
            float d = D[idx(i-1, j-1)], u = D[idx(i-1, j)], l = D[idx(i, j-1)];
            if (d <= u && d <= l) { i--; j--; }
            else if (u <= l)      { i--; }
            else                  { j--; }
        }
        text_indices.push_back(i);
        time_indices.push_back(j);
    }
    std::reverse(text_indices.begin(), text_indices.end());
    std::reverse(time_indices.begin(), time_indices.end());
}

// Forward declaration (defined below in word splitting section)
static std::vector<std::vector<whisper_token>> split_to_words(
    struct whisper_context * ctx,
    const std::vector<whisper_token> & tokens,
    const std::string & language = "en");

// DTW alignment result for word-level timestamps
struct dtw_result {
    float last_word_end   = -1.0f;  // end time of last word
    float last_word_start = -1.0f;  // start time of last word
    float second_last_word_start = -1.0f; // start time of second-to-last word
    int   tokens_excl_last_word = 0; // token count excluding last word
    int   tokens_excl_last_2_words = 0; // token count excluding last 2 words
    bool  valid = false;
};

// Compute word-level timestamps using DTW alignment.
// Appends a dummy token (uniform attention) to absorb trailing frames,
// analogous to EOT in Whisper's DTW — so the last real word gets a proper end time.
static dtw_result compute_dtw_alignment(
    struct whisper_context * ctx,
    const std::vector<whisper_token> & tokens,
    const std::vector<std::vector<float>> & token_attns,
    float frame_duration,
    bool debug,
    const std::string & language = "en")
{
    dtw_result result;
    if (tokens.empty()) return result;

    // Find n_audio_ctx from first valid attention vector
    int n_audio_ctx = 0;
    for (auto & a : token_attns) {
        if (!a.empty()) { n_audio_ctx = (int)a.size(); break; }
    }
    if (n_audio_ctx == 0) return result;

    int n_tokens = (int)tokens.size();

    // Verify all tokens have valid attention
    for (int i = 0; i < n_tokens; i++) {
        if (i >= (int)token_attns.size() || (int)token_attns[i].size() != n_audio_ctx) {
            if (debug) fprintf(stderr, "  [DTW] Token %d missing attention, falling back to peak\n", i);
            return result;
        }
    }

    // Build matrix: (n_tokens + 1 dummy) × n_audio_ctx
    int N = n_tokens + 1;  // +1 for dummy (absorbs trailing frames like EOT)
    int M = n_audio_ctx;
    std::vector<std::vector<float>> matrix(N, std::vector<float>(M));
    for (int i = 0; i < n_tokens; i++) matrix[i] = token_attns[i];
    // Dummy token: uniform attention
    for (int j = 0; j < M; j++) matrix[n_tokens][j] = 1.0f / M;

    // Z-score normalize across token dimension (per frame) — matches Whisper timing.py
    for (int j = 0; j < M; j++) {
        float sum = 0, sum2 = 0;
        for (int i = 0; i < N; i++) { sum += matrix[i][j]; sum2 += matrix[i][j] * matrix[i][j]; }
        float mean = sum / N;
        float std_v = sqrtf(sum2 / N - mean * mean + 1e-8f);
        for (int i = 0; i < N; i++) matrix[i][j] = (matrix[i][j] - mean) / std_v;
    }

    // Median filter per token (width=7, matches Whisper default)
    for (int i = 0; i < N; i++) median_filter_1d(matrix[i], 7);

    // Negate for DTW (minimize cost = maximize attention)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++)
            matrix[i][j] = -matrix[i][j];

    // Run DTW
    std::vector<int> ti, tj;
    dtw_compute(matrix, N, M, ti, tj);

    // Extract jump_times: one entry per unique text_index (= N entries)
    // jump_times[i] = audio time where token i first appears in DTW path
    std::vector<float> jump_times;
    for (int k = 0; k < (int)ti.size(); k++) {
        if (k == 0 || ti[k] != ti[k-1]) {
            jump_times.push_back(tj[k] * frame_duration);
        }
    }

    if ((int)jump_times.size() != N) {
        if (debug) fprintf(stderr, "  [DTW] jump_times size %d != N %d\n", (int)jump_times.size(), N);
        return result;
    }

    // Split tokens into words
    auto words = split_to_words(ctx, tokens, language);
    if (words.empty()) return result;

    // Word boundaries: cumulative token count
    int boundary = 0;
    for (int w = 0; w < (int)words.size(); w++) {
        float word_start = (boundary < (int)jump_times.size()) ? jump_times[boundary] : -1;
        boundary += (int)words[w].size();
        float word_end = (boundary < (int)jump_times.size()) ? jump_times[boundary] : -1;

        if (w == (int)words.size() - 2) {
            result.second_last_word_start = word_start;
            result.tokens_excl_last_2_words = boundary - (int)words[w].size();
        }
        if (w == (int)words.size() - 1) {
            result.last_word_end = word_end;
            result.last_word_start = word_start;
            result.tokens_excl_last_word = n_tokens - (int)words[w].size();
            if ((int)words.size() == 1) {
                // Only one word — no second-to-last
                result.tokens_excl_last_2_words = 0;
            }
        }
    }

    result.valid = (result.last_word_end >= 0);

    // Detect DTW failure: all word timestamps collapsed to 0
    if (result.valid && result.last_word_end <= 0.0f && result.last_word_start <= 0.0f) {
        if (debug) fprintf(stderr, "  [DTW] WARNING: all timestamps collapsed to 0 (n_tokens=%d, n_audio_ctx=%d)\n", n_tokens, M);
    }

    if (debug && result.valid) {
        int b = 0;
        for (int w = 0; w < (int)words.size(); w++) {
            float start = (b < (int)jump_times.size()) ? jump_times[b] : -1;
            b += (int)words[w].size();
            float end = (b < (int)jump_times.size()) ? jump_times[b] : -1;
            std::string ws;
            for (auto t : words[w]) {
                const char * s = whisper_token_to_str(ctx, t);
                if (s) ws += s;
            }
            fprintf(stderr, "  [DTW] word[%d]='%s' %.2fs-%.2fs\n", w, ws.c_str(), start, end);
        }
    }

    return result;
}

// =====================================================================
// Token classification helpers
// =====================================================================

static bool is_punctuation_token(struct whisper_context * ctx, whisper_token tok) {
    const char * str = whisper_token_to_str(ctx, tok);
    if (!str || strlen(str) == 0) return true;
    // Strip leading space
    const char * p = str;
    if (*p == ' ') p++;
    if (*p == '\0') return true;
    // Check if remaining is 1-2 chars and non-alphanumeric
    int len = (int)strlen(p);
    if (len <= 2) {
        bool all_non_alnum = true;
        for (int i = 0; i < len; i++) {
            if (isalnum((unsigned char)p[i])) { all_non_alnum = false; break; }
        }
        if (all_non_alnum) return true;
    }
    return false;
}

static bool is_cjk_language(const std::string & lang) {
    return lang == "ko" || lang == "zh";
}

static bool is_subword_token(struct whisper_context * ctx, whisper_token tok, const std::string & language = "en") {
    // Chinese: no subword concept (no spaces between words) — always return false
    if (language == "zh") return false;
    const char * str = whisper_token_to_str(ctx, tok);
    if (!str || strlen(str) == 0) return false;
    // Subword = no leading space AND has alphabetic content
    if (str[0] == ' ') return false;
    for (const char * p = str; *p; p++) {
        if (isalpha((unsigned char)*p)) return true;
        // Korean: UTF-8 multibyte chars count as alphabetic
        if (language == "ko" && ((unsigned char)*p >= 0x80)) return true;
    }
    return false;
}

// =====================================================================
// Word splitting and conservative emission (same as simul_streaming)
// =====================================================================

static std::vector<std::vector<whisper_token>> split_to_words(
    struct whisper_context * ctx,
    const std::vector<whisper_token> & tokens,
    const std::string & language)
{
    std::vector<std::vector<whisper_token>> words;

    // Chinese: each token is its own "word" (no spaces between characters)
    if (language == "zh") {
        for (auto tok : tokens) {
            words.push_back({tok});
        }
        return words;
    }

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

static std::vector<whisper_token> emit_conservative(
    struct whisper_context * ctx,
    const std::vector<whisper_token> & decoded_tokens,
    bool is_last,
    const std::string & language = "en")
{
    if (is_last) return decoded_tokens;

    auto words = split_to_words(ctx, decoded_tokens, language);
    if (words.size() <= 1) return {};

    std::vector<whisper_token> emitted;
    for (size_t i = 0; i < words.size() - 1; i++) {
        for (auto t : words[i]) emitted.push_back(t);
    }
    return emitted;
}

// =====================================================================
// UTF-8 helpers
// =====================================================================

static bool ends_with_incomplete_utf8(const std::string & s) {
    if (s.empty()) return false;
    int i = (int)s.size() - 1;
    int cont = 0;
    while (i >= 0 && (s[i] & 0xC0) == 0x80) { cont++; i--; }
    if (i < 0) return true;
    unsigned char lead = (unsigned char)s[i];
    int expected = 0;
    if      ((lead & 0x80) == 0x00) expected = 0;
    else if ((lead & 0xE0) == 0xC0) expected = 1;
    else if ((lead & 0xF0) == 0xE0) expected = 2;
    else if ((lead & 0xF8) == 0xF0) expected = 3;
    return cont < expected;
}

// =====================================================================
// Chunk-boundary dedup helpers
// =====================================================================

static std::string normalize_word_for_dedup(const std::string & w) {
    std::string result;
    for (char c : w) {
        unsigned char uc = static_cast<unsigned char>(c);
        if (std::isalnum(uc)) {
            result += std::tolower(uc);
        } else if (uc >= 0x80) {
            // Preserve UTF-8 multibyte chars (CJK, Korean, etc.)
            result += c;
        }
    }
    return result;
}

// Extract normalized words from tokens, each with its starting token index
struct DedupeWordInfo {
    std::string norm;   // normalized word
    int token_idx;      // first token index of this word in the token vector
};

static std::vector<DedupeWordInfo> extract_words_for_dedup(
    struct whisper_context * ctx,
    const std::vector<whisper_token> & tokens)
{
    std::vector<DedupeWordInfo> words;
    std::string current;
    int word_start = 0;

    for (int i = 0; i < (int)tokens.size(); i++) {
        const char * s = whisper_token_to_str(ctx, tokens[i]);
        if (!s) continue;
        std::string tok(s);

        // Leading space = new word boundary
        if (!tok.empty() && tok[0] == ' ') {
            if (!current.empty()) {
                std::string norm = normalize_word_for_dedup(current);
                if (!norm.empty()) {
                    words.push_back({norm, word_start});
                }
            }
            current.clear();
            word_start = i;
            tok = tok.substr(1);
        }

        if (!tok.empty()) {
            if (current.empty()) word_start = i;
            current += tok;
        }
    }
    if (!current.empty()) {
        std::string norm = normalize_word_for_dedup(current);
        if (!norm.empty()) {
            words.push_back({norm, word_start});
        }
    }
    return words;
}

// Dedup: find overlap between prev_tail (last 3 words) and curr prefix (first 3 words).
// If a suffix of prev_tail (min 2 words) matches inside curr prefix,
// Normalize text for comparison: lowercase, strip leading/trailing whitespace and punctuation
static std::string normalize_for_compare(const std::string & s) {
    std::string result;
    for (char c : s) {
        unsigned char uc = (unsigned char)c;
        if (std::isalnum(uc)) {
            result += std::tolower(uc);
        } else if (uc >= 0x80) {
            result += c;
        }
    }
    return result;
}

// skip tokens up to end of match (removes junk like "daughter" between chunks).
// Returns number of tokens to skip from emitted, or 0 if no dedup needed.
static int find_dedup_skip(
    struct whisper_context * ctx,
    const std::vector<std::string> & prev_tail,
    const std::vector<whisper_token> & emitted,
    bool debug)
{
    if (prev_tail.empty() || emitted.empty()) return 0;

    auto curr_words = extract_words_for_dedup(ctx, emitted);
    int check_n = std::min(3, (int)curr_words.size());
    int prev_n = (int)prev_tail.size();

    for (int slen = std::min(prev_n, 3); slen >= 2; slen--) {
        for (int start = 0; start + slen <= check_n; start++) {
            bool match = true;
            for (int k = 0; k < slen; k++) {
                if (prev_tail[prev_n - slen + k] != curr_words[start + k].norm) {
                    match = false;
                    break;
                }
            }
            if (match) {
                int end_word = start + slen;
                int skip_tokens = (end_word < (int)curr_words.size())
                    ? curr_words[end_word].token_idx
                    : (int)emitted.size();
                if (debug) {
                    fprintf(stderr, "[DEDUP] prev_tail suffix matched %d words at curr pos %d, skipping %d tokens\n",
                            slen, start, skip_tokens);
                }
                return skip_tokens;
            }
        }
    }
    return 0;
}

// Update prev_tail_words with last 3 normalized words from emitted tokens
static void update_prev_tail(
    struct whisper_context * ctx,
    const std::vector<whisper_token> & emitted,
    std::vector<std::string> & prev_tail)
{
    auto words = extract_words_for_dedup(ctx, emitted);
    prev_tail.clear();
    int start = std::max(0, (int)words.size() - 3);
    for (int i = start; i < (int)words.size(); i++) {
        prev_tail.push_back(words[i].norm);
    }
}

// =====================================================================
// Output helpers (matching simul_streaming format for comparison)
// =====================================================================

void print_committed(const std::vector<std::string> & committed) {
    fprintf(stderr, "\n=== Final transcript ===\n");
    for (const auto & w : committed) {
        fprintf(stderr, "%s", w.c_str());
    }
    fprintf(stderr, "\n========================\n");
}

void print_tsw_with_latency(const std::vector<std::tuple<double, double, std::string, double, double>> & records,
                            const std::string & language = "en") {
    double latency_sum = 0.0;
    int count = 0;

    // For Korean: merge tokens with incomplete UTF-8 before printing
    if (is_cjk_language(language)) {
        std::string pending_text;
        double pending_start = 0, pending_end = 0, pending_lat = 0, pending_emit = 0;
        for (const auto & entry : records) {
            if (pending_text.empty()) {
                pending_start = std::get<0>(entry);
                pending_end   = std::get<1>(entry);
                pending_lat   = std::get<3>(entry);
                pending_emit  = std::get<4>(entry);
            }
            pending_text += std::get<2>(entry);
            pending_end = std::get<1>(entry);
            if (!ends_with_incomplete_utf8(pending_text)) {
                printf("Start Time: %g, End Time: %g, Transcript: %s, Latency: %g, Emission Time: %g\n",
                       pending_start, pending_end, pending_text.c_str(), pending_lat, pending_emit);
                latency_sum += pending_lat;
                count++;
                pending_text.clear();
            }
        }
        if (!pending_text.empty()) {
            printf("Start Time: %g, End Time: %g, Transcript: %s, Latency: %g, Emission Time: %g\n",
                   pending_start, pending_end, pending_text.c_str(), pending_lat, pending_emit);
            latency_sum += pending_lat;
            count++;
        }
    } else {
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
    }
    if (count > 0) {
        printf("Average latency: %g\n", latency_sum / count);
    }
}


// =====================================================================
// Reinfer: skip EOT after sentence-ending punctuation (.?!), log top-5
// =====================================================================
struct reinfer_eot_log_data {
    bool debug;
    float audio_seconds;  // effective audio duration for min-token EOT suppression
    // Record each EOT event: (step, top5 info string, suppressed)
    std::vector<std::tuple<int, std::string, bool>> eot_events;
    // Track last generated token text to detect sentence-ending punctuation
    std::string last_token_text;
};

static void reinfer_logits_filter(
        struct whisper_context * ctx,
        struct whisper_state * /*state*/,
        const whisper_token_data * tokens,
        int   n_tokens,
        float * logits,
        void * user_data) {
    auto * data = (reinfer_eot_log_data *)user_data;
    const int eot_id = whisper_token_eot(ctx);
    const int n_vocab = whisper_n_vocab(ctx);

    // Track last token text
    if (n_tokens > 0) {
        const char * s = whisper_token_to_str(ctx, tokens[n_tokens - 1].id);
        data->last_token_text = s ? s : "";
    }

    // Find argmax
    int best_id = 0;
    for (int v = 1; v < n_vocab; v++) {
        if (logits[v] > logits[best_id]) best_id = v;
    }

    // Only act when EOT is the top token
    if (best_id != eot_id) return;

    // Find top 5 by logit value (excluding timestamp tokens)
    const int beg_id = whisper_token_beg(ctx);
    std::vector<std::pair<float, int>> top5;
    for (int v = 0; v < n_vocab; v++) {
        if (v >= beg_id && v != eot_id) continue;  // skip timestamp tokens
        if ((int)top5.size() < 5) {
            top5.push_back({logits[v], v});
            std::sort(top5.begin(), top5.end(), [](auto & a, auto & b){ return a.first > b.first; });
        } else if (logits[v] > top5.back().first) {
            top5.back() = {logits[v], v};
            std::sort(top5.begin(), top5.end(), [](auto & a, auto & b){ return a.first > b.first; });
        }
    }

    // Build info string
    std::string info;
    for (int i = 0; i < (int)top5.size(); i++) {
        int vid = top5[i].second;
        float val = top5[i].first;
        const char * tok_str = (vid == eot_id) ? "[EOT]" : whisper_token_to_str(ctx, vid);
        if (!tok_str) tok_str = "?";
        char buf[128];
        snprintf(buf, sizeof(buf), "%s'%s'=%.2f", (i > 0 ? ", " : ""), tok_str, val);
        info += buf;
    }

    // Check if last token ends with sentence-ending punctuation
    bool after_sentence_end = false;
    if (!data->last_token_text.empty()) {
        char last_char = data->last_token_text.back();
        if (last_char == '.' || last_char == '?' || last_char == '!') {
            after_sentence_end = true;
        }
    }

    // Avoid duplicate logs for the same step
    bool duplicate = false;
    for (const auto & prev : data->eot_events) {
        if (std::get<0>(prev) == n_tokens) { duplicate = true; break; }
    }

    // Suppress EOT if too few tokens relative to audio duration (~1.5 tokens/sec minimum)
    float min_tokens = data->audio_seconds * 1.5f;
    bool too_few_tokens = (n_tokens < (int)min_tokens);

    // Suppress EOT only if too few tokens for the audio duration.
    // after_sentence_end alone is not enough — repeated sentences would suppress EOT forever.
    if (too_few_tokens) {
        // Suppress EOT
        logits[eot_id] = -INFINITY;
        if (!duplicate) {
            const char * reason = after_sentence_end ? "min_tokens+sentence_boundary" : "min_tokens";
            data->eot_events.push_back({n_tokens, info, true});
            if (data->debug) {
                fprintf(stderr, "  [REINFER] EOT after '%s' at step %d — SKIPPED (%s, need >= %.0f tokens for %.1fs audio), top5: %s\n",
                        data->last_token_text.c_str(), n_tokens, reason, min_tokens, data->audio_seconds, info.c_str());
            }
        }
    } else {
        // Allow EOT
        if (!duplicate) {
            data->eot_events.push_back({n_tokens, info, false});
            if (data->debug) {
                fprintf(stderr, "  [REINFER] EOT at step %d, top5: %s\n", n_tokens, info.c_str());
            }
        }
    }
}

// =====================================================================
// Main
// =====================================================================
int main(int argc, char ** argv) {
    ours_streaming_params params;

    if (!params_parse(argc, argv, params)) {
        return 1;
    }

    if (params.fname_inp.empty()) {
        fprintf(stderr, "error: no input file specified\n");
        print_usage(argc, argv, params);
        return 1;
    }

    if (params.flash_attn) {
        fprintf(stderr, "error: --flash-attn is not supported (need KQ_soft_max access for backward peak detection)\n");
        return 1;
    }

    // Init whisper context
    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu    = params.use_gpu;
    cparams.flash_attn = false;

    // Enable alignment heads DTW if carryover mode uses DTW (mode 1, 2)
    bool use_aheads_dtw = (params.carryover_mode == 1 || params.carryover_mode == 2 || params.carryover_mode == 3);
    if (use_aheads_dtw) {
        cparams.dtw_token_timestamps = true;

        // Auto-detect model preset from model path if not specified
        std::string dtw_preset = params.dtw_model;
        if (dtw_preset.empty()) {
            // Extract from model filename: ggml-base.bin → base
            std::string mpath = params.model;
            auto slash = mpath.find_last_of('/');
            if (slash != std::string::npos) mpath = mpath.substr(slash + 1);
            // Parse "ggml-{model}.bin" or "ggml-{model}.en.bin"
            if (mpath.find("tiny.en") != std::string::npos) dtw_preset = "tiny.en";
            else if (mpath.find("tiny") != std::string::npos) dtw_preset = "tiny";
            else if (mpath.find("base.en") != std::string::npos) dtw_preset = "base.en";
            else if (mpath.find("base") != std::string::npos) dtw_preset = "base";
            else if (mpath.find("small.en") != std::string::npos) dtw_preset = "small.en";
            else if (mpath.find("small") != std::string::npos) dtw_preset = "small";
            else if (mpath.find("medium.en") != std::string::npos) dtw_preset = "medium.en";
            else if (mpath.find("medium") != std::string::npos) dtw_preset = "medium";
            else if (mpath.find("large-v3") != std::string::npos) dtw_preset = "large.v3";
            else if (mpath.find("large-v2") != std::string::npos) dtw_preset = "large.v2";
            else if (mpath.find("large-v1") != std::string::npos || mpath.find("large") != std::string::npos) dtw_preset = "large.v1";
        }

        cparams.dtw_aheads_preset = WHISPER_AHEADS_NONE;
        if (dtw_preset == "tiny")       cparams.dtw_aheads_preset = WHISPER_AHEADS_TINY;
        if (dtw_preset == "tiny.en")    cparams.dtw_aheads_preset = WHISPER_AHEADS_TINY_EN;
        if (dtw_preset == "base")       cparams.dtw_aheads_preset = WHISPER_AHEADS_BASE;
        if (dtw_preset == "base.en")    cparams.dtw_aheads_preset = WHISPER_AHEADS_BASE_EN;
        if (dtw_preset == "small")      cparams.dtw_aheads_preset = WHISPER_AHEADS_SMALL;
        if (dtw_preset == "small.en")   cparams.dtw_aheads_preset = WHISPER_AHEADS_SMALL_EN;
        if (dtw_preset == "medium")     cparams.dtw_aheads_preset = WHISPER_AHEADS_MEDIUM;
        if (dtw_preset == "medium.en")  cparams.dtw_aheads_preset = WHISPER_AHEADS_MEDIUM_EN;
        if (dtw_preset == "large.v1")   cparams.dtw_aheads_preset = WHISPER_AHEADS_LARGE_V1;
        if (dtw_preset == "large.v2")   cparams.dtw_aheads_preset = WHISPER_AHEADS_LARGE_V2;
        if (dtw_preset == "large.v3")   cparams.dtw_aheads_preset = WHISPER_AHEADS_LARGE_V3;

        if (cparams.dtw_aheads_preset == WHISPER_AHEADS_NONE) {
            fprintf(stderr, "error: could not detect alignment heads preset from model '%s'. Use --dtw <model>\n", params.model.c_str());
            return 1;
        }

        fprintf(stderr, "%s: DTW alignment heads preset: %s\n", __func__, dtw_preset.c_str());
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
    fprintf(stderr, "%s: Ours (Backward Peak Detection) streaming mode\n", __func__);
    fprintf(stderr, "%s: model = %s, language = %s\n", __func__, params.model.c_str(), params.language.c_str());
    fprintf(stderr, "%s: audio = %.1f sec, step = %d ms\n",
            __func__, (float)pcmf32_all.size() / WHISPER_SAMPLE_RATE, params.step_ms);
    fprintf(stderr, "%s: smoothing = %d, median_filter = %d, cross_attn_layer = %d\n",
            __func__, params.smoothing_window, params.median_filter_window, params.cross_attn_layer);
    fprintf(stderr, "%s: peak_margin = %.2f s, carryover_overlap = %.2f s, min_chunk = %.2f s\n",
            __func__, params.peak_margin_sec, params.carryover_overlap_sec, params.min_chunk_sec);
    fprintf(stderr, "%s: prompt_prefill = %d, skip_eot = %s\n",
            __func__, params.prompt_prefill_n, params.skip_eot ? "true" : "false");
    fprintf(stderr, "%s: carryover_mode = %d (%s), word_end_offset = %.2f s\n",
            __func__, params.carryover_mode,
            params.carryover_mode == 0 ? "peak" : (params.carryover_mode == 1 ? "prefill_last" :
             (params.carryover_mode == 2 ? "emit_all_overlap" : "emit_all_overlap+reinfer")),
            params.word_end_offset_sec);
    fprintf(stderr, "\n");

    // State
    std::vector<float> audio_buffer;               // accumulated audio
    std::vector<std::string> committed_words;      // committed word strings
    std::vector<whisper_token> committed_tokens;    // committed token IDs
    std::vector<std::string> prev_tail_words;      // last 3 normalized words for dedup
    int64_t buffer_start_sample = 0;               // absolute sample offset

    // Carryover state
    std::vector<float> carryover_audio;            // audio carried from previous chunk
    int64_t carryover_abs_start_sample = 0;        // absolute start sample of carryover audio
    std::vector<whisper_token> prev_chunk_token_ids; // for prompt prefill

    // Speculative emission state (for DTW failure recovery)
    bool prev_speculative = false;                   // previous round was speculative
    int speculative_committed_count = 0;             // number of speculatively committed tokens
    std::vector<whisper_token> pre_speculative_prefill; // prefill from before speculative round

    // Mode 3: track repeated backward_peak at same step/token for reinfer trigger
    int prev_bp_step = -1;
    whisper_token prev_bp_token = -1;
    int consecutive_same_bp_count = 0;


    // Mode 5 overlap dedup: last word from previous chunk to be matched/removed
    int olap_prev_last_word_token_count = 0;         // number of tokens in previous chunk's last word
    std::string olap_prev_last_word_text;             // normalized text of previous chunk's last word
    double olap_replace_emission_time = 0.0;         // preserved emission time for replace (no match)
    int olap_replace_token_count = 0;                // number of cur first word tokens to apply prev emission time

    // Latency tracking: (start_time, end_time, transcript, latency)
    std::vector<std::tuple<double, double, std::string, double, double>> latency_record;

    const int n_samples_step = (params.step_ms * WHISPER_SAMPLE_RATE) / 1000;
    int64_t pcmf32_index = 0;
    int n_iter = 0;
    int backward_peak_count = 0;
    int peak_margin_count = 0;
    int carryover_count = 0;
    const auto t_start = std::chrono::high_resolution_clock::now();

    while (pcmf32_index < (int64_t)pcmf32_all.size()) {
        bool is_last = false;

        int64_t chunk_size;
        if (params.no_realtime) {
            // No-realtime mode: take exactly one step worth of audio, no sleep
            chunk_size = std::min((int64_t)n_samples_step, (int64_t)pcmf32_all.size() - pcmf32_index);
        } else {
            // Simulate real-time audio arrival
            double target_time = (double)(pcmf32_index + n_samples_step) / WHISPER_SAMPLE_RATE;
            auto t_now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(t_now - t_start).count();
            if (elapsed < target_time) {
                double sleep_sec = target_time - elapsed;
                std::this_thread::sleep_for(std::chrono::microseconds((int64_t)(sleep_sec * 1e6)));
            }

            // Get next audio chunk (elapsed-time-based: consume all audio that has "arrived")
            auto t_after_sleep = std::chrono::high_resolution_clock::now();
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

        // Prepend carryover audio if available
        std::vector<float> effective_audio;
        float carryover_duration = 0.0f;
        int64_t effective_abs_start_sample = 0;  // absolute start sample of effective_audio
        if (!carryover_audio.empty()) {
            effective_audio = carryover_audio;
            effective_audio.insert(effective_audio.end(), audio_buffer.begin(), audio_buffer.end());
            carryover_duration = (float)carryover_audio.size() / WHISPER_SAMPLE_RATE;
            effective_abs_start_sample = carryover_abs_start_sample;
            carryover_audio.clear();
        } else {
            effective_abs_start_sample = pcmf32_index - chunk_size;
            effective_audio = audio_buffer;
        }

        // Trim to 30s max (Whisper positional embedding limit)
        while (effective_audio.size() > 30 * WHISPER_SAMPLE_RATE) {
            int remove_samples = effective_audio.size() - 30 * WHISPER_SAMPLE_RATE;
            effective_audio.erase(effective_audio.begin(), effective_audio.begin() + remove_samples);
            buffer_start_sample += remove_samples;
        }

        float effective_seconds = (float)effective_audio.size() / WHISPER_SAMPLE_RATE;

        // Skip if audio too short
        if (params.min_chunk_sec > 0 && effective_seconds < params.min_chunk_sec && !is_last) {
            if (params.debug) {
                fprintf(stderr, "[iter=%d] audio too short (%.2fs < %.2fs), skipping\n",
                        n_iter, effective_seconds, params.min_chunk_sec);
            }
            carryover_audio = effective_audio;
            carryover_abs_start_sample = effective_abs_start_sample;
            audio_buffer.clear();
            n_iter++;
            continue;
        }

        // Buffer stats
        printf("[BUFFER_STATS] iter=%d buffer_sec=%.3f buffer_samples=%ld\n",
               n_iter, effective_seconds, (long)effective_audio.size());

        // Content mel len for this chunk (no padding)
        int content_mel_len = (int)effective_audio.size() / 320;
        if (content_mel_len < 1) content_mel_len = 1;

        // Input stats
        printf("[INPUT_STATS] iter=%d input_sec=%.3f input_samples=%ld content_mel_len=%d\n",
               n_iter, effective_seconds, (long)effective_audio.size(), content_mel_len);

        if (params.debug) {
            float abs_start_sec = (float)effective_abs_start_sample / WHISPER_SAMPLE_RATE;
            float abs_end_sec = abs_start_sec + effective_seconds;
            fprintf(stderr, "[iter=%d] effective=%.2fs, carryover=%.2fs, is_last=%d, abs=%.2f-%.2fs\n",
                    n_iter, effective_seconds, carryover_duration, is_last, abs_start_sec, abs_end_sec);
        }

        // =====================================================
        // Token-by-token decoding with backward peak detection
        // Matches Python: pcm_to_mel → encode → decode loop
        // =====================================================

        // Step 1: PCM → mel spectrogram (no padding)
        auto t_mel_s = std::chrono::high_resolution_clock::now();
        if (whisper_pcm_to_mel(ctx, effective_audio.data(), effective_audio.size(), params.n_threads) != 0) {
            fprintf(stderr, "%s: failed to compute mel spectrogram\n", __func__);
            audio_buffer.clear();
            n_iter++;
            continue;
        }
        auto t_mel_e = std::chrono::high_resolution_clock::now();
        double iter_mel_us = std::chrono::duration<double, std::micro>(t_mel_e - t_mel_s).count();

        // Set audio_ctx to match actual audio length BEFORE encode (no 30s padding)
        // This ensures both encoder and decoder use the correct audio context size.
        // In Python no_padding: encoder processes only actual mel frames, not 1500.
        whisper_set_audio_ctx(ctx, -1);
        {
            const int mel_shape_frames = whisper_n_len(ctx);
            const int encoder_ctx = ((mel_shape_frames / 2 + 7) / 8) * 8;
            const int encoder_mel_frames = encoder_ctx * 2;
            const float encoder_input_sec = (float) encoder_mel_frames / 100.0f;
            printf("[ENCODER_INPUT_STATS] iter=%d mel_shape_frames=%d encoder_ctx=%d encoder_mel_frames=%d encoder_input_sec=%.3f\n",
                   n_iter, mel_shape_frames, encoder_ctx, encoder_mel_frames, encoder_input_sec);
        }

        if (params.debug) {
            fprintf(stderr, "  mel.n_len=%d, content_mel_len=%d\n",
                    whisper_n_len(ctx), content_mel_len);
        }

        // Step 2: Encode
        auto t_enc_s = std::chrono::high_resolution_clock::now();
        if (whisper_encode(ctx, 0, params.n_threads) != 0) {
            fprintf(stderr, "%s: failed to encode\n", __func__);
            audio_buffer.clear();
            n_iter++;
            continue;
        }
        auto t_enc_e = std::chrono::high_resolution_clock::now();
        double iter_encode_us = std::chrono::duration<double, std::micro>(t_enc_e - t_enc_s).count();

        // Step 3: Build initial token sequence [sot, lang, transcribe, notimestamps] + prefill
        std::vector<whisper_token> prompt_tokens;
        if (!prev_chunk_token_ids.empty()) {
            if (params.carryover_mode == 2 || params.carryover_mode == 3) {
                // Mode 2: prefill with tokens excluding last word (set by mode 2 logic)
                prompt_tokens = prev_chunk_token_ids;
            } else if (params.carryover_mode == 1) {
                // Mode 1: always prefill with last word tokens (set by mode 1 logic)
                prompt_tokens = prev_chunk_token_ids;
            } else if (!params.no_postprocess && params.prompt_prefill_n > 0) {
                int start = std::max(0, (int)prev_chunk_token_ids.size() - params.prompt_prefill_n);
                prompt_tokens.insert(prompt_tokens.end(),
                                     prev_chunk_token_ids.begin() + start,
                                     prev_chunk_token_ids.end());
            }
        }

        std::vector<whisper_token> token_seq;
        token_seq.push_back(whisper_token_sot(ctx));
        if (whisper_is_multilingual(ctx)) {
            token_seq.push_back(whisper_token_lang(ctx, whisper_lang_id(params.language.c_str())));
            if (params.translate) {
                token_seq.push_back(whisper_token_translate(ctx));
            } else {
                token_seq.push_back(whisper_token_transcribe(ctx));
            }
        }
        token_seq.push_back(whisper_token_not(ctx));  // no_timestamps
        for (auto t : prompt_tokens) {
            token_seq.push_back(t);  // prefill after <notimestamps>
        }
        int sot_sequence_len = (int)token_seq.size();

        if (params.debug) {
            fprintf(stderr, "  Initial tokens (%d):", sot_sequence_len);
            for (auto t : token_seq) {
                fprintf(stderr, " %d('%s')", t, whisper_token_to_str(ctx, t));
            }
            fprintf(stderr, "\n");
        }

        // Enable layer cross-attention capture for every decode call (backward peak)
        whisper_set_layer_cross_attention_capture(ctx, params.cross_attn_layer);

        // Enable alignment heads capture for DTW (carryover modes 1, 2)
        if (use_aheads_dtw) {
            whisper_set_aheads_cross_attention_capture(ctx, true);
        }

        // Decode the initial prompt tokens (prefill, feed all at once with n_past=0)
        auto t_prefill_s = std::chrono::high_resolution_clock::now();
        if (whisper_decode(ctx, token_seq.data(), token_seq.size(), 0, params.n_threads) != 0) {
            fprintf(stderr, "%s: failed to decode initial tokens\n", __func__);
            audio_buffer.clear();
            n_iter++;
            continue;
        }
        auto t_prefill_e = std::chrono::high_resolution_clock::now();
        double iter_prefill_us = std::chrono::duration<double, std::micro>(t_prefill_e - t_prefill_s).count();

        // Step 4: Token-by-token generation with backward peak detection
        const float frame_duration = 0.02f;  // 20ms per encoder frame (50fps)
        std::vector<whisper_token> decoded_tokens;
        int stop_token_idx = -1;
        bool stopped_by_backward_peak = false;
        bool stopped_by_peak_margin = false;
        bool stopped_by_repetition = false;
        bool stopped_by_eot = false;
        float last_max_peak_time = -1.0f;

        // Per-chunk delta state (reset each chunk)
        std::vector<float> prev_non_punct_attn;
        bool is_first_delta = true;

        int max_tokens = params.max_tokens > 0 ? params.max_tokens : 224;  // whisper_n_text_ctx/2
        int n_past = sot_sequence_len;

        // Track how many tokens were in the last decode call (for logits offset)
        int last_decode_n_tokens = sot_sequence_len;

        // Per-token absolute attention (for backward peak detection — single layer, all heads avg)
        std::vector<std::vector<float>> per_token_attn;

        // Per-token alignment heads attention (for DTW — proper alignment heads)
        std::vector<std::vector<float>> per_token_aheads_attn;

        // Cross-attention extraction timing
        double cross_attn_copy_us = 0.0;
        double cross_attn_calc_us = 0.0;
        double iter_decode_us = 0.0;
        int n_decoded_tokens = 0;

        for (int step = 0; step < max_tokens; step++) {
            // Get logits from last decode call and pick next token (greedy)
            // logits layout: [last_decode_n_tokens * n_vocab], last token's logits at offset (last_decode_n_tokens-1)*n_vocab
            float * logits_all = whisper_get_logits(ctx);
            int n_vocab = whisper_n_vocab(ctx);
            float * logits = logits_all + (last_decode_n_tokens - 1) * n_vocab;

            // Korean: suppress special tokens during decoding
            if (is_cjk_language(params.language)) {
                logits[whisper_token_sot(ctx)]        = -INFINITY;
                logits[whisper_token_transcribe(ctx)]  = -INFINITY;
                logits[whisper_token_translate(ctx)]   = -INFINITY;
                logits[whisper_token_not(ctx)]          = -INFINITY;
                int lang_id = whisper_lang_id(params.language.c_str());
                if (lang_id >= 0) {
                    logits[whisper_token_lang(ctx, lang_id)] = -INFINITY;
                }
            }

            // Find argmax (greedy decoding)
            whisper_token next_token = 0;
            float max_logit = logits[0];
            for (int v = 1; v < n_vocab; v++) {
                if (logits[v] > max_logit) {
                    max_logit = logits[v];
                    next_token = v;
                }
            }

            // EOT handling
            // Mode 0 without skip_eot: stop immediately
            // Mode 0 with skip_eot, mode 1, mode 2: skip EOT, try 2nd best
            if (next_token == whisper_token_eot(ctx)) {
                // Mode 0 without skip_eot: stop on EOT
                if (params.carryover_mode == 0 && !params.skip_eot) {
                    stopped_by_eot = true;
                    if (params.debug) {
                        fprintf(stderr, "  [STOP EOT] step %d\n", step);
                    }
                    break;
                }
                // Skip EOT, try 2nd best
                logits[whisper_token_eot(ctx)] = -INFINITY;
                next_token = 0;
                max_logit = logits[0];
                for (int v = 1; v < n_vocab; v++) {
                    if (logits[v] > max_logit) {
                        max_logit = logits[v];
                        next_token = v;
                    }
                }
                if (params.debug) {
                    fprintf(stderr, "  [SKIP EOT] step %d, using next best: '%s' (id=%d)\n",
                            step, whisper_token_to_str(ctx, next_token), next_token);
                }
                // Check if next best is a special/garbage token
                bool is_garbage = false;
                if (next_token >= whisper_token_eot(ctx)) {
                    is_garbage = true;
                } else {
                    const char * tok_str = whisper_token_to_str(ctx, next_token);
                    if (tok_str) {
                        bool has_valid = false;
                        for (const char * p = tok_str; *p; p++) {
                            if (std::isalnum((unsigned char)*p)) {
                                has_valid = true;
                                break;
                            }
                            if (is_cjk_language(params.language) && ((unsigned char)*p >= 0x80)) {
                                has_valid = true;
                                break;
                            }
                        }
                        if (!has_valid) {
                            is_garbage = true;
                        }
                    }
                }
                if (is_garbage) {
                    if (params.debug) {
                        fprintf(stderr, "  [STOP EOT] step %d: next best is garbage/special, stopping\n", step);
                    }
                    stopped_by_eot = true;
                    stop_token_idx = (int)decoded_tokens.size();
                    break;
                }
            }

            // Add token
            decoded_tokens.push_back(next_token);
            per_token_attn.push_back({});  // placeholder, updated when attention is available

            // Decode this single token (incremental, n_past = previous context length)
            auto t_dec_s = std::chrono::high_resolution_clock::now();
            if (whisper_decode(ctx, &next_token, 1, n_past, params.n_threads) != 0) {
                fprintf(stderr, "%s: failed to decode token at step %d\n", __func__, step);
                break;
            }
            auto t_dec_e = std::chrono::high_resolution_clock::now();
            iter_decode_us += std::chrono::duration<double, std::micro>(t_dec_e - t_dec_s).count();
            n_decoded_tokens++;
            n_past++;
            last_decode_n_tokens = 1;  // subsequent decodes are always 1 token

            // Copy layer cross-attention data from GPU (timed)
            {
                auto t_copy_s = std::chrono::high_resolution_clock::now();
                int copy_ret = whisper_copy_layer_cross_attention_from_ctx(ctx);
                auto t_copy_e = std::chrono::high_resolution_clock::now();
                cross_attn_copy_us += std::chrono::duration<double, std::micro>(t_copy_e - t_copy_s).count();
                if (copy_ret != 0) {
                    if (params.debug) {
                        fprintf(stderr, "  [WARN] could not copy cross-attention at step %d\n", step);
                    }
                    continue;
                }
            }

            // Get cross-attention data + head averaging (timed)
            int attn_n_tokens, attn_n_audio_ctx, attn_n_heads;
            const float * attn_data;
            std::vector<float> avg_attn;
            {
                auto t_calc_s = std::chrono::high_resolution_clock::now();

                attn_data = whisper_get_layer_cross_attention_data_from_ctx(
                    ctx, &attn_n_tokens, &attn_n_audio_ctx, &attn_n_heads);

                if (attn_data != nullptr) {
                    avg_attn = get_avg_attn_for_token(
                        attn_data, attn_n_audio_ctx, attn_n_tokens, attn_n_heads, 0);
                }

                auto t_calc_e = std::chrono::high_resolution_clock::now();
                cross_attn_calc_us += std::chrono::duration<double, std::micro>(t_calc_e - t_calc_s).count();
            }

            if (attn_data == nullptr) continue;

            // Store layer attention for backward peak
            if (!per_token_attn.empty()) {
                per_token_attn.back() = avg_attn;
            }

            // Extract alignment heads attention for DTW (carryover modes 1, 2)
            if (use_aheads_dtw) {
                int aheads_copy_ret = whisper_copy_aheads_cross_attention_from_ctx(ctx);
                if (aheads_copy_ret == 0) {
                    int ah_n_tokens, ah_n_audio_ctx, ah_n_aheads;
                    const float * ah_data = whisper_get_aheads_cross_attention_data_from_ctx(
                        ctx, &ah_n_tokens, &ah_n_audio_ctx, &ah_n_aheads);
                    if (ah_data != nullptr && ah_n_aheads > 0) {
                        // Average across alignment heads
                        // Layout: data[t + a * n_tokens + h * n_tokens * n_audio_ctx]
                        // For single-token decode: n_tokens=1, t=0 → data[a + h * n_audio_ctx]
                        std::vector<float> ah_avg(ah_n_audio_ctx, 0.0f);
                        for (int h = 0; h < ah_n_aheads; h++) {
                            for (int a = 0; a < ah_n_audio_ctx; a++) {
                                ah_avg[a] += ah_data[a + h * ah_n_audio_ctx];
                            }
                        }
                        float inv = 1.0f / ah_n_aheads;
                        for (auto & v : ah_avg) v *= inv;
                        per_token_aheads_attn.push_back(std::move(ah_avg));
                    } else {
                        per_token_aheads_attn.push_back({});
                    }
                } else {
                    per_token_aheads_attn.push_back({});
                }
            }

            if (params.debug && step == 0) {
                fprintf(stderr, "  [ATTN] n_tokens=%d, n_audio_ctx=%d, n_heads=%d\n",
                        attn_n_tokens, attn_n_audio_ctx, attn_n_heads);
                fprintf(stderr, "  [ATTN RAW] per-head sum (token=0):");
                for (int h = 0; h < attn_n_heads; h++) {
                    float hsum = 0.0f;
                    for (int a = 0; a < attn_n_audio_ctx; a++) {
                        hsum += attn_data[a + 0 * attn_n_audio_ctx + h * attn_n_audio_ctx * attn_n_tokens];
                    }
                    fprintf(stderr, " h%d=%.4f", h, hsum);
                }
                fprintf(stderr, "\n");
            }

            if (params.debug) {
                int best = 0;
                for (int a = 1; a < attn_n_audio_ctx; a++) {
                    if (avg_attn[a] > avg_attn[best]) best = a;
                }
                float attn_sum = 0.0f;
                for (int a = 0; a < attn_n_audio_ctx; a++) attn_sum += avg_attn[a];
                fprintf(stderr, "  [AVG_ATTN] step=%d argmax=%d (%.2fs) sum=%.4f first10:",
                        step, best, best * 0.02f, attn_sum);
                for (int a = 0; a < std::min(10, attn_n_audio_ctx); a++) {
                    fprintf(stderr, " %.4f", avg_attn[a]);
                }
                fprintf(stderr, "\n");
            }

            // Token classification
            int t = (int)decoded_tokens.size() - 1;
            whisper_token tok = next_token;
            bool is_punct = params.skip_punctuation_delta && is_punctuation_token(ctx, tok);
            bool is_subw = params.skip_subword_delta && is_subword_token(ctx, tok, params.language);
            bool is_eot_tok = (tok >= whisper_token_eot(ctx));
            bool is_special = is_punct || is_subw || is_eot_tok;

            if (params.debug) {
                fprintf(stderr, "  [step %d] token='%s' (id=%d)%s\n",
                        step, whisper_token_to_str(ctx, tok), tok,
                        is_special ? " [SPECIAL]" : "");
            }

            // Check consecutive token repetition (3+ same tokens in a row)
            if (t >= 2 && decoded_tokens[t] == decoded_tokens[t-1] && decoded_tokens[t] == decoded_tokens[t-2]) {
                if (params.debug) {
                    fprintf(stderr, "  [REPETITION] Token '%s' repeated 3+ times at %d\n",
                            whisper_token_to_str(ctx, tok), t);
                }
                stopped_by_repetition = true;
                stop_token_idx = t - 2;
                break;
            }

            // Skip delta for special tokens
            if (is_special) {
                if (params.debug) {
                    fprintf(stderr, "  [SKIP DELTA] Token '%s' (special)\n",
                            whisper_token_to_str(ctx, tok));
                }
                // Don't update prev_non_punct_attn
                continue;
            }

            // Compute delta if we have previous non-special attention
            if (!prev_non_punct_attn.empty()) {
                std::vector<float> delta(attn_n_audio_ctx);
                for (int a = 0; a < attn_n_audio_ctx; a++) {
                    delta[a] = avg_attn[a] - prev_non_punct_attn[a];
                }

                // Apply median filter
                if (params.median_filter_window > 0) {
                    median_filter_1d(delta, params.median_filter_window);
                }

                // Apply smoothing
                if (params.smoothing_window > 0) {
                    moving_average(delta, params.smoothing_window);
                }

                // Debug: dump delta after smoothing
                if (params.debug) {
                    fprintf(stderr, "  [DELTA] step=%d token='%s' n_audio_ctx=%d values(first 20):",
                            step, whisper_token_to_str(ctx, tok), attn_n_audio_ctx);
                    for (int a = 0; a < std::min(20, attn_n_audio_ctx); a++) {
                        fprintf(stderr, " %.4f", delta[a]);
                    }
                    fprintf(stderr, "\n");
                    // Also dump around peak region
                    fprintf(stderr, "  [DELTA] values(last 20 of %d):", attn_n_audio_ctx);
                    for (int a = std::max(0, attn_n_audio_ctx - 20); a < attn_n_audio_ctx; a++) {
                        fprintf(stderr, " [%d]=%.4f", a, delta[a]);
                    }
                    fprintf(stderr, "\n");
                }

                // Exclude first exclude_time seconds from max peak search
                int exclude_idx = (int)(params.exclude_time / frame_duration);
                if (exclude_idx >= attn_n_audio_ctx) exclude_idx = attn_n_audio_ctx - 1;
                if (exclude_idx < 0) exclude_idx = 0;

                // Find max peak in [exclude_idx, end)
                int max_peak_idx = exclude_idx;
                for (int a = exclude_idx + 1; a < attn_n_audio_ctx; a++) {
                    if (delta[a] > delta[max_peak_idx]) max_peak_idx = a;
                }
                float max_peak_time = max_peak_idx * frame_duration;

                // Find min peak in entire range
                int min_peak_idx = 0;
                for (int a = 1; a < attn_n_audio_ctx; a++) {
                    if (delta[a] < delta[min_peak_idx]) min_peak_idx = a;
                }

                if (params.debug) {
                    fprintf(stderr, "  [PEAK] exclude_idx=%d, max_peak_idx=%d (%.2fs, val=%.4f), min_peak_idx=%d (%.2fs, val=%.4f)\n",
                            exclude_idx, max_peak_idx, max_peak_time, delta[max_peak_idx],
                            min_peak_idx, min_peak_idx * frame_duration, delta[min_peak_idx]);
                }

                float min_peak_time;
                if (is_first_delta) {
                    min_peak_time = 0.0f;
                    is_first_delta = false;
                } else {
                    min_peak_time = min_peak_idx * frame_duration;
                }

                // Backward peak detection: max before min = hallucination
                if (max_peak_time < min_peak_time) {
                    // Safety check: skip stopping for special tokens
                    bool is_special_char2 = is_punctuation_token(ctx, tok);
                    bool is_timestamp2 = (tok >= 50364);
                    bool is_eot_tok2 = (tok >= whisper_token_eot(ctx));
                    bool is_subword2 = is_subword_token(ctx, tok, params.language);

                    if (is_special_char2 || is_timestamp2 || is_eot_tok2 || is_subword2) {
                        if (params.debug) {
                            fprintf(stderr, "  [SKIP] Backward peak detected but token '%s' is special (not stopping)\n",
                                    whisper_token_to_str(ctx, tok));
                        }
                    } else {
                        if (params.debug) {
                            fprintf(stderr, "  [STOP] Backward peak at step %d '%s': max=%.2fs < min=%.2fs\n",
                                    step, whisper_token_to_str(ctx, tok), max_peak_time, min_peak_time);
                        }
                        stopped_by_backward_peak = true;
                        backward_peak_count++;
                        stop_token_idx = t;  // drop this token
                        break;
                    }
                }

                // Track last valid max peak time
                last_max_peak_time = max_peak_time;

                // Peak margin: stop if peak enters margin zone near audio end
                if (params.peak_margin_sec > 0 && !is_last) {
                    float audio_end_time = effective_seconds;
                    if (max_peak_time > audio_end_time - params.peak_margin_sec) {
                        if (params.debug) {
                            fprintf(stderr, "  [PEAK MARGIN] Step %d '%s': peak=%.2fs in margin (%.2fs-%.2fs)\n",
                                    step, whisper_token_to_str(ctx, tok), max_peak_time,
                                    audio_end_time - params.peak_margin_sec, audio_end_time);
                        }
                        stopped_by_peak_margin = true;
                        peak_margin_count++;
                        stop_token_idx = t + 1;  // include this token (still valid, just in margin zone)
                        break;
                    }
                }
            }

            // Update previous attention for next delta
            prev_non_punct_attn = avg_attn;
        }

        // Print timings
        whisper_print_timings(ctx);

        fprintf(stderr, "[ITER_TIMING] iter=%d mel=%.2f ms encode=%.2f ms prefill=%.2f ms decode=%.2f ms n_tokens=%d\n",
                n_iter, iter_mel_us / 1000.0, iter_encode_us / 1000.0,
                iter_prefill_us / 1000.0, iter_decode_us / 1000.0, n_decoded_tokens);

        fprintf(stderr, "[CROSS_ATTN_TIME] iter=%d copy=%.2f ms calc=%.2f ms\n",
                n_iter, cross_attn_copy_us / 1000.0, cross_attn_calc_us / 1000.0);

        if (decoded_tokens.empty()) {
            // Carry over entire effective audio for next chunk (don't clear)
            float max_carryover_sec = 2.0f * params.step_ms / 1000.0f;
            if (effective_seconds > max_carryover_sec) {
                int cap_start_sample = (int)((effective_seconds - max_carryover_sec) * WHISPER_SAMPLE_RATE);
                carryover_audio.assign(
                    effective_audio.begin() + cap_start_sample,
                    effective_audio.end());
                carryover_abs_start_sample = effective_abs_start_sample + cap_start_sample;
                if (params.debug) {
                    fprintf(stderr, "[iter=%d] no tokens decoded — carryover capped: %.2fs -> %.2fs (dropped front %.2fs)\n",
                            n_iter, effective_seconds, max_carryover_sec, effective_seconds - max_carryover_sec);
                }
            } else {
                carryover_audio = effective_audio;
                carryover_abs_start_sample = effective_abs_start_sample;
                if (params.debug) {
                    fprintf(stderr, "[iter=%d] no tokens decoded — carrying over entire %.2fs\n",
                            n_iter, effective_seconds);
                }
            }
            audio_buffer.clear();
            n_iter++;
            continue;
        }

        // =====================================================
        // Determine tokens for emission
        // No conservative emission needed — backward peak detection
        // already determines the exact stop point.
        // =====================================================
        std::vector<whisper_token> emitted;
        if (is_last) {
            // Skip last chunk (e.g. zero-padding region)
            if (params.debug) {
                fprintf(stderr, "[iter=%d] last chunk — skipping emission\n", n_iter);
            }
        } else if (stop_token_idx >= 0) {
            emitted.insert(emitted.end(),
                           decoded_tokens.begin(),
                           decoded_tokens.begin() + stop_token_idx);
        } else {
            emitted = decoded_tokens;
        }

        // Debug: print emitted text for this round
        if (params.debug && !emitted.empty()) {
            std::string round_text;
            for (auto t : emitted) {
                const char * s = whisper_token_to_str(ctx, t);
                if (s) round_text += s;
            }
            if (is_cjk_language(params.language)) {
                fprintf(stderr, "[iter=%d] emitted: \"%s\" incomplete_utf8=%s\n", n_iter, round_text.c_str(),
                        ends_with_incomplete_utf8(round_text) ? "YES" : "no");
            } else {
                fprintf(stderr, "[iter=%d] emitted: \"%s\"\n", n_iter, round_text.c_str());
            }
        }

        bool reinfer_done = false;


        // Compute DTW alignment (for carryover mode 1, 2, 3)
        dtw_result dtw = {};
        if ((params.carryover_mode == 1 || params.carryover_mode == 2 || params.carryover_mode == 3) && !emitted.empty() &&
            (stopped_by_backward_peak || stopped_by_peak_margin || stopped_by_repetition || stopped_by_eot))
        {
            int emit_count = (stop_token_idx >= 0) ? stop_token_idx : (int)decoded_tokens.size();
            // Use alignment heads attention for DTW (more accurate than single layer)
            auto & attn_source = use_aheads_dtw ? per_token_aheads_attn : per_token_attn;
            std::vector<std::vector<float>> emit_attns(
                attn_source.begin(),
                attn_source.begin() + std::min(emit_count, (int)attn_source.size()));
            dtw = compute_dtw_alignment(
                ctx, emitted, emit_attns, frame_duration, params.debug, params.language);
            if (dtw.valid && params.debug) {
                fprintf(stderr, "  [DTW] last word: start=%.2fs end=%.2fs (vs peak=%.2fs)\n",
                        dtw.last_word_start, dtw.last_word_end, last_max_peak_time);
            }
        }

        // Detect unreliable DTW (all-zero or last word end too early)
        bool dtw_unreliable = false;
        if ((params.carryover_mode == 1 || params.carryover_mode == 2 || params.carryover_mode == 3) &&
            (stopped_by_backward_peak || stopped_by_peak_margin || stopped_by_repetition || stopped_by_eot) && !emitted.empty())
        {
            if (!dtw.valid) {
                dtw_unreliable = true;
            } else if (dtw.last_word_end <= 0.0f && dtw.last_word_start <= 0.0f) {
                dtw_unreliable = true;
            } else if (effective_seconds > 0 && dtw.last_word_end < effective_seconds * 0.3f) {
                dtw_unreliable = true;
            }
            if (dtw_unreliable && params.debug) {
                fprintf(stderr, "  [SPECULATIVE] DTW unreliable — speculative emission, carryover entire audio\n");
            }
        }

        // Mode 3: track backward_peak step/token for reinfer trigger
        if (params.carryover_mode == 3 && stopped_by_backward_peak && stop_token_idx >= 0) {
            whisper_token bp_tok = decoded_tokens[stop_token_idx];
            if (bp_tok == prev_bp_token && stop_token_idx == prev_bp_step) {
                consecutive_same_bp_count++;
            } else {
                consecutive_same_bp_count = 1;
            }
            prev_bp_step = stop_token_idx;
            prev_bp_token = bp_tok;
        } else if (params.carryover_mode == 3 && !stopped_by_backward_peak) {
            consecutive_same_bp_count = 0;
            prev_bp_step = -1;
            prev_bp_token = -1;
        }

        // Mode 3: reinfer with 30s padding if 2+ consecutive same step/token backward_peak
        if (params.carryover_mode == 3 && !reinfer_done &&
            consecutive_same_bp_count >= 2)
        {
            // 1. Undo all speculative tokens from previous rounds
            if (prev_speculative && speculative_committed_count > 0) {
                int to_remove = std::min(speculative_committed_count, (int)committed_tokens.size());
                committed_tokens.erase(committed_tokens.end() - to_remove, committed_tokens.end());
                committed_words.erase(committed_words.end() - to_remove, committed_words.end());
                latency_record.erase(latency_record.end() - to_remove, latency_record.end());
                if (params.debug) {
                    fprintf(stderr, "  [REINFER] Undoing %d speculative tokens before reinfer\n", to_remove);
                }
                printf("\n[REPLACE_PREVIOUS %d]\n", to_remove);
                fflush(stdout);
            }

            // 2. No prefill for reinfer — let whisper_full decode from audio alone
            if (params.debug) {
                fprintf(stderr, "  [REINFER] %d consecutive backward_peak at same step/token, re-inferring with 30s padding (no prefill)\n",
                        consecutive_same_bp_count);
            }

            // 3. Pad audio to 30s and run whisper_full (no DTW needed)
            std::vector<float> padded_audio(WHISPER_SAMPLE_RATE * 30, 0.0f);
            std::copy(effective_audio.begin(), effective_audio.end(), padded_audio.begin());

            struct whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
            wparams.print_progress   = false;
            wparams.print_timestamps = false;
            wparams.print_special    = false;
            wparams.print_realtime   = false;
            wparams.n_threads        = params.n_threads;
            wparams.language         = params.language.c_str();
            wparams.max_len          = 0;
            wparams.single_segment  = true;
            wparams.suppress_non_speech_tokens = true;
            // No initial_prompt — decode from audio alone

            // Log top-5 logits when EOT is selected (do not suppress)
            reinfer_eot_log_data eot_data = { params.debug, effective_seconds, {}, "" };
            wparams.logits_filter_callback = reinfer_logits_filter;
            wparams.logits_filter_callback_user_data = &eot_data;

            if (whisper_full(ctx, wparams, padded_audio.data(), padded_audio.size()) == 0) {
                emitted.clear();
                int n_segments = whisper_full_n_segments(ctx);
                for (int s = 0; s < n_segments; s++) {
                    int n_tok = whisper_full_n_tokens(ctx, s);
                    for (int t = 0; t < n_tok; t++) {
                        whisper_token tid = whisper_full_get_token_id(ctx, s, t);
                        if (tid < whisper_token_eot(ctx)) {
                            emitted.push_back(tid);
                        }
                    }
                }

                std::string reinfer_text;
                for (auto t : emitted) {
                    const char * s = whisper_token_to_str(ctx, t);
                    if (s) reinfer_text += s;
                }

                int eot_skipped = 0, eot_allowed = 0;
                for (const auto & evt : eot_data.eot_events) {
                    if (std::get<2>(evt)) eot_skipped++; else eot_allowed++;
                }
                fprintf(stderr, "[iter=%d] [REINFER] Re-inferred %zu tokens (no carryover, %d EOT events, %d skipped): \"%s\"\n",
                        n_iter, emitted.size(), (int)eot_data.eot_events.size(), eot_skipped, reinfer_text.c_str());
                for (const auto & evt : eot_data.eot_events) {
                    fprintf(stderr, "[iter=%d] [REINFER] EOT at step %d%s: top5: %s\n",
                            n_iter, std::get<0>(evt),
                            std::get<2>(evt) ? " [SKIPPED:sentence_end]" : "",
                            std::get<1>(evt).c_str());
                }

                // Dedup: remove leading words in reinfer result that match
                // trailing words in committed_tokens (overlap from carryover audio)
                if (!emitted.empty() && !committed_tokens.empty()) {
                    auto reinfer_words = extract_words_for_dedup(ctx, emitted);
                    auto committed_words_dedup = extract_words_for_dedup(ctx, committed_tokens);

                    // Compare last 1-2 committed words with first 1-2 reinfer words
                    int max_match = std::min(2, std::min((int)reinfer_words.size(), (int)committed_words_dedup.size()));
                    int matched = 0;
                    for (int try_n = max_match; try_n >= 1; try_n--) {
                        bool all_match = true;
                        for (int k = 0; k < try_n; k++) {
                            int ci = (int)committed_words_dedup.size() - try_n + k;
                            if (committed_words_dedup[ci].norm != reinfer_words[k].norm) {
                                all_match = false;
                                break;
                            }
                        }
                        if (all_match) {
                            matched = try_n;
                            break;
                        }
                    }

                    if (matched > 0) {
                        // Remove matched leading tokens from emitted
                        int skip_tokens = reinfer_words[matched].token_idx;  // start of first non-matched word
                        if (params.debug) {
                            std::string skipped_text;
                            for (int i = 0; i < skip_tokens; i++) {
                                const char * s = whisper_token_to_str(ctx, emitted[i]);
                                if (s) skipped_text += s;
                            }
                            fprintf(stderr, "[iter=%d] [REINFER] Dedup: removed %d leading words (%d tokens): \"%s\"\n",
                                    n_iter, matched, skip_tokens, skipped_text.c_str());
                        }
                        emitted.erase(emitted.begin(), emitted.begin() + skip_tokens);

                        // Rebuild reinfer_text for log
                        reinfer_text.clear();
                        for (auto t : emitted) {
                            const char * s = whisper_token_to_str(ctx, t);
                            if (s) reinfer_text += s;
                        }
                        fprintf(stderr, "[iter=%d] [REINFER] After dedup (%zu tokens): \"%s\"\n",
                                n_iter, emitted.size(), reinfer_text.c_str());
                    }
                }

                // Reset all state: no carryover, no prefill, no speculative, no olap dedup
                carryover_audio.clear();
                prev_chunk_token_ids.clear();
                prev_speculative = false;
                speculative_committed_count = 0;
                reinfer_done = true;
                dtw_unreliable = false;
                consecutive_same_bp_count = 0;
                prev_bp_step = -1;
                prev_bp_token = -1;
                olap_prev_last_word_token_count = 0;
                olap_prev_last_word_text.clear();
            }
        }

        // If previous round was speculative, undo those committed tokens
        if (prev_speculative && speculative_committed_count > 0) {
            // Remove speculatively committed tokens
            int to_remove = std::min(speculative_committed_count, (int)committed_tokens.size());
            committed_tokens.erase(committed_tokens.end() - to_remove, committed_tokens.end());
            committed_words.erase(committed_words.end() - to_remove, committed_words.end());
            latency_record.erase(latency_record.end() - to_remove, latency_record.end());
            if (params.debug) {
                fprintf(stderr, "  [SPECULATIVE] Replacing %d speculative tokens from previous round\n", to_remove);
            }
            // Output replacement marker
            printf("\n[REPLACE_PREVIOUS %d]\n", to_remove);
            fflush(stdout);
        }
        prev_speculative = false;
        speculative_committed_count = 0;

        // Save previous round's olap dedup state before mode 2 setup may overwrite it
        int olap_dedup_count_this_round = olap_prev_last_word_token_count;
        std::string olap_dedup_text_this_round = olap_prev_last_word_text;

        // Mode 1 (prefill_last): emit all tokens, carryover from last word start,
        // and use last word tokens as prompt prefill for next chunk
        // NOTE: Must happen BEFORE olap dedup which modifies emitted
        if (params.carryover_mode == 1 && !dtw_unreliable && dtw.valid &&
            (stopped_by_backward_peak || stopped_by_peak_margin || stopped_by_repetition || stopped_by_eot))
        {
            // Extract last word tokens for prefill
            std::vector<whisper_token> last_word_tokens(
                emitted.begin() + dtw.tokens_excl_last_word,
                emitted.end());
            // Force prefill with last word tokens (override prev_chunk_token_ids)
            prev_chunk_token_ids.clear();
            for (auto t : last_word_tokens) {
                if (t < whisper_token_eot(ctx)) {
                    prev_chunk_token_ids.push_back(t);
                }
            }
            if (params.debug) {
                std::string last_word_text;
                for (auto t : last_word_tokens) {
                    const char * s = whisper_token_to_str(ctx, t);
                    if (s) last_word_text += s;
                }
                fprintf(stderr, "  [MODE1] prefill last word: \"%s\" (%zu tokens)%s\n",
                        last_word_text.c_str(), last_word_tokens.size(),
                        (is_cjk_language(params.language) && ends_with_incomplete_utf8(last_word_text)) ? " incomplete_utf8=YES" : "");
            }
        }

        // Mode 2 (emit_all_overlap): emit all, carryover from 2nd-to-last word start,
        // prefill with 2nd-to-last word only (matches carryover audio start).
        // Last word is emitted now but will be deduped against next chunk's first word.
        // NOTE: Must happen BEFORE olap dedup which modifies emitted
        if ((params.carryover_mode == 2 || params.carryover_mode == 3) && !dtw_unreliable && dtw.valid && !reinfer_done &&
            (stopped_by_backward_peak || stopped_by_peak_margin || stopped_by_repetition || stopped_by_eot))
        {
            // Record last word token count and text for overlap dedup in next round
            int last_word_n_tokens = (int)emitted.size() - dtw.tokens_excl_last_word;
            olap_prev_last_word_token_count = last_word_n_tokens;
            olap_prev_last_word_text.clear();
            for (int i = dtw.tokens_excl_last_word; i < (int)emitted.size(); i++) {
                const char * s = whisper_token_to_str(ctx, emitted[i]);
                if (s) olap_prev_last_word_text += s;
            }

            // Prefill = 2nd-to-last word tokens only (audio/prefill alignment)
            prev_chunk_token_ids.clear();
            if (dtw.tokens_excl_last_2_words >= 0 && dtw.tokens_excl_last_word > dtw.tokens_excl_last_2_words) {
                for (int i = dtw.tokens_excl_last_2_words; i < dtw.tokens_excl_last_word && i < (int)emitted.size(); i++) {
                    if (emitted[i] < whisper_token_eot(ctx)) {
                        prev_chunk_token_ids.push_back(emitted[i]);
                    }
                }
            }
            if (params.debug) {
                std::string prefill_text;
                for (auto t : prev_chunk_token_ids) {
                    const char * s = whisper_token_to_str(ctx, t);
                    if (s) prefill_text += s;
                }
                fprintf(stderr, "  [MODE%d] prefill (2nd-last word only): \"%s\" (%zu tokens)%s\n",
                        params.carryover_mode,
                        prefill_text.c_str(), prev_chunk_token_ids.size(),
                        (is_cjk_language(params.language) && ends_with_incomplete_utf8(prefill_text)) ? " incomplete_utf8=YES" : "");
                fprintf(stderr, "  [MODE2] carryover from 2nd-last word start: %.2fs\n",
                        dtw.second_last_word_start);
                fprintf(stderr, "  [MODE2] last word tokens to dedup next round: %d\n",
                        last_word_n_tokens);
            }
        }

        // Mode 2: if only 1 word generated (all tokens are "last word"), treat as speculative
        // Single-word chunks are unreliable — let next chunk with more context regenerate
        if ((params.carryover_mode == 2 || params.carryover_mode == 3) && !dtw_unreliable && dtw.valid && !emitted.empty() && !reinfer_done &&
            dtw.tokens_excl_last_word == 0 &&
            (stopped_by_backward_peak || stopped_by_peak_margin || stopped_by_repetition || stopped_by_eot))
        {
            dtw_unreliable = true;
            if (params.debug) {
                fprintf(stderr, "  [MODE2] Only 1 word generated — treating as speculative\n");
            }
        }

        // Speculative: emit but mark for replacement, carryover entire audio, keep old prefill
        if (dtw_unreliable && !emitted.empty()) {
            prev_speculative = true;
            speculative_committed_count = (int)emitted.size();
            pre_speculative_prefill = prev_chunk_token_ids; // save current prefill
            // Don't update prev_chunk_token_ids — next round uses the pre-speculative prefill
            prev_chunk_token_ids = pre_speculative_prefill;
            // Preserve olap dedup state — defer to next (non-speculative) round
            olap_prev_last_word_token_count = olap_dedup_count_this_round;
            olap_prev_last_word_text = olap_dedup_text_this_round;
            if (params.debug && olap_dedup_count_this_round > 0) {
                fprintf(stderr, "  [MODE2 DEDUP] Deferred olap dedup (%d tokens) — speculative round\n",
                        olap_dedup_count_this_round);
            }
        }

        // Mode 2 overlap dedup: remove previous chunk's last word if next chunk reproduces it
        // NOTE: Must happen AFTER mode 2 setup (original emitted + DTW) and AFTER speculative check
        // Skipped during speculative rounds — deferred to next round
        if ((params.carryover_mode == 2 || params.carryover_mode == 3) && olap_dedup_count_this_round > 0 && !emitted.empty() && !prev_speculative) {
            // Compare previous chunk's last word text with current chunk's first word text
            // Use DTW to find first word boundary in current chunk, or fall back to token count
            int cur_first_word_n = olap_dedup_count_this_round; // default: same token count
            if (dtw.valid && dtw.tokens_excl_last_word > 0) {
                // DTW available: first word = tokens before 2nd word
                // tokens_excl_last_2_words gives count before 2nd-to-last word
                // For first word, we need to find where word[1] starts
                // Simpler: use the number of tokens that make up the first DTW word
                // Word boundaries from DTW: word[0] = tokens[0..first_boundary)
                // In compute_dtw_alignment, words are grouped by leading space
                // Just build text from first N tokens until we find a word boundary
                cur_first_word_n = 0;
                std::string first_text;
                for (int i = 0; i < (int)emitted.size(); i++) {
                    const char * s = whisper_token_to_str(ctx, emitted[i]);
                    if (s) {
                        std::string tok_str(s);
                        // Word boundary: token starts with space (and not the first token)
                        if (i > 0 && !tok_str.empty() && tok_str[0] == ' ') {
                            break;
                        }
                        first_text += tok_str;
                    }
                    cur_first_word_n++;
                }
            }
            cur_first_word_n = std::min(cur_first_word_n, (int)emitted.size());

            // Build current chunk's first word text
            std::string cur_first_word_text;
            for (int i = 0; i < cur_first_word_n; i++) {
                const char * s = whisper_token_to_str(ctx, emitted[i]);
                if (s) cur_first_word_text += s;
            }

            // Normalize and compare (use saved text from previous round, not current)
            std::string prev_norm = normalize_for_compare(olap_dedup_text_this_round);
            std::string cur_norm = normalize_for_compare(cur_first_word_text);

            bool match = !prev_norm.empty() && prev_norm == cur_norm;

            if (params.debug) {
                fprintf(stderr, "  [MODE2 DEDUP] prev=\"%s\" (%d tokens) vs cur=\"%s\" (%d tokens) norm: \"%s\" vs \"%s\" -> %s\n",
                        olap_dedup_text_this_round.c_str(), olap_dedup_count_this_round,
                        cur_first_word_text.c_str(), cur_first_word_n,
                        prev_norm.c_str(), cur_norm.c_str(),
                        match ? "MATCH" : "NO MATCH");
            }

            if (match) {
                // First word matches — keep previous emission (earlier timestamp), skip duplicate in current
                emitted.erase(emitted.begin(), emitted.begin() + cur_first_word_n);
                if (params.debug) {
                    fprintf(stderr, "  [MODE2 DEDUP] Skipped %d duplicate tokens (match with prev last word)\n", cur_first_word_n);
                }
            } else {
                // No match — replace: remove prev last word from committed, keep current first word
                // Preserve the earlier emission time so the replacement word keeps the original timestamp
                int to_remove = std::min(olap_dedup_count_this_round, (int)committed_tokens.size());
                double prev_emission_time = 0.0;
                if (to_remove > 0) {
                    // Save emission time from the first token of the prev last word
                    prev_emission_time = std::get<4>(latency_record[latency_record.size() - to_remove]);
                    committed_tokens.erase(committed_tokens.end() - to_remove, committed_tokens.end());
                    committed_words.erase(committed_words.end() - to_remove, committed_words.end());
                    latency_record.erase(latency_record.end() - to_remove, latency_record.end());
                    printf("\n[REPLACE_PREVIOUS %d]\n", to_remove);
                    fflush(stdout);
                }
                // Mark cur first word tokens to use prev emission time
                olap_replace_emission_time = prev_emission_time;
                olap_replace_token_count = cur_first_word_n;
                if (params.debug) {
                    fprintf(stderr, "  [MODE2 DEDUP] No match — replacing prev \"%s\" (%d tokens) with cur \"%s\", preserving emission_time=%.3f\n",
                            olap_dedup_text_this_round.c_str(), to_remove, cur_first_word_text.c_str(), prev_emission_time);
                }
            }

            // If olap dedup emptied emitted, invalidate mode 2 setup for this round
            if (emitted.empty()) {
                olap_prev_last_word_token_count = 0;
                olap_prev_last_word_text.clear();
                if (params.debug) {
                    fprintf(stderr, "  [MODE2 DEDUP] emitted empty after dedup — reset olap state\n");
                }
            }
        }

        // Chunk-boundary dedup: remove overlapping prefix (e.g. hallucinated words)
        // Only for mode 0 (peak): modes 1/2 handle overlap via DTW-based carryover
        if (!params.no_postprocess && params.carryover_mode == 0) {
            int skip = find_dedup_skip(ctx, prev_tail_words, emitted, params.debug);
            if (skip > 0 && skip < (int)emitted.size()) {
                emitted.erase(emitted.begin(), emitted.begin() + skip);
            } else if (skip >= (int)emitted.size()) {
                emitted.clear();
            }

            // Update prev_tail_words for next iteration's dedup
            if (!emitted.empty()) {
                update_prev_tail(ctx, emitted, prev_tail_words);
            }
        }

        // Compute emission wall time
        auto t_now = std::chrono::high_resolution_clock::now();
        double emission_wall_time = std::chrono::duration<double>(t_now - t_start).count();

        // Record emitted tokens (Start/End Time = 0; latency computed offline with ground truth)
        if (!emitted.empty()) {
            for (int ei = 0; ei < (int)emitted.size(); ei++) {
                committed_tokens.push_back(emitted[ei]);
                const char * str = whisper_token_to_str(ctx, emitted[ei]);
                if (str) {
                    committed_words.push_back(str);

                    // For replaced tokens (olap no-match), use the original (earlier) emission time
                    double tok_emission_time = emission_wall_time;
                    if (olap_replace_token_count > 0 && ei < olap_replace_token_count) {
                        tok_emission_time = olap_replace_emission_time;
                    }
                    latency_record.push_back(std::make_tuple(0.0, 0.0, std::string(str), 0.0, tok_emission_time));

                    printf("%s", str);
                    fflush(stdout);
                }
            }
            // Reset olap replace state after recording
            olap_replace_token_count = 0;
            olap_replace_emission_time = 0.0;
        }

        // =====================================================
        // Determine carryover for next chunk
        // =====================================================
        if (!is_last && !reinfer_done) {
            if (emitted.empty()) {
                // No text tokens emitted — carry over entire effective audio
                // But cap to prevent snowballing (max 2x step size)
                float max_carryover_sec = 2.0f * params.step_ms / 1000.0f;
                if (effective_seconds > max_carryover_sec) {
                    int cap_start_sample = (int)((effective_seconds - max_carryover_sec) * WHISPER_SAMPLE_RATE);
                    carryover_audio.assign(
                        effective_audio.begin() + cap_start_sample,
                        effective_audio.end());
                    carryover_abs_start_sample = effective_abs_start_sample + cap_start_sample;
                    if (params.debug) {
                        fprintf(stderr, "[iter=%d] CARRYOVER (no emission, capped): %.2fs -> %.2fs (max %.2fs)\n",
                                n_iter, effective_seconds, max_carryover_sec, max_carryover_sec);
                    }
                } else {
                    carryover_audio = effective_audio;
                    carryover_abs_start_sample = effective_abs_start_sample;
                    if (params.debug) {
                        fprintf(stderr, "[iter=%d] CARRYOVER (no emission): entire %.2fs\n",
                                n_iter, effective_seconds);
                    }
                }
                carryover_count++;
            } else if (dtw_unreliable) {
                // Speculative: carry over entire effective audio for re-decoding (no cap)
                carryover_audio = effective_audio;
                carryover_abs_start_sample = effective_abs_start_sample;
                if (params.debug) {
                    fprintf(stderr, "[iter=%d] CARRYOVER (speculative): entire %.2fs\n",
                            n_iter, effective_seconds);
                }
                carryover_count++;
            } else if ((stopped_by_backward_peak || stopped_by_peak_margin || stopped_by_repetition || stopped_by_eot) &&
                       (last_max_peak_time > 0 || (dtw.valid && (params.carryover_mode == 1 || params.carryover_mode == 2 || params.carryover_mode == 3)))) {
                // Determine carryover start based on mode
                float carryover_start_sec;
                if ((params.carryover_mode == 2 || params.carryover_mode == 3) && dtw.valid && dtw.second_last_word_start >= 0) {
                    // Mode 2: carryover from 2nd-to-last word START + offset
                    carryover_start_sec = dtw.second_last_word_start + params.word_end_offset_sec;
                    if (params.debug) {
                        fprintf(stderr, "  [CARRYOVER MODE] overlap: 2nd_last_start=%.2fs + offset=%.2fs = %.2fs\n",
                                dtw.second_last_word_start, params.word_end_offset_sec, carryover_start_sec);
                    }
                } else if (params.carryover_mode == 1 && dtw.valid) {
                    // Mode 1: carryover from last word START + offset
                    carryover_start_sec = dtw.last_word_start + params.word_end_offset_sec;
                    if (params.debug) {
                        fprintf(stderr, "  [CARRYOVER MODE] prefill_last: last_word_start=%.2fs + offset=%.2fs = %.2fs\n",
                                dtw.last_word_start, params.word_end_offset_sec, carryover_start_sec);
                    }
                } else {
                    // Mode 0 (peak): carryover from peak - overlap
                    carryover_start_sec = last_max_peak_time - params.carryover_overlap_sec;
                }
                carryover_start_sec = std::max(0.0f, carryover_start_sec);
                // Cap carryover to max 2x step size to prevent snowballing
                float max_carryover_sec = 2.0f * params.step_ms / 1000.0f;
                float min_start = effective_seconds - max_carryover_sec;
                if (min_start > carryover_start_sec) {
                    if (params.debug) {
                        fprintf(stderr, "  [CARRYOVER CAP] %.2fs -> %.2fs (max %.2fs)\n",
                                carryover_start_sec, min_start, max_carryover_sec);
                    }
                    carryover_start_sec = min_start;
                }
                int carryover_start_sample = (int)(carryover_start_sec * WHISPER_SAMPLE_RATE);
                if (carryover_start_sample < (int)effective_audio.size()) {
                    carryover_audio.assign(
                        effective_audio.begin() + carryover_start_sample,
                        effective_audio.end());
                    carryover_abs_start_sample = effective_abs_start_sample + carryover_start_sample;
                    carryover_count++;
                    if (params.debug) {
                        fprintf(stderr, "[iter=%d] CARRYOVER (%s): from %.2fs, %.2fs carried\n",
                                n_iter,
                                stopped_by_backward_peak ? "backward_peak" : "peak_margin",
                                carryover_start_sec,
                                (float)carryover_audio.size() / WHISPER_SAMPLE_RATE);
                    }
                }
            }
        }

        // Update prev_chunk_token_ids for prompt prefill
        // Mode 1/2/3 set this directly above — skip overwrite here
        if (params.carryover_mode != 1 && params.carryover_mode != 2 && params.carryover_mode != 3) {
            if (params.prompt_prefill_n > 0) {
                if (!emitted.empty()) {
                    // Filter out special tokens
                    prev_chunk_token_ids.clear();
                    for (auto t : emitted) {
                        if (t < whisper_token_eot(ctx)) {
                            prev_chunk_token_ids.push_back(t);
                        }
                    }
                } else {
                    // No tokens emitted — clear prefill so next round uses plain SOT
                    prev_chunk_token_ids.clear();
                }
            }
        } else if (emitted.empty()) {
            // Mode 1/2: no emission — keep previous prefill as-is
            // (audio is carried over entirely, so prefill from previous chunk is still valid)
        }

        // Clear audio buffer (it was consumed by effective_audio)
        audio_buffer.clear();

        fprintf(stderr, "[iter=%d] emitted %zu tokens, backward_peak=%d, peak_margin=%d\n",
                n_iter, emitted.size(), stopped_by_backward_peak, stopped_by_peak_margin);

        n_iter++;
    }

    printf("\n");
    // Print stats
    fprintf(stderr, "\n=== Ours Streaming Stats ===\n");
    fprintf(stderr, "Total iterations: %d\n", n_iter);
    fprintf(stderr, "Backward peak stops: %d\n", backward_peak_count);
    fprintf(stderr, "Peak margin stops: %d\n", peak_margin_count);
    fprintf(stderr, "Carryover events: %d\n", carryover_count);
    fprintf(stderr, "============================\n");

    // Print committed tokens with latency
    print_tsw_with_latency(latency_record, params.language);
    print_committed(committed_words);

    whisper_free(ctx);

    return 0;
}
