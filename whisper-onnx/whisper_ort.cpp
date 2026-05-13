#include "whisper_ort.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <numeric>
#include <unordered_map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// ═══════════════════════════════════════════════════════════════════════════════
// Alignment heads presets (from whisper.cpp)
// ═══════════════════════════════════════════════════════════════════════════════

static const whisper_ahead g_aheads_tiny_en[]   = { {1,0},{2,0},{2,5},{3,0},{3,1},{3,2},{3,3},{3,4} };
static const whisper_ahead g_aheads_tiny[]      = { {2,2},{3,0},{3,2},{3,3},{3,4},{3,5} };
static const whisper_ahead g_aheads_base_en[]   = { {3,3},{4,7},{5,1},{5,5},{5,7} };
static const whisper_ahead g_aheads_base[]      = { {3,1},{4,2},{4,3},{4,7},{5,1},{5,2},{5,4},{5,6} };
static const whisper_ahead g_aheads_small_en[]  = { {6,6},{7,0},{7,3},{7,8},{8,2},{8,5},{8,7},{9,0},{9,4},{9,8},{9,10},{10,0},{10,1},{10,2},{10,3},{10,6},{10,11},{11,2},{11,4} };
static const whisper_ahead g_aheads_small[]     = { {5,3},{5,9},{8,0},{8,4},{8,7},{8,8},{9,0},{9,7},{9,9},{10,5} };
static const whisper_ahead g_aheads_medium_en[] = { {11,4},{14,1},{14,12},{14,14},{15,4},{16,0},{16,4},{16,9},{17,12},{17,14},{18,7},{18,10},{18,15},{20,0},{20,3},{20,9},{20,14},{21,12} };
static const whisper_ahead g_aheads_medium[]    = { {13,15},{15,4},{15,15},{16,1},{20,0},{23,4} };
static const whisper_ahead g_aheads_large_v1[]  = { {9,19},{11,2},{11,4},{11,17},{22,7},{22,11},{22,17},{23,2},{23,15} };
static const whisper_ahead g_aheads_large_v2[]  = { {10,12},{13,17},{16,11},{16,12},{16,13},{17,15},{17,16},{18,4},{18,11},{18,19},{19,11},{21,2},{21,3},{22,3},{22,9},{22,12},{23,5},{23,7},{23,13},{25,5},{26,1},{26,12},{27,15} };
static const whisper_ahead g_aheads_large_v3[]  = { {7,0},{10,17},{12,18},{13,12},{16,1},{17,14},{19,11},{21,4},{24,1},{25,6} };

struct aheads_preset_entry { size_t n; const whisper_ahead * heads; };
static const std::map<whisper_alignment_heads_preset, aheads_preset_entry> g_aheads_presets = {
    { WHISPER_AHEADS_TINY_EN,   {  8, g_aheads_tiny_en   } },
    { WHISPER_AHEADS_TINY,      {  6, g_aheads_tiny      } },
    { WHISPER_AHEADS_BASE_EN,   {  5, g_aheads_base_en   } },
    { WHISPER_AHEADS_BASE,      {  8, g_aheads_base      } },
    { WHISPER_AHEADS_SMALL_EN,  { 19, g_aheads_small_en  } },
    { WHISPER_AHEADS_SMALL,     { 10, g_aheads_small     } },
    { WHISPER_AHEADS_MEDIUM_EN, { 18, g_aheads_medium_en } },
    { WHISPER_AHEADS_MEDIUM,    {  6, g_aheads_medium    } },
    { WHISPER_AHEADS_LARGE_V1,  {  9, g_aheads_large_v1  } },
    { WHISPER_AHEADS_LARGE_V2,  { 23, g_aheads_large_v2  } },
    { WHISPER_AHEADS_LARGE_V3,  { 10, g_aheads_large_v3  } },
};

// ═══════════════════════════════════════════════════════════════════════════════
// Internal types
// ═══════════════════════════════════════════════════════════════════════════════

struct whisper_mel {
    int n_len;    // number of frames
    int n_mel;    // number of mel bins (80)
    std::vector<float> data;  // [n_mel * n_len], row-major: data[mel * n_len + t]
};

struct whisper_segment {
    int64_t     t0;
    int64_t     t1;
    std::string text;
    std::vector<whisper_token_data> tokens;
    bool        speaker_turn_next = false;
};

struct whisper_dims {
    int n_mels;
    int n_audio_ctx;
    int n_audio_state;
    int n_audio_head;
    int n_audio_layer;
    int n_vocab;
    int n_text_ctx;
    int n_text_state;
    int n_text_head;
    int n_text_layer;
    int head_dim;
    bool is_multilingual;
};

struct whisper_context {
    // ORT
    Ort::Env         env;
    Ort::Session     encoder_session;
    Ort::Session     prefill_session;
    Ort::Session     step_session;
    Ort::MemoryInfo  mem_info;

    // model dimensions
    whisper_dims dims;

    // vocab: token id → string
    std::vector<std::string> vocab;
    // vocab: string → token id (reverse lookup)
    std::unordered_map<std::string, int> vocab_reverse;

    // language id → token id map  (e.g. "en" → 50259)
    std::map<std::string, int> lang_to_token;

    // mel filterbank loaded from file (matches whisper.cpp / Python whisper)
    struct {
        int n_mel = 0;
        int n_fft = 0;  // = N_FFT/2 + 1 = 201
        std::vector<float> data;  // [n_mel x n_fft]
    } filters;

    // mel spectrogram (from whisper_pcm_to_mel)
    whisper_mel mel;

    // encoder output [1, n_audio_ctx, n_audio_state]
    std::vector<float>   enc_out;
    int                  enc_n_frames = 0;
    int                  enc_n_state  = 0;

    // KV cache
    // self_kv  : [n_layers, 2, 1, n_head, past_len, head_dim]
    // cross_kv : [n_layers, 2, 1, n_head, n_audio_ctx, head_dim]
    std::vector<float> self_kv;
    std::vector<float> cross_kv;
    std::vector<int64_t> self_kv_shape;
    std::vector<int64_t> cross_kv_shape;

    // logits from last decode [n_vocab]
    std::vector<float> logits;

    // cross-attention capture
    int  capture_layer  = -1;   // -1 = disabled
    bool capture_aheads = false;
    std::vector<float> cross_attn_data;
    int cross_attn_n_heads    = 0;
    int cross_attn_n_audio_ctx = 0;

    // alignment heads selection (for DTW)
    std::vector<whisper_ahead> aheads_list;  // selected (layer, head) pairs
    std::vector<float> aheads_attn_data;     // extracted selected heads data
    int aheads_n_heads     = 0;
    int aheads_n_audio_ctx = 0;

    // cross-attention weights from decoder_step.onnx (real softmax attention)
    // shape: [1, n_layers, n_head, n_audio_ctx]  (batch always 1)
    std::vector<float> step_cross_attn_w;

    // results from whisper_full
    std::vector<whisper_segment> segments;

    // timings
    int64_t t_encode_us = 0;
    int64_t t_decode_us = 0;
    int64_t t_start_us  = 0;

    whisper_context()
        : env(ORT_LOGGING_LEVEL_WARNING, "whisper_ort")
        , encoder_session(nullptr)
        , prefill_session(nullptr)
        , step_session(nullptr)
        , mem_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
    {}
};

// ═══════════════════════════════════════════════════════════════════════════════
// Mel spectrogram (adapted from whisper.cpp)
// ═══════════════════════════════════════════════════════════════════════════════

// DFT fallback for odd-sized inputs (O(N^2) but correct)
static void dft(const std::vector<float> & in, std::vector<float> & out) {
    const int N = (int)in.size();
    out.resize(2 * N, 0.0f);
    for (int k = 0; k < N; ++k) {
        float re = 0.0f, im = 0.0f;
        for (int n = 0; n < N; ++n) {
            float angle = -2.0f * (float)M_PI * k * n / N;
            re += in[n] * std::cos(angle);
            im += in[n] * std::sin(angle);  // angle is negative, so sin(angle) = -sin(2πkn/N)
        }
        out[2 * k]     = re;
        out[2 * k + 1] = im;
    }
}

// Cooley-Tukey FFT with DFT fallback for odd-sized sub-problems.
// Matches whisper.cpp behavior: if N is odd, fall back to O(N^2) DFT.
static void fft(const std::vector<float> & in, std::vector<float> & out) {
    const int N = (int)in.size();
    out.resize(2 * N, 0.0f);

    if (N == 1) {
        out[0] = in[0];
        out[1] = 0.0f;
        return;
    }

    if (N % 2 == 1) {
        dft(in, out);
        return;
    }

    std::vector<float> even(N / 2), odd(N / 2);
    for (int i = 0; i < N / 2; ++i) {
        even[i] = in[2 * i];
        odd[i]  = in[2 * i + 1];
    }

    std::vector<float> even_fft, odd_fft;
    fft(even, even_fft);
    fft(odd,  odd_fft);

    for (int k = 0; k < N / 2; ++k) {
        float angle   = -2.0f * (float)M_PI * k / N;
        float re      = std::cos(angle);
        float im      = std::sin(angle);
        float re_odd  = odd_fft[2 * k];
        float im_odd  = odd_fft[2 * k + 1];

        out[2 * k + 0]         = even_fft[2 * k] + re * re_odd - im * im_odd;
        out[2 * k + 1]         = even_fft[2 * k + 1] + re * im_odd + im * re_odd;
        out[2 * (k + N/2) + 0] = even_fft[2 * k] - re * re_odd + im * im_odd;
        out[2 * (k + N/2) + 1] = even_fft[2 * k + 1] - re * im_odd - im * re_odd;
    }
}

static void hann_window(int n, std::vector<float> & w) {
    w.resize(n);
    for (int i = 0; i < n; ++i) {
        w[i] = 0.5f * (1.0f - std::cos(2.0f * (float)M_PI * i / n));
    }
}

// Build mel filterbank weights.
// Returns matrix [n_mels x (n_fft/2+1)], row-major.
static std::vector<float> mel_filterbank(int n_mels, int n_fft, int sample_rate) {
    // Hz → mel
    auto hz_to_mel = [](float f) { return 2595.0f * std::log10(1.0f + f / 700.0f); };
    auto mel_to_hz = [](float m) { return 700.0f * (std::pow(10.0f, m / 2595.0f) - 1.0f); };

    float f_min  = 0.0f;
    float f_max  = sample_rate / 2.0f;
    int   n_bins = n_fft / 2 + 1;

    float mel_min = hz_to_mel(f_min);
    float mel_max = hz_to_mel(f_max);

    // n_mels + 2 equally spaced mel points
    std::vector<float> mel_pts(n_mels + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
        mel_pts[i] = mel_to_hz(mel_min + (mel_max - mel_min) * i / (n_mels + 1));
    }

    // FFT bin frequencies
    std::vector<float> freq(n_bins);
    for (int i = 0; i < n_bins; ++i) {
        freq[i] = (float)i * sample_rate / n_fft;
    }

    std::vector<float> fb(n_mels * n_bins, 0.0f);
    for (int m = 0; m < n_mels; ++m) {
        float f_lo = mel_pts[m];
        float f_ctr= mel_pts[m + 1];
        float f_hi = mel_pts[m + 2];
        for (int k = 0; k < n_bins; ++k) {
            float f = freq[k];
            if (f >= f_lo && f <= f_ctr) {
                fb[m * n_bins + k] = (f - f_lo) / (f_ctr - f_lo);
            } else if (f > f_ctr && f <= f_hi) {
                fb[m * n_bins + k] = (f_hi - f) / (f_hi - f_ctr);
            }
        }
    }
    return fb;
}

struct mel_compute_params {
    const float        * samples;
    int                  n_samples;
    int                  frame_size;  // n_fft = 400
    int                  frame_step;  // hop = 160
    int                  n_mels;
    const std::vector<float> * hann;
    const std::vector<float> * filterbank; // [n_mels x (frame_size/2+1)]
    whisper_mel        * mel;
    int                  ith;
    int                  n_threads;
};

static void mel_worker(const mel_compute_params & p) {
    const int n_fft_bins = p.frame_size / 2 + 1;
    std::vector<float> fft_in(p.frame_size, 0.0f);
    std::vector<float> fft_out;

    for (int i = p.ith; i < p.mel->n_len; i += p.n_threads) {
        const int offset = i * p.frame_step;

        // Apply Hanning window
        for (int j = 0; j < p.frame_size; ++j) {
            if (offset + j < p.n_samples) {
                fft_in[j] = (*p.hann)[j] * p.samples[offset + j];
            } else {
                fft_in[j] = 0.0f;
            }
        }

        fft(fft_in, fft_out);

        // Power spectrum (magnitude squared)
        for (int j = 0; j < n_fft_bins; ++j) {
            float re = fft_out[2 * j];
            float im = fft_out[2 * j + 1];
            fft_in[j] = re * re + im * im;  // reuse fft_in as power buf
        }

        // Apply mel filterbank (double accumulation, matches whisper.cpp)
        for (int m = 0; m < p.n_mels; ++m) {
            double sum = 0.0;
            const float * row = p.filterbank->data() + m * n_fft_bins;
            for (int k = 0; k < n_fft_bins; ++k) {
                sum += (double)row[k] * fft_in[k];
            }
            p.mel->data[m * p.mel->n_len + i] = std::log10(std::max(sum, 1e-10));
        }
    }
}

static bool compute_log_mel_spectrogram(
    const float * samples, int n_samples,
    int n_threads, int n_mels,
    whisper_mel & mel,
    const std::vector<float> * loaded_filters = nullptr,
    int loaded_n_fft = 0)
{
    const int frame_size = WHISPER_N_FFT;    // 400
    const int frame_step = WHISPER_HOP_LENGTH; // 160

    // Hanning window (periodic, matches whisper.cpp)
    std::vector<float> hann;
    hann_window(frame_size, hann);

    // Padding: reflective pad at beginning, zero pad at end (matches whisper.cpp / Python)
    const int pad = frame_size / 2;  // 200
    std::vector<float> padded(n_samples + 2 * pad, 0.0f);
    std::copy(samples, samples + n_samples, padded.begin() + pad);
    // Reflective pad: copy samples[1..pad] reversed to beginning
    std::reverse_copy(samples + 1, samples + 1 + pad, padded.begin());

    mel.n_mel     = n_mels;
    mel.n_len     = (int)(padded.size() - frame_size) / frame_step;
    mel.data.assign(n_mels * mel.n_len, 0.0f);

    // Filterbank: use loaded file if available, else compute at runtime
    const std::vector<float> * fb_ptr = loaded_filters;
    static std::vector<float> fb_cache;
    static int fb_n_mels = 0;
    int n_fft_bins = loaded_n_fft;
    if (!fb_ptr || fb_ptr->empty()) {
        if (fb_n_mels != n_mels) {
            fb_cache  = mel_filterbank(n_mels, frame_size, WHISPER_SAMPLE_RATE);
            fb_n_mels = n_mels;
        }
        fb_ptr = &fb_cache;
        n_fft_bins = frame_size / 2 + 1;
    }

    // Multi-threaded mel computation
    std::vector<std::thread> threads;
    std::vector<mel_compute_params> params(n_threads);
    int n_samples_padded = n_samples + pad;  // effective sample count for worker bounds
    for (int t = 0; t < n_threads; ++t) {
        params[t] = { padded.data(), (int)padded.size(), frame_size, frame_step,
                      n_mels, &hann, fb_ptr, &mel, t, n_threads };
        if (t > 0) threads.emplace_back(mel_worker, std::cref(params[t]));
    }
    mel_worker(params[0]);
    for (auto & th : threads) th.join();

    // Clamping and normalization (matches whisper.cpp)
    double mmax = -1e20;
    for (int i = 0; i < n_mels * mel.n_len; i++) {
        if (mel.data[i] > mmax) mmax = mel.data[i];
    }
    mmax -= 8.0;
    for (int i = 0; i < n_mels * mel.n_len; i++) {
        if (mel.data[i] < mmax) mel.data[i] = mmax;
        mel.data[i] = (mel.data[i] + 4.0) / 4.0;
    }

    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Dims + vocab loading
// ═══════════════════════════════════════════════════════════════════════════════

static bool load_dims(const std::string & model_dir, whisper_dims & dims) {
    std::ifstream f(model_dir + "/dims.txt");
    if (!f.is_open()) return false;
    std::string line;
    while (std::getline(f, line)) {
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);
        if (key == "n_mels")         dims.n_mels         = std::stoi(val);
        else if (key == "n_audio_ctx")    dims.n_audio_ctx    = std::stoi(val);
        else if (key == "n_audio_state")  dims.n_audio_state  = std::stoi(val);
        else if (key == "n_audio_head")   dims.n_audio_head   = std::stoi(val);
        else if (key == "n_audio_layer")  dims.n_audio_layer  = std::stoi(val);
        else if (key == "n_vocab")        dims.n_vocab        = std::stoi(val);
        else if (key == "n_text_ctx")     dims.n_text_ctx     = std::stoi(val);
        else if (key == "n_text_state")   dims.n_text_state   = std::stoi(val);
        else if (key == "n_text_head")    dims.n_text_head    = std::stoi(val);
        else if (key == "n_text_layer")   dims.n_text_layer   = std::stoi(val);
        else if (key == "head_dim")       dims.head_dim       = std::stoi(val);
        else if (key == "is_multilingual") dims.is_multilingual = std::stoi(val) != 0;
    }
    return true;
}

static bool load_vocab(const std::string & model_dir, std::vector<std::string> & vocab,
                       std::unordered_map<std::string, int> & vocab_reverse) {
    std::ifstream f(model_dir + "/vocab.txt");
    if (!f.is_open()) return false;
    std::string line;
    while (std::getline(f, line)) {
        // unescape \n \r
        std::string s;
        for (size_t i = 0; i < line.size(); ++i) {
            if (line[i] == '\\' && i + 1 < line.size()) {
                if (line[i+1] == 'n')  { s += '\n'; ++i; }
                else if (line[i+1] == 'r') { s += '\r'; ++i; }
                else { s += line[i]; }
            } else {
                s += line[i];
            }
        }
        int id = (int)vocab.size();
        vocab_reverse[s] = id;
        vocab.push_back(s);
    }
    return true;
}

// Build language → token id map.
// Multilingual whisper reserves tokens starting at sot+1 for languages.
static void build_lang_map(whisper_context * ctx) {
    // Language list in Whisper order (index i → token sot+1+i)
    static const char * LANGS[] = {
        "en","zh","de","es","ru","ko","fr","ja","pt","tr","pl","ca","nl","ar",
        "sv","it","id","hi","fi","vi","iw","uk","el","ms","cs","ro","da","hu",
        "ta","no","th","ur","hr","bg","lt","la","mi","ml","cy","sk","te","fa",
        "lv","bn","sr","az","sl","kn","et","mk","br","eu","is","hy","ne","mn",
        "bs","kk","sq","sw","gl","mr","pa","si","km","sn","yo","so","af","oc",
        "ka","be","tg","sd","gu","am","yi","lo","uz","fo","ht","ps","tk","nn",
        "mt","sa","lb","my","bo","tl","mg","as","tt","haw","ln","ha","ba","jw","su",
        nullptr
    };
    int sot = ctx->dims.n_vocab - 110 - 6; // approximate; overridden below
    // sot token is n_vocab - (n_special + n_lang + n_timestamps)
    // For base: sot = 50258, lang starts at 50259
    // We find sot by looking for "<|startoftranscript|>" in vocab
    for (int i = 0; i < (int)ctx->vocab.size(); ++i) {
        if (ctx->vocab[i] == "<|startoftranscript|>") { sot = i; break; }
    }
    for (int i = 0; LANGS[i] != nullptr; ++i) {
        ctx->lang_to_token[LANGS[i]] = sot + 1 + i;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ORT session helpers
// ═══════════════════════════════════════════════════════════════════════════════

static Ort::Session make_session(Ort::Env & env, const std::string & path, bool use_gpu) {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    opts.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    if (use_gpu) {
        OrtCUDAProviderOptions cuda_opts{};
        try { opts.AppendExecutionProvider_CUDA(cuda_opts); fprintf(stderr, "using CUDA backend\n"); }
        catch (...) { fprintf(stderr, "cuda_init() failed\n"); }
    }
    return Ort::Session(env, path.c_str(), opts);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Lifecycle
// ═══════════════════════════════════════════════════════════════════════════════

whisper_context_params whisper_context_default_params() {
    return { false, 0, false };
}

whisper_context * whisper_init_from_file_with_params(
    const char * model_path, whisper_context_params params)
{
    std::string dir(model_path);

    auto * ctx = new whisper_context();

    try {
        ctx->encoder_session = make_session(ctx->env, dir + "/encoder.onnx",        params.use_gpu);
        ctx->prefill_session = make_session(ctx->env, dir + "/decoder_prefill.onnx", params.use_gpu);
        ctx->step_session    = make_session(ctx->env, dir + "/decoder_step.onnx",    params.use_gpu);
    } catch (const Ort::Exception & e) {
        fprintf(stderr, "%s: failed to load ORT sessions: %s\n", __func__, e.what());
        delete ctx;
        return nullptr;
    }

    if (!load_dims(dir, ctx->dims)) {
        fprintf(stderr, "%s: failed to load dims.txt\n", __func__);
        delete ctx;
        return nullptr;
    }

    if (!load_vocab(dir, ctx->vocab, ctx->vocab_reverse)) {
        fprintf(stderr, "%s: failed to load vocab.txt\n", __func__);
        delete ctx;
        return nullptr;
    }

    // Load mel filterbank from file (matches whisper.cpp / Python whisper)
    {
        std::string fb_path = dir + "/mel_filters.bin";
        std::ifstream fb_f(fb_path, std::ios::binary);
        if (fb_f.is_open()) {
            int32_t n_mel, n_fft;
            fb_f.read(reinterpret_cast<char*>(&n_mel), sizeof(n_mel));
            fb_f.read(reinterpret_cast<char*>(&n_fft), sizeof(n_fft));
            ctx->filters.n_mel = n_mel;
            ctx->filters.n_fft = n_fft;
            ctx->filters.data.resize(n_mel * n_fft);
            fb_f.read(reinterpret_cast<char*>(ctx->filters.data.data()), n_mel * n_fft * sizeof(float));
            fprintf(stderr, "%s: loaded mel filterbank (%d x %d) from %s\n",
                    __func__, n_mel, n_fft, fb_path.c_str());
        } else {
            fprintf(stderr, "%s: mel_filters.bin not found at %s, using runtime-computed filterbank\n",
                    __func__, fb_path.c_str());
        }
    }

    build_lang_map(ctx);

    ctx->t_start_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();

    fprintf(stderr, "%s: loaded model from '%s'\n", __func__, model_path);
    fprintf(stderr, "%s: n_vocab=%d, n_text_layer=%d, n_audio_layer=%d, multilingual=%d\n",
        __func__, ctx->dims.n_vocab, ctx->dims.n_text_layer,
        ctx->dims.n_audio_layer, ctx->dims.is_multilingual);

    return ctx;
}

void whisper_free(whisper_context * ctx) {
    delete ctx;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Model info
// ═══════════════════════════════════════════════════════════════════════════════

int  whisper_n_vocab          (whisper_context * ctx) { return ctx->dims.n_vocab;        }
int  whisper_n_text_ctx       (whisper_context * ctx) { return ctx->dims.n_text_ctx;     }
int  whisper_n_audio_ctx      (whisper_context * ctx) { return ctx->dims.n_audio_ctx;    }
bool whisper_is_multilingual  (whisper_context * ctx) { return ctx->dims.is_multilingual;}
int  whisper_model_n_audio_state(whisper_context * ctx) { return ctx->dims.n_audio_state;}
int  whisper_model_n_audio_head (whisper_context * ctx) { return ctx->dims.n_audio_head; }
int  whisper_model_n_audio_layer(whisper_context * ctx) { return ctx->dims.n_audio_layer;}
int  whisper_model_n_text_state (whisper_context * ctx) { return ctx->dims.n_text_state; }

// ═══════════════════════════════════════════════════════════════════════════════
// Special tokens
// ═══════════════════════════════════════════════════════════════════════════════

static int find_token(whisper_context * ctx, const std::string & s) {
    for (int i = 0; i < (int)ctx->vocab.size(); ++i) {
        if (ctx->vocab[i] == s) return i;
    }
    return -1;
}

whisper_token whisper_token_eot  (whisper_context * ctx) { return find_token(ctx, "<|endoftext|>"); }
whisper_token whisper_token_sot  (whisper_context * ctx) { return find_token(ctx, "<|startoftranscript|>"); }
whisper_token whisper_token_prev (whisper_context * ctx) { return find_token(ctx, "<|startofprev|>"); }
whisper_token whisper_token_solm (whisper_context * ctx) { return find_token(ctx, "<|startoflm|>"); }
whisper_token whisper_token_not  (whisper_context * ctx) { return find_token(ctx, "<|notimestamps|>"); }
whisper_token whisper_token_beg  (whisper_context * ctx) { return find_token(ctx, "<|0.00|>"); }
whisper_token whisper_token_translate  (whisper_context * ctx) { return find_token(ctx, "<|translate|>"); }
whisper_token whisper_token_transcribe (whisper_context * ctx) { return find_token(ctx, "<|transcribe|>"); }

whisper_token whisper_token_lang(whisper_context * ctx, int lang_id) {
    // lang_id is the index into the language list
    whisper_token sot = whisper_token_sot(ctx);
    return sot + 1 + lang_id;
}

int whisper_lang_id(const char * lang) {
    static const char * LANGS[] = {
        "en","zh","de","es","ru","ko","fr","ja","pt","tr","pl","ca","nl","ar",
        "sv","it","id","hi","fi","vi","iw","uk","el","ms","cs","ro","da","hu",
        "ta","no","th","ur","hr","bg","lt","la","mi","ml","cy","sk","te","fa",
        "lv","bn","sr","az","sl","kn","et","mk","br","eu","is","hy","ne","mn",
        "bs","kk","sq","sw","gl","mr","pa","si","km","sn","yo","so","af","oc",
        "ka","be","tg","sd","gu","am","yi","lo","uz","fo","ht","ps","tk","nn",
        "mt","sa","lb","my","bo","tl","mg","as","tt","haw","ln","ha","ba","jw","su",
        nullptr
    };
    for (int i = 0; LANGS[i]; ++i) {
        if (std::string(lang) == LANGS[i]) return i;
    }
    return -1;
}

const char * whisper_token_to_str(whisper_context * ctx, whisper_token token) {
    if (token < 0 || token >= (int)ctx->vocab.size()) return "";
    return ctx->vocab[token].c_str();
}

int whisper_token_to_id(whisper_context * ctx, const char * str) {
    auto it = ctx->vocab_reverse.find(str);
    if (it == ctx->vocab_reverse.end()) return -1;
    return it->second;
}

int whisper_n_len(whisper_context * ctx) {
    return ctx->mel.n_len;
}

void whisper_set_audio_ctx(whisper_context * /*ctx*/, int /*audio_ctx*/) {
    // No-op in ORT backend: encoder always processes actual mel frames.
}

const char * whisper_print_system_info() {
    return "whisper_ort (ONNX Runtime backend)";
}

// ═══════════════════════════════════════════════════════════════════════════════
// Low-level encode / decode
// ═══════════════════════════════════════════════════════════════════════════════

int whisper_pcm_to_mel(whisper_context * ctx, const float * samples, int n_samples, int n_threads) {
    const std::vector<float> * fb = nullptr;
    int fb_n_fft = 0;
    if (!ctx->filters.data.empty()) {
        fb = &ctx->filters.data;
        fb_n_fft = ctx->filters.n_fft;
    }
    if (!compute_log_mel_spectrogram(samples, n_samples, n_threads, ctx->dims.n_mels, ctx->mel, fb, fb_n_fft)) {
        fprintf(stderr, "%s: failed to compute mel spectrogram\n", __func__);
        return -1;
    }
    return 0;
}

int whisper_encode(whisper_context * ctx, int /*offset*/, int /*n_threads*/) {
    auto t0 = std::chrono::steady_clock::now();

    const int n_mel    = ctx->mel.n_mel;
    const int n_frames = ctx->mel.n_len;

    // Encoder ONNX accepts variable-length mel (dynamic n_frames axis).
    // Pass actual mel frames directly — no padding needed.
    std::vector<int64_t> mel_shape = {1, n_mel, n_frames};

    Ort::Value mel_tensor = Ort::Value::CreateTensor<float>(
        ctx->mem_info,
        ctx->mel.data.data(), ctx->mel.data.size(),
        mel_shape.data(), mel_shape.size());

    const char * in_names[]  = {"mel"};
    const char * out_names[] = {"encoder_output"};

    auto outputs = ctx->encoder_session.Run(
        Ort::RunOptions{nullptr}, in_names, &mel_tensor, 1, out_names, 1);

    auto & out = outputs[0];
    auto shape = out.GetTensorTypeAndShapeInfo().GetShape();
    // shape: [1, n_audio_ctx, n_audio_state]
    ctx->enc_n_frames = (int)shape[1];
    ctx->enc_n_state  = (int)shape[2];

    size_t total = (size_t)shape[1] * shape[2];
    ctx->enc_out.resize(total);
    std::memcpy(ctx->enc_out.data(), out.GetTensorData<float>(), total * sizeof(float));

    auto t1 = std::chrono::steady_clock::now();
    ctx->t_encode_us += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    return 0;
}

int whisper_decode(whisper_context * ctx,
                   const whisper_token * tokens, int n_tokens,
                   int n_past, int n_threads)
{
    (void)n_threads;
    auto t0 = std::chrono::steady_clock::now();

    const int n_layers   = ctx->dims.n_text_layer;
    const int n_head     = ctx->dims.n_text_head;
    const int head_dim   = ctx->dims.head_dim;
    const int n_audio_ctx = ctx->enc_n_frames;
    const int n_vocab    = ctx->dims.n_vocab;

    // Convert tokens to int64
    std::vector<int64_t> token_ids(tokens, tokens + n_tokens);

    if (n_past == 0) {
        // ── Prefill ──────────────────────────────────────────────────────────
        std::vector<int64_t> tok_shape = {1, n_tokens};
        std::vector<int64_t> enc_shape = {1, (int64_t)ctx->enc_n_frames, ctx->enc_n_state};

        Ort::Value tok_tensor = Ort::Value::CreateTensor<int64_t>(
            ctx->mem_info, token_ids.data(), token_ids.size(),
            tok_shape.data(), tok_shape.size());
        Ort::Value enc_tensor = Ort::Value::CreateTensor<float>(
            ctx->mem_info, ctx->enc_out.data(), ctx->enc_out.size(),
            enc_shape.data(), enc_shape.size());

        std::vector<Ort::Value> inputs;
        inputs.push_back(std::move(tok_tensor));
        inputs.push_back(std::move(enc_tensor));

        const char * in_names[]  = {"tokens", "encoder_output"};
        const char * out_names[] = {"logits", "self_kv", "cross_kv"};

        auto outputs = ctx->prefill_session.Run(
            Ort::RunOptions{nullptr},
            in_names, inputs.data(), 2,
            out_names, 3);

        // logits: [1, n_tokens, n_vocab] → store all tokens (ggml-compatible:
        // callers use logits_all + (n_tokens-1)*n_vocab to get the last token)
        {
            const float * data = outputs[0].GetTensorData<float>();
            ctx->logits.assign(data, data + n_tokens * n_vocab);
        }

        // self_kv: [n_layers, 2, 1, n_head, n_tokens, head_dim]
        {
            auto shape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
            size_t total = 1;
            for (auto s : shape) total *= s;
            ctx->self_kv.resize(total);
            std::memcpy(ctx->self_kv.data(), outputs[1].GetTensorData<float>(), total * sizeof(float));
            ctx->self_kv_shape = shape;
        }

        // cross_kv: [n_layers, 2, 1, n_head, n_audio_ctx, head_dim]
        {
            auto shape = outputs[2].GetTensorTypeAndShapeInfo().GetShape();
            size_t total = 1;
            for (auto s : shape) total *= s;
            ctx->cross_kv.resize(total);
            std::memcpy(ctx->cross_kv.data(), outputs[2].GetTensorData<float>(), total * sizeof(float));
            ctx->cross_kv_shape = shape;
        }

    } else {
        // ── Step (single token) ───────────────────────────────────────────────
        assert(n_tokens == 1);

        std::vector<int64_t> tok_shape = {1, 1};

        Ort::Value tok_tensor = Ort::Value::CreateTensor<int64_t>(
            ctx->mem_info, token_ids.data(), 1,
            tok_shape.data(), tok_shape.size());
        Ort::Value self_kv_tensor = Ort::Value::CreateTensor<float>(
            ctx->mem_info, ctx->self_kv.data(), ctx->self_kv.size(),
            ctx->self_kv_shape.data(), ctx->self_kv_shape.size());
        Ort::Value cross_kv_tensor = Ort::Value::CreateTensor<float>(
            ctx->mem_info, ctx->cross_kv.data(), ctx->cross_kv.size(),
            ctx->cross_kv_shape.data(), ctx->cross_kv_shape.size());

        std::vector<Ort::Value> inputs;
        inputs.push_back(std::move(tok_tensor));
        inputs.push_back(std::move(self_kv_tensor));
        inputs.push_back(std::move(cross_kv_tensor));

        const char * in_names[]  = {"token", "past_self_kv", "cross_kv"};
        const char * out_names[] = {"logits", "new_self_kv", "cross_attn_weights"};

        auto outputs = ctx->step_session.Run(
            Ort::RunOptions{nullptr},
            in_names, inputs.data(), 3,
            out_names, 3);

        // logits: [1, 1, n_vocab]
        {
            const float * data = outputs[0].GetTensorData<float>();
            ctx->logits.assign(data, data + n_vocab);
        }

        // new_self_kv: [n_layers, 2, 1, n_head, past_len+1, head_dim]
        {
            auto shape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
            size_t total = 1;
            for (auto s : shape) total *= s;
            ctx->self_kv.resize(total);
            std::memcpy(ctx->self_kv.data(), outputs[1].GetTensorData<float>(), total * sizeof(float));
            ctx->self_kv_shape = shape;
        }

        // cross_attn_weights: [1, n_layers, n_head, n_audio_ctx]
        {
            auto shape = outputs[2].GetTensorTypeAndShapeInfo().GetShape();
            size_t total = 1;
            for (auto s : shape) total *= s;
            ctx->step_cross_attn_w.resize(total);
            std::memcpy(ctx->step_cross_attn_w.data(), outputs[2].GetTensorData<float>(), total * sizeof(float));
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    ctx->t_decode_us += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    return 0;
}

float * whisper_get_logits(whisper_context * ctx) {
    return ctx->logits.data();
}

const float * whisper_get_encoder_output_from_ctx(
    whisper_context * ctx, int * n_frames_out, int * n_state_out)
{
    if (n_frames_out) *n_frames_out = ctx->enc_n_frames;
    if (n_state_out)  *n_state_out  = ctx->enc_n_state;
    return ctx->enc_out.data();
}

// ═══════════════════════════════════════════════════════════════════════════════
// Cross-attention capture
// (Currently captures cross_kv from last decode and computes Q·K attention scores)
// ═══════════════════════════════════════════════════════════════════════════════

void whisper_set_layer_cross_attention_capture(whisper_context * ctx, int layer) {
    ctx->capture_layer  = layer;
    ctx->capture_aheads = false;
}

void whisper_set_aheads_cross_attention_capture(whisper_context * ctx, bool enable) {
    ctx->capture_aheads = enable;
    ctx->capture_layer  = enable ? -1 : -2;
}

int whisper_copy_layer_cross_attention_from_ctx(whisper_context * ctx) {
    const int n_layers    = ctx->dims.n_text_layer;
    const int n_head      = ctx->dims.n_text_head;
    const int n_audio_ctx = ctx->enc_n_frames;

    int layer = ctx->capture_layer;
    if (layer < 0) layer = n_layers + layer;  // -1 → last layer
    if (layer < 0 || layer >= n_layers) return -1;

    ctx->cross_attn_n_heads     = n_head;
    ctx->cross_attn_n_audio_ctx = n_audio_ctx;
    ctx->cross_attn_data.assign(n_head * n_audio_ctx, 0.0f);

    // Use real cross-attention weights from decoder_step.onnx output.
    // step_cross_attn_w shape: [1, n_layers, n_head, n_audio_ctx]
    if (ctx->step_cross_attn_w.empty()) return -1;

    const int layer_stride = n_head * n_audio_ctx;
    // skip batch dim (size 1): data starts at offset 0
    const float * layer_w = ctx->step_cross_attn_w.data() + layer * layer_stride;
    std::memcpy(ctx->cross_attn_data.data(), layer_w, n_head * n_audio_ctx * sizeof(float));
    return 0;
}

void whisper_set_aheads_preset(whisper_context * ctx, whisper_alignment_heads_preset preset) {
    ctx->aheads_list.clear();
    auto it = g_aheads_presets.find(preset);
    if (it != g_aheads_presets.end()) {
        for (size_t i = 0; i < it->second.n; i++) {
            ctx->aheads_list.push_back(it->second.heads[i]);
        }
    }
}

int whisper_copy_aheads_cross_attention_from_ctx(whisper_context * ctx) {
    // If alignment heads are selected, extract only those heads from step_cross_attn_w
    if (!ctx->aheads_list.empty()) {
        const int n_layers    = ctx->dims.n_text_layer;
        const int n_head      = ctx->dims.n_text_head;
        const int n_audio_ctx = ctx->enc_n_frames;

        if (ctx->step_cross_attn_w.empty()) return -1;

        const int n_selected = (int)ctx->aheads_list.size();
        ctx->aheads_n_heads     = n_selected;
        ctx->aheads_n_audio_ctx = n_audio_ctx;
        ctx->aheads_attn_data.resize(n_selected * n_audio_ctx);

        // step_cross_attn_w shape: [1, n_layers, n_head, n_audio_ctx]
        const int layer_stride = n_head * n_audio_ctx;
        for (int i = 0; i < n_selected; i++) {
            int layer = ctx->aheads_list[i].n_text_layer;
            int head  = ctx->aheads_list[i].n_head;
            if (layer < 0 || layer >= n_layers || head < 0 || head >= n_head) continue;
            const float * src = ctx->step_cross_attn_w.data() + layer * layer_stride + head * n_audio_ctx;
            std::memcpy(ctx->aheads_attn_data.data() + i * n_audio_ctx, src, n_audio_ctx * sizeof(float));
        }
        return 0;
    }
    // Fallback: use single layer (all heads)
    return whisper_copy_layer_cross_attention_from_ctx(ctx);
}

const float * whisper_get_layer_cross_attention_data_from_ctx(
    whisper_context * ctx, int * n_tokens_out, int * n_audio_ctx_out, int * n_heads_out)
{
    if (n_tokens_out)    *n_tokens_out    = 1;  // always 1 token in ORT step decode
    if (n_audio_ctx_out) *n_audio_ctx_out = ctx->cross_attn_n_audio_ctx;
    if (n_heads_out)     *n_heads_out     = ctx->cross_attn_n_heads;
    return ctx->cross_attn_data.data();
}

const float * whisper_get_aheads_cross_attention_data_from_ctx(
    whisper_context * ctx, int * n_tokens_out, int * n_audio_ctx_out, int * n_heads_out)
{
    // If alignment heads are selected, return the selected heads data
    if (!ctx->aheads_list.empty() && !ctx->aheads_attn_data.empty()) {
        if (n_tokens_out)    *n_tokens_out    = 1;
        if (n_audio_ctx_out) *n_audio_ctx_out = ctx->aheads_n_audio_ctx;
        if (n_heads_out)     *n_heads_out     = ctx->aheads_n_heads;
        return ctx->aheads_attn_data.data();
    }
    // Fallback: return single-layer data (all heads)
    return whisper_get_layer_cross_attention_data_from_ctx(ctx, n_tokens_out, n_audio_ctx_out, n_heads_out);
}

// ═══════════════════════════════════════════════════════════════════════════════
// High-level whisper_full (greedy + beam search)
// Used by whisper_streaming_cpp
// ═══════════════════════════════════════════════════════════════════════════════

whisper_full_params whisper_full_default_params(whisper_sampling_strategy strategy) {
    whisper_full_params p{};
    p.strategy          = strategy;
    p.n_threads         = std::min(4, (int)std::thread::hardware_concurrency());
    p.n_max_text_ctx    = 16384;
    p.language          = "en";
    p.beam_search.beam_size = 5;
    p.beam_search.patience  = 1.0f;
    p.greedy.best_of        = 1;
    p.temperature           = 0.0f;
    p.temperature_inc       = 0.2f;
    p.entropy_thold         = 2.4f;
    p.logprob_thold         = -1e9f;  // disabled by default; set > -100 to enable per-token guard
    p.no_speech_thold       = 0.6f;
    p.token_timestamps      = false;
    p.thold_pt              = 0.01f;
    p.thold_ptsum           = 0.01f;
    p.max_len               = 0;
    p.split_on_word         = false;
    p.max_tokens            = 0;
    p.suppress_non_speech_tokens = false;
    p.logits_filter_callback     = nullptr;
    p.logits_filter_callback_user_data = nullptr;
    return p;
}

// Build initial prompt token sequence
static std::vector<whisper_token> build_sot_sequence(whisper_context * ctx,
                                                      const whisper_full_params & params) {
    std::vector<whisper_token> seq;
    seq.push_back(whisper_token_sot(ctx));
    if (ctx->dims.is_multilingual) {
        int lang_id = whisper_lang_id(params.language ? params.language : "en");
        if (lang_id < 0) lang_id = 0;
        seq.push_back(whisper_token_lang(ctx, lang_id));
        seq.push_back(params.translate ? whisper_token_translate(ctx) : whisper_token_transcribe(ctx));
    }
    if (params.no_timestamps) {
        seq.push_back(whisper_token_not(ctx));
    }
    return seq;
}

int whisper_full(whisper_context * ctx,
                 whisper_full_params params,
                 const float * samples, int n_samples)
{
    ctx->segments.clear();

    // whisper_streaming uses whisper_full with a growing context window.
    // 30s zero-padding is required for stable inference (prevents hallucination),
    // matching the whisper-streaming paper implementation.
    const int N_SAMPLES_30S = WHISPER_CHUNK_SIZE * WHISPER_SAMPLE_RATE;  // 480000
    std::vector<float> padded_samples;
    const float * enc_samples   = samples;
    int           enc_n_samples = n_samples;
    if (n_samples < N_SAMPLES_30S) {
        padded_samples.assign(samples, samples + n_samples);
        padded_samples.resize(N_SAMPLES_30S, 0.0f);
        enc_samples   = padded_samples.data();
        enc_n_samples = N_SAMPLES_30S;
    }

    if (whisper_pcm_to_mel(ctx, enc_samples, enc_n_samples, params.n_threads) != 0) return -1;
    // Encode
    if (whisper_encode(ctx, 0, params.n_threads) != 0) return -1;

    const whisper_token eot = whisper_token_eot(ctx);
    const int n_vocab = ctx->dims.n_vocab;
    int max_tokens = params.max_tokens > 0 ? params.max_tokens : ctx->dims.n_text_ctx;

    // ── Greedy decode ─────────────────────────────────────────────────────────
    auto sot_seq = build_sot_sequence(ctx, params);

    if (whisper_decode(ctx, sot_seq.data(), (int)sot_seq.size(), 0, params.n_threads) != 0)
        return -1;

    whisper_segment seg;
    seg.t0 = 0;
    std::vector<whisper_token> decoded;
    int n_past       = (int)sot_seq.size();
    int n_last_batch = n_past;  // tracks how many tokens were in the last decode call

    // Cap max_tokens so that n_past never reaches n_text_ctx (positional embedding limit).
    const int n_text_ctx = ctx->dims.n_text_ctx;
    if (max_tokens > n_text_ctx - n_past - 1)
        max_tokens = n_text_ctx - n_past - 1;

    // Build token_data array for logits_filter_callback (tracks decoded tokens)
    std::vector<whisper_token_data> token_data_for_cb;
    for (auto t : sot_seq) {
        whisper_token_data td{}; td.id = t; td.p = 1.0f;
        token_data_for_cb.push_back(td);
    }

    for (int step = 0; step < max_tokens; ++step) {
        // logits buffer holds n_last_batch * n_vocab values; last token is at the end.
        float * logits = whisper_get_logits(ctx) + (n_last_batch - 1) * n_vocab;

        // Suppress special tokens above EOT (timestamps etc.) — always.
        for (int i = eot + 1; i < n_vocab; ++i) logits[i] = -1e9f;

        // Call logits_filter_callback if set (allows conditional EOT suppression)
        if (params.logits_filter_callback) {
            params.logits_filter_callback(
                ctx, nullptr,
                token_data_for_cb.data(), (int)token_data_for_cb.size(),
                logits, params.logits_filter_callback_user_data);
        } else {
            // Default: suppress EOT (original behavior)
            logits[eot] = -1e9f;
        }

        whisper_token next = (whisper_token)(std::max_element(logits, logits + n_vocab) - logits);
        if (next >= eot) break;
        decoded.push_back(next);

        whisper_token_data td{}; td.id = next; td.p = 1.0f;
        token_data_for_cb.push_back(td);

        if (whisper_decode(ctx, &next, 1, n_past, params.n_threads) != 0) break;
        ++n_past;
        n_last_batch = 1;  // step decode always processes 1 token
    }

    for (whisper_token t : decoded) {
        seg.text += ctx->vocab[t];
        whisper_token_data td{};
        td.id = t;
        td.p  = 1.0f;
        seg.tokens.push_back(td);
    }
    seg.t1 = (int64_t)(n_samples * 100LL / WHISPER_SAMPLE_RATE);
    ctx->segments.push_back(seg);
    return 0;
}

// ── Result accessors ──────────────────────────────────────────────────────────

int whisper_full_n_segments(whisper_context * ctx) {
    return (int)ctx->segments.size();
}

const char * whisper_full_get_segment_text(whisper_context * ctx, int i) {
    return ctx->segments[i].text.c_str();
}

int64_t whisper_full_get_segment_t0(whisper_context * ctx, int i) {
    return ctx->segments[i].t0;
}

int64_t whisper_full_get_segment_t1(whisper_context * ctx, int i) {
    return ctx->segments[i].t1;
}

bool whisper_full_get_segment_speaker_turn_next(whisper_context * ctx, int i) {
    return ctx->segments[i].speaker_turn_next;
}

int whisper_full_n_tokens(whisper_context * ctx, int i) {
    return (int)ctx->segments[i].tokens.size();
}

whisper_token_data whisper_full_get_token_data(whisper_context * ctx, int i, int j) {
    return ctx->segments[i].tokens[j];
}

whisper_token whisper_full_get_token_id(whisper_context * ctx, int i, int j) {
    return ctx->segments[i].tokens[j].id;
}

const char * whisper_full_get_token_text(whisper_context * ctx, int i, int j) {
    return whisper_token_to_str(ctx, ctx->segments[i].tokens[j].id);
}

float whisper_full_get_token_p(whisper_context * ctx, int i, int j) {
    return ctx->segments[i].tokens[j].p;
}

// ── Timings ───────────────────────────────────────────────────────────────────

void whisper_print_timings(whisper_context * ctx) {
    fprintf(stderr, "\n");
    fprintf(stderr, "%s:     encode time = %8.2f ms\n", __func__, ctx->t_encode_us / 1000.0);
    fprintf(stderr, "%s:     decode time = %8.2f ms\n", __func__, ctx->t_decode_us / 1000.0);
}

void whisper_reset_timings(whisper_context * ctx) {
    ctx->t_encode_us = 0;
    ctx->t_decode_us = 0;
}
