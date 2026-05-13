#define _USE_MATH_DEFINES  // M_PI on MSVC
#include "whisper_ort.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <unordered_map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

static int64_t elapsed_us(
    std::chrono::steady_clock::time_point t0,
    std::chrono::steady_clock::time_point t1) {
    return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
}

static double us_to_ms(int64_t us) {
    return us / 1000.0;
}

static const char * npu_decode_phase_name(int phase) {
    switch (phase) {
        case 1: return "prefill";
        case 2: return "generation";
        default: return "unknown";
    }
}

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

    struct NpuRunProfile {
        int64_t encoder_calls = 0;
        int64_t encoder_total_us = 0;
        int64_t encoder_prepare_us = 0;
        int64_t encoder_run_us = 0;
        int64_t encoder_parse_us = 0;

        int64_t chunk_calls = 0;
        int64_t chunk_total_us = 0;
        int64_t chunk_prepare_us = 0;
        int64_t chunk_run_us = 0;
        int64_t chunk_parse_us = 0;

        int64_t one_step_calls = 0;
        int64_t one_step_total_us = 0;
        int64_t one_step_prepare_us = 0;
        int64_t one_step_run_us = 0;
        int64_t one_step_parse_us = 0;

        int64_t prefill_calls = 0;
        int64_t prefill_total_us = 0;
        int64_t generation_calls = 0;
        int64_t generation_total_us = 0;
    } npu_profile;

    // ═══════════════════════════════════════════════════════════════════════════
    // NPU N/K unrolled decoder state
    // ═══════════════════════════════════════════════════════════════════════════
    int backend = 0;  // 0 = ONNX, 1 = NPU_NK

    // NPU config (from config.json)
    int npu_num_blocks = 0;
    int npu_num_heads  = 0;
    int npu_head_dim   = 0;
    int npu_num_chunks = 0;    // 6 for the full N=30 K=5 design
    bool npu_use_1step = false;
    int npu_decoder_kv_cache_size = 199;
    std::string npu_model_dir;
    bool npu_use_cpu_ep = false;
    bool npu_has_30s_reinfer = false;
    int npu_reinfer_kv_cache_size = 100;
    std::vector<float> npu_buckets_sec;
    std::map<float, int> npu_bucket_audio_emb_len;

    // Per-chain K table (chunks 0..num_chunks-1).
    //   normal chain  (no prev degen, no carryover): K = [4, 6, 5, 5, 5, 5]
    //   prefill chain variants preserve up to N carryover tokens by choosing:
    //     chunk_0 = 4 + prompt_len, chunk_1 = 10 - chunk_0, chunks 2..5 shared.
    // chunks 2..5 are byte-identical between chains and weight-shared in the same bin.
    std::vector<int> npu_K_normal;
    std::vector<int> npu_K_prefill;
    std::vector<int> npu_prefill_prompt_token_lens;

    // Per-bucket NPU sessions
    struct NpuBucketSessions {
        Ort::Session encoder{nullptr};
        Ort::Session decoder_1step{nullptr};
        bool has_1step = false;
        // Chain-specific chunks 0 and 1.
        //  normal chain exists for empty-prompt starts.
        //  prefill chain variants preserve carryover prompt lengths.
        Ort::Session chunk_0_normal{nullptr};
        Ort::Session chunk_1_normal{nullptr};
        bool has_normal = false;
        std::map<int, Ort::Session> chunk_0_prefill_variants;
        std::map<int, Ort::Session> chunk_1_prefill_variants;
        // Shared chunks 2..num_chunks-1 (one session each, shared across both chains).
        std::vector<Ort::Session> chunks_shared;
    };
    std::map<float, NpuBucketSessions> npu_sessions;
    std::set<float> npu_preloaded_buckets;  // buckets that should never be evicted

    // 30s reinference sessions (mode 3 only)
    struct Npu30sSessions {
        Ort::Session encoder{nullptr};
        Ort::Session decoder_1step{nullptr};
        int kv_cache_size = 100;  // fixed KV cache for 1-step decoder
    };
    std::unique_ptr<Npu30sSessions> npu_30s;

    // Per-block KV cache (NPU format: separate per block)
    // KV cache stored as raw bytes (fp16 for NPU, fp32 for ONNX)
    struct NpuBlockKV {
        std::vector<std::vector<char>> k_cache;  // [num_blocks], raw tensor bytes
        std::vector<std::vector<char>> v_cache;  // [num_blocks], raw tensor bytes
        int audio_ctx = 0;  // for cross-KV: audio_emb_len
        int valid_audio_ctx = 0;  // actual non-padded encoder length at 50 fps
        int seq_len = 0;    // for self-KV: current sequence length
    };
    NpuBlockKV npu_cross_kv;
    NpuBlockKV npu_self_kv;

    int npu_n_alignment_heads = 0;  // 0 = legacy (1 head avg), >0 = per-head alignment

    // N/K buffer from the last chunk NPU call.
    std::vector<std::vector<float>> npu_chunk_logits;      // K × [vocab_size]
    std::vector<std::vector<float>> npu_chunk_peak_attn;   // K × [audio_ctx]
    std::vector<std::vector<float>> npu_chunk_cross_attn;  // K × [n_aheads * audio_ctx]
    int npu_buffer_read_idx    = 0;   // next position to read (0..K-1)
    int npu_current_chunk_idx  = 0;   // current chunk (0..num_chunks-1)
    float npu_current_bucket   = 0;   // active bucket seconds
    // Active chain for the in-flight decode round.
    //   0 = normal  (no prev degen, n_tokens=4 prefill, K=[4,6,5,5,5,5])
    //   1 = prefill (prev degen + carryover prompt tokens, chunk_0/1 chosen by length)
    int  npu_active_chain = 0;
    int  npu_active_prefill_prompt_tokens = 0;
    bool npu_force_30s = false;         // force 30s bucket for reinference
    int  npu_decode_phase = 0;          // 0=unknown, 1=prefill, 2=generation

    // ── Phase A: shadow argmax counters ───────────────────────────────────────
    // Compare host's chosen next token against what NPU's internal speculative
    // argmax would have produced for the same step. Mismatch = the buffered
    // logits/cross_attn at the next step were computed under a history the
    // host did NOT accept. Cumulative across the whole run unless reset.
    int64_t shadow_n_compares             = 0;
    int64_t shadow_n_match                = 0;
    int64_t shadow_n_diverge_in_chunk     = 0;
    int64_t shadow_n_diverge_at_chunk_end = 0;

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
    // CRITICAL: open in BINARY mode. On Windows, the default text-mode ifstream
    // treats byte 0x1A (Ctrl+Z, DOS EOF marker) as end-of-file, and Whisper's
    // vocab.txt contains 0x1A as a normal BPE token at index ~26. Without binary
    // mode, vocab loading silently truncates after the first 26 tokens, all
    // special tokens (<|startoftranscript|> etc.) are missing, find_token()
    // returns -1, and the decoder receives input_ids=-1.
    std::ifstream f(model_dir + "/vocab.txt", std::ios::binary);
    if (!f.is_open()) return false;
    std::string line;
    while (std::getline(f, line)) {
        // In binary mode std::getline does not strip the trailing '\r' from
        // CRLF line endings — strip it manually so token strings stay clean.
        if (!line.empty() && line.back() == '\r') line.pop_back();
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

#ifdef _WIN32
static std::wstring to_wstring(const std::string & s) {
    std::wstring ws(s.begin(), s.end());
    return ws;
}
#define ORT_PATH(s) to_wstring(s).c_str()
#else
#define ORT_PATH(s) (s).c_str()
#endif

static Ort::Session make_session(Ort::Env & env, const std::string & path, bool use_gpu) {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    opts.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    if (use_gpu) {
        OrtCUDAProviderOptions cuda_opts{};
        try { opts.AppendExecutionProvider_CUDA(cuda_opts); fprintf(stderr, "using CUDA backend\n"); }
        catch (...) { fprintf(stderr, "cuda_init() failed\n"); }
    }
    return Ort::Session(env, ORT_PATH(path), opts);
}

// NPU session: QNN EP or CPU EP for debugging NPU models
// Global: model directory for locating QNN DLLs
static std::string g_npu_model_dir;

// Global: path to QnnHtp.dll (set from config.json or auto-detected from ORT capi)
static std::string g_npu_qnn_htp_path;
static std::string g_npu_qnn_profile_level;
static std::string g_npu_qnn_profile_file;
static std::string g_npu_qnn_profile_include;
static int g_npu_qnn_vtcm_mb = -1;
static int g_npu_qnn_profile_session_idx = 0;
static bool g_npu_qnn_share_ep_contexts = true;
static bool g_npu_eager_preload = true;

static std::string qnn_profile_file_for_session(const std::string & path) {
    if (path.empty()) {
        return path;
    }
    const int idx = g_npu_qnn_profile_session_idx++;
    const std::string suffix = "_s" + std::to_string(idx);
    const size_t slash = path.find_last_of("/\\");
    const size_t dot = path.find_last_of('.');
    if (dot != std::string::npos && (slash == std::string::npos || dot > slash)) {
        return path.substr(0, dot) + suffix + path.substr(dot);
    }
    return path + suffix;
}

static bool qnn_profile_matches_include(const std::string & model_path) {
    if (g_npu_qnn_profile_include.empty()) {
        return true;
    }
    size_t start = 0;
    while (start <= g_npu_qnn_profile_include.size()) {
        size_t comma = g_npu_qnn_profile_include.find(',', start);
        std::string needle = g_npu_qnn_profile_include.substr(
            start, comma == std::string::npos ? std::string::npos : comma - start);
        needle.erase(needle.begin(), std::find_if(needle.begin(), needle.end(),
            [](unsigned char ch) { return !std::isspace(ch); }));
        needle.erase(std::find_if(needle.rbegin(), needle.rend(),
            [](unsigned char ch) { return !std::isspace(ch); }).base(), needle.end());
        if (!needle.empty() && model_path.find(needle) != std::string::npos) {
            return true;
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return false;
}

static Ort::Session make_npu_session(Ort::Env & env, const std::string & path, bool use_cpu_ep) {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    if (!use_cpu_ep) {
        // Match Python stream_transcribe.py settings exactly:
        //   - backend_path / enable_htp_fp16_precision / htp_performance_mode (provider opts)
        //   - session.disable_cpu_ep_fallback=1 (session opt, BEFORE EP append)
        //   - ep.share_ep_contexts=1 (session opt, BEFORE EP append) — required
        //     to fit multiple bucket bins in HTP memory. Without this, each
        //     session creates its own QNN context and ~3 buckets × 8 chunks
        //     blows past the NPU memory limit (Error code 1007). With it, all
        //     sessions referencing the same bin file share one NPU context.
        //     The earlier deadlock that made me remove this was caused by
        //     setting it AFTER AppendExecutionProvider — see #12 in NPU_DEBUG_LOG.
        opts.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
        if (g_npu_qnn_share_ep_contexts) {
            opts.AddConfigEntry("ep.share_ep_contexts", "1");
        }

        std::unordered_map<std::string, std::string> qnn_opts;
        qnn_opts["backend_path"] = g_npu_qnn_htp_path;
        qnn_opts["enable_htp_fp16_precision"] = "1";
        qnn_opts["htp_performance_mode"] = "burst";
        std::string qnn_profile_file;
        const bool profile_this_session =
            !g_npu_qnn_profile_level.empty() &&
            g_npu_qnn_profile_level != "off" &&
            qnn_profile_matches_include(path);
        if (profile_this_session) {
            qnn_profile_file = qnn_profile_file_for_session(g_npu_qnn_profile_file);
            qnn_opts["profiling_level"] = g_npu_qnn_profile_level;
            qnn_opts["profiling_file_path"] = qnn_profile_file;
        }
        if (g_npu_qnn_vtcm_mb > 0) {
            qnn_opts["vtcm_mb"] = std::to_string(g_npu_qnn_vtcm_mb);
        }
        printf("[NPU] make_npu_session: backend_path=%s\n", g_npu_qnn_htp_path.c_str());
        printf("[NPU] QNN share_ep_contexts=%s\n", g_npu_qnn_share_ep_contexts ? "1" : "0");
        if (profile_this_session) {
            printf("[NPU] QNN profiling: level=%s file=%s\n",
                   g_npu_qnn_profile_level.c_str(), qnn_profile_file.c_str());
        } else if (!g_npu_qnn_profile_level.empty() && g_npu_qnn_profile_level != "off") {
            printf("[NPU] QNN profiling: skipped for %s (include=%s)\n",
                   path.c_str(), g_npu_qnn_profile_include.c_str());
        }
        if (g_npu_qnn_vtcm_mb > 0) {
            printf("[NPU] QNN vtcm_mb=%d\n", g_npu_qnn_vtcm_mb);
        }
        fflush(stdout);
        try {
            opts.AppendExecutionProvider("QNN", qnn_opts);
            printf("[NPU] AppendExecutionProvider QNN succeeded\n");
            fflush(stdout);
        } catch (const Ort::Exception & e) {
            printf("[NPU] QNN EP init failed: %s\n", e.what());
            fflush(stdout);
        }
    }
    return Ort::Session(env, ORT_PATH(path), opts);
}

// ── NPU config.json loader ───────────────────────────────────────────────────

struct NpuConfig {
    std::string model;
    std::string decoder_mode = "nk";
    int num_blocks = 6;
    int num_heads  = 8;
    int head_dim   = 64;
    int num_chunks = 6;
    // Per-chain K table (chunks 0..num_chunks-1).
    std::vector<int> K_normal;   // [4, 6, 5, 5, 5, 5]
    std::vector<int> K_prefill;  // [6, 4, 5, 5, 5, 5]
    std::vector<int> prefill_prompt_token_lens;  // e.g. [1,2,3,4,5]
    std::vector<float> buckets_sec;
    std::vector<int> bucket_audio_emb_len;
    int n_alignment_heads = 0;
    int decoder_1step_kv_cache_size = 199;
    bool has_30s_reinfer = false;
    int reinfer_kv_cache_size = 100;
};

static bool load_npu_config(const std::string & dir, NpuConfig & cfg) {
    std::string path = dir + "/config.json";
    std::ifstream f(path);
    if (!f.is_open()) return false;

    // Simple JSON parser for flat config
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    auto get_int = [&](const std::string & key) -> int {
        auto pos = content.find("\"" + key + "\"");
        if (pos == std::string::npos) return 0;
        pos = content.find(":", pos);
        return std::stoi(content.substr(pos + 1));
    };
    auto get_str = [&](const std::string & key) -> std::string {
        auto pos = content.find("\"" + key + "\"");
        if (pos == std::string::npos) return "";
        pos = content.find("\"", content.find(":", pos) + 1);
        auto end = content.find("\"", pos + 1);
        return content.substr(pos + 1, end - pos - 1);
    };

    cfg.model      = get_str("model");
    cfg.decoder_mode = get_str("decoder_mode");
    if (cfg.decoder_mode.empty()) cfg.decoder_mode = "nk";
    cfg.num_blocks = get_int("num_blocks");
    cfg.num_heads  = get_int("num_heads");
    cfg.head_dim   = get_int("head_dim");
    cfg.num_chunks = get_int("num_chunks");

    // Parse chunk_K_normal array, e.g. [4,6,5,5,5,5]
    auto parse_int_array = [&](const std::string & key, std::vector<int> & out) {
        auto pos = content.find("\"" + key + "\"");
        if (pos == std::string::npos) return;
        auto arr_start = content.find("[", pos);
        auto arr_end = content.find("]", arr_start);
        std::string arr = content.substr(arr_start + 1, arr_end - arr_start - 1);
        std::istringstream ss(arr);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            try { out.push_back(std::stoi(tok)); } catch (...) {}
        }
    };
    parse_int_array("K_normal", cfg.K_normal);
    parse_int_array("K_prefill", cfg.K_prefill);
    parse_int_array("prefill_prompt_token_lens", cfg.prefill_prompt_token_lens);
    parse_int_array("bucket_audio_emb_len", cfg.bucket_audio_emb_len);

    // Parse buckets_sec array
    {
        auto pos = content.find("\"buckets_sec\"");
        if (pos != std::string::npos) {
            auto arr_start = content.find("[", pos);
            auto arr_end = content.find("]", arr_start);
            std::string arr = content.substr(arr_start + 1, arr_end - arr_start - 1);
            std::istringstream ss(arr);
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                try { cfg.buckets_sec.push_back(std::stof(tok)); } catch (...) {}
            }
        }
    }

    cfg.n_alignment_heads = get_int("n_alignment_heads");
    cfg.decoder_1step_kv_cache_size = get_int("decoder_1step_kv_cache_size");
    if (cfg.decoder_1step_kv_cache_size <= 0) cfg.decoder_1step_kv_cache_size = 199;

    cfg.has_30s_reinfer = (content.find("\"has_30s_reinfer\"") != std::string::npos &&
                           content.find("true", content.find("\"has_30s_reinfer\"")) != std::string::npos);
    if (cfg.has_30s_reinfer) {
        cfg.reinfer_kv_cache_size = get_int("reinfer_kv_cache_size");
        if (cfg.reinfer_kv_cache_size <= 0) cfg.reinfer_kv_cache_size = 100;
    }

    fprintf(stderr, "%s: loaded NPU config: model=%s, decoder_mode=%s, chunks=%d, K_normal=[",
            __func__, cfg.model.c_str(), cfg.decoder_mode.c_str(), cfg.num_chunks);
    for (int i = 0; i < (int)cfg.K_normal.size(); i++)
        fprintf(stderr, "%s%d", i ? "," : "", cfg.K_normal[i]);
    fprintf(stderr, "], K_prefill=[");
    for (int i = 0; i < (int)cfg.K_prefill.size(); i++)
        fprintf(stderr, "%s%d", i ? "," : "", cfg.K_prefill[i]);
    fprintf(stderr, "], prefill_prompt_lens=[");
    for (int i = 0; i < (int)cfg.prefill_prompt_token_lens.size(); i++)
        fprintf(stderr, "%s%d", i ? "," : "", cfg.prefill_prompt_token_lens[i]);
    fprintf(stderr, "], buckets=[");
    for (int i = 0; i < (int)cfg.buckets_sec.size(); i++)
        fprintf(stderr, "%s%.0f", i ? "," : "", cfg.buckets_sec[i]);
    fprintf(stderr, "]s, emb=[");
    for (int i = 0; i < (int)cfg.bucket_audio_emb_len.size(); i++)
        fprintf(stderr, "%s%d", i ? "," : "", cfg.bucket_audio_emb_len[i]);
    fprintf(stderr, "], 30s_reinfer=%s\n", cfg.has_30s_reinfer ? "yes" : "no");

    return true;
}

// Select smallest bucket >= required duration
static float select_bucket(const std::vector<float> & buckets, float required_sec) {
    for (float b : buckets) {
        if (b >= required_sec) return b;
    }
    return buckets.back();  // fallback to largest
}

static bool npu_supports_prefill_prompt_len(const whisper_context * ctx, int prompt_len) {
    return std::find(
        ctx->npu_prefill_prompt_token_lens.begin(),
        ctx->npu_prefill_prompt_token_lens.end(),
        prompt_len) != ctx->npu_prefill_prompt_token_lens.end();
}

static int npu_shared_prefix_decode_steps(const whisper_context * ctx) {
    if ((int)ctx->npu_K_normal.size() >= 2) {
        return ctx->npu_K_normal[0] + ctx->npu_K_normal[1];
    }
    if ((int)ctx->npu_K_prefill.size() >= 2) {
        return ctx->npu_K_prefill[0] + ctx->npu_K_prefill[1];
    }
    return 10;
}

static int npu_get_chunk_k(const whisper_context * ctx, int chunk_idx) {
    if (ctx->npu_active_chain == 0 || ctx->npu_use_1step) {
        if (chunk_idx >= 0 && chunk_idx < (int)ctx->npu_K_normal.size()) {
            return ctx->npu_K_normal[chunk_idx];
        }
        return 0;
    }

    if (chunk_idx == 0) {
        return ctx->npu_K_normal[0] + ctx->npu_active_prefill_prompt_tokens;
    }
    if (chunk_idx == 1) {
        return npu_shared_prefix_decode_steps(ctx) - npu_get_chunk_k(ctx, 0);
    }
    if (chunk_idx >= 0 && chunk_idx < (int)ctx->npu_K_normal.size()) {
        return ctx->npu_K_normal[chunk_idx];
    }
    return 0;
}

static bool npu_load_bucket_sessions(whisper_context * ctx, float bucket) {
    // Multi-bucket cache: preloaded buckets are kept permanently;
    // at most one non-preloaded ("transient") bucket is allowed at a time.
    if (ctx->npu_sessions.count(bucket) > 0) {
        return true;
    }

    // Evict any existing transient (non-preloaded) bucket to stay within
    // HTP memory limits. Preloaded buckets are never evicted.
    for (auto it = ctx->npu_sessions.begin(); it != ctx->npu_sessions.end(); ) {
        if (ctx->npu_preloaded_buckets.count(it->first) == 0) {
            fprintf(stderr, "%s: evicting transient bucket %.0fs\n",
                    __func__, it->first);
            it = ctx->npu_sessions.erase(it);
        } else {
            ++it;
        }
    }

    const int bs = (int)bucket;
    const std::string bs_str = std::to_string(bs);

    try {
        whisper_context::NpuBucketSessions bsess;
        const std::string dir = ctx->npu_model_dir;

        bsess.encoder = make_npu_session(
            ctx->env, dir + "/encoder_" + bs_str + "s.onnx", ctx->npu_use_cpu_ep);

        if (ctx->npu_use_1step) {
            bsess.decoder_1step = make_npu_session(
                ctx->env, dir + "/decoder_1step_" + bs_str + "s.onnx", ctx->npu_use_cpu_ep);
            bsess.has_1step = true;
        } else {
            for (int prompt_len : ctx->npu_prefill_prompt_token_lens) {
                const std::string suffix = "_p" + std::to_string(prompt_len);
                bsess.chunk_0_prefill_variants.emplace(
                    prompt_len,
                    make_npu_session(
                        ctx->env,
                        dir + "/decoder_" + bs_str + "s_chunk_0_prefill" + suffix + ".onnx",
                        ctx->npu_use_cpu_ep));
                bsess.chunk_1_prefill_variants.emplace(
                    prompt_len,
                    make_npu_session(
                        ctx->env,
                        dir + "/decoder_" + bs_str + "s_chunk_1_prefill" + suffix + ".onnx",
                        ctx->npu_use_cpu_ep));
            }

            const std::string chunk0_normal_path =
                dir + "/decoder_" + bs_str + "s_chunk_0_normal.onnx";
            const std::string chunk1_normal_path =
                dir + "/decoder_" + bs_str + "s_chunk_1_normal.onnx";
            if (std::filesystem::exists(chunk0_normal_path) &&
                std::filesystem::exists(chunk1_normal_path)) {
                bsess.chunk_0_normal = make_npu_session(
                    ctx->env, chunk0_normal_path,
                    ctx->npu_use_cpu_ep);
                bsess.chunk_1_normal = make_npu_session(
                    ctx->env, chunk1_normal_path,
                    ctx->npu_use_cpu_ep);
                bsess.has_normal = true;
            }

            for (int c = 2; c < ctx->npu_num_chunks; ++c) {
                bsess.chunks_shared.push_back(make_npu_session(
                    ctx->env,
                    dir + "/decoder_" + bs_str + "s_chunk_" + std::to_string(c) + ".onnx",
                    ctx->npu_use_cpu_ep));
            }
        }

        ctx->npu_sessions.emplace(bucket, std::move(bsess));
        fprintf(stderr,
                "%s: loaded NPU bucket %.0fs (%s) [cached=%zu]\n",
                __func__, bucket, ctx->npu_use_1step ? "1-step" : "N/K",
                ctx->npu_sessions.size());
        return true;
    } catch (const Ort::Exception & e) {
        fprintf(stderr,
                "%s: failed to load NPU bucket %.0fs sessions: %s\n",
                __func__, bucket, e.what());
        ctx->npu_sessions.erase(bucket);
        return false;
    }
}

static bool npu_ensure_bucket_sessions(whisper_context * ctx, float bucket) {
    if (bucket <= 0.0f || bucket == 30.0f) {
        return true;
    }
    return npu_load_bucket_sessions(ctx, bucket);
}

static bool npu_ensure_30s_sessions(whisper_context * ctx) {
    if (!ctx->npu_has_30s_reinfer) {
        return false;
    }
    if (ctx->npu_30s) {
        return true;
    }

    try {
        auto sess = std::make_unique<whisper_context::Npu30sSessions>();
        sess->encoder = make_npu_session(
            ctx->env, ctx->npu_model_dir + "/encoder_30s.onnx", ctx->npu_use_cpu_ep);
        sess->decoder_1step = make_npu_session(
            ctx->env, ctx->npu_model_dir + "/decoder_1step_30s.onnx", ctx->npu_use_cpu_ep);
        sess->kv_cache_size = ctx->npu_reinfer_kv_cache_size;
        ctx->npu_30s = std::move(sess);
        fprintf(stderr, "%s: lazily loaded 30s reinference sessions (KV=%d)\n",
                __func__, ctx->npu_reinfer_kv_cache_size);
        return true;
    } catch (const Ort::Exception & e) {
        fprintf(stderr, "%s: failed to load 30s reinference sessions: %s\n",
                __func__, e.what());
        ctx->npu_30s.reset();
        return false;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Lifecycle
// ═══════════════════════════════════════════════════════════════════════════════

whisper_context_params whisper_context_default_params() {
    whisper_context_params p{};
    p.use_gpu    = false;
    p.gpu_device = 0;
    p.flash_attn = false;
    p.use_npu    = false;
    p.use_cpu_ep = false;
    p.qnn_htp_path = nullptr;
    p.qnn_profile_level = nullptr;
    p.qnn_profile_file = nullptr;
    p.qnn_profile_include = nullptr;
    p.qnn_vtcm_mb = -1;
    p.qnn_share_ep_contexts = true;
    p.npu_eager_preload = true;
    return p;
}

whisper_context * whisper_init_from_file_with_params(
    const char * model_path, whisper_context_params params)
{
    std::string dir(model_path);

    auto * ctx = new whisper_context();

    // ── NPU N/K mode ─────────────────────────────────────────────────────────
    if (params.use_npu) {
        NpuConfig cfg;
        if (!load_npu_config(dir, cfg)) {
            fprintf(stderr, "%s: config.json not found in '%s', NPU mode requires it\n", __func__, dir.c_str());
            delete ctx;
            return nullptr;
        }

        ctx->backend         = 1 /*BACKEND_NPU_NK*/;
        g_npu_model_dir      = dir;
        ctx->npu_model_dir   = dir;
        ctx->npu_use_cpu_ep  = params.use_cpu_ep;

        // Set QnnHtp.dll path for NPU backend
        if (params.qnn_htp_path && params.qnn_htp_path[0]) {
            g_npu_qnn_htp_path = params.qnn_htp_path;
        } else {
            // Default: look in model directory
            g_npu_qnn_htp_path = dir + "/QnnHtp.dll";
        }
        printf("[NPU] QnnHtp path: %s\n", g_npu_qnn_htp_path.c_str());
        g_npu_qnn_profile_level = (
            params.qnn_profile_level && params.qnn_profile_level[0])
            ? params.qnn_profile_level
            : "off";
        g_npu_qnn_profile_file = (
            params.qnn_profile_file && params.qnn_profile_file[0])
            ? params.qnn_profile_file
            : "";
        g_npu_qnn_profile_include = (
            params.qnn_profile_include && params.qnn_profile_include[0])
            ? params.qnn_profile_include
            : "";
        g_npu_qnn_vtcm_mb = params.qnn_vtcm_mb;
        g_npu_qnn_share_ep_contexts = params.qnn_share_ep_contexts;
        g_npu_eager_preload = params.npu_eager_preload;
        if (g_npu_qnn_profile_level != "off") {
            if (g_npu_qnn_profile_file.empty()) {
                fprintf(stderr, "%s: qnn_profile_file is required when QNN profiling is enabled\n", __func__);
                delete ctx;
                return nullptr;
            }
            printf("[NPU] QNN profile level: %s\n", g_npu_qnn_profile_level.c_str());
            printf("[NPU] QNN profile file: %s\n", g_npu_qnn_profile_file.c_str());
            if (!g_npu_qnn_profile_include.empty()) {
                printf("[NPU] QNN profile include: %s\n", g_npu_qnn_profile_include.c_str());
            }
        }
        if (g_npu_qnn_vtcm_mb > 0) {
            printf("[NPU] QNN vtcm_mb: %d\n", g_npu_qnn_vtcm_mb);
        }
        printf("[NPU] QNN share_ep_contexts: %s\n", g_npu_qnn_share_ep_contexts ? "1" : "0");
        printf("[NPU] NPU eager preload: %s\n", g_npu_eager_preload ? "1" : "0");
        fflush(stdout);
        ctx->npu_num_blocks  = cfg.num_blocks;
        ctx->npu_num_heads   = cfg.num_heads;
        ctx->npu_head_dim    = cfg.head_dim;
        ctx->npu_num_chunks  = cfg.num_chunks;
        ctx->npu_use_1step   = (cfg.decoder_mode == "1step");
        ctx->npu_decoder_kv_cache_size = cfg.decoder_1step_kv_cache_size;
        ctx->npu_has_30s_reinfer = cfg.has_30s_reinfer;
        ctx->npu_reinfer_kv_cache_size = cfg.reinfer_kv_cache_size > 0 ? cfg.reinfer_kv_cache_size : 100;
        ctx->npu_K_normal    = cfg.K_normal;
        ctx->npu_K_prefill   = cfg.K_prefill;
        ctx->npu_prefill_prompt_token_lens = cfg.prefill_prompt_token_lens;
        if (ctx->npu_prefill_prompt_token_lens.empty() && !ctx->npu_use_1step) {
            // Backward-compatible legacy deploys only support the old 2-token prefill.
            ctx->npu_prefill_prompt_token_lens = {2};
        }
        // Restrict runtime buckets to 3-6s. Audio >6s falls back to 30s.
        ctx->npu_buckets_sec = {3.0f, 4.0f, 5.0f, 6.0f};
        fprintf(stderr, "%s: runtime buckets restricted to [3s,4s,5s,6s], >6s -> 30s fallback\n", __func__);
        ctx->npu_n_alignment_heads = cfg.n_alignment_heads;
        if (!cfg.bucket_audio_emb_len.empty() &&
            cfg.bucket_audio_emb_len.size() != cfg.buckets_sec.size()) {
            fprintf(stderr, "%s: bucket_audio_emb_len size mismatch with buckets_sec\n",
                    __func__);
            delete ctx;
            return nullptr;
        }
        for (size_t i = 0; i < cfg.buckets_sec.size(); ++i) {
            float bucket = cfg.buckets_sec[i];
            int raw_emb = (int)(bucket * 50.0f);
            int default_emb = ((raw_emb + 7) / 8) * 8;
            int emb = (i < cfg.bucket_audio_emb_len.size()) ? cfg.bucket_audio_emb_len[i] : default_emb;
            ctx->npu_bucket_audio_emb_len[bucket] = emb;
        }

        if (!ctx->npu_use_1step) {
            if ((int)cfg.K_normal.size() != cfg.num_chunks ||
                (int)cfg.K_prefill.size() != cfg.num_chunks) {
                fprintf(stderr, "%s: K_normal/K_prefill size mismatch with num_chunks=%d\n",
                        __func__, cfg.num_chunks);
                delete ctx;
                return nullptr;
            }
            // Both chains must converge: cumulative K through chunk_1 must be equal.
            if (cfg.K_normal[0] + cfg.K_normal[1] != cfg.K_prefill[0] + cfg.K_prefill[1]) {
                fprintf(stderr, "%s: chains do not converge after chunk_1: "
                        "K_normal[0]+[1]=%d K_prefill[0]+[1]=%d\n",
                        __func__,
                        cfg.K_normal[0] + cfg.K_normal[1],
                        cfg.K_prefill[0] + cfg.K_prefill[1]);
                delete ctx;
                return nullptr;
            }
        }

        // QNN EP is built-in to pip onnxruntime-qnn; no RegisterExecutionProviderLibrary needed.
        // backend_path (QnnHtp.dll) must be in the onnxruntime capi directory where all
        // QNN DLL dependencies (QnnHtpV73Stub, QnnSystem, libQnnHtpV73Skel.so) reside.
        fprintf(stderr, "%s: NPU %s mode initialized (%d buckets)\n",
                __func__, ctx->npu_use_1step ? "1-step" : "N/K",
                (int)cfg.buckets_sec.size());

        // Eagerly load ALL runtime buckets (3-6s) + 30s reinference by default.
        // Profiling-isolation runs can disable this to avoid holding many QNN
        // contexts while debugging one session.
        if (g_npu_eager_preload) {
            fprintf(stderr, "%s: eager-loading all runtime buckets [3s,4s,5s,6s] ...\n", __func__);
            for (float b : ctx->npu_buckets_sec) {
                if (!npu_load_bucket_sessions(ctx, b)) {
                    fprintf(stderr, "%s: WARNING: failed to load bucket %.0fs\n", __func__, b);
                } else {
                    ctx->npu_preloaded_buckets.insert(b);
                }
            }
            if (ctx->npu_has_30s_reinfer) {
                fprintf(stderr, "%s: eager-loading 30s reinference sessions ...\n", __func__);
                npu_ensure_30s_sessions(ctx);
            }
            fprintf(stderr, "%s: eager-load done (%zu bucket sessions + 30s)\n",
                    __func__, ctx->npu_sessions.size());
        } else {
            fprintf(stderr, "%s: eager preload disabled; buckets/30s sessions will load lazily\n",
                    __func__);
        }

    // ── Standard ONNX mode ───────────────────────────────────────────────────
    } else {
        try {
            ctx->encoder_session = make_session(ctx->env, dir + "/encoder.onnx",        params.use_gpu);
            ctx->prefill_session = make_session(ctx->env, dir + "/decoder_prefill.onnx", params.use_gpu);
            ctx->step_session    = make_session(ctx->env, dir + "/decoder_step.onnx",    params.use_gpu);
        } catch (const Ort::Exception & e) {
            fprintf(stderr, "%s: failed to load ORT sessions: %s\n", __func__, e.what());
            delete ctx;
            return nullptr;
        }
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
    // Phase A debug: dump computed mel to a binary file so it can be diffed
    // against the HF/librosa reference for the same wav. Header: int32 n_mel,
    // int32 n_len, then n_mel*n_len float32 in row-major (mel, frame). Each
    // call writes a separate file with an iteration counter suffix so that
    // the first iter's mel survives later overwrites.
    if (const char * dump_dir = std::getenv("WHISPER_DUMP_MEL")) {
        static int dump_iter = 0;
        char path[1024];
        std::snprintf(path, sizeof(path), "%s.%03d.bin", dump_dir, dump_iter);
        FILE * f = std::fopen(path, "wb");
        if (f) {
            int32_t n_mel = ctx->mel.n_mel;
            int32_t n_len = ctx->mel.n_len;
            std::fwrite(&n_mel, sizeof(int32_t), 1, f);
            std::fwrite(&n_len, sizeof(int32_t), 1, f);
            std::fwrite(ctx->mel.data.data(), sizeof(float),
                        (size_t)n_mel * (size_t)n_len, f);
            std::fclose(f);
            fprintf(stderr, "[DUMP] mel saved: %s (n_mel=%d n_len=%d)\n",
                    path, n_mel, n_len);
        }
        dump_iter++;
    }
    return 0;
}

int whisper_encode(whisper_context * ctx, int /*offset*/, int /*n_threads*/) {
    auto t0 = std::chrono::steady_clock::now();

    const int n_mel    = ctx->mel.n_mel;
    const int n_frames = ctx->mel.n_len;

    // ── NPU N/K mode ─────────────────────────────────────────────────────────
    if (ctx->backend == 1 /*BACKEND_NPU_NK*/) {
        // 1. Select bucket: smallest bucket >= audio duration
        float audio_sec = (float)n_frames / 100.0f;  // 100 mel frames per second
        float bucket;
        if (ctx->npu_force_30s && ctx->npu_has_30s_reinfer) {
            bucket = 30.0f;  // forced 30s for reinference
        } else {
            bucket = select_bucket(ctx->npu_buckets_sec, audio_sec);
            // If audio exceeds the largest runtime bucket (6s), fall back to
            // 30s encoder + 1-step decoder so no audio is truncated.
            if (bucket < audio_sec && ctx->npu_has_30s_reinfer) {
                bucket = 30.0f;
            }
        }
        ctx->npu_current_bucket = bucket;

        // Use the exact per-bucket audio_emb_len from config.json when present.
        // This keeps runtime bucket selection aligned with the compiled encoder
        // and decoder wrappers instead of re-deriving 2s as a special case.
        int raw_emb        = (int)(bucket * 50.0f);
        int audio_emb_len  = ((raw_emb + 7) / 8) * 8;
        if (bucket == 30.0f) {
            // 30s reinference artifacts are compiled at Whisper's native
            // audio context length (1500), not the rounded-up 1504 bucket size.
            audio_emb_len = 1500;
        }
        auto emb_it = ctx->npu_bucket_audio_emb_len.find(bucket);
        if (emb_it != ctx->npu_bucket_audio_emb_len.end()) {
            audio_emb_len = emb_it->second;
        }
        int bucket_mel_len = audio_emb_len * 2;

        // 2. Pad mel to exact bucket size
        std::vector<float> padded_mel(n_mel * bucket_mel_len, 0.0f);
        for (int i = 0; i < n_mel; i++) {
            int copy_len = std::min(n_frames, bucket_mel_len);
            std::memcpy(padded_mel.data() + i * bucket_mel_len,
                       ctx->mel.data.data() + i * n_frames,
                       copy_len * sizeof(float));
        }

        // 3. Run NPU encoder
        printf("[NPU] Encoding: bucket=%.0fs, mel=[1,%d,%d]\n", bucket, n_mel, bucket_mel_len);
        fflush(stdout);

        if (bucket == 30.0f) {
            if (!npu_ensure_30s_sessions(ctx)) {
                fprintf(stderr, "%s: failed to prepare 30s reinference sessions\n", __func__);
                return -1;
            }
        } else {
            if (!npu_ensure_bucket_sessions(ctx, bucket)) {
                fprintf(stderr, "%s: failed to prepare bucket %.0fs sessions\n",
                        __func__, bucket);
                return -1;
            }
        }

        // Convert mel to float16 for NPU (EPContext wrapper expects fp16 I/O)
        std::vector<Ort::Float16_t> padded_mel_fp16(padded_mel.size());
        for (size_t i = 0; i < padded_mel.size(); i++) {
            padded_mel_fp16[i] = Ort::Float16_t(padded_mel[i]);
        }

        std::vector<int64_t> mel_shape = {1, (int64_t)n_mel, (int64_t)bucket_mel_len};
        Ort::Value mel_tensor = Ort::Value::CreateTensor<Ort::Float16_t>(
            ctx->mem_info, padded_mel_fp16.data(), padded_mel_fp16.size(),
            mel_shape.data(), mel_shape.size());

        const char * in_names[] = {"input_features"};

        // Build output names: k_cache_cross_0, v_cache_cross_0, ..., k_cache_cross_5, v_cache_cross_5
        const int nb = ctx->npu_num_blocks;
        std::vector<std::string> out_name_strs;
        for (int i = 0; i < nb; i++) {
            out_name_strs.push_back("k_cache_cross_" + std::to_string(i));
            out_name_strs.push_back("v_cache_cross_" + std::to_string(i));
        }
        std::vector<const char *> out_names;
        for (auto & s : out_name_strs) out_names.push_back(s.c_str());

        // Select encoder session: 30s uses special npu_30s, others use per-bucket
        Ort::Session & enc_sess = (bucket == 30.0f && ctx->npu_30s)
            ? ctx->npu_30s->encoder
            : ctx->npu_sessions.at(bucket).encoder;
        auto t_run0 = std::chrono::steady_clock::now();
        auto outputs = enc_sess.Run(
            Ort::RunOptions{nullptr}, in_names, &mel_tensor, 1,
            out_names.data(), out_names.size());
        auto t_run1 = std::chrono::steady_clock::now();

        printf("[NPU] Encoder completed: %zu outputs\n", outputs.size());
        fflush(stdout);

        // 4. Parse outputs into per-block cross-KV cache
        ctx->npu_cross_kv.k_cache.resize(nb);
        ctx->npu_cross_kv.v_cache.resize(nb);
        ctx->npu_cross_kv.audio_ctx = audio_emb_len;
        ctx->npu_cross_kv.valid_audio_ctx = std::max(1, std::min(audio_emb_len, (n_frames + 1) / 2));

        for (int i = 0; i < nb; i++) {
            // k_cache_cross_{i}: [num_heads, 1, head_dim, audio_emb_len] — fp16
            auto & k_out = outputs[i * 2];
            auto k_info = k_out.GetTensorTypeAndShapeInfo();
            size_t k_bytes = k_info.GetElementCount() * sizeof(Ort::Float16_t);
            ctx->npu_cross_kv.k_cache[i].resize(k_bytes);
            std::memcpy(ctx->npu_cross_kv.k_cache[i].data(),
                       k_out.GetTensorData<Ort::Float16_t>(), k_bytes);

            // v_cache_cross_{i}: [num_heads, 1, audio_emb_len, head_dim] — fp16
            auto & v_out = outputs[i * 2 + 1];
            auto v_info = v_out.GetTensorTypeAndShapeInfo();
            size_t v_bytes = v_info.GetElementCount() * sizeof(Ort::Float16_t);
            ctx->npu_cross_kv.v_cache[i].resize(v_bytes);
            std::memcpy(ctx->npu_cross_kv.v_cache[i].data(),
                       v_out.GetTensorData<Ort::Float16_t>(), v_bytes);
        }

        // 5. Set encoder output info for compatibility
        ctx->enc_n_frames = audio_emb_len;
        ctx->enc_n_state  = ctx->npu_num_heads * ctx->npu_head_dim;  // 512

        // 6. Reset NPU decode state
        ctx->npu_current_chunk_idx  = 0;
        ctx->npu_buffer_read_idx    = 0;
        ctx->npu_active_chain       = 0;
        ctx->npu_chunk_logits.clear();
        ctx->npu_chunk_peak_attn.clear();
        ctx->npu_chunk_cross_attn.clear();
        ctx->npu_self_kv.k_cache.clear();
        ctx->npu_self_kv.v_cache.clear();
        ctx->npu_self_kv.seq_len = 0;

        auto t1 = std::chrono::steady_clock::now();
        const int64_t total_us = elapsed_us(t0, t1);
        const int64_t run_us = elapsed_us(t_run0, t_run1);
        const int64_t parse_us = elapsed_us(t_run1, t1);
        const int64_t prepare_us = std::max<int64_t>(0, total_us - run_us - parse_us);
        ctx->t_encode_us += total_us;
        ctx->npu_profile.encoder_calls++;
        ctx->npu_profile.encoder_total_us += total_us;
        ctx->npu_profile.encoder_prepare_us += prepare_us;
        ctx->npu_profile.encoder_run_us += run_us;
        ctx->npu_profile.encoder_parse_us += parse_us;
        fprintf(stderr,
                "[NPU_RUN] phase=encoder bucket_sec=%.0f raw_mel_frames=%d "
                "encoder_mel_frames=%d valid_audio_ctx=%d total_ms=%.3f "
                "prepare_ms=%.3f run_ms=%.3f parse_ms=%.3f\n",
                bucket, n_frames, bucket_mel_len, ctx->npu_cross_kv.valid_audio_ctx,
                us_to_ms(total_us), us_to_ms(prepare_us), us_to_ms(run_us),
                us_to_ms(parse_us));
        return 0;
    }

    // ── Standard ONNX mode ───────────────────────────────────────────────────
    // Match GGML audio_ctx padding: pad mel so encoder output is multiple of 8 frames.
    const int max_audio_ctx = ctx->dims.n_audio_ctx;
    int target_ctx    = ((int(n_frames / 2.0) + 7) / 8) * 8;
    if (target_ctx < 8) target_ctx = 8;
    if (target_ctx > max_audio_ctx) target_ctx = max_audio_ctx;
    const int padded_frames = target_ctx * 2;

    std::vector<float> padded_mel;
    const float * mel_ptr = ctx->mel.data.data();
    int mel_frames = n_frames;

    if (padded_frames > n_frames) {
        padded_mel.resize(n_mel * padded_frames, 0.0f);
        for (int i = 0; i < n_mel; i++) {
            std::memcpy(padded_mel.data() + i * padded_frames,
                       ctx->mel.data.data() + i * n_frames,
                       n_frames * sizeof(float));
        }
        mel_ptr    = padded_mel.data();
        mel_frames = padded_frames;
    }

    std::vector<int64_t> mel_shape = {1, n_mel, mel_frames};

    Ort::Value mel_tensor = Ort::Value::CreateTensor<float>(
        ctx->mem_info,
        const_cast<float *>(mel_ptr), n_mel * mel_frames,
        mel_shape.data(), mel_shape.size());

    const char * in_names[]  = {"mel"};
    const char * out_names[] = {"encoder_output"};

    auto outputs = ctx->encoder_session.Run(
        Ort::RunOptions{nullptr}, in_names, &mel_tensor, 1, out_names, 1);

    auto & out = outputs[0];
    auto shape = out.GetTensorTypeAndShapeInfo().GetShape();
    ctx->enc_n_frames = (int)shape[1];
    ctx->enc_n_state  = (int)shape[2];

    size_t total = (size_t)shape[1] * shape[2];
    ctx->enc_out.resize(total);
    std::memcpy(ctx->enc_out.data(), out.GetTensorData<float>(), total * sizeof(float));

    auto t1 = std::chrono::steady_clock::now();
    ctx->t_encode_us += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    return 0;
}

// ── Phase A: shadow argmax helper ────────────────────────────────────────────
// Mirrors compile_decoder_unroll_k_xplus.py UnrolledKChunkDecoder's
// internal_argmax_bias: argmax over logits with EOT biased to -inf.
// Produces the token NPU's chunk-internal speculative argmax would have picked
// at the step that emitted these logits. Caller compares to host's actual
// next token to detect proposal-vs-host divergence.
static int npu_proposal_argmax(const std::vector<float> & logits, int eot_id) {
    int best = 0;
    float bestv = -INFINITY;
    const int V = (int)logits.size();
    for (int v = 0; v < V; v++) {
        if (v == eot_id) continue;  // mirror NPU graph's EOT bias
        if (logits[v] > bestv) { bestv = logits[v]; best = v; }
    }
    return best;
}

// ── NPU: run a single N/K decoder chunk, buffer logits + peak/align attn ─────
//
// input_tokens / n_input_tokens:
//   - chunk_0 normal chain  (active_chain=0): 4 tokens (4 SOT positions)
//   - chunk_0 prefill chain (active_chain=1): 4 + prompt_len tokens
//   - chunks ≥ 1: always 1 token (autoregressive single-token feed)
//
// ctx->npu_active_chain (0=normal, 1=prefill) selects which chunk_0/chunk_1
// session pair to use; chunks 2..num_chunks-1 are the same session for both
// chains (graphs converge at chunk_2 with cache_in=11 in both K tables).
static void npu_run_chunk(whisper_context * ctx, int chunk_idx,
                          const int32_t * input_tokens, int n_input_tokens) {
    auto t_total0 = std::chrono::steady_clock::now();
    const int nb = ctx->npu_num_blocks;
    const int nh = ctx->npu_num_heads;
    const int hd = ctx->npu_head_dim;
    const int audio_ctx = ctx->npu_cross_kv.audio_ctx;
    const int active_chain = ctx->npu_active_chain;  // 0=normal, 1=prefill

    // K_internal for this chunk under the active chain.
    if (chunk_idx < 0 || chunk_idx >= ctx->npu_num_chunks) {
        fprintf(stderr, "[NPU] FATAL: chunk_idx=%d out of range (num_chunks=%zu)\n",
                chunk_idx, ctx->npu_K_normal.size());
        fflush(stderr);
        return;
    }
    int K = npu_get_chunk_k(ctx, chunk_idx);

    // Sanity: chunk_0 takes K_table[0] tokens; chunks ≥ 1 take exactly 1.
    int expected_ids_len = (chunk_idx == 0) ? K : 1;
    if (n_input_tokens != expected_ids_len) {
        fprintf(stderr,
                "[NPU] FATAL: chunk_%d (active_chain=%d) expects n_input_tokens=%d but got %d\n",
                chunk_idx, active_chain, expected_ids_len, n_input_tokens);
        fflush(stderr);
        ctx->npu_chunk_logits.assign(K, std::vector<float>(ctx->dims.n_vocab, -1e9f));
        ctx->npu_chunk_peak_attn.assign(K, std::vector<float>(audio_ctx, 0.0f));
        ctx->npu_chunk_cross_attn.assign(K, std::vector<float>(audio_ctx, 0.0f));
        ctx->npu_buffer_read_idx = 0;
        return;
    }

    printf("[NPU] npu_run_chunk: chunk=%d, active_chain=%s, n_in=%d, token0=%d, K=%d, seq_in=%d, audio_ctx=%d\n",
           chunk_idx, active_chain == 0 ? "normal" : "prefill",
           n_input_tokens, input_tokens[0], K,
           ctx->npu_self_kv.seq_len, audio_ctx);
    fflush(stdout);

    // Validate every input token: invalid (-1) or out-of-range tokens are passed
    // straight to nn.Embedding inside the NPU graph and have caused QNN HTP to
    // hang on Run(). Common cause: vocab.txt missing special token strings →
    // find_token returns -1 → whisper_token_sot()/_lang()/_not() return -1.
    for (int t = 0; t < n_input_tokens; t++) {
        if (input_tokens[t] < 0 || input_tokens[t] >= ctx->dims.n_vocab) {
            fprintf(stderr,
                    "[NPU] FATAL: invalid input_tokens[%d]=%d (n_vocab=%d). "
                    "Likely cause: deploy vocab.txt is missing special tokens. "
                    "Aborting decoder run to avoid QNN hang.\n",
                    t, input_tokens[t], ctx->dims.n_vocab);
            fflush(stderr);
            ctx->npu_chunk_logits.assign(K, std::vector<float>(ctx->dims.n_vocab, -1e9f));
            int eot = whisper_token_eot(ctx);
            if (eot >= 0 && eot < ctx->dims.n_vocab) {
                for (int k = 0; k < K; k++) ctx->npu_chunk_logits[k][eot] = 0.0f;
            }
            ctx->npu_chunk_peak_attn.assign(K, std::vector<float>(audio_ctx, 0.0f));
            ctx->npu_chunk_cross_attn.assign(K, std::vector<float>(audio_ctx, 0.0f));
            ctx->npu_buffer_read_idx = 0;
            return;
        }
    }

    // Select session: chunk_0/1 from the active chain, chunks 2..N-1 shared.
    auto & bucket_sess = ctx->npu_sessions.at(ctx->npu_current_bucket);
    Ort::Session * session_ptr = nullptr;
    if (chunk_idx == 0) {
        if (active_chain == 0) {
            if (!bucket_sess.has_normal) {
                fprintf(stderr, "[NPU] FATAL: normal chain not loaded for bucket %.0fs\n",
                        ctx->npu_current_bucket);
                fflush(stderr);
                ctx->npu_chunk_logits.assign(K, std::vector<float>(ctx->dims.n_vocab, -1e9f));
                ctx->npu_chunk_peak_attn.assign(K, std::vector<float>(audio_ctx, 0.0f));
                ctx->npu_chunk_cross_attn.assign(K, std::vector<float>(audio_ctx, 0.0f));
                ctx->npu_buffer_read_idx = 0;
                return;
            }
            session_ptr = &bucket_sess.chunk_0_normal;
        } else {
            auto it = bucket_sess.chunk_0_prefill_variants.find(ctx->npu_active_prefill_prompt_tokens);
            if (it == bucket_sess.chunk_0_prefill_variants.end()) {
                fprintf(stderr, "[NPU] FATAL: chunk_0 prefill variant p%d not loaded for bucket %.0fs\n",
                        ctx->npu_active_prefill_prompt_tokens, ctx->npu_current_bucket);
                fflush(stderr);
                ctx->npu_chunk_logits.assign(K, std::vector<float>(ctx->dims.n_vocab, -1e9f));
                ctx->npu_chunk_peak_attn.assign(K, std::vector<float>(audio_ctx, 0.0f));
                ctx->npu_chunk_cross_attn.assign(K, std::vector<float>(audio_ctx, 0.0f));
                ctx->npu_buffer_read_idx = 0;
                return;
            }
            session_ptr = &it->second;
        }
    } else if (chunk_idx == 1) {
        if (active_chain == 0) {
            if (!bucket_sess.has_normal) {
                fprintf(stderr, "[NPU] FATAL: normal chain chunk_1 not loaded for bucket %.0fs\n",
                        ctx->npu_current_bucket);
                fflush(stderr);
                ctx->npu_chunk_logits.assign(K, std::vector<float>(ctx->dims.n_vocab, -1e9f));
                ctx->npu_chunk_peak_attn.assign(K, std::vector<float>(audio_ctx, 0.0f));
                ctx->npu_chunk_cross_attn.assign(K, std::vector<float>(audio_ctx, 0.0f));
                ctx->npu_buffer_read_idx = 0;
                return;
            }
            session_ptr = &bucket_sess.chunk_1_normal;
        } else {
            auto it = bucket_sess.chunk_1_prefill_variants.find(ctx->npu_active_prefill_prompt_tokens);
            if (it == bucket_sess.chunk_1_prefill_variants.end()) {
                fprintf(stderr, "[NPU] FATAL: chunk_1 prefill variant p%d not loaded for bucket %.0fs\n",
                        ctx->npu_active_prefill_prompt_tokens, ctx->npu_current_bucket);
                fflush(stderr);
                ctx->npu_chunk_logits.assign(K, std::vector<float>(ctx->dims.n_vocab, -1e9f));
                ctx->npu_chunk_peak_attn.assign(K, std::vector<float>(audio_ctx, 0.0f));
                ctx->npu_chunk_cross_attn.assign(K, std::vector<float>(audio_ctx, 0.0f));
                ctx->npu_buffer_read_idx = 0;
                return;
            }
            session_ptr = &it->second;
        }
    } else {
        // chunks 2..num_chunks-1 are shared between both chains.
        int shared_idx = chunk_idx - 2;
        if (shared_idx < 0 || shared_idx >= (int)bucket_sess.chunks_shared.size()) {
            fprintf(stderr, "[NPU] FATAL: shared chunk index %d out of range (size=%zu)\n",
                    shared_idx, bucket_sess.chunks_shared.size());
            fflush(stderr);
            ctx->npu_chunk_logits.assign(K, std::vector<float>(ctx->dims.n_vocab, -1e9f));
            ctx->npu_chunk_peak_attn.assign(K, std::vector<float>(audio_ctx, 0.0f));
            ctx->npu_chunk_cross_attn.assign(K, std::vector<float>(audio_ctx, 0.0f));
            ctx->npu_buffer_read_idx = 0;
            return;
        }
        session_ptr = &bucket_sess.chunks_shared[shared_idx];
    }
    auto & session = *session_ptr;

    // Build input feeds
    std::vector<Ort::Value> inputs;
    std::vector<std::string> in_name_strs;

    // input_ids: [1, n_input_tokens] — caller bytes copied into a local buffer
    // so the lifetime is the function scope (Ort::Value does not own data).
    std::vector<int32_t> ids_buf(input_tokens, input_tokens + n_input_tokens);
    std::vector<int64_t> ids_shape = {1, (int64_t)n_input_tokens};
    inputs.push_back(Ort::Value::CreateTensor<int32_t>(
        ctx->mem_info, ids_buf.data(), ids_buf.size(),
        ids_shape.data(), ids_shape.size()));
    in_name_strs.push_back("input_ids");

    // cross_attention_mask: [1, 1, 1, audio_ctx]
    const Ort::Float16_t CROSS_MASK_NEG = Ort::Float16_t(-65504.0f);
    std::vector<Ort::Float16_t> cross_mask(audio_ctx, Ort::Float16_t(0.0f));
    const int valid_audio_ctx = std::max(1, std::min(audio_ctx, ctx->npu_cross_kv.valid_audio_ctx));
    for (int i = valid_audio_ctx; i < audio_ctx; ++i) cross_mask[i] = CROSS_MASK_NEG;
    std::vector<int64_t> cross_mask_shape = {1, 1, 1, (int64_t)audio_ctx};
    inputs.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(
        ctx->mem_info, cross_mask.data(), cross_mask.size(),
        cross_mask_shape.data(), cross_mask_shape.size()));
    in_name_strs.push_back("cross_attention_mask");

    // Self-KV cache inputs (per-block) — fp16 raw bytes
    int seq_in = ctx->npu_self_kv.seq_len;
    for (int i = 0; i < nb; i++) {
        std::vector<int64_t> k_shape = {nh, 1, hd, seq_in};
        size_t k_elems = nh * 1 * hd * seq_in;
        inputs.push_back(Ort::Value::CreateTensor(
            ctx->mem_info, ctx->npu_self_kv.k_cache[i].data(),
            ctx->npu_self_kv.k_cache[i].size(),
            k_shape.data(), k_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16));
        in_name_strs.push_back("k_cache_self_" + std::to_string(i) + "_in");

        std::vector<int64_t> v_shape = {nh, 1, seq_in, hd};
        inputs.push_back(Ort::Value::CreateTensor(
            ctx->mem_info, ctx->npu_self_kv.v_cache[i].data(),
            ctx->npu_self_kv.v_cache[i].size(),
            v_shape.data(), v_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16));
        in_name_strs.push_back("v_cache_self_" + std::to_string(i) + "_in");
    }

    // Cross-KV cache inputs (per-block, constant from encoder) — fp16 raw bytes
    for (int i = 0; i < nb; i++) {
        std::vector<int64_t> kc_shape = {nh, 1, hd, audio_ctx};
        inputs.push_back(Ort::Value::CreateTensor(
            ctx->mem_info, ctx->npu_cross_kv.k_cache[i].data(),
            ctx->npu_cross_kv.k_cache[i].size(),
            kc_shape.data(), kc_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16));
        in_name_strs.push_back("k_cache_cross_" + std::to_string(i));

        std::vector<int64_t> vc_shape = {nh, 1, audio_ctx, hd};
        inputs.push_back(Ort::Value::CreateTensor(
            ctx->mem_info, ctx->npu_cross_kv.v_cache[i].data(),
            ctx->npu_cross_kv.v_cache[i].size(),
            vc_shape.data(), vc_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16));
        in_name_strs.push_back("v_cache_cross_" + std::to_string(i));
    }

    std::vector<const char *> in_names;
    for (auto & s : in_name_strs) in_names.push_back(s.c_str());

    // Build output names: K logits + nb*2 self-KV out + K peak_attn + K cross_attn
    std::vector<std::string> out_name_strs;
    for (int k = 0; k < K; k++)
        out_name_strs.push_back("logits_" + std::to_string(k));
    for (int i = 0; i < nb; i++) {
        out_name_strs.push_back("k_cache_self_" + std::to_string(i) + "_out");
        out_name_strs.push_back("v_cache_self_" + std::to_string(i) + "_out");
    }
    for (int k = 0; k < K; k++)
        out_name_strs.push_back("peak_attn_" + std::to_string(k));
    for (int k = 0; k < K; k++)
        out_name_strs.push_back("cross_attn_" + std::to_string(k));

    std::vector<const char *> out_names;
    for (auto & s : out_name_strs) out_names.push_back(s.c_str());

    // Run NPU decoder chunk
    printf("[NPU] Running decoder chunk %d: %zu inputs, %zu outputs\n",
           chunk_idx, inputs.size(), out_names.size());
    fflush(stdout);

    auto t_run0 = std::chrono::steady_clock::now();
    auto outputs = session.Run(
        Ort::RunOptions{nullptr},
        in_names.data(), inputs.data(), inputs.size(),
        out_names.data(), out_names.size());
    auto t_run1 = std::chrono::steady_clock::now();

    // Parse K logits: outputs[0..K-1], each [1, vocab_size, 1, 1] — fp16 → fp32
    const int n_vocab = ctx->dims.n_vocab;
    ctx->npu_chunk_logits.resize(K);
    for (int k = 0; k < K; k++) {
        const Ort::Float16_t * fp16_data = outputs[k].GetTensorData<Ort::Float16_t>();
        ctx->npu_chunk_logits[k].resize(n_vocab);
        for (int v = 0; v < n_vocab; v++) {
            ctx->npu_chunk_logits[k][v] = static_cast<float>(fp16_data[v]);
        }
    }

    // Parse updated self-KV: outputs[K..K+nb*2-1] — fp16 raw bytes
    int seq_out = seq_in + K;
    ctx->npu_self_kv.seq_len = seq_out;
    ctx->npu_self_kv.k_cache.resize(nb);
    ctx->npu_self_kv.v_cache.resize(nb);
    int kv_start = K;
    for (int i = 0; i < nb; i++) {
        auto & k_out = outputs[kv_start + i * 2];
        auto k_info = k_out.GetTensorTypeAndShapeInfo();
        size_t k_bytes = k_info.GetElementCount() * sizeof(Ort::Float16_t);
        ctx->npu_self_kv.k_cache[i].resize(k_bytes);
        std::memcpy(ctx->npu_self_kv.k_cache[i].data(), k_out.GetTensorData<Ort::Float16_t>(), k_bytes);

        auto & v_out = outputs[kv_start + i * 2 + 1];
        auto v_info = v_out.GetTensorTypeAndShapeInfo();
        size_t v_bytes = v_info.GetElementCount() * sizeof(Ort::Float16_t);
        ctx->npu_self_kv.v_cache[i].resize(v_bytes);
        std::memcpy(ctx->npu_self_kv.v_cache[i].data(), v_out.GetTensorData<Ort::Float16_t>(), v_bytes);
    }

    // Parse K peak_attn: fp16 → fp32
    int peak_start = K + nb * 2;
    ctx->npu_chunk_peak_attn.resize(K);
    for (int k = 0; k < K; k++) {
        auto & peak_out = outputs[peak_start + k];
        size_t peak_total = peak_out.GetTensorTypeAndShapeInfo().GetElementCount();
        const Ort::Float16_t * fp16_data = peak_out.GetTensorData<Ort::Float16_t>();
        ctx->npu_chunk_peak_attn[k].resize(peak_total);
        for (size_t j = 0; j < peak_total; j++) {
            ctx->npu_chunk_peak_attn[k][j] = static_cast<float>(fp16_data[j]);
        }
    }

    // Parse K cross_attn (alignment heads): fp16 → fp32
    int ca_start = peak_start + K;
    ctx->npu_chunk_cross_attn.resize(K);
    for (int k = 0; k < K; k++) {
        auto & ca_out = outputs[ca_start + k];
        size_t ca_total = ca_out.GetTensorTypeAndShapeInfo().GetElementCount();
        const Ort::Float16_t * fp16_data = ca_out.GetTensorData<Ort::Float16_t>();
        ctx->npu_chunk_cross_attn[k].resize(ca_total);
        for (size_t j = 0; j < ca_total; j++) {
            ctx->npu_chunk_cross_attn[k][j] = static_cast<float>(fp16_data[j]);
        }
    }

    ctx->npu_buffer_read_idx = 0;
    auto t_total1 = std::chrono::steady_clock::now();
    const int64_t total_us = elapsed_us(t_total0, t_total1);
    const int64_t run_us = elapsed_us(t_run0, t_run1);
    const int64_t parse_us = elapsed_us(t_run1, t_total1);
    const int64_t prepare_us = std::max<int64_t>(0, total_us - run_us - parse_us);
    ctx->npu_profile.chunk_calls++;
    ctx->npu_profile.chunk_total_us += total_us;
    ctx->npu_profile.chunk_prepare_us += prepare_us;
    ctx->npu_profile.chunk_run_us += run_us;
    ctx->npu_profile.chunk_parse_us += parse_us;
    fprintf(stderr,
            "[NPU_RUN] phase=decoder_chunk decode_phase=%s bucket_sec=%.0f chunk=%d chain=%s "
            "K=%d seq_in=%d audio_ctx=%d total_ms=%.3f prepare_ms=%.3f "
            "run_ms=%.3f parse_ms=%.3f\n",
            npu_decode_phase_name(ctx->npu_decode_phase),
            ctx->npu_current_bucket, chunk_idx,
            active_chain == 0 ? "normal" : "prefill",
            K, seq_in, audio_ctx,
            us_to_ms(total_us), us_to_ms(prepare_us), us_to_ms(run_us),
            us_to_ms(parse_us));
}

// ── NPU: populate step_cross_attn_w from buffered NPU cross-attn ─────────────
static void npu_store_cross_attn(whisper_context * ctx, int buf_idx) {
    const int audio_ctx = ctx->npu_cross_kv.audio_ctx;
    const auto & peak_raw = ctx->npu_chunk_peak_attn[buf_idx];
    const auto & raw = ctx->npu_chunk_cross_attn[buf_idx];
    const int n_ah = ctx->npu_n_alignment_heads;

    fprintf(stderr,
            "[NPU_ATTN_STORE] buf_idx=%d audio_ctx=%d peak_raw=%zu raw=%zu n_ah=%d\n",
            buf_idx, audio_ctx, peak_raw.size(), raw.size(), n_ah);

    if ((int)peak_raw.size() == audio_ctx) {
        // Backward peak detection uses a dedicated last-layer all-head average.
        ctx->cross_attn_data = peak_raw;
        ctx->cross_attn_n_heads = 1;
        ctx->cross_attn_n_audio_ctx = audio_ctx;
        ctx->step_cross_attn_w = ctx->cross_attn_data;
    } else if (n_ah > 0 && (int)raw.size() == n_ah * audio_ctx) {
        // Backward-compatible fallback for old decoders with only alignment heads.
        ctx->cross_attn_data.resize(audio_ctx);
        for (int a = 0; a < audio_ctx; a++) {
            float sum = 0.0f;
            for (int h = 0; h < n_ah; h++) sum += raw[h * audio_ctx + a];
            ctx->cross_attn_data[a] = sum / n_ah;
        }
        ctx->cross_attn_n_heads = 1;
        ctx->cross_attn_n_audio_ctx = audio_ctx;
        ctx->step_cross_attn_w = ctx->cross_attn_data;
    } else {
        ctx->step_cross_attn_w = raw;
        ctx->cross_attn_n_heads = 1;
        ctx->cross_attn_n_audio_ctx = audio_ctx;
        ctx->cross_attn_data = raw;
    }

    if (n_ah > 0 && (int)raw.size() == n_ah * audio_ctx) {
        // aheads_attn_data for DTW: per-head data
        ctx->aheads_attn_data = raw;  // [n_aheads * audio_ctx]
        ctx->aheads_n_heads = n_ah;
        ctx->aheads_n_audio_ctx = audio_ctx;
    } else {
        ctx->aheads_attn_data.clear();
        ctx->aheads_n_heads = 0;
        ctx->aheads_n_audio_ctx = 0;
    }

    fprintf(stderr,
            "[NPU_ATTN_STORE] result cross_attn_n_audio_ctx=%d aheads_n_audio_ctx=%d cross_heads=%d aheads_heads=%d\n",
            ctx->cross_attn_n_audio_ctx,
            ctx->aheads_n_audio_ctx,
            ctx->cross_attn_n_heads,
            ctx->aheads_n_heads);
}

// ── NPU 1-step decoder helper ────────────────────────────────────────────────
static void npu_run_1step_session(
    whisper_context * ctx,
    Ort::Session & sess,
    int cache_size,
    int32_t input_token,
    int position) {
    auto t_total0 = std::chrono::steady_clock::now();
    const int nb = ctx->npu_num_blocks;
    const int nh = ctx->npu_num_heads;
    const int hd = ctx->npu_head_dim;
    const int audio_ctx = ctx->npu_cross_kv.audio_ctx;
    const int n_vocab = ctx->dims.n_vocab;
    const float MASK_NEG = -65504.0f;  // fp16 min

    std::vector<Ort::Value> inputs;
    std::vector<std::string> in_strs;

    // input_ids: [1, 1]
    int32_t ids_val = input_token;
    std::vector<int64_t> ids_shape = {1, 1};
    inputs.push_back(Ort::Value::CreateTensor<int32_t>(
        ctx->mem_info, &ids_val, 1, ids_shape.data(), ids_shape.size()));
    in_strs.push_back("input_ids");

    // attention_mask: [1, 1, 1, cache_size+1] — sliding window
    int mask_len = cache_size + 1;
    std::vector<Ort::Float16_t> mask(mask_len, Ort::Float16_t(MASK_NEG));
    int valid = std::min(position + 1, mask_len);
    for (int i = mask_len - valid; i < mask_len; i++) mask[i] = Ort::Float16_t(0.0f);
    std::vector<int64_t> mask_shape = {1, 1, 1, (int64_t)mask_len};
    inputs.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(
        ctx->mem_info, mask.data(), mask.size(), mask_shape.data(), mask_shape.size()));
    in_strs.push_back("attention_mask");

    // cross_attention_mask: [1, 1, 1, audio_ctx]
    std::vector<Ort::Float16_t> cross_mask(audio_ctx, Ort::Float16_t(0.0f));
    const int valid_audio_ctx = std::max(1, std::min(audio_ctx, ctx->npu_cross_kv.valid_audio_ctx));
    for (int i = valid_audio_ctx; i < audio_ctx; ++i) cross_mask[i] = Ort::Float16_t(MASK_NEG);
    std::vector<int64_t> cross_mask_shape = {1, 1, 1, (int64_t)audio_ctx};
    inputs.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(
        ctx->mem_info, cross_mask.data(), cross_mask.size(), cross_mask_shape.data(), cross_mask_shape.size()));
    in_strs.push_back("cross_attention_mask");

    // Self-KV cache (fixed size = cache_size) — fp16 raw bytes
    for (int i = 0; i < nb; i++) {
        std::vector<int64_t> k_shape = {nh, 1, hd, cache_size};
        inputs.push_back(Ort::Value::CreateTensor(
            ctx->mem_info, ctx->npu_self_kv.k_cache[i].data(),
            ctx->npu_self_kv.k_cache[i].size(),
            k_shape.data(), k_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16));
        in_strs.push_back("k_cache_self_" + std::to_string(i) + "_in");

        std::vector<int64_t> v_shape = {nh, 1, cache_size, hd};
        inputs.push_back(Ort::Value::CreateTensor(
            ctx->mem_info, ctx->npu_self_kv.v_cache[i].data(),
            ctx->npu_self_kv.v_cache[i].size(),
            v_shape.data(), v_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16));
        in_strs.push_back("v_cache_self_" + std::to_string(i) + "_in");
    }

    // Cross-KV cache — fp16 raw bytes
    for (int i = 0; i < nb; i++) {
        std::vector<int64_t> kc_shape = {nh, 1, hd, audio_ctx};
        inputs.push_back(Ort::Value::CreateTensor(
            ctx->mem_info, ctx->npu_cross_kv.k_cache[i].data(),
            ctx->npu_cross_kv.k_cache[i].size(),
            kc_shape.data(), kc_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16));
        in_strs.push_back("k_cache_cross_" + std::to_string(i));

        std::vector<int64_t> vc_shape = {nh, 1, audio_ctx, hd};
        inputs.push_back(Ort::Value::CreateTensor(
            ctx->mem_info, ctx->npu_cross_kv.v_cache[i].data(),
            ctx->npu_cross_kv.v_cache[i].size(),
            vc_shape.data(), vc_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16));
        in_strs.push_back("v_cache_cross_" + std::to_string(i));
    }

    // position_ids: [1]
    int32_t pos_val = (int32_t)position;
    std::vector<int64_t> pos_shape = {1};
    inputs.push_back(Ort::Value::CreateTensor<int32_t>(
        ctx->mem_info, &pos_val, 1, pos_shape.data(), pos_shape.size()));
    in_strs.push_back("position_ids");

    std::vector<const char *> in_names;
    for (auto & s : in_strs) in_names.push_back(s.c_str());

    // Output: logits + nb*2 self-KV + peak_attn + cross_attn
    std::vector<std::string> out_strs;
    out_strs.push_back("logits");
    for (int i = 0; i < nb; i++) {
        out_strs.push_back("k_cache_self_" + std::to_string(i) + "_out");
        out_strs.push_back("v_cache_self_" + std::to_string(i) + "_out");
    }
    out_strs.push_back("peak_attn");
    out_strs.push_back("cross_attn");
    std::vector<const char *> out_names;
    for (auto & s : out_strs) out_names.push_back(s.c_str());

    auto t_run0 = std::chrono::steady_clock::now();
    auto outputs = sess.Run(
        Ort::RunOptions{nullptr},
        in_names.data(), inputs.data(), inputs.size(),
        out_names.data(), out_names.size());
    auto t_run1 = std::chrono::steady_clock::now();

    // Parse logits: [1, vocab_size, 1, 1] — fp16
    {
        const auto * data = outputs[0].GetTensorData<Ort::Float16_t>();
        ctx->logits.resize(n_vocab);
        for (int i = 0; i < n_vocab; ++i) {
            ctx->logits[i] = static_cast<float>(data[i]);
        }
    }

    // Parse updated self-KV (same fixed size, raw fp16 bytes)
    for (int i = 0; i < nb; i++) {
        auto & k_out = outputs[1 + i * 2];
        size_t k_total = k_out.GetTensorTypeAndShapeInfo().GetElementCount();
        size_t k_bytes = k_total * sizeof(Ort::Float16_t);
        ctx->npu_self_kv.k_cache[i].resize(k_bytes);
        std::memcpy(ctx->npu_self_kv.k_cache[i].data(), k_out.GetTensorData<Ort::Float16_t>(), k_bytes);

        auto & v_out = outputs[1 + i * 2 + 1];
        size_t v_total = v_out.GetTensorTypeAndShapeInfo().GetElementCount();
        size_t v_bytes = v_total * sizeof(Ort::Float16_t);
        ctx->npu_self_kv.v_cache[i].resize(v_bytes);
        std::memcpy(ctx->npu_self_kv.v_cache[i].data(), v_out.GetTensorData<Ort::Float16_t>(), v_bytes);
    }

    // peak_attn: [1, 1, 1, audio_ctx]
    {
        const auto * data = outputs[1 + nb * 2].GetTensorData<Ort::Float16_t>();
        ctx->step_cross_attn_w.resize(audio_ctx);
        ctx->cross_attn_data.resize(audio_ctx);
        for (int i = 0; i < audio_ctx; ++i) {
            float v = static_cast<float>(data[i]);
            ctx->step_cross_attn_w[i] = v;
            ctx->cross_attn_data[i] = v;
        }
        ctx->cross_attn_n_heads = 1;
        ctx->cross_attn_n_audio_ctx = audio_ctx;
    }

    // cross_attn: [1, n_alignment_heads, 1, audio_ctx]
    {
        const auto * data = outputs[2 + nb * 2].GetTensorData<Ort::Float16_t>();
        const int n_ah = std::max(1, ctx->npu_n_alignment_heads);
        ctx->aheads_n_heads = n_ah;
        ctx->aheads_n_audio_ctx = audio_ctx;
        ctx->aheads_attn_data.resize((size_t)n_ah * audio_ctx);
        for (size_t i = 0; i < ctx->aheads_attn_data.size(); ++i) {
            ctx->aheads_attn_data[i] = static_cast<float>(data[i]);
        }
    }

    auto t_total1 = std::chrono::steady_clock::now();
    const int64_t total_us = elapsed_us(t_total0, t_total1);
    const int64_t run_us = elapsed_us(t_run0, t_run1);
    const int64_t parse_us = elapsed_us(t_run1, t_total1);
    const int64_t prepare_us = std::max<int64_t>(0, total_us - run_us - parse_us);
    ctx->npu_profile.one_step_calls++;
    ctx->npu_profile.one_step_total_us += total_us;
    ctx->npu_profile.one_step_prepare_us += prepare_us;
    ctx->npu_profile.one_step_run_us += run_us;
    ctx->npu_profile.one_step_parse_us += parse_us;
    fprintf(stderr,
            "[NPU_RUN] phase=decoder_1step decode_phase=%s bucket_sec=%.0f position=%d "
            "cache_size=%d audio_ctx=%d total_ms=%.3f prepare_ms=%.3f "
            "run_ms=%.3f parse_ms=%.3f\n",
            npu_decode_phase_name(ctx->npu_decode_phase),
            ctx->npu_current_bucket, position, cache_size, audio_ctx,
            us_to_ms(total_us), us_to_ms(prepare_us), us_to_ms(run_us),
            us_to_ms(parse_us));
}

// ── NPU 30s: run 1-step decoder for reinference ─────────────────────────────
static void npu_run_1step_30s(whisper_context * ctx, int32_t input_token, int position) {
    if (!ctx->npu_30s) return;
    npu_run_1step_session(
        ctx, ctx->npu_30s->decoder_1step, ctx->npu_30s->kv_cache_size,
        input_token, position);
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

    // ── NPU N/K mode ─────────────────────────────────────────────────────────
    if (ctx->backend == 1 /*BACKEND_NPU_NK*/) {

        // ── 1-step decoder path: GGML-equivalent host-driven greedy ───────
        const bool use_30s_decoder = ctx->npu_force_30s ||
                                     ctx->npu_current_bucket == 30.0f;
        if (use_30s_decoder || ctx->npu_use_1step) {
            ctx->npu_decode_phase = (n_past == 0) ? 1 : 2;
            const int nb = ctx->npu_num_blocks;
            const int nh = ctx->npu_num_heads;
            const int hd = ctx->npu_head_dim;
            int cache_size = ctx->npu_decoder_kv_cache_size;
            Ort::Session * decoder_sess = nullptr;

            if (use_30s_decoder) {
                if (!ctx->npu_30s) {
                    fprintf(stderr, "[NPU] FATAL: 30s reinference requested but 30s decoder is unavailable\n");
                    return -1;
                }
                decoder_sess = &ctx->npu_30s->decoder_1step;
                cache_size = ctx->npu_30s->kv_cache_size;
            } else {
                auto & bucket_sess = ctx->npu_sessions.at(ctx->npu_current_bucket);
                if (!bucket_sess.has_1step) {
                    fprintf(stderr, "[NPU] FATAL: 1-step decoder not loaded for bucket %.0fs\n",
                            ctx->npu_current_bucket);
                    return -1;
                }
                decoder_sess = &bucket_sess.decoder_1step;
            }

            if (n_past == 0) {
                // Init self-KV zeros (fixed cache_size)
                ctx->npu_self_kv.seq_len = 0;
                ctx->npu_self_kv.k_cache.resize(nb);
                ctx->npu_self_kv.v_cache.resize(nb);
                for (int i = 0; i < nb; i++) {
                    ctx->npu_self_kv.k_cache[i].assign(
                        nh * 1 * hd * cache_size * (int)sizeof(Ort::Float16_t), 0);
                    ctx->npu_self_kv.v_cache[i].assign(
                        nh * 1 * cache_size * hd * (int)sizeof(Ort::Float16_t), 0);
                }
                // Run each SOT token through 1-step decoder
                for (int t = 0; t < n_tokens; t++) {
                    npu_run_1step_session(ctx, *decoder_sess, cache_size, (int32_t)tokens[t], t);
                    ctx->npu_self_kv.seq_len = t + 1;
                }
                // logits from last token are already in ctx->logits
            } else {
                npu_run_1step_session(ctx, *decoder_sess, cache_size, (int32_t)tokens[0], n_past);
                ctx->npu_self_kv.seq_len = n_past + 1;
            }

            auto t1 = std::chrono::steady_clock::now();
            const int64_t decode_us = elapsed_us(t0, t1);
            ctx->t_decode_us += decode_us;
            if (n_past == 0) {
                ctx->npu_profile.prefill_calls++;
                ctx->npu_profile.prefill_total_us += decode_us;
            } else {
                ctx->npu_profile.generation_calls++;
                ctx->npu_profile.generation_total_us += decode_us;
            }
            return 0;
        }

        // ── Normal N/K chunked decode ────────────────────────────────────
        if (n_past == 0) {
            ctx->npu_decode_phase = 1;
            printf("[NPU] decode: n_past=0, n_tokens=%d (prefill)\n", n_tokens);
            fflush(stdout);

            // Strict chain selection by n_tokens:
            //   n_tokens == K_normal[0]           → normal chain  (4 SOT tokens, no carryover)
            //   n_tokens == 4 + prompt_prefill    → matching prefill variant
            // Anything else is a programming error in the caller.
            int K_norm_0 = ctx->npu_K_normal[0];
            int chosen_chain = -1;
            int prompt_prefill_tokens = std::max(0, n_tokens - K_norm_0);
            if (n_tokens == K_norm_0) {
                chosen_chain = 0;
                prompt_prefill_tokens = 0;
            } else if (n_tokens > K_norm_0 &&
                       npu_supports_prefill_prompt_len(ctx, prompt_prefill_tokens)) {
                chosen_chain = 1;
            }
            if (chosen_chain < 0) {
                fprintf(stderr,
                        "[NPU] FATAL: unsupported prefill n_tokens=%d. "
                        "normal K_chunk_0=%d, supported prefill prompt lens=[",
                        n_tokens, K_norm_0);
                for (size_t i = 0; i < ctx->npu_prefill_prompt_token_lens.size(); ++i) {
                    fprintf(stderr, "%s%d", i ? "," : "", ctx->npu_prefill_prompt_token_lens[i]);
                }
                fprintf(stderr, "]\n");
                fflush(stderr);
                ctx->logits.assign((size_t)n_tokens * n_vocab, -1e9f);
                return -1;
            }
            // Empty-prompt starts must use the normal chain. Falling back to a
            // synthetic prefill chain changes mode3 semantics relative to ggml.
            auto & bucket_sess = ctx->npu_sessions.at(ctx->npu_current_bucket);
            if (chosen_chain == 0 && !bucket_sess.has_normal) {
                fprintf(stderr,
                        "[NPU] FATAL: normal chain requested at bucket %.0fs, but "
                        "no normal wrapper is available.\n",
                        ctx->npu_current_bucket);
                fflush(stderr);
                ctx->logits.assign((size_t)n_tokens * n_vocab, -1e9f);
                return -1;
            }
            ctx->npu_active_chain = chosen_chain;
            ctx->npu_active_prefill_prompt_tokens = prompt_prefill_tokens;

            int K0 = npu_get_chunk_k(ctx, 0);
            printf("[NPU] decode: chose chain=%s, prompt_prefill_tokens=%d, K_chunk_0=%d, n_tokens=%d, bucket=%.0fs\n",
                   chosen_chain == 0 ? "normal" : "prefill",
                   prompt_prefill_tokens,
                   K0, n_tokens, ctx->npu_current_bucket);
            fflush(stdout);

            const int nb = ctx->npu_num_blocks;
            const int nh = ctx->npu_num_heads;
            const int hd = ctx->npu_head_dim;
            ctx->npu_self_kv.seq_len = 1;
            ctx->npu_self_kv.k_cache.resize(nb);
            ctx->npu_self_kv.v_cache.resize(nb);
            for (int i = 0; i < nb; i++) {
                ctx->npu_self_kv.k_cache[i].assign(nh * 1 * hd * 1 * sizeof(Ort::Float16_t), 0);
                ctx->npu_self_kv.v_cache[i].assign(nh * 1 * 1 * hd * sizeof(Ort::Float16_t), 0);
            }

            // chunk_0 takes all n_tokens caller-provided tokens.
            std::vector<int32_t> chunk0_ids;
            chunk0_ids.reserve(n_tokens);
            for (int i = 0; i < n_tokens; i++) chunk0_ids.push_back((int32_t)tokens[i]);
            npu_run_chunk(ctx, 0, chunk0_ids.data(), (int)chunk0_ids.size());

            // Caller (ours_streaming.cpp) reads prefill logits as
            //   logits = whisper_get_logits(ctx) + (n_tokens - 1) * n_vocab;
            // so ctx->logits must hold n_tokens * n_vocab floats and entry
            // [n_tokens-1] must be NPU chunk_0 output[n_tokens-1] = the
            // prediction at position n_tokens. When a synthetic prefill token is
            // appended to enter the p1 chain on buckets > 2s, only the first
            // n_tokens outputs correspond to the caller-visible prefix.
            ctx->logits.resize((size_t)n_tokens * n_vocab);
            for (int p = 0; p < n_tokens; p++) {
                std::memcpy(ctx->logits.data() + (size_t)p * n_vocab,
                            ctx->npu_chunk_logits[p].data(),
                            (size_t)n_vocab * sizeof(float));
            }
            int last_idx = n_tokens - 1;
            npu_store_cross_attn(ctx, last_idx);
            // chunk_0 buffer is fully consumed by the prefill — first step decode
            // will run chunk_1.
            ctx->npu_buffer_read_idx = K0;
            ctx->npu_current_chunk_idx = 0;

        } else {
            ctx->npu_decode_phase = 2;
            // ── Step decode: return from buffer or run next chunk ─────────────
            int current_K = npu_get_chunk_k(ctx, ctx->npu_current_chunk_idx);

            if (ctx->npu_buffer_read_idx < current_K) {
                int idx = ctx->npu_buffer_read_idx;
                // Phase A: host's tokens[0] (this call) was picked from the
                // logits returned LAST call (= chunk_logits[idx-1]). NPU's
                // chunk-internal argmax of the same logits became the input
                // to this step, so chunk_logits[idx] is only "valid under
                // host history" if the two argmax decisions agree.
                if (idx >= 1) {
                    int proposal = npu_proposal_argmax(
                        ctx->npu_chunk_logits[idx - 1], whisper_token_eot(ctx));
                    ctx->shadow_n_compares++;
                    if ((int)tokens[0] == proposal) ctx->shadow_n_match++;
                    else                            ctx->shadow_n_diverge_in_chunk++;
                }
                ctx->logits = ctx->npu_chunk_logits[idx];
                npu_store_cross_attn(ctx, idx);
                ctx->npu_buffer_read_idx++;
            } else {
                // Phase A: about to launch next chunk with host's tokens[0],
                // which was picked from the last buffered logit of the
                // current chunk (chunk_logits[current_K - 1]). Compare to
                // what the NPU's chunk-internal argmax would have produced.
                if (current_K >= 1
                    && (int)ctx->npu_chunk_logits.size() >= current_K) {
                    int proposal = npu_proposal_argmax(
                        ctx->npu_chunk_logits[current_K - 1],
                        whisper_token_eot(ctx));
                    ctx->shadow_n_compares++;
                    if ((int)tokens[0] == proposal) ctx->shadow_n_match++;
                    else                            ctx->shadow_n_diverge_at_chunk_end++;
                }
                // Buffer exhausted → advance to next chunk
                ctx->npu_current_chunk_idx++;
                if (ctx->npu_current_chunk_idx >= ctx->npu_num_chunks) {
                    // All chunks used → force EOT
                    ctx->logits.assign(n_vocab, -1e9f);
                    int eot = whisper_token_eot(ctx);
                    if (eot >= 0 && eot < n_vocab) ctx->logits[eot] = 0.0f;
                } else {
                    // chunks ≥ 1 always take a single token
                    int32_t tok = (int32_t)tokens[0];
                    npu_run_chunk(ctx, ctx->npu_current_chunk_idx, &tok, 1);
                    ctx->logits = ctx->npu_chunk_logits[0];
                    npu_store_cross_attn(ctx, 0);
                    ctx->npu_buffer_read_idx = 1;
                }
            }
        }

        auto t1 = std::chrono::steady_clock::now();
        const int64_t decode_us = elapsed_us(t0, t1);
        ctx->t_decode_us += decode_us;
        if (n_past == 0) {
            ctx->npu_profile.prefill_calls++;
            ctx->npu_profile.prefill_total_us += decode_us;
        } else {
            ctx->npu_profile.generation_calls++;
            ctx->npu_profile.generation_total_us += decode_us;
        }
        return 0;
    }

    // ── Standard ONNX mode ───────────────────────────────────────────────────

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
    // NPU mode: npu_store_cross_attn() already populated cross_attn_data
    if (ctx->backend == 1 /*BACKEND_NPU_NK*/) {
        // cross_attn_data is already set by npu_store_cross_attn (single all-heads avg)
        return ctx->cross_attn_data.empty() ? -1 : 0;
    }

    // Standard ONNX mode: extract from step_cross_attn_w [1, n_layers, n_head, n_audio_ctx]
    const int n_layers    = ctx->dims.n_text_layer;
    const int n_head      = ctx->dims.n_text_head;
    const int n_audio_ctx = ctx->enc_n_frames;

    int layer = ctx->capture_layer;
    if (layer < 0) layer = n_layers + layer;
    if (layer < 0 || layer >= n_layers) return -1;

    ctx->cross_attn_n_heads     = n_head;
    ctx->cross_attn_n_audio_ctx = n_audio_ctx;
    ctx->cross_attn_data.assign(n_head * n_audio_ctx, 0.0f);

    if (ctx->step_cross_attn_w.empty()) return -1;

    const int layer_stride = n_head * n_audio_ctx;
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
    // NPU mode:
    //   - Preferred: use the real per-head alignment attention captured by
    //     npu_store_cross_attn() from the decoder outputs.
    //   - Fallback: if only a single averaged attention vector is available,
    //     replicate it so the DTW caller still receives the expected shape.
    if (ctx->backend == 1 /*BACKEND_NPU_NK*/) {
        if (!ctx->aheads_list.empty() &&
            !ctx->aheads_attn_data.empty() &&
            ctx->aheads_n_heads > 0 &&
            ctx->aheads_n_audio_ctx > 0) {
            return 0;
        }

        if (ctx->cross_attn_data.empty()) return -1;
        const int n_audio_ctx = ctx->cross_attn_n_audio_ctx;
        if (!ctx->aheads_list.empty()) {
            const int n_selected = (int)ctx->aheads_list.size();
            ctx->aheads_n_heads     = n_selected;
            ctx->aheads_n_audio_ctx = n_audio_ctx;
            ctx->aheads_attn_data.resize(n_selected * n_audio_ctx);
            // Legacy fallback: replicate single avg for each alignment head.
            for (int i = 0; i < n_selected; i++) {
                std::memcpy(ctx->aheads_attn_data.data() + i * n_audio_ctx,
                           ctx->cross_attn_data.data(), n_audio_ctx * sizeof(float));
            }
            return 0;
        }
        return whisper_copy_layer_cross_attention_from_ctx(ctx);
    }

    // Standard ONNX mode: extract specific (layer, head) pairs from step_cross_attn_w
    if (!ctx->aheads_list.empty()) {
        const int n_layers    = ctx->dims.n_text_layer;
        const int n_head      = ctx->dims.n_text_head;
        const int n_audio_ctx = ctx->enc_n_frames;

        if (ctx->step_cross_attn_w.empty()) return -1;

        const int n_selected = (int)ctx->aheads_list.size();
        ctx->aheads_n_heads     = n_selected;
        ctx->aheads_n_audio_ctx = n_audio_ctx;
        ctx->aheads_attn_data.resize(n_selected * n_audio_ctx);

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

bool whisper_prefill_requires_exact_two_tokens(whisper_context * ctx) {
    if (!(ctx && ctx->backend == 1 /*BACKEND_NPU_NK*/) || ctx->npu_use_1step) {
        return false;
    }
    return ctx->npu_prefill_prompt_token_lens.size() == 1 &&
           ctx->npu_prefill_prompt_token_lens[0] == 2;
}

int whisper_get_npu_max_prefill_prompt_tokens(whisper_context * ctx) {
    if (!(ctx && ctx->backend == 1 /*BACKEND_NPU_NK*/) || ctx->npu_use_1step ||
        ctx->npu_prefill_prompt_token_lens.empty()) {
        return -1;
    }
    return *std::max_element(
        ctx->npu_prefill_prompt_token_lens.begin(),
        ctx->npu_prefill_prompt_token_lens.end());
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
    p.no_context        = true;
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
    p.suppress_blank        = true;
    p.max_initial_ts        = 1.0f;
    p.suppress_non_speech_tokens = false;
    p.logits_filter_callback     = nullptr;
    p.logits_filter_callback_user_data = nullptr;
    return p;
}

static const std::vector<std::string> g_non_speech_tokens = {
    "\"", "#", "(", ")", "*", "+", "/", ":", ";", "<", "=", ">", "@", "[", "\\", "]", "^",
    "_", "`", "{", "|", "}", "~", "「", "」", "『", "』", "<<", ">>", "<<<", ">>>", "--",
    "---", "-(", "-[", "('", "(\"", "((", "))", "(((", ")))", "[[", "]]", "{{", "}}", "♪♪",
    "♪♪♪","♩", "♪", "♫", "♬", "♭", "♮", "♯"
};

static void suppress_token_if_present(whisper_context * ctx, float * logits, const char * token, float value) {
    const int tid = whisper_token_to_id(ctx, token);
    if (tid >= 0 && tid < ctx->dims.n_vocab) {
        logits[tid] = value;
    }
}

static void apply_whisper_full_logits_filters(
        whisper_context * ctx,
        const whisper_full_params & params,
        const std::vector<whisper_token_data> & token_data_for_cb,
        int step,
        float * logits) {
    const int n_vocab = ctx->dims.n_vocab;
    const int eot = whisper_token_eot(ctx);
    const int sot = whisper_token_sot(ctx);
    const int nots = whisper_token_not(ctx);
    const int beg = whisper_token_beg(ctx);
    const int transcribe = whisper_token_transcribe(ctx);
    const int translate = whisper_token_translate(ctx);
    const int prev = whisper_token_prev(ctx);
    const int solm = whisper_token_solm(ctx);
    const int nosp = whisper_token_to_id(ctx, "<|nospeech|>");
    const bool is_initial = (step == 0);
    constexpr float kNegInf = -1e9f;

    if (params.suppress_blank && is_initial) {
        if (eot >= 0 && eot < n_vocab) logits[eot] = kNegInf;
        suppress_token_if_present(ctx, logits, " ", kNegInf);
    }

    if (nots >= 0 && nots < n_vocab) logits[nots] = kNegInf;
    if (params.no_timestamps && beg >= 0 && beg < n_vocab) {
        for (int i = beg; i < n_vocab; ++i) logits[i] = kNegInf;
    }

    if (sot >= 0 && sot < n_vocab) logits[sot] = kNegInf;
    if (nosp >= 0 && nosp < n_vocab) logits[nosp] = kNegInf;
    if (solm >= 0 && solm < n_vocab) logits[solm] = kNegInf;
    if (translate >= 0 && translate < n_vocab) logits[translate] = kNegInf;
    if (transcribe >= 0 && transcribe < n_vocab) logits[transcribe] = kNegInf;
    if (prev >= 0 && prev < n_vocab) logits[prev] = kNegInf;

    if (ctx->dims.is_multilingual) {
        for (int i = 0; i < 99; ++i) {
            const int lang = whisper_token_lang(ctx, i);
            if (lang >= 0 && lang < n_vocab) logits[lang] = kNegInf;
        }
    }

    if (params.logits_filter_callback) {
        params.logits_filter_callback(
            ctx,
            nullptr,
            token_data_for_cb.data(),
            (int)token_data_for_cb.size(),
            logits,
            params.logits_filter_callback_user_data);
    }

    if (params.suppress_non_speech_tokens) {
        for (const std::string & token : g_non_speech_tokens) {
            const std::string suppress_tokens[] = {token, " " + token};
            for (const std::string & suppress_token : suppress_tokens) {
                const int tid = whisper_token_to_id(ctx, suppress_token.c_str());
                if (tid >= 0 && tid < n_vocab) logits[tid] = kNegInf;
            }
        }
        suppress_token_if_present(ctx, logits, " -", kNegInf);
        suppress_token_if_present(ctx, logits, " '", kNegInf);
    }

    if (beg >= 0 && beg < n_vocab) {
        const bool last_was_timestamp =
            !token_data_for_cb.empty() && token_data_for_cb.back().id >= beg;
        const bool penultimate_was_timestamp =
            token_data_for_cb.size() < 2 || token_data_for_cb[token_data_for_cb.size() - 2].id >= beg;

        if (last_was_timestamp) {
            if (penultimate_was_timestamp) {
                for (int i = beg; i < n_vocab; ++i) logits[i] = kNegInf;
            } else {
                for (int i = 0; i < eot; ++i) logits[i] = kNegInf;
            }
        }

        if (is_initial && params.max_initial_ts > 0.0f) {
            const float precision = float(WHISPER_CHUNK_SIZE) / ctx->dims.n_audio_ctx;
            const int tid0 = std::round(params.max_initial_ts / precision);
            for (int i = beg + tid0 + 1; i < n_vocab; ++i) logits[i] = kNegInf;
        }

        float logit_max = -INFINITY;
        for (int i = 0; i < n_vocab; ++i) {
            if (logits[i] > logit_max) logit_max = logits[i];
        }
        if (logit_max > -INFINITY) {
            float logsumexp = 0.0f;
            for (int i = 0; i < n_vocab; ++i) {
                if (logits[i] > -INFINITY) {
                    logsumexp += expf(logits[i] - logit_max);
                }
            }
            if (logsumexp > 0.0f) {
                const float full_logsumexp = logf(logsumexp) + logit_max;
                float timestamp_logprob = -INFINITY;
                float max_ts_logprob = -INFINITY;
                for (int i = beg; i < n_vocab; ++i) {
                    if (logits[i] > -INFINITY) {
                        const float lp = logits[i] - full_logsumexp;
                        if (lp > max_ts_logprob) max_ts_logprob = lp;
                    }
                }
                if (max_ts_logprob > -INFINITY) {
                    float ts_sum = 0.0f;
                    for (int i = beg; i < n_vocab; ++i) {
                        if (logits[i] > -INFINITY) {
                            const float lp = logits[i] - full_logsumexp;
                            ts_sum += expf(lp - max_ts_logprob);
                        }
                    }
                    if (ts_sum > 0.0f) {
                        timestamp_logprob = logf(ts_sum) + max_ts_logprob;
                    }
                }

                float max_text_logprob = -INFINITY;
                for (int i = 0; i < beg; ++i) {
                    if (logits[i] > -INFINITY) {
                        const float lp = logits[i] - full_logsumexp;
                        if (lp > max_text_logprob) max_text_logprob = lp;
                    }
                }

                if (timestamp_logprob > max_text_logprob) {
                    for (int i = 0; i < beg; ++i) logits[i] = kNegInf;
                }
            }
        }
    }
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

    // For NPU mode reinference: force the dedicated 30s encoder + 1-step decoder.
    // The first reinfer call starts from a fresh context, so ctx->npu_30s may not
    // be loaded yet. Key off config availability, not the already-loaded pointer.
    bool was_force_30s = false;
    if (ctx->backend == 1 /*BACKEND_NPU_NK*/ && ctx->npu_has_30s_reinfer) {
        was_force_30s = ctx->npu_force_30s;
        ctx->npu_force_30s = true;
    }

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

        apply_whisper_full_logits_filters(ctx, params, token_data_for_cb, step, logits);

        whisper_token next = (whisper_token)(std::max_element(logits, logits + n_vocab) - logits);
        if (next == eot) break;

        whisper_token_data td{}; td.id = next; td.p = 1.0f;
        token_data_for_cb.push_back(td);

        if (next < eot) {
            decoded.push_back(next);
        }

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

    // Restore 30s flag
    if (ctx->backend == 1 /*BACKEND_NPU_NK*/) {
        ctx->npu_force_30s = was_force_30s;
    }

    return 0;
}

// ── Phase A: shadow argmax stats accessors ────────────────────────────────────

void whisper_print_shadow_stats(whisper_context * ctx) {
    if (!ctx) return;
    int64_t n   = ctx->shadow_n_compares;
    int64_t m   = ctx->shadow_n_match;
    int64_t din = ctx->shadow_n_diverge_in_chunk;
    int64_t den = ctx->shadow_n_diverge_at_chunk_end;
    double  rate = (n > 0) ? (100.0 * (double)(n - m) / (double)n) : 0.0;
    fprintf(stderr,
            "[SHADOW] compares=%lld match=%lld diverge_in_chunk=%lld "
            "diverge_at_chunk_end=%lld diverge_rate=%.2f%%\n",
            (long long)n, (long long)m, (long long)din, (long long)den, rate);
}

void whisper_reset_shadow_stats(whisper_context * ctx) {
    if (!ctx) return;
    ctx->shadow_n_compares             = 0;
    ctx->shadow_n_match                = 0;
    ctx->shadow_n_diverge_in_chunk     = 0;
    ctx->shadow_n_diverge_at_chunk_end = 0;
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
    if (ctx->backend == 1 /*BACKEND_NPU_NK*/) {
        auto avg_ms = [](int64_t total_us, int64_t calls) -> double {
            return calls > 0 ? us_to_ms(total_us) / (double)calls : 0.0;
        };
        const auto & p = ctx->npu_profile;
        fprintf(stderr,
                "[NPU_PROFILE_SUMMARY] encoder_calls=%lld encoder_total_ms=%.3f "
                "encoder_run_ms=%.3f encoder_prepare_ms=%.3f encoder_parse_ms=%.3f "
                "encoder_run_avg_ms=%.3f\n",
                (long long)p.encoder_calls,
                us_to_ms(p.encoder_total_us),
                us_to_ms(p.encoder_run_us),
                us_to_ms(p.encoder_prepare_us),
                us_to_ms(p.encoder_parse_us),
                avg_ms(p.encoder_run_us, p.encoder_calls));
        fprintf(stderr,
                "[NPU_PROFILE_SUMMARY] chunk_calls=%lld chunk_total_ms=%.3f "
                "chunk_run_ms=%.3f chunk_prepare_ms=%.3f chunk_parse_ms=%.3f "
                "chunk_run_avg_ms=%.3f\n",
                (long long)p.chunk_calls,
                us_to_ms(p.chunk_total_us),
                us_to_ms(p.chunk_run_us),
                us_to_ms(p.chunk_prepare_us),
                us_to_ms(p.chunk_parse_us),
                avg_ms(p.chunk_run_us, p.chunk_calls));
        fprintf(stderr,
                "[NPU_PROFILE_SUMMARY] one_step_calls=%lld one_step_total_ms=%.3f "
                "one_step_run_ms=%.3f one_step_prepare_ms=%.3f one_step_parse_ms=%.3f "
                "one_step_run_avg_ms=%.3f\n",
                (long long)p.one_step_calls,
                us_to_ms(p.one_step_total_us),
                us_to_ms(p.one_step_run_us),
                us_to_ms(p.one_step_prepare_us),
                us_to_ms(p.one_step_parse_us),
                avg_ms(p.one_step_run_us, p.one_step_calls));
        fprintf(stderr,
                "[NPU_PROFILE_SUMMARY] prefill_calls=%lld prefill_total_ms=%.3f "
                "prefill_avg_ms=%.3f generation_calls=%lld generation_total_ms=%.3f "
                "generation_avg_ms=%.3f\n",
                (long long)p.prefill_calls,
                us_to_ms(p.prefill_total_us),
                avg_ms(p.prefill_total_us, p.prefill_calls),
                (long long)p.generation_calls,
                us_to_ms(p.generation_total_us),
                avg_ms(p.generation_total_us, p.generation_calls));
    }
}

void whisper_reset_timings(whisper_context * ctx) {
    ctx->t_encode_us = 0;
    ctx->t_decode_us = 0;
    ctx->npu_profile = {};
}
