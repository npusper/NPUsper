#pragma once

#include <cstdint>
#include <string>
#include <vector>

// ── Constants (same as whisper.cpp) ──────────────────────────────────────────
#define WHISPER_SAMPLE_RATE 16000
#define WHISPER_N_FFT       400
#define WHISPER_HOP_LENGTH  160
#define WHISPER_CHUNK_SIZE  30

// ── Opaque context ────────────────────────────────────────────────────────────
struct whisper_context;

// ── Token type ────────────────────────────────────────────────────────────────
typedef int whisper_token;

struct whisper_token_data {
    whisper_token id;
    whisper_token tid;  // timestamp token
    float   p;          // probability
    float   plog;       // log probability
    float   pt;         // probability of timestamp token
    float   ptsum;      // sum of probabilities of timestamp tokens
    int64_t t0;         // start time (in 1/100th of a second)
    int64_t t1;         // end time
    float   vlen;       // voice length
};

// ── Init params ───────────────────────────────────────────────────────────────
struct whisper_context_params {
    bool use_gpu;
    int  gpu_device;
    bool flash_attn;  // ignored, for API compat
};

whisper_context_params whisper_context_default_params();

// ── Full params ───────────────────────────────────────────────────────────────
enum whisper_sampling_strategy {
    WHISPER_SAMPLING_GREEDY,
    WHISPER_SAMPLING_BEAM_SEARCH,
};

struct whisper_full_params {
    whisper_sampling_strategy strategy;

    int   n_threads;
    int   n_max_text_ctx;
    int   offset_ms;
    int   duration_ms;

    bool  translate;
    bool  no_context;
    bool  no_timestamps;
    bool  single_segment;
    bool  print_special;
    bool  print_progress;
    bool  print_realtime;
    bool  print_timestamps;

    const char * language;
    const char * initial_prompt;

    // beam search
    struct {
        int beam_size;
        float patience;
    } beam_search;

    // greedy
    struct {
        int best_of;
    } greedy;

    float temperature;
    float temperature_inc;
    float entropy_thold;
    float logprob_thold;
    float no_speech_thold;

    // token timestamps
    bool  token_timestamps;
    float thold_pt;
    float thold_ptsum;
    int   max_len;
    bool  split_on_word;
    int   max_tokens;

    bool  suppress_non_speech_tokens;

    // Logits filter callback — called after each decode step before sampling.
    // Can modify logits in-place (e.g. suppress EOT conditionally).
    // Signature matches whisper.cpp: (ctx, state=NULL, tokens, n_tokens, logits, user_data)
    typedef void (*whisper_logits_filter_callback)(
            struct whisper_context * ctx,
            void                  * state,       // unused (NULL), for API compat with whisper.cpp
            const whisper_token_data * tokens,
            int                     n_tokens,
            float                 * logits,
            void                  * user_data);

    whisper_logits_filter_callback logits_filter_callback;
    void * logits_filter_callback_user_data;

    // callbacks (unused, for API compat)
    void * new_segment_callback;
    void * new_segment_callback_user_data;
    void * progress_callback;
    void * progress_callback_user_data;
};

whisper_full_params whisper_full_default_params(whisper_sampling_strategy strategy);

// ── Lifecycle ─────────────────────────────────────────────────────────────────
// model_path: directory containing encoder.onnx, decoder_prefill.onnx,
//             decoder_step.onnx, vocab.txt, dims.txt
whisper_context * whisper_init_from_file_with_params(
    const char * model_path,
    whisper_context_params params);

void whisper_free(whisper_context * ctx);

// ── Model info ────────────────────────────────────────────────────────────────
int  whisper_n_vocab          (whisper_context * ctx);
int  whisper_n_text_ctx       (whisper_context * ctx);
int  whisper_n_audio_ctx      (whisper_context * ctx);
bool whisper_is_multilingual  (whisper_context * ctx);
int  whisper_model_n_audio_state(whisper_context * ctx);
int  whisper_model_n_audio_head (whisper_context * ctx);
int  whisper_model_n_audio_layer(whisper_context * ctx);
int  whisper_model_n_text_state (whisper_context * ctx);

// ── Special tokens ────────────────────────────────────────────────────────────
whisper_token whisper_token_eot  (whisper_context * ctx);
whisper_token whisper_token_sot  (whisper_context * ctx);
whisper_token whisper_token_prev (whisper_context * ctx);
whisper_token whisper_token_solm (whisper_context * ctx);
whisper_token whisper_token_not  (whisper_context * ctx);  // no timestamps
whisper_token whisper_token_beg  (whisper_context * ctx);
whisper_token whisper_token_lang (whisper_context * ctx, int lang_id);
whisper_token whisper_token_translate  (whisper_context * ctx);
whisper_token whisper_token_transcribe (whisper_context * ctx);

// ── Vocab ─────────────────────────────────────────────────────────────────────
const char * whisper_token_to_str(whisper_context * ctx, whisper_token token);
int          whisper_token_to_id (whisper_context * ctx, const char * str);  // -1 if not found
int          whisper_lang_id     (const char * lang);  // -1 if not found

// ── Low-level encode / decode ─────────────────────────────────────────────────
// Must call whisper_pcm_to_mel (or whisper_set_mel) before whisper_encode.

// Number of mel frames from last whisper_pcm_to_mel call.
int whisper_n_len(whisper_context * ctx);

// Set effective audio context (no-op in ORT; always uses actual audio length).
void whisper_set_audio_ctx(whisper_context * ctx, int audio_ctx);

// System info string (returns empty string in ORT backend).
const char * whisper_print_system_info();

// Compute log-mel spectrogram from raw PCM samples.
int whisper_pcm_to_mel(
    whisper_context * ctx,
    const float     * samples,
    int               n_samples,
    int               n_threads);

// Run the encoder. offset = 0 for standard usage.
int whisper_encode(whisper_context * ctx, int offset, int n_threads);

// Run the decoder.
//   n_past == 0 → decoder_prefill (processes all tokens, builds full KV cache)
//   n_past >  0 → decoder_step   (processes single token, updates self_kv)
// tokens must have exactly 1 token when n_past > 0.
int whisper_decode(
    whisper_context     * ctx,
    const whisper_token * tokens,
    int                   n_tokens,
    int                   n_past,
    int                   n_threads);

// Pointer to logits from the last whisper_decode call.
// Valid until the next whisper_decode call.
// Length: whisper_n_vocab(ctx)
float * whisper_get_logits(whisper_context * ctx);

// Encoder output from last whisper_encode call.
// n_frames_out: number of encoder output frames (n_audio_frames / 2)
// n_state_out:  encoder hidden state size
const float * whisper_get_encoder_output_from_ctx(
    whisper_context * ctx,
    int             * n_frames_out,
    int             * n_state_out);

// ── Alignment heads presets (same as whisper.cpp) ────────────────────────────
enum whisper_alignment_heads_preset {
    WHISPER_AHEADS_NONE,
    WHISPER_AHEADS_TINY_EN,
    WHISPER_AHEADS_TINY,
    WHISPER_AHEADS_BASE_EN,
    WHISPER_AHEADS_BASE,
    WHISPER_AHEADS_SMALL_EN,
    WHISPER_AHEADS_SMALL,
    WHISPER_AHEADS_MEDIUM_EN,
    WHISPER_AHEADS_MEDIUM,
    WHISPER_AHEADS_LARGE_V1,
    WHISPER_AHEADS_LARGE_V2,
    WHISPER_AHEADS_LARGE_V3,
};

struct whisper_ahead {
    int n_text_layer;
    int n_head;
};

// Set alignment heads preset for DTW. Must be called before whisper_set_aheads_cross_attention_capture.
void whisper_set_aheads_preset(whisper_context * ctx, whisper_alignment_heads_preset preset);

// ── Cross-attention capture ───────────────────────────────────────────────────
// Called before whisper_decode to enable per-layer cross-attention capture.
// layer == -1 means the last layer.
void whisper_set_layer_cross_attention_capture(whisper_context * ctx, int layer);
void whisper_set_aheads_cross_attention_capture(whisper_context * ctx, bool enable);

// Copy cross-attention weights captured during the last whisper_decode into
// internal storage. Returns 0 on success.
int whisper_copy_layer_cross_attention_from_ctx(whisper_context * ctx);
int whisper_copy_aheads_cross_attention_from_ctx(whisper_context * ctx);

// Access the stored cross-attention data.
// Shape: [n_heads, n_tokens, n_audio_ctx], n_tokens always 1 in ORT backend.
const float * whisper_get_layer_cross_attention_data_from_ctx(
    whisper_context * ctx,
    int             * n_tokens_out,
    int             * n_audio_ctx_out,
    int             * n_heads_out);

const float * whisper_get_aheads_cross_attention_data_from_ctx(
    whisper_context * ctx,
    int             * n_tokens_out,
    int             * n_audio_ctx_out,
    int             * n_heads_out);

// ── High-level transcription (used by whisper_streaming_cpp) ─────────────────
int whisper_full(
    whisper_context    * ctx,
    whisper_full_params  params,
    const float        * samples,
    int                  n_samples);

int          whisper_full_n_segments              (whisper_context * ctx);
const char * whisper_full_get_segment_text        (whisper_context * ctx, int i_segment);
int64_t      whisper_full_get_segment_t0          (whisper_context * ctx, int i_segment);
int64_t      whisper_full_get_segment_t1          (whisper_context * ctx, int i_segment);
bool         whisper_full_get_segment_speaker_turn_next(whisper_context * ctx, int i_segment);
int          whisper_full_n_tokens                (whisper_context * ctx, int i_segment);
whisper_token_data whisper_full_get_token_data    (whisper_context * ctx, int i_segment, int i_token);
whisper_token      whisper_full_get_token_id      (whisper_context * ctx, int i_segment, int i_token);
const char *       whisper_full_get_token_text    (whisper_context * ctx, int i_segment, int i_token);
float              whisper_full_get_token_p       (whisper_context * ctx, int i_segment, int i_token);

// ── Misc ──────────────────────────────────────────────────────────────────────
void whisper_print_timings(whisper_context * ctx);
void whisper_reset_timings(whisper_context * ctx);
