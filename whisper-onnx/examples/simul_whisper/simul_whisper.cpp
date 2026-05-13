// SimulWhisper C++ implementation (ONNX Runtime backend)
// Ported from whisperflow-ggml/examples/simul_whisper/
// Changes: whisper.h → whisper_ort.h, removed ggml DTW cparams
//
#include "common.h"
#include "whisper_ort.h"

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


// =====================================================================
// CIF model: Linear(n_audio_state, 1) for truncation detection
// =====================================================================
struct cif_model {
    int n_state;
    std::vector<float> weight;
    float bias;
};

bool load_cif_model(const std::string & path, cif_model & cif) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin.is_open()) {
        fprintf(stderr, "error: failed to open CIF model: %s\n", path.c_str());
        return false;
    }
    int32_t n_state;
    fin.read(reinterpret_cast<char *>(&n_state), sizeof(int32_t));
    cif.n_state = n_state;
    cif.weight.resize(n_state);
    fin.read(reinterpret_cast<char *>(cif.weight.data()), n_state * sizeof(float));
    fin.read(reinterpret_cast<char *>(&cif.bias), sizeof(float));
    if (!fin) {
        fprintf(stderr, "error: failed to read CIF model data from %s\n", path.c_str());
        return false;
    }
    return true;
}

bool fire_at_boundary(const float * enc_out, int content_mel_len, int n_state, const cif_model & cif) {
    if (content_mel_len < 2) return false;

    std::vector<float> alphas(content_mel_len);
    for (int t = 0; t < content_mel_len; t++) {
        float dot = 0.0f;
        const float * enc_t = enc_out + t * n_state;
        for (int i = 0; i < n_state; i++) dot += cif.weight[i] * enc_t[i];
        alphas[t] = 1.0f / (1.0f + expf(-(dot + cif.bias)));
    }

    float alpha_sum = 0.0f;
    for (int t = 0; t < content_mel_len; t++) alpha_sum += alphas[t];
    int decode_length = (int)roundf(alpha_sum);
    if (decode_length < 1) decode_length = 1;

    float scale = (float)decode_length / (alpha_sum + 1e-8f);
    for (int t = 0; t < content_mel_len; t++) alphas[t] *= scale;

    const float threshold = 0.999f;
    for (int iter = 0; iter < 10; iter++) {
        bool found = false;
        for (int t = 0; t < content_mel_len; t++) {
            if (alphas[t] > threshold) { found = true; break; }
        }
        if (!found) break;
        float mask_sum = 0.0f; int mask_count = 0;
        for (int t = 0; t < content_mel_len; t++) {
            if (alphas[t] != 0.0f) { mask_sum += alphas[t]; mask_count++; }
        }
        if (mask_count == 0) break;
        float mean = 0.5f * mask_sum / mask_count;
        for (int t = 0; t < content_mel_len; t++) {
            if (alphas[t] > threshold) {
                float mask_val = (alphas[t] != 0.0f) ? 1.0f : 0.0f;
                alphas[t] = alphas[t] * 0.5f + mean * mask_val;
            }
        }
    }

    int cum_len = content_mel_len - 1;
    if (cum_len < 1) return false;

    std::vector<float> integrate(cum_len);
    integrate[0] = alphas[0];
    for (int t = 1; t < cum_len; t++) integrate[t] = integrate[t - 1] + alphas[t];

    float exceed_count = floorf(integrate[cum_len - 1] / threshold);
    for (int t = 0; t < cum_len; t++) integrate[t] -= exceed_count;

    int first_pos = -1;
    for (int t = 0; t < cum_len; t++) {
        if (integrate[t] >= 0.0f) { first_pos = t; break; }
    }

    if (first_pos < 0) return false;
    return first_pos >= content_mel_len - 2;
}


// =====================================================================
// Command-line parameters
// =====================================================================
struct simul_whisper_params {
    int32_t n_threads  = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t step_ms    = 1000;
    int32_t max_tokens = 0;
    int32_t audio_ctx  = 0;

    int32_t frame_threshold  = 4;
    int32_t rewind_threshold = 200;

    bool translate     = false;
    bool print_special = false;
    bool no_timestamps = false;
    bool use_gpu       = true;
    bool debug         = false;
    bool no_realtime   = false;  // skip sleep, process chunks as fast as possible

    std::string language  = "en";
    std::string model     = "models/base";
    std::string dtw       = "base";  // kept for CLI compat; no DTW preset in ORT
    std::string cif_model_path = "";

    std::vector<std::string> fname_inp = {};
    std::string fname_out = "";
};

void print_usage(int argc, char ** argv, const simul_whisper_params & params);

bool params_parse(int argc, char ** argv, simul_whisper_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg[0] != '-') { params.fname_inp.push_back(arg); continue; }
        if (arg == "-h" || arg == "--help") { print_usage(argc, argv, params); exit(0); }
        else if (arg == "-t"   || arg == "--threads")          { params.n_threads        = std::stoi(argv[++i]); }
        else if (               arg == "--step")               { params.step_ms          = std::stoi(argv[++i]); }
        else if (arg == "-mt"  || arg == "--max-tokens")       { params.max_tokens       = std::stoi(argv[++i]); }
        else if (arg == "-ac"  || arg == "--audio-ctx")        { params.audio_ctx        = std::stoi(argv[++i]); }
        else if (arg == "-ft"  || arg == "--frame-threshold")  { params.frame_threshold  = std::stoi(argv[++i]); }
        else if (arg == "-rt"  || arg == "--rewind-threshold") { params.rewind_threshold = std::stoi(argv[++i]); }
        else if (arg == "-cf"  || arg == "--cif-model")        { params.cif_model_path   = argv[++i]; }
        else if (arg == "-tr"  || arg == "--translate")        { params.translate        = true; }
        else if (arg == "-ps"  || arg == "--print-special")    { params.print_special    = true; }
        else if (arg == "-dtw" || arg == "--dtw")              { params.dtw              = argv[++i]; }
        else if (arg == "-l"   || arg == "--language")         { params.language         = argv[++i]; }
        else if (arg == "-m"   || arg == "--model")            { params.model            = argv[++i]; }
        else if (arg == "-of"  || arg == "--output-file")      { params.fname_out        = argv[++i]; }
        else if (arg == "-ng"  || arg == "--no-gpu")           { params.use_gpu          = false; }
        else if (               arg == "--debug")              { params.debug            = true; }
        else if (               arg == "--no-realtime")        { params.no_realtime      = true; }
        else { fprintf(stderr, "error: unknown argument: %s\n", arg.c_str()); print_usage(argc, argv, params); exit(1); }
    }
    return true;
}

void print_usage(int /*argc*/, char ** argv, const simul_whisper_params & params) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options] input.wav\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "SimulWhisper: AlignAtt + CIF truncation detection (ONNX RT)\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,       --help              show this help message\n");
    fprintf(stderr, "  -t N,     --threads N         [%-7d] number of threads\n",                params.n_threads);
    fprintf(stderr, "            --step N            [%-7d] audio step size in ms\n",             params.step_ms);
    fprintf(stderr, "  -mt N,    --max-tokens N      [%-7d] max tokens per chunk (0=auto)\n",    params.max_tokens);
    fprintf(stderr, "  -ac N,    --audio-ctx N       [%-7d] audio context size\n",               params.audio_ctx);
    fprintf(stderr, "  -ft N,    --frame-threshold N [%-7d] AlignAtt frame threshold\n",         params.frame_threshold);
    fprintf(stderr, "  -rt N,    --rewind-threshold N[%-7d] AlignAtt rewind threshold\n",        params.rewind_threshold);
    fprintf(stderr, "  -cf FNAME,--cif-model FNAME   [%-7s] CIF model weights (required)\n",    params.cif_model_path.c_str());
    fprintf(stderr, "  -tr,      --translate         [%-7s] translate to English\n",             params.translate ? "true" : "false");
    fprintf(stderr, "  -dtw M,   --dtw M             [%-7s] model preset (no-op in ORT)\n",      params.dtw.c_str());
    fprintf(stderr, "  -l LANG,  --language LANG     [%-7s] spoken language\n",                  params.language.c_str());
    fprintf(stderr, "  -m FNAME, --model FNAME       [%-7s] model path\n",                       params.model.c_str());
    fprintf(stderr, "  -of FNAME,--output-file FNAME           output file path\n");
    fprintf(stderr, "  -ng,      --no-gpu            [%-7s] disable GPU\n",                      params.use_gpu ? "false" : "true");
    fprintf(stderr, "            --debug             [%-7s] print debug info\n",                 params.debug ? "true" : "false");
    fprintf(stderr, "            --no-realtime       [%-7s] skip sleep, process as fast as possible\n", params.no_realtime ? "true" : "false");
    fprintf(stderr, "\n");
}

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

static void zscore_normalize(std::vector<float> & data, int n_heads, int n_tokens, int n_audio_ctx) {
    for (int h = 0; h < n_heads; h++) {
        for (int a = 0; a < n_audio_ctx; a++) {
            float sum = 0.0f, sum2 = 0.0f;
            for (int t = 0; t < n_tokens; t++) {
                float v = data[h * n_tokens * n_audio_ctx + t * n_audio_ctx + a];
                sum += v; sum2 += v * v;
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

static int argmax(const float * arr, int len) {
    int best = 0;
    for (int i = 1; i < len; i++) if (arr[i] > arr[best]) best = i;
    return best;
}

static std::vector<std::vector<whisper_token>> split_to_words(
    struct whisper_context * ctx, const std::vector<whisper_token> & tokens)
{
    std::vector<std::vector<whisper_token>> words;
    std::vector<whisper_token> current_word;
    for (auto tok : tokens) {
        const char * str = whisper_token_to_str(ctx, tok);
        if (str && str[0] == ' ' && !current_word.empty()) {
            words.push_back(current_word); current_word.clear();
        }
        current_word.push_back(tok);
    }
    if (!current_word.empty()) words.push_back(current_word);
    return words;
}

static std::vector<whisper_token> emit_with_cif(
    struct whisper_context * ctx,
    const std::vector<whisper_token> & decoded_tokens,
    bool is_last, bool fire_detected)
{
    if (is_last || fire_detected) return decoded_tokens;
    auto words = split_to_words(ctx, decoded_tokens);
    if (words.size() <= 1) return {};
    std::vector<whisper_token> emitted;
    for (size_t i = 0; i < words.size() - 1; i++)
        for (auto t : words[i]) emitted.push_back(t);
    return emitted;
}

void print_committed(const std::vector<std::string> & committed) {
    fprintf(stderr, "\n=== Final transcript ===\n");
    for (const auto & w : committed) fprintf(stderr, "%s", w.c_str());
    fprintf(stderr, "\n========================\n");
}

void print_tsw_with_latency(const std::vector<std::tuple<double, double, std::string, double, double>> & records) {
    double latency_sum = 0.0; int count = 0;
    for (const auto & entry : records) {
        printf("Start Time: %g, End Time: %g, Transcript: %s, Latency: %g, Emission Time: %g\n",
               std::get<0>(entry), std::get<1>(entry), std::get<2>(entry).c_str(),
               std::get<3>(entry), std::get<4>(entry));
        latency_sum += std::get<3>(entry); count++;
    }
    if (count > 0) printf("Average latency: %g\n", latency_sum / count);
}

int main(int argc, char ** argv) {
    simul_whisper_params params;
    if (!params_parse(argc, argv, params)) return 1;

    if (params.fname_inp.empty()) {
        fprintf(stderr, "error: no input file specified\n");
        print_usage(argc, argv, params); return 1;
    }
    if (params.cif_model_path.empty()) {
        fprintf(stderr, "error: --cif-model is required for SimulWhisper\n"); return 1;
    }

    cif_model cif;
    if (!load_cif_model(params.cif_model_path, cif)) return 2;
    fprintf(stderr, "%s: loaded CIF model: n_state=%d, bias=%.6f\n", __func__, cif.n_state, cif.bias);

    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu    = params.use_gpu;
    cparams.flash_attn = false;
    // Note: DTW preset selection not available in ORT backend; all heads used.

    struct whisper_context * ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "error: failed to initialize whisper context\n"); return 4;
    }

    int model_n_state = whisper_model_n_audio_state(ctx);
    if (cif.n_state != model_n_state) {
        fprintf(stderr, "error: CIF n_state (%d) != model n_audio_state (%d)\n", cif.n_state, model_n_state);
        whisper_free(ctx); return 5;
    }

    std::vector<float> pcmf32_all;
    std::vector<std::vector<float>> pcmf32s;
    if (!::read_wav(params.fname_inp[0], pcmf32_all, pcmf32s, false)) {
        fprintf(stderr, "error: failed to read WAV file '%s'\n", params.fname_inp[0].c_str()); return 6;
    }

    fprintf(stderr, "\n%s: SimulWhisper (AlignAtt + CIF) mode\n", __func__);
    fprintf(stderr, "%s: model = %s, language = %s\n", __func__, params.model.c_str(), params.language.c_str());
    fprintf(stderr, "%s: cif_model = %s (n_state=%d)\n", __func__, params.cif_model_path.c_str(), cif.n_state);
    fprintf(stderr, "%s: audio = %.1f sec, step = %d ms, frame_threshold = %d, rewind_threshold = %d\n",
            __func__, (float)pcmf32_all.size() / WHISPER_SAMPLE_RATE, params.step_ms,
            params.frame_threshold, params.rewind_threshold);
    fprintf(stderr, "\n");

    std::vector<float> audio_buffer;
    std::vector<std::string> committed_words;
    std::vector<whisper_token> committed_tokens;
    std::vector<std::vector<whisper_token>> token_segments;
    int n_segments_in_buffer = 0;
    int last_attend_frame = -params.rewind_threshold;
    int64_t buffer_start_sample = 0;

    int cif_fire_count = 0, cif_drop_count = 0;
    std::vector<std::tuple<double, double, std::string, double, double>> latency_record;

    const int n_samples_step = (params.step_ms * WHISPER_SAMPLE_RATE) / 1000;
    int64_t pcmf32_index = 0;
    int n_iter = 0;
    const auto t_start = std::chrono::high_resolution_clock::now();

    while (pcmf32_index < (int64_t)pcmf32_all.size()) {
        bool is_last = false;

        int64_t chunk_size;
        if (params.no_realtime) {
            chunk_size = std::min((int64_t)n_samples_step, (int64_t)pcmf32_all.size() - pcmf32_index);
        } else {
            double target_time = (double)(pcmf32_index + n_samples_step) / WHISPER_SAMPLE_RATE;
            auto t_now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(t_now - t_start).count();
            if (elapsed < target_time) {
                double sleep_sec = target_time - elapsed;
                std::this_thread::sleep_for(std::chrono::microseconds((int64_t)(sleep_sec * 1e6)));
            }

            auto t_after_sleep = std::chrono::high_resolution_clock::now();
            double elapsed_after = std::chrono::duration<double>(t_after_sleep - t_start).count();
            int64_t available_samples = std::min((int64_t)(elapsed_after * WHISPER_SAMPLE_RATE), (int64_t)pcmf32_all.size());
            chunk_size = available_samples - pcmf32_index;
            if (chunk_size <= 0) chunk_size = std::min((int64_t)n_samples_step, (int64_t)pcmf32_all.size() - pcmf32_index);
        }
        std::vector<float> pcmf32_new;
        pcmf32_new.insert(pcmf32_new.end(), pcmf32_all.begin() + pcmf32_index, pcmf32_all.begin() + pcmf32_index + chunk_size);
        pcmf32_index += chunk_size;
        is_last = (pcmf32_index >= (int64_t)pcmf32_all.size());

        audio_buffer.insert(audio_buffer.end(), pcmf32_new.begin(), pcmf32_new.end());
        n_segments_in_buffer++;

        const float buffer_len_sec = 30.0f;
        const float step_sec = (float)params.step_ms / 1000.0f;
        while ((float)n_segments_in_buffer * step_sec >= buffer_len_sec && n_segments_in_buffer > 1) {
            int remove_samples = std::min((int)(n_samples_step), (int)audio_buffer.size());
            audio_buffer.erase(audio_buffer.begin(), audio_buffer.begin() + remove_samples);
            buffer_start_sample += remove_samples;
            int removed_frames = remove_samples / (WHISPER_HOP_LENGTH * 2);
            last_attend_frame -= removed_frames;
            n_segments_in_buffer--;
            if (!token_segments.empty()) token_segments.erase(token_segments.begin());
        }

        float buffer_seconds = (float)audio_buffer.size() / WHISPER_SAMPLE_RATE;
        fprintf(stderr, "[BUFFER_STATS] iter=%d buffer_sec=%.3f buffer_samples=%ld\n",
               n_iter, buffer_seconds, (long)audio_buffer.size());

        int content_mel_len = (int)audio_buffer.size() / 320;
        if (content_mel_len < 1) content_mel_len = 1;

        std::vector<float> pcmf32_padded = audio_buffer;
        const size_t n_samples_30s = WHISPER_SAMPLE_RATE * 30;
        if (pcmf32_padded.size() < n_samples_30s) pcmf32_padded.resize(n_samples_30s, 0.0f);

        fprintf(stderr, "[INPUT_STATS] iter=%d input_sec=%.3f input_samples=%ld content_mel_len=%d\n",
               n_iter, (float)pcmf32_padded.size() / WHISPER_SAMPLE_RATE,
               (long)pcmf32_padded.size(), content_mel_len);

        auto t_mel_s = std::chrono::high_resolution_clock::now();
        if (whisper_pcm_to_mel(ctx, pcmf32_padded.data(), pcmf32_padded.size(), params.n_threads) != 0) {
            fprintf(stderr, "%s: failed to compute mel spectrogram\n", __func__); return 7;
        }
        auto t_mel_e = std::chrono::high_resolution_clock::now();
        double iter_mel_us = std::chrono::duration<double, std::micro>(t_mel_e - t_mel_s).count();

        auto t_enc_s = std::chrono::high_resolution_clock::now();
        if (whisper_encode(ctx, 0, params.n_threads) != 0) {
            fprintf(stderr, "%s: failed to encode\n", __func__); return 7;
        }
        auto t_enc_e = std::chrono::high_resolution_clock::now();
        double iter_encode_us = std::chrono::duration<double, std::micro>(t_enc_e - t_enc_s).count();

        int enc_n_frames = 0, enc_n_state = 0;
        const float * enc_out = whisper_get_encoder_output_from_ctx(ctx, &enc_n_frames, &enc_n_state);
        bool fire_detected = false;
        if (enc_out && content_mel_len >= 2) {
            auto t_cif_start = std::chrono::high_resolution_clock::now();
            fire_detected = fire_at_boundary(enc_out, content_mel_len, enc_n_state, cif);
            auto t_cif_end = std::chrono::high_resolution_clock::now();
            double cif_us = std::chrono::duration<double, std::micro>(t_cif_end - t_cif_start).count();
            fprintf(stderr, "[iter=%d] CIF inference: %.1f us (content_mel_len=%d)\n", n_iter, cif_us, content_mel_len);
            if (fire_detected) cif_fire_count++;
            else               cif_drop_count++;
            if (params.debug) {
                fprintf(stderr, "[iter=%d] CIF fire=%s (fire_count=%d, drop_count=%d)\n",
                        n_iter, fire_detected ? "true" : "false", cif_fire_count, cif_drop_count);
            }
        }

        const int n_text_ctx = whisper_n_text_ctx(ctx);
        const int n_sot_special = whisper_is_multilingual(ctx) ? 4 : 2;
        const int max_context = n_text_ctx - n_sot_special;

        std::vector<whisper_token> prompt_tokens;
        for (const auto & seg : token_segments) for (auto t : seg) prompt_tokens.push_back(t);
        if ((int)prompt_tokens.size() > max_context) {
            int excess = (int)prompt_tokens.size() - max_context;
            prompt_tokens.erase(prompt_tokens.begin(), prompt_tokens.begin() + excess);
        }

        std::vector<whisper_token> sot_seq;
        sot_seq.push_back(whisper_token_sot(ctx));
        if (whisper_is_multilingual(ctx)) {
            sot_seq.push_back(whisper_token_lang(ctx, whisper_lang_id(params.language.c_str())));
            if (params.translate) sot_seq.push_back(whisper_token_translate(ctx));
            else                  sot_seq.push_back(whisper_token_transcribe(ctx));
        }
        sot_seq.push_back(whisper_token_not(ctx));
        for (auto t : prompt_tokens) sot_seq.push_back(t);
        int sot_sequence_len = (int)sot_seq.size();

        whisper_set_aheads_cross_attention_capture(ctx, true);

        auto t_prefill_s = std::chrono::high_resolution_clock::now();
        if (whisper_decode(ctx, sot_seq.data(), sot_seq.size(), 0, params.n_threads) != 0) {
            fprintf(stderr, "%s: failed to decode SOT sequence\n", __func__); return 7;
        }
        auto t_prefill_e = std::chrono::high_resolution_clock::now();
        double iter_prefill_us = std::chrono::duration<double, std::micro>(t_prefill_e - t_prefill_s).count();

        int n_past = sot_sequence_len;
        int last_decode_n_tokens = sot_sequence_len;
        int max_tokens = params.max_tokens > 0 ? params.max_tokens : (n_text_ctx - sot_sequence_len);
        int n_vocab = whisper_n_vocab(ctx);

        std::vector<whisper_token> decoded_tokens;
        std::vector<std::vector<float>> per_token_avg_attn;
        int attn_n_audio_ctx_saved = 0;

        double cross_attn_copy_us = 0.0, cross_attn_calc_us = 0.0, iter_decode_us = 0.0;
        int n_decoded_tokens = 0;

        int stop_token_idx = -1;
        int final_most_attended = last_attend_frame;
        bool rewind_detected = false;
        std::vector<int> token_attend_frames;

        for (int step = 0; step < max_tokens; step++) {
            float * logits_all = whisper_get_logits(ctx);
            float * logits = logits_all + (last_decode_n_tokens - 1) * n_vocab;

            whisper_token next_token = 0;
            float max_logit = logits[0];
            for (int v = 1; v < n_vocab; v++) {
                if (logits[v] > max_logit) { max_logit = logits[v]; next_token = v; }
            }
            if (next_token >= whisper_token_eot(ctx)) break;
            decoded_tokens.push_back(next_token);

            auto t_dec_s = std::chrono::high_resolution_clock::now();
            if (whisper_decode(ctx, &next_token, 1, n_past, params.n_threads) != 0) {
                fprintf(stderr, "%s: failed to decode token at step %d\n", __func__, step); break;
            }
            auto t_dec_e = std::chrono::high_resolution_clock::now();
            iter_decode_us += std::chrono::duration<double, std::micro>(t_dec_e - t_dec_s).count();
            n_decoded_tokens++;
            n_past++;
            last_decode_n_tokens = 1;

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
                    std::vector<float> avg(attn_n_audio_ctx, 0.0f);
                    for (int h = 0; h < attn_n_aheads; h++)
                        for (int a = 0; a < attn_n_audio_ctx; a++)
                            avg[a] += attn_data[a + h * attn_n_audio_ctx];
                    float inv = 1.0f / attn_n_aheads;
                    for (auto & v : avg) v *= inv;
                    per_token_avg_attn.push_back(std::move(avg));
                    attn_ok = true;
                }
                auto t_calc_e = std::chrono::high_resolution_clock::now();
                cross_attn_calc_us += std::chrono::duration<double, std::micro>(t_calc_e - t_calc_s).count();
            }

            if (attn_ok) {
                int n_tok = (int)per_token_avg_attn.size();
                int n_ctx = attn_n_audio_ctx_saved;
                std::vector<float> attn_mat(n_tok * n_ctx, 0.0f);
                for (int t = 0; t < n_tok; t++)
                    for (int a = 0; a < n_ctx && a < (int)per_token_avg_attn[t].size(); a++)
                        attn_mat[t * n_ctx + a] = per_token_avg_attn[t][a];

                if (n_tok >= 2) zscore_normalize(attn_mat, 1, n_tok, n_ctx);
                median_filter(attn_mat, 1, n_tok, n_ctx, 7);

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

                if (!is_last && last_attend_frame - most_attended > params.rewind_threshold) {
                    whisper_token prev_token = (decoded_tokens.size() >= 2)
                        ? decoded_tokens[decoded_tokens.size() - 2] : sot_seq.back();
                    if (prev_token >= whisper_token_eot(ctx)) {
                        if (params.debug) fprintf(stderr, "  [token %d] omit rewinding from special tokens\n", step);
                    } else {
                        fprintf(stderr, "[iter=%d] rewind detected: last_attend=%d, most_attended=%d\n",
                                n_iter, last_attend_frame, most_attended);
                        rewind_detected = true;
                        last_attend_frame = -params.rewind_threshold;
                        break;
                    }
                }
                final_most_attended = most_attended;

                int threshold = is_last ? 4 : params.frame_threshold;
                if (content_mel_len - most_attended <= threshold) {
                    if (params.debug) {
                        fprintf(stderr, "  [token %d] STOP: %d - %d = %d <= %d\n",
                                step, content_mel_len, most_attended, content_mel_len - most_attended, threshold);
                    }
                    stop_token_idx = n_tok - 1;
                    break;
                }
            }
            next_step:;
        }

        whisper_print_timings(ctx);
        fprintf(stderr, "[ITER_TIMING] iter=%d mel=%.2f ms encode=%.2f ms prefill=%.2f ms decode=%.2f ms n_tokens=%d\n",
                n_iter, iter_mel_us / 1000.0, iter_encode_us / 1000.0,
                iter_prefill_us / 1000.0, iter_decode_us / 1000.0, n_decoded_tokens);
        fprintf(stderr, "[CROSS_ATTN_TIME] iter=%d copy=%.2f ms calc=%.2f ms\n",
                n_iter, cross_attn_copy_us / 1000.0, cross_attn_calc_us / 1000.0);

        if (decoded_tokens.empty()) {
            if (params.debug) fprintf(stderr, "[iter=%d] no tokens decoded\n", n_iter);
            token_segments.push_back({});
            n_iter++; continue;
        }
        if (rewind_detected) {
            token_segments.push_back({});
            n_iter++; continue;
        }

        std::vector<whisper_token> tokens_for_emission;
        if (stop_token_idx >= 0) {
            tokens_for_emission.insert(tokens_for_emission.end(),
                                        decoded_tokens.begin(), decoded_tokens.begin() + stop_token_idx);
        } else {
            tokens_for_emission = decoded_tokens;
        }

        std::vector<whisper_token> emitted = emit_with_cif(ctx, tokens_for_emission, is_last, fire_detected);

        auto t_now = std::chrono::high_resolution_clock::now();
        double emission_wall_time = std::chrono::duration<double>(t_now - t_start).count();

        token_segments.push_back(emitted);

        if (!emitted.empty()) {
            for (int ei = 0; ei < (int)emitted.size(); ei++) {
                committed_tokens.push_back(emitted[ei]);
                const char * str = whisper_token_to_str(ctx, emitted[ei]);
                if (str) {
                    committed_words.push_back(str);
                    int attend_frame = (ei < (int)token_attend_frames.size()) ? token_attend_frames[ei] : -1;
                    double token_start_time = 0.0, token_end_time = 0.0;
                    if (attend_frame >= 0) {
                        int64_t abs_sample = buffer_start_sample + (int64_t)attend_frame * 320;
                        token_start_time = (double)abs_sample / WHISPER_SAMPLE_RATE;
                        token_end_time   = (double)(abs_sample + 320) / WHISPER_SAMPLE_RATE;
                    }
                    double latency = std::max(0.0, emission_wall_time - token_end_time);
                    latency_record.push_back(std::make_tuple(token_start_time, token_end_time, std::string(str), latency, emission_wall_time));
                    printf("%s", str);
                    fflush(stdout);
                }
            }
        }

        if (!rewind_detected && final_most_attended >= 0) last_attend_frame = final_most_attended;

        fprintf(stderr, "[iter=%d] fire=%s, emitted %zu tokens, last_attend=%d, content_mel_len=%d\n",
                n_iter, fire_detected ? "true" : "false", emitted.size(), last_attend_frame, content_mel_len);

        n_iter++;
    }

    printf("\n");
    print_tsw_with_latency(latency_record);
    print_committed(committed_words);
    fprintf(stderr, "\n%s: CIF stats: fire=%d, drop=%d (total=%d)\n",
            __func__, cif_fire_count, cif_drop_count, cif_fire_count + cif_drop_count);

    whisper_free(ctx);
    return 0;
}
