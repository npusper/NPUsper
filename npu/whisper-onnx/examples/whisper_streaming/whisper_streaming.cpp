// whisper-streaming C++ implementation (ONNX Runtime backend)
// Ported from whisperflow-ggml/examples/whisper_streaming_cpp/
// Changes: whisper.h → whisper_ort.h, removed ggml.h/DTW/ggml_time_us
//
#include "common.h"
#include "whisper_ort.h"
#include "buffer.h"

#include <cassert>
#include <cstdio>
#include <regex>
#include <string>
#include <thread>
#include <vector>
#include <fstream>
#include <cstring>
#include <chrono>

static int64_t get_time_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// command-line parameters
struct whisper_params {
    int32_t n_threads  = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t step_ms    = 3000;
    int32_t length_ms  = 10000;
    int32_t keep_ms    = 200;
    int32_t capture_id = -1;
    int32_t max_tokens = 32;
    int32_t best_of       = whisper_full_default_params(WHISPER_SAMPLING_GREEDY).greedy.best_of;
    int32_t beam_size     = whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH).beam_search.beam_size;
    int32_t max_round_decode = 9999;

    float vad_thold    = 0.6f;
    float freq_thold   = 100.0f;

    bool translate     = false;
    bool no_fallback   = false;
    bool print_special = false;
    bool no_context    = true;
    bool no_timestamps = false;
    bool use_gpu       = true;
    bool no_realtime   = false;  // skip sleep, process chunks as fast as possible

    std::string language  = "en";
    std::string model     = "models/base";

    std::string audio_tag = "";

    std::vector<std::string> fname_inp = {};
    std::vector<std::string> fname_out = {};
};

void whisper_print_usage(int argc, char ** argv, const whisper_params & params);

bool whisper_params_parse(int argc, char ** argv, whisper_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-"){
            params.fname_inp.push_back(arg);
            continue;
        }

        if (arg[0] != '-') {
            params.fname_inp.push_back(arg);
            continue;
        }

        if (arg == "-h" || arg == "--help") {
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
        else if (arg == "-t"    || arg == "--threads")       { params.n_threads     = std::stoi(argv[++i]); }
        else if (                  arg == "--step")          { params.step_ms       = std::stoi(argv[++i]); }
        else if (                  arg == "--length")        { params.length_ms     = std::stoi(argv[++i]); }
        else if (                  arg == "--keep")          { params.keep_ms       = std::stoi(argv[++i]); }
        else if (arg == "-c"    || arg == "--capture")       { params.capture_id    = std::stoi(argv[++i]); }
        else if (arg == "-mt"   || arg == "--max-tokens")    { params.max_tokens    = std::stoi(argv[++i]); }
        else if (arg == "-bo"   || arg == "--best-of")         { params.best_of         = std::stoi(argv[++i]); }
        else if (arg == "-bs"   || arg == "--beam-size")       { params.beam_size       = std::stoi(argv[++i]); }
        else if (arg == "-mrd" || arg == "--max-round-decode")  { params.max_round_decode = std::stoi(argv[++i]); }
        else if (arg == "-vth"  || arg == "--vad-thold")     { params.vad_thold     = std::stof(argv[++i]); }
        else if (arg == "-fth"  || arg == "--freq-thold")    { params.freq_thold    = std::stof(argv[++i]); }
        else if (arg == "-tr"   || arg == "--translate")     { params.translate     = true; }
        else if (arg == "-nf"   || arg == "--no-fallback")   { params.no_fallback   = true; }
        else if (arg == "-ps"   || arg == "--print-special") { params.print_special = true; }
        else if (arg == "-kc"   || arg == "--keep-context")  { params.no_context    = false; }
        else if (arg == "-at"   || arg == "--audio-tag")       { params.audio_tag       = argv[++i];}
        else if (arg == "-l"    || arg == "--language")      { params.language      = argv[++i]; }
        else if (arg == "-m"    || arg == "--model")         { params.model         = argv[++i]; }
        else if (arg == "-of"   || arg == "--output-file")     { params.fname_out.emplace_back(argv[++i]); }
        else if (arg == "-ng"   || arg == "--no-gpu")        { params.use_gpu       = false; }
        else if (                  arg == "--no-realtime")   { params.no_realtime   = true; }
        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}

void whisper_print_usage(int /*argc*/, char ** argv, const whisper_params & params) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,       --help          show this help message and exit\n");
    fprintf(stderr, "  -t N,     --threads N     [%-7d] number of threads to use during computation\n",    params.n_threads);
    fprintf(stderr, "            --step N        [%-7d] audio step size in milliseconds\n",                params.step_ms);
    fprintf(stderr, "            --length N      [%-7d] audio length in milliseconds\n",                   params.length_ms);
    fprintf(stderr, "            --keep N        [%-7d] audio to keep from previous step in ms\n",         params.keep_ms);
    fprintf(stderr, "  -c ID,    --capture ID    [%-7d] capture device ID\n",                              params.capture_id);
    fprintf(stderr, "  -mt N,    --max-tokens N  [%-7d] maximum number of tokens per audio chunk\n",       params.max_tokens);
    fprintf(stderr, "  -mrd N,   --max-round-decode N [%-7d] max decode rounds per chunk\n",               params.max_round_decode);
    fprintf(stderr, "  -vth N,   --vad-thold N   [%-7.2f] voice activity detection threshold\n",           params.vad_thold);
    fprintf(stderr, "  -fth N,   --freq-thold N  [%-7.2f] high-pass frequency cutoff\n",                   params.freq_thold);
    fprintf(stderr, "  -tr,      --translate     [%-7s] translate from source language to english\n",      params.translate ? "true" : "false");
    fprintf(stderr, "  -nf,      --no-fallback   [%-7s] do not use temperature fallback while decoding\n", params.no_fallback ? "true" : "false");
    fprintf(stderr, "  -ps,      --print-special [%-7s] print special tokens\n",                           params.print_special ? "true" : "false");
    fprintf(stderr, "  -kc,      --keep-context  [%-7s] keep context between audio chunks\n",              params.no_context ? "false" : "true");
    fprintf(stderr, "  -at AUDIO_TAG_PATH --audio-tag AUDIO_TAG_PATH attach audio tag\n");
    fprintf(stderr, "  -l LANG,  --language LANG [%-7s] spoken language\n",                                params.language.c_str());
    fprintf(stderr, "  -m FNAME, --model FNAME   [%-7s] model path (ONNX model directory)\n",              params.model.c_str());
    fprintf(stderr, "  -of FNAME, --output-file FNAME output file path\n");
    fprintf(stderr, "  -ng,      --no-gpu        [%-7s] disable GPU inference\n",                          params.use_gpu ? "false" : "true");
    fprintf(stderr, "            --no-realtime   [%-7s] skip sleep, process as fast as possible\n",        params.no_realtime ? "true" : "false");
    fprintf(stderr, "\n");
}

bool get_audio_chunk(const std::vector<float> &pcmf32_all, std::vector<float> &pcmf32_new, int64_t pcmf32_index, int step_ms, int sample_rate) {
    int64_t pcmf32_index_sample = (pcmf32_index * sample_rate) / 1000;
    int num_samples = (step_ms * sample_rate) / 1000;
    pcmf32_new.clear();

    if (pcmf32_index + num_samples > (int64_t)pcmf32_all.size()) {
        num_samples = pcmf32_all.size() - pcmf32_index;
    }
    printf("Get new chunk of audio start from %lld and end with %lld.\n", (long long)pcmf32_index_sample, (long long)pcmf32_index_sample + num_samples);
    bool has_more_audio = true;
    if (pcmf32_index_sample + num_samples >= (int64_t)pcmf32_all.size()) {
        has_more_audio = false;
        pcmf32_new.insert(pcmf32_new.end(), pcmf32_all.begin()+pcmf32_index_sample, pcmf32_all.end());
    } else {
        pcmf32_new.insert(pcmf32_new.end(), pcmf32_all.begin()+pcmf32_index_sample,
                                            pcmf32_all.begin()+pcmf32_index_sample+num_samples);
    }
    return has_more_audio;
}

void precise_sleep(double seconds) {
    struct timespec start, end;
    double elapsed;
    clock_gettime(CLOCK_MONOTONIC, &start);
    do {
        clock_gettime(CLOCK_MONOTONIC, &end);
        elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    } while (elapsed < seconds);
}

std::vector<float> readCSVToVector(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<float> data;
    std::string line;
    if (!file.is_open()) {
        fprintf(stderr, "error: failed to read audio tag file\n");
        return data;
    }
    while (std::getline(file, line)) {
        try { data.push_back(std::stof(line)); }
        catch (const std::invalid_argument&) { fprintf(stderr, "error: failed to read data in audio tag\n"); }
    }
    file.close();
    return data;
}

void print_tsw(const std::vector<std::tuple<double, double, std::string>>& committed) {
    for (const auto& entry : committed) {
        double start_time, end_time;
        std::string transcript;
        std::tie(start_time, end_time, transcript) = entry;
        std::cout << "Start Time: " << start_time << ", End Time: " << end_time
                  << ", Transcript: " << transcript << std::endl;
    }
}

void print_tsw_with_token_latency(const std::vector<std::tuple<double, double, std::string, double, double>>& committed) {
    int count = 0;
    double latency_sum = 0;
    for (const auto& entry : committed) {
        double start_time = std::get<0>(entry);
        double end_time   = std::get<1>(entry);
        const std::string & transcript = std::get<2>(entry);
        double latency       = std::get<3>(entry);
        double emission_time = std::get<4>(entry);
        std::cout << "Start Time: " << start_time << ", End Time: " << end_time
                  << ", Transcript: " << transcript << ", Latency: " << latency
                  << ", Emission Time: " << emission_time << std::endl;
        count++;
        latency_sum += latency;
    }
    if (count > 0) std::cout << "Average latency: " << latency_sum / count << std::endl;
}

std::vector<std::tuple<double, double, std::string>> output_word_level_timestamp(
                struct whisper_context * ctx,
                const whisper_params & /*params*/,
                bool   full) {
    const int n_segments = whisper_full_n_segments(ctx);
    std::vector<std::tuple<double, double, std::string>> records;
    for (int i = 0; i < n_segments; ++i) {
        if (full) {
            const int n = whisper_full_n_tokens(ctx, i);
            for (int j = 0; j < n; ++j) {
                auto token = whisper_full_get_token_data(ctx, i, j);
                auto word_tmp = whisper_token_to_str(ctx, token.id);
                // Use t0 as timestamp (t_dtw not available in ORT backend)
                auto time_start = token.t0 / 100.0 - 0.01;
                auto time_end   = token.t0 / 100.0 + 0.01;
                records.push_back(std::make_tuple(time_start, time_end, word_tmp));
            }
        }
    }
    return records;
}

std::vector<whisper_token> prompt(struct whisper_context* ctx,
    const std::vector<std::tuple<double, double, std::string>>& committed,
    double buffer_time_offset)
{
    size_t k = committed.size() > 0 ? committed.size() - 1 : 0;
    while (k > 0 && std::get<1>(committed[k-1]) > buffer_time_offset) k--;

    std::vector<std::string> p;
    for (size_t i = 0; i < k; i++) p.push_back(std::get<2>(committed[i]));

    std::vector<std::string> prompt_vec;
    int l = 0;
    while (!p.empty() && l < 200) {
        std::string x = p.back(); p.pop_back(); l++;
        prompt_vec.push_back(x);
    }
    std::reverse(prompt_vec.begin(), prompt_vec.end());

    std::vector<whisper_token> token_ids;
    for (const auto& token : prompt_vec) {
        auto it = whisper_token_to_id(ctx, token.c_str());
        if (it != -1) token_ids.push_back(it);
        else fprintf(stderr, "%s: Token not found in vocab %s\n", __func__, token.c_str());
    }
    return token_ids;
}

std::vector<int64_t> get_end_time_of_res(struct whisper_context* ctx) {
    std::vector<int64_t> segment_end_time;
    int n_segments = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_segments; ++i) {
        segment_end_time.push_back(whisper_full_get_segment_t1(ctx, i));
    }
    return segment_end_time;
}

void chunk_completed_segment(std::vector<int64_t>& segment_end_time,
                             std::vector<std::tuple<double, double, std::string>>& commited,
                             std::vector<float>& audio_buffer,
                             HypothesisBuffer &transcript_buffer,
                             double& buffer_time_offset) {
    if (commited.empty()) return;

    std::vector<int64_t>& ends = segment_end_time;
    double t = std::get<1>(commited.back());

    if (ends.size() > 1) {
        double e = ends[ends.size() - 2] / 100.0 + buffer_time_offset;
        while (ends.size() > 2 && e > t) {
            ends.pop_back();
            e = ends[ends.size() - 2] / 100.0 + buffer_time_offset;
        }
        if (e <= t) {
            fprintf(stderr, "%s: segement chunked at %f\n", __func__, e);
            transcript_buffer.pop_committed(e);
            double cut_seconds = e - buffer_time_offset;
            audio_buffer.erase(audio_buffer.begin(), audio_buffer.begin() + static_cast<int>(cut_seconds * WHISPER_SAMPLE_RATE));
            buffer_time_offset = e;
        } else {
            fprintf(stderr, "%s: last segment not within committed area\n", __func__);
        }
    } else {
        fprintf(stderr, "%s: not enough segments to chunk\n", __func__);
    }
}

std::tuple<double, double, std::string> to_flush(
    const std::vector<std::tuple<double, double, std::string>>& sents,
    double offset = 0.0)
{
    if (sents.empty()) return std::make_tuple(0, 0, "");
    std::string concatenated;
    for (size_t i = 0; i < sents.size(); ++i) {
        if (i > 0) concatenated += " ";
        concatenated += std::get<2>(sents[i]);
    }
    double beg = offset + std::get<0>(sents[0]);
    double end = offset + std::get<1>(sents.back());
    return std::make_tuple(beg, end, concatenated);
}

int main(int argc, char ** argv) {
    whisper_params params;

    if (whisper_params_parse(argc, argv, params) == false) return 1;

    params.keep_ms   = std::min(params.keep_ms,   params.step_ms);
    params.length_ms = std::max(params.length_ms, params.step_ms);

    const int n_samples_step = (1e-3*params.step_ms  )*WHISPER_SAMPLE_RATE;
    const int n_samples_len  = (1e-3*params.length_ms)*WHISPER_SAMPLE_RATE;
    const int n_samples_keep = (1e-3*params.keep_ms  )*WHISPER_SAMPLE_RATE;
    const int n_samples_30s  = (1e-3*30000.0         )*WHISPER_SAMPLE_RATE;
    (void)n_samples_len; (void)n_samples_keep;

    const bool use_vad = n_samples_step <= 0;
    const int n_new_line = 1;

    params.no_timestamps = true;   // ORT whisper_full is simple greedy; no timestamp token parsing
    params.no_context    |= use_vad;
    params.max_tokens     = 0;

    if (params.language != "auto" && whisper_lang_id(params.language.c_str()) == -1){
        fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
        whisper_print_usage(argc, argv, params);
        exit(0);
    }

    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu    = params.use_gpu;
    cparams.flash_attn = false;

    struct whisper_context * ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "error: failed to initialize whisper context\n");
        return 1;
    }

    std::vector<float> pcmf32    (n_samples_30s, 0.0f);
    std::vector<float> pcmf32_new(n_samples_30s, 0.0f);
    std::vector<whisper_token> prompt_tokens;

    {
        fprintf(stderr, "\n");
        if (!whisper_is_multilingual(ctx)) {
            if (params.language != "en" || params.translate) {
                params.language = "en"; params.translate = false;
                fprintf(stderr, "%s: WARNING: model is not multilingual, ignoring language and translation options\n", __func__);
            }
        }
        fprintf(stderr, "%s: processing %d samples (step = %.1f sec / len = %.1f sec / keep = %.1f sec), %d threads, lang = %s, task = %s, timestamps = %d ...\n",
                __func__, n_samples_step,
                float(n_samples_step)/WHISPER_SAMPLE_RATE,
                float(params.length_ms*16)/1000.0f,
                float(n_samples_keep)/WHISPER_SAMPLE_RATE,
                params.n_threads, params.language.c_str(),
                params.translate ? "translate" : "transcribe",
                params.no_timestamps ? 0 : 1);
        if (!use_vad) fprintf(stderr, "%s: n_new_line = %d, no_context = %d\n", __func__, n_new_line, params.no_context);
        fprintf(stderr, "\n");
    }

    int n_iter = 0;
    bool is_running = true;

    fflush(stdout);

    const auto t_start_us = get_time_us();

    const auto fname_inp = params.fname_inp[0];
    const auto fname_out = 0 < (int)params.fname_out.size() && !params.fname_out[0].empty() ? params.fname_out[0] : params.fname_inp[0];
    (void)fname_out;

    std::vector<float> pcmf32_all;
    std::vector<std::vector<float>> pcmf32s;

    if (!::read_wav(fname_inp, pcmf32_all, pcmf32s, false)) {
        fprintf(stderr, "error: failed to read WAV file '%s'\n", fname_inp.c_str());
    }

    std::vector<float> pcmf32_audio_tag;
    if (params.audio_tag != "") {
        pcmf32_audio_tag = readCSVToVector(params.audio_tag);
    }

    {
        fprintf(stderr, "\n");
        fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
                params.n_threads, (int)std::thread::hardware_concurrency(),
                whisper_print_system_info());

        fprintf(stderr, "\n");
        fprintf(stderr, "%s: processing '%s' (%d samples, %.1f sec), %d threads, %d beams + best of %d, lang = %s, task = %s ...\n",
                __func__, fname_inp.c_str(), int(pcmf32_all.size()), float(pcmf32_all.size())/WHISPER_SAMPLE_RATE,
                params.n_threads, params.beam_size, params.best_of,
                params.language.c_str(),
                params.translate ? "translate" : "transcribe");
        fprintf(stderr, "\n");
    }

    int64_t pcmf32_index = 0;
    const auto t_ref_us = get_time_us();
    int64_t start = t_ref_us / 1000 - pcmf32_index;
    int64_t pcmf32_index_end = 0;
    int64_t now = 0;

    std::vector<float> pcmf32_audio_buffer;
    HypothesisBuffer transcript_buffer;
    double buffer_time_offset = 0;
    std::vector<std::tuple<double, double, std::string>> committed;
    std::vector<std::tuple<double, double, std::string, double, double>> latency_record;

    bool is_last_iter = false;
    while (is_running || !is_last_iter) {

        if (!use_vad) {
            if (!is_running) {
                is_last_iter = true;
                fprintf(stderr, "[iter=%d] Audio ended, final inference on buffer (%.2fs)\n",
                        n_iter, (float)pcmf32_audio_buffer.size()/WHISPER_SAMPLE_RATE);
            } else {
                if (params.no_realtime) {
                    pcmf32_index_end += params.step_ms;
                } else {
                    now = get_time_us() / 1000 - start;
                    if (now < pcmf32_index_end + params.step_ms) {
                        precise_sleep((params.step_ms + pcmf32_index_end - now) / 1000.0);
                    }
                    pcmf32_index_end = get_time_us() / 1000 - start;
                }
                is_running = get_audio_chunk(pcmf32_all, pcmf32_new, pcmf32_index, pcmf32_index_end - pcmf32_index, WHISPER_SAMPLE_RATE);
                pcmf32_index = pcmf32_index_end;
                pcmf32_audio_buffer.insert(pcmf32_audio_buffer.end(), pcmf32_new.begin(), pcmf32_new.end());
            }
            fprintf(stderr, "[BUFFER_STATS] iter=%d buffer_sec=%.3f buffer_samples=%ld\n", n_iter, (float)pcmf32_audio_buffer.size()/WHISPER_SAMPLE_RATE, (long)pcmf32_audio_buffer.size());

            if (pcmf32_audio_buffer.size() > 30 * WHISPER_SAMPLE_RATE) {
                fprintf(stderr, "%s: WARNING: audio buffer exceeded 30s (%.1fs), stopping.\n", __func__, (float)pcmf32_audio_buffer.size()/WHISPER_SAMPLE_RATE);
                is_running = false;
                break;
            }

            pcmf32 = pcmf32_audio_buffer;
            prompt_tokens = prompt(ctx, committed, buffer_time_offset);
        } else {
            fprintf(stderr, "The use_vad version is not implemented for now.\n");
        }

        {
            whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
            wparams.strategy = (params.beam_size > 1) ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY;
            wparams.print_progress   = false;
            wparams.print_special    = params.print_special;
            wparams.print_realtime   = false;
            wparams.print_timestamps = !params.no_timestamps;
            wparams.translate        = params.translate;
            wparams.single_segment   = false;
            wparams.max_tokens       = params.max_tokens;
            wparams.language         = params.language.c_str();
            wparams.n_threads        = params.n_threads;
            wparams.greedy.best_of        = params.best_of;
            wparams.beam_search.beam_size = params.beam_size;
            wparams.temperature_inc  = params.no_fallback ? 0.0f : wparams.temperature_inc;

            if (params.audio_tag != "") {
                std::vector<float> audio_with_tag;
                audio_with_tag.reserve(pcmf32.size() + pcmf32_audio_tag.size());
                audio_with_tag.insert(audio_with_tag.end(), pcmf32.begin(), pcmf32.end());
                audio_with_tag.insert(audio_with_tag.end(), pcmf32_audio_tag.begin(), pcmf32_audio_tag.end());
                pcmf32 = audio_with_tag;
            }

            if (pcmf32.size() < (size_t)n_samples_step) {
                pcmf32.resize(n_samples_step, 0.0f);
            }

            printf("\n");
            printf("Start new round of inference, data length %ld, buffer offset %f.\n", (long)pcmf32.size(), buffer_time_offset);

            int content_mel_len = (int)(pcmf32.size() / 320);
            fprintf(stderr, "[INPUT_STATS] iter=%d input_sec=%.3f input_samples=%ld content_mel_len=%d\n",
                   n_iter, (float)pcmf32.size()/WHISPER_SAMPLE_RATE, (long)pcmf32.size(), content_mel_len);

            if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
                fprintf(stderr, "%s: failed to process audio\n", argv[0]);
                return 6;
            }

            whisper_print_timings(ctx);

            std::vector<std::tuple<double, double, std::string>> tsw = output_word_level_timestamp(ctx, params, true);
            printf("New round of transcript length: %d\n", (int)tsw.size());
            transcript_buffer.insert(tsw, buffer_time_offset);
            transcript_buffer.print_info();
            std::vector<std::tuple<double, double, std::string>> o = transcript_buffer.flush();
            committed.insert(committed.end(), o.begin(), o.end());

            std::vector<std::tuple<double, double, std::string>> r_o = transcript_buffer.complete();
            if (is_last_iter) {
                committed.insert(committed.end(), r_o.begin(), r_o.end());
            }
            std::tuple<double, double, std::string> completed = to_flush(o, buffer_time_offset);
            std::tuple<double, double, std::string> the_rest  = to_flush(r_o, buffer_time_offset);
            printf("COMPLETE NOW: %s\n", std::get<2>(completed).c_str());
            printf("INCOMPLETE: %s\n",   std::get<2>(the_rest).c_str());

            double step_end = (get_time_us() / 1000.0 - start) / 1000.0;
            for (auto& token : o) {
                double start_time, end_time;
                std::string transcript;
                std::tie(start_time, end_time, transcript) = token;
                double latency = std::max(0.0, step_end - end_time);
                latency_record.push_back(std::make_tuple(start_time, end_time, transcript, latency, step_end));
            }

            int64_t s = 15;
            std::vector<int64_t> segment_end_time = get_end_time_of_res(ctx);
            if (pcmf32_audio_buffer.size() > (size_t)(s * WHISPER_SAMPLE_RATE)) {
                chunk_completed_segment(segment_end_time, committed, pcmf32_audio_buffer, transcript_buffer, buffer_time_offset);
                fprintf(stderr, "[BUFFER_STATS_AFTER_TRIM] iter=%d buffer_sec=%.3f\n", n_iter, (float)pcmf32_audio_buffer.size()/WHISPER_SAMPLE_RATE);
            }

            now = get_time_us() / 1000 - start;

            {
                if (!use_vad) {
                    printf("\33[2K\r");
                    printf("%s", std::string(100, ' ').c_str());
                    printf("\33[2K\r");
                }

                const int n_segments = whisper_full_n_segments(ctx);
                for (int i = 0; i < n_segments; ++i) {
                    const char * text = whisper_full_get_segment_text(ctx, i);
                    if (params.no_timestamps) {
                        printf("%s", text);
                        fflush(stdout);
                    } else {
                        const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
                        const int64_t t1 = whisper_full_get_segment_t1(ctx, i);
                        std::string output = "[" + to_timestamp(t0, false) + " --> " + to_timestamp(t1, false) + "]  " + text;
                        if (whisper_full_get_segment_speaker_turn_next(ctx, i)) output += " [SPEAKER_TURN]";
                        output += "\n";
                        printf("%s", output.c_str());
                        fflush(stdout);
                    }
                }
            }

            ++n_iter;
            fflush(stdout);

            if (is_last_iter) break;
        }
    }

    print_tsw(committed);
    print_tsw_with_token_latency(latency_record);
    whisper_print_timings(ctx);
    whisper_free(ctx);

    return 0;
}
