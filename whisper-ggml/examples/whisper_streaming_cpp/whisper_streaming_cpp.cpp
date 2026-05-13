// Real-time speech recognition of input from a microphone
//
// A very quick-n-dirty implementation serving mainly as a proof of concept.
//
#include "common.h"
#include "whisper.h"
#include "ggml.h"
#include "examples/whisper_streaming_cpp/buffer.h"

#include <cassert>
#include <cstdio>
#include <regex>
#include <string>
#include <thread>
#include <vector>
#include <fstream>
#include <cstring>


// command-line parameters
struct whisper_params {
    int32_t n_threads  = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t step_ms    = 3000;
    int32_t length_ms  = 10000;
    int32_t keep_ms    = 200;
    int32_t capture_id = -1;
    int32_t max_tokens = 32;
    int32_t audio_ctx  = 0;
    int32_t best_of       = whisper_full_default_params(WHISPER_SAMPLING_GREEDY).greedy.best_of;
    int32_t beam_size     = whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH).beam_search.beam_size;
    int32_t max_round_decode = 9999;

    float vad_thold    = 0.6f;
    float freq_thold   = 100.0f;

    bool speed_up      = false;
    bool translate     = false;
    bool no_fallback   = false;
    bool print_special = false;
    bool no_context    = true;
    bool no_timestamps = false;
    bool tinydiarize   = false;
    bool save_audio    = false; // save audio to wav file
    bool use_gpu       = true;
    bool flash_attn    = false;
    bool no_realtime   = false;  // skip sleep, process chunks as fast as possible

    std::string language  = "en";
    std::string model     = "models/ggml-base.en.bin";

    std::string dtw = "";
    std::string audio_tag = "";

    std::vector<std::string> fname_inp = {};
    std::vector<std::string> fname_out = {};
};

void whisper_print_usage(int argc, char ** argv, const whisper_params & params);

bool whisper_params_parse(int argc, char ** argv, whisper_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        // For reading audio files
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
        else if (arg == "-ac"   || arg == "--audio-ctx")     { params.audio_ctx     = std::stoi(argv[++i]); }
        else if (arg == "-mrd" || arg == "--max-round-decode")  { params.max_round_decode = std::stoi(argv[++i]); }
        else if (arg == "-vth"  || arg == "--vad-thold")     { params.vad_thold     = std::stof(argv[++i]); }
        else if (arg == "-fth"  || arg == "--freq-thold")    { params.freq_thold    = std::stof(argv[++i]); }
        else if (arg == "-su"   || arg == "--speed-up")      { params.speed_up      = true; }
        else if (arg == "-tr"   || arg == "--translate")     { params.translate     = true; }
        else if (arg == "-nf"   || arg == "--no-fallback")   { params.no_fallback   = true; }
        else if (arg == "-ps"   || arg == "--print-special") { params.print_special = true; }
        else if (arg == "-kc"   || arg == "--keep-context")  { params.no_context    = false; }
        else if (arg == "-dtw"  || arg == "--dtw")             { params.dtw             = argv[++i]; }
        else if (arg == "-at"   || arg == "--audio-tag")       { params.audio_tag       = argv[++i];}
        else if (arg == "-l"    || arg == "--language")      { params.language      = argv[++i]; }
        else if (arg == "-m"    || arg == "--model")         { params.model         = argv[++i]; }
        //else if (arg == "-f"    || arg == "--file")          { params.fname_out     = argv[++i]; }
        else if (arg == "-of"   || arg == "--output-file")     { params.fname_out.emplace_back(argv[++i]); }
        else if (arg == "-tdrz" || arg == "--tinydiarize")   { params.tinydiarize   = true; }
        else if (arg == "-sa"   || arg == "--save-audio")    { params.save_audio    = true; }
        else if (arg == "-ng"   || arg == "--no-gpu")        { params.use_gpu       = false; }
        else if (arg == "-fa"   || arg == "--flash-attn")    { params.flash_attn    = true; }
        else if (               arg == "--no-realtime")      { params.no_realtime   = true; }

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
    fprintf(stderr, "  -h,       --help          [default] show this help message and exit\n");
    fprintf(stderr, "  -t N,     --threads N     [%-7d] number of threads to use during computation\n",    params.n_threads);
    fprintf(stderr, "            --step N        [%-7d] audio step size in milliseconds\n",                params.step_ms);
    fprintf(stderr, "            --length N      [%-7d] audio length in milliseconds\n",                   params.length_ms);
    fprintf(stderr, "            --keep N        [%-7d] audio to keep from previous step in ms\n",         params.keep_ms);
    fprintf(stderr, "  -c ID,    --capture ID    [%-7d] capture device ID\n",                              params.capture_id);
    fprintf(stderr, "  -mt N,    --max-tokens N  [%-7d] maximum number of tokens per audio chunk\n",       params.max_tokens);
    fprintf(stderr, "  -mrd N,    --max-round-decode N  [%-7d] maximum round of decoding per audio chunk\n",       params.max_round_decode);
    fprintf(stderr, "  -ac N,    --audio-ctx N   [%-7d] audio context size (0 - all)\n",                   params.audio_ctx);
    fprintf(stderr, "  -vth N,   --vad-thold N   [%-7.2f] voice activity detection threshold\n",           params.vad_thold);
    fprintf(stderr, "  -fth N,   --freq-thold N  [%-7.2f] high-pass frequency cutoff\n",                   params.freq_thold);
    fprintf(stderr, "  -su,      --speed-up      [%-7s] speed up audio by x2 (reduced accuracy)\n",        params.speed_up ? "true" : "false");
    fprintf(stderr, "  -tr,      --translate     [%-7s] translate from source language to english\n",      params.translate ? "true" : "false");
    fprintf(stderr, "  -nf,      --no-fallback   [%-7s] do not use temperature fallback while decoding\n", params.no_fallback ? "true" : "false");
    fprintf(stderr, "  -ps,      --print-special [%-7s] print special tokens\n",                           params.print_special ? "true" : "false");
    fprintf(stderr, "  -kc,      --keep-context  [%-7s] keep context between audio chunks\n",              params.no_context ? "false" : "true");
    fprintf(stderr, "  -dtw MODEL --dtw MODEL         [%-7s] compute token-level timestamps\n",                 params.dtw.c_str());
    fprintf(stderr, "  -at AUDIO_TAG_PATH --audio-tag AUDIO_TAG_PATH [%s] attach audio tag at the end of the original audio\n", params.audio_tag.c_str());
    fprintf(stderr, "  -l LANG,  --language LANG [%-7s] spoken language\n",                                params.language.c_str());
    fprintf(stderr, "  -m FNAME, --model FNAME   [%-7s] model path\n",                                     params.model.c_str());
    fprintf(stderr, "  -of FNAME, --output-file FNAME [%-7s] output file path (without file extension)\n",      "");
    fprintf(stderr, "  -tdrz,    --tinydiarize   [%-7s] enable tinydiarize (requires a tdrz model)\n",     params.tinydiarize ? "true" : "false");
    fprintf(stderr, "  -sa,      --save-audio    [%-7s] save the recorded audio to a file\n",              params.save_audio ? "true" : "false");
    fprintf(stderr, "  -ng,      --no-gpu        [%-7s] disable GPU inference\n",                          params.use_gpu ? "false" : "true");
    fprintf(stderr, "  -fa,      --flash-attn    [%-7s] flash attention during inference\n",               params.flash_attn ? "true" : "false");
    fprintf(stderr, "            --no-realtime   [%-7s] skip sleep, process as fast as possible\n",        params.no_realtime ? "true" : "false");
    fprintf(stderr, "\n");
}

// get audio chunk from the full audio read from the file.
// need to initialize the index as pcmf32_index = 0 to make sure it start from the very beginning.
bool get_audio_chunk(const std::vector<float> &pcmf32_all, std::vector<float> &pcmf32_new, int64_t pcmf32_index, int step_ms, int sample_rate) {
    int64_t pcmf32_index_sample = (pcmf32_index * sample_rate) / 1000;
    int num_samples = (step_ms * sample_rate) / 1000;
    pcmf32_new.clear();
    
    if (pcmf32_index + num_samples > pcmf32_all.size()) {
        num_samples = pcmf32_all.size() - pcmf32_index;
    }
    printf("Get new chunk of audio start from %lld and end with %lld.\n", pcmf32_index_sample, pcmf32_index_sample + num_samples);
    // there is no stopping mechanism for now, so the program will only terminate when it reaches the end of the audio
    // with a segfault (access out-of-bound memory)
    bool has_more_audio = true;
    if (pcmf32_index_sample + num_samples >= pcmf32_all.size()) {
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

    // Get the current time
    clock_gettime(CLOCK_MONOTONIC, &start);

    do {
        // Get the current time again
        clock_gettime(CLOCK_MONOTONIC, &end);

        // Calculate the elapsed time in seconds
        elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    } while (elapsed < seconds);
}

// Function to read numbers from a .csv file and store them in a vector
std::vector<float> readCSVToVector(const std::string& filename) {
    std::ifstream file(filename);  // Open the file
    std::vector<float> data;       // Vector to store the numbers
    std::string line;

    // Check if the file was opened successfully
    if (!file.is_open()) {
        fprintf(stderr, "error: failed to read audio tag file\n");
        return data;  // Return an empty vector if the file could not be opened
    }

    // Read each line from the file
    while (std::getline(file, line)) {
        try {
            // Convert the line to a float and store it in the vector
            data.push_back(std::stof(line));
        } catch (const std::invalid_argument& e) {
            fprintf(stderr, "error: failed to read data in audio tag\n");
        }
    }

    // Close the file
    file.close();

    return data;
}

void print_tsw(const std::vector<std::tuple<double, double, std::string>>& committed) {
    for (const auto& entry : committed) {
        double start_time, end_time;
        std::string transcript;
        std::tie(start_time, end_time, transcript) = entry;

        std::cout << "Start Time: " << start_time
                  << ", End Time: " << end_time
                  << ", Transcript: " << transcript
                  << std::endl;
    }
}

void print_tsw_with_token_latency(const std::vector<std::tuple<double, double, std::string, double, double>>& committed) {
    int count = 0;
    double latency_sum = 0;
    for (const auto& entry : committed) {
        double start_time = std::get<0>(entry);
        double end_time = std::get<1>(entry);
        const std::string & transcript = std::get<2>(entry);
        double latency = std::get<3>(entry);
        double emission_time = std::get<4>(entry);

        std::cout << "Start Time: " << start_time
                  << ", End Time: " << end_time
                  << ", Transcript: " << transcript
                  << ", Latency: " << latency
                  << ", Emission Time: " << emission_time
                  << std::endl;
        count++;
        latency_sum += latency;
    }
    if (count > 0) {
        std::cout << "Average latency: " << latency_sum / count << std::endl;
    }
}

std::vector<std::tuple<double, double, std::string>> output_word_level_timestamp(
                struct whisper_context * ctx,
                const whisper_params & params,
                bool   full) {
    const int n_segments = whisper_full_n_segments(ctx);
    std::vector<std::tuple<double, double, std::string>> records;
    for (int i = 0; i < n_segments; ++i) {
        const char * text = whisper_full_get_segment_text(ctx, i);

        const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
        const int64_t t1 = whisper_full_get_segment_t1(ctx, i);

        if (full) {
            const int n = whisper_full_n_tokens(ctx, i);
            for (int j = 0; j < n; ++j) {
                auto token = whisper_full_get_token_data(ctx, i, j);
                auto word_tmp = whisper_token_to_str(ctx, token.id);
                auto time_start = token.t_dtw / 100.0 - 0.01;  // tentatively adopt the estimation of word time as 0.02s. The estimation is from https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/notebooks/Multilingual_ASR.ipynb
                auto time_end = token.t_dtw / 100.0 + 0.01;
                /* if(token.t0 > -1 && token.t1 > -1) {
                    fprintf(stdout, "Begin time: %.2lld, End time: %.2lld, Word: %s\n", time_start, time_end, word_tmp);
                } else {
                    fprintf(stdout, "token timestamps are -1.\n");
                } */
                //fprintf(stdout, "Begin time: %.2f, End time: %.2f, Word: %s\n", time_start, time_end, word_tmp);
                records.push_back(std::make_tuple(time_start, time_end, word_tmp));
            }
        }
    }
    return records;
}


// whisper_streaming function OnlineASRProcessor.prompt
// check committed part and the buffer_time_offset to only select the committed part that before the audio buffer and within a length limit
// Function to get the token IDs
std::vector<whisper_token> prompt(struct whisper_context* ctx,
    const std::vector<std::tuple<double, double, std::string>>& committed, 
    double buffer_time_offset
    ) 
{
    // Initialize k to the last index of the committed vector
    size_t k = committed.size() > 0 ? committed.size() - 1 : 0;

    // Adjust k based on the buffer_time_offset
    while (k > 0 && std::get<1>(committed[k-1]) > buffer_time_offset) {
        k--;
    }

    // Get the committed part up to k
    std::vector<std::string> p;
    for (size_t i = 0; i < k; i++) {
        p.push_back(std::get<2>(committed[i]));
    }

    // Create the prompt by accumulating the last 200 words
    std::vector<std::string> prompt;
    int l = 0;
    while (!p.empty() && l < 200) {
        std::string x = p.back();
        p.pop_back();
        l++;  // count words, not characters
        prompt.push_back(x);
    }

    // Reverse the order of the prompt (from end to beginning)
    std::reverse(prompt.begin(), prompt.end());

    // Convert prompt to token IDs

    //whisper_vocab * vocab = ctx->vocab;
    //whisper_vocab& vocab = ctx->vocab;

    std::vector<whisper_token> token_ids;
    for (const auto& token : prompt) {
        // Look up the token in the vocabulary and add its corresponding ID
        //auto it = vocab.token_to_id.find(token);
        auto it = whisper_token_to_id(ctx, token); // whisper_token_to_id return vocab.token_to_id.find(token)
        if (it != -1) { //whisper_token_end return vocab.token_to_id.end()
            token_ids.push_back(it);
        } else {
            fprintf(stderr, "%s: Token not found in vocab %s\n", __func__, token.c_str());
        }
    }

    return token_ids;
}


// whisper_streaming helper function
// extract the end time point of each segment and return the vector, for chunk_completed_segment function usage
std::vector<int64_t> get_end_time_of_res(struct whisper_context* ctx) {
    std::vector<int64_t> segment_end_time;
    int n_segments = whisper_full_n_segments(ctx);
    int64_t end_time_tmp;
    for (int i = 0; i < n_segments; ++i) {
        end_time_tmp = whisper_full_get_segment_t1(ctx, i);
        segment_end_time.push_back(end_time_tmp);
    }
    return segment_end_time;
}

// whisper_streaming audio buffer management
// 
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
            //std::cout << "--- segment chunked at " << e << std::endl;
            fprintf(stderr, "%s: segement chunked at %f\n", __func__, e);
            // Assuming you have a mechanism to manage transcript_buffer
            transcript_buffer.pop_committed(e); // Add your logic for transcript management
            double cut_seconds = e - buffer_time_offset;
            audio_buffer.erase(audio_buffer.begin(), audio_buffer.begin() + static_cast<int>(cut_seconds * WHISPER_SAMPLE_RATE));
            buffer_time_offset = e;
        } else {
            // std::cout << "--- last segment not within committed area" << std::endl;
            fprintf(stderr, "%s: last segment not within committed area\n", __func__);
        }
    } else {
        //std::cout << "--- not enough segments to chunk" << std::endl;
        fprintf(stderr, "%s: not enough segments to chunk\n", __func__);
    }
}

// whisper_streaming helper function
// to_flush() transferring [(beg, end, word), ...] to the transcript string
std::tuple<double, double, std::string> to_flush(
    const std::vector<std::tuple<double, double, std::string>>& sents,
    double offset = 0.0
) {
    // Handle empty case
    if (sents.empty()) {
        return std::make_tuple(0, 0, "");
    }

    // Concatenate all the strings from sents with the given separator
    std::string concatenated;
    for (size_t i = 0; i < sents.size(); ++i) {
        if (i > 0) {
            concatenated += " "; // Add separator between sentences
        }
        concatenated += std::get<2>(sents[i]);
    }

    // Calculate the beginning and end timestamps
    double beg = offset + std::get<0>(sents[0]);
    double end = offset + std::get<1>(sents.back());

    return std::make_tuple(beg, end, concatenated);
}

int main(int argc, char ** argv) {
    whisper_params params;

    if (whisper_params_parse(argc, argv, params) == false) {
        return 1;
    }

    params.keep_ms   = std::min(params.keep_ms,   params.step_ms);
    params.length_ms = std::max(params.length_ms, params.step_ms);

    const int n_samples_step = (1e-3*params.step_ms  )*WHISPER_SAMPLE_RATE;
    const int n_samples_len  = (1e-3*params.length_ms)*WHISPER_SAMPLE_RATE;
    const int n_samples_keep = (1e-3*params.keep_ms  )*WHISPER_SAMPLE_RATE;
    const int n_samples_30s  = (1e-3*30000.0         )*WHISPER_SAMPLE_RATE;

    const bool use_vad = n_samples_step <= 0; // sliding window mode uses VAD

    //const int n_new_line = !use_vad ? std::max(1, params.length_ms / params.step_ms - 1) : 1; // number of steps to print new line
    // make n_new_line = 1 to update pcmf32_old and prompt every time for now
    const int n_new_line = 1;

    //params.no_timestamps  = !use_vad;
    params.no_timestamps = false;
    params.no_context    |= use_vad;
    params.max_tokens     = 0;

    // init audio
    /* 
    audio_async audio(params.length_ms);
    if (!audio.init(params.capture_id, WHISPER_SAMPLE_RATE)) {
        fprintf(stderr, "%s: audio.init() failed!\n", __func__);
        return 1;
    }

    audio.resume();
    */
    // whisper init
    if (params.language != "auto" && whisper_lang_id(params.language.c_str()) == -1){
        fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
        whisper_print_usage(argc, argv, params);
        exit(0);
    }

    struct whisper_context_params cparams = whisper_context_default_params();

    cparams.use_gpu    = params.use_gpu;
    cparams.flash_attn = params.flash_attn;

    if (!params.dtw.empty()) {
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
    }

    struct whisper_context * ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);

    std::vector<float> pcmf32    (n_samples_30s, 0.0f);
    std::vector<float> pcmf32_old;
    std::vector<float> pcmf32_new(n_samples_30s, 0.0f);

    std::vector<whisper_token> prompt_tokens;

    // print some info about the processing
    {
        fprintf(stderr, "\n");
        if (!whisper_is_multilingual(ctx)) {
            if (params.language != "en" || params.translate) {
                params.language = "en";
                params.translate = false;
                fprintf(stderr, "%s: WARNING: model is not multilingual, ignoring language and translation options\n", __func__);
            }
        }
        fprintf(stderr, "%s: processing %d samples (step = %.1f sec / len = %.1f sec / keep = %.1f sec), %d threads, lang = %s, task = %s, timestamps = %d ...\n",
                __func__,
                n_samples_step,
                float(n_samples_step)/WHISPER_SAMPLE_RATE,
                float(n_samples_len )/WHISPER_SAMPLE_RATE,
                float(n_samples_keep)/WHISPER_SAMPLE_RATE,
                params.n_threads,
                params.language.c_str(),
                params.translate ? "translate" : "transcribe",
                params.no_timestamps ? 0 : 1);

        if (!use_vad) {
            fprintf(stderr, "%s: n_new_line = %d, no_context = %d\n", __func__, n_new_line, params.no_context);
        } else {
            fprintf(stderr, "%s: using VAD, will transcribe on speech activity\n", __func__);
        }

        fprintf(stderr, "\n");
    }

    int n_iter = 0;

    bool is_running = true;

    /*
    std::ofstream fout;
    if (params.fname_out.length() > 0) {
        fout.open(params.fname_out);
        if (!fout.is_open()) {
            fprintf(stderr, "%s: failed to open output file '%s'!\n", __func__, params.fname_out.c_str());
            return 1;
        }
    }
    */
    //wav_writer wavWriter;
    // save wav file
    /*
    if (params.save_audio) {
        // Get current date/time for filename
        time_t now = time(0);
        char buffer[80];
        strftime(buffer, sizeof(buffer), "%Y%m%d%H%M%S", localtime(&now));
        std::string filename = std::string(buffer) + ".wav";

        wavWriter.open(filename, WHISPER_SAMPLE_RATE, 16, 1);
    }
    printf("[Start speaking]\n");
    */
    fflush(stdout);

    auto t_last  = std::chrono::high_resolution_clock::now();
    const auto t_start = t_last;



    // Modify to read data from file
    const auto fname_inp = params.fname_inp[0];
    const auto fname_out = 0 < (int) params.fname_out.size() && !params.fname_out[0].empty() ? params.fname_out[0] : params.fname_inp[0];

    std::vector<float> pcmf32_all;               // mono-channel F32 PCM
    std::vector<std::vector<float>> pcmf32s; // stereo-channel F32 PCM

    if (!::read_wav(fname_inp, pcmf32_all, pcmf32s, false)) {
        fprintf(stderr, "error: failed to read WAV file '%s'\n", fname_inp.c_str());
    }
    std::vector<float> pcmf32_audio_tag;
    if (params.audio_tag != "") {
        pcmf32_audio_tag = readCSVToVector(params.audio_tag);
    }
    /*
    if (!whisper_is_multilingual(ctx)) {
        if (params.language != "en" || params.translate) {
            params.language = "en";
            params.translate = false;
            fprintf(stderr, "%s: WARNING: model is not multilingual, ignoring language and translation options\n", __func__);
        }
    }
    if (params.detect_language) {
        params.language = "auto";
    }
    */
    if (true) {
        // print system information
        fprintf(stderr, "\n");
        fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
                params.n_threads, std::thread::hardware_concurrency(), whisper_print_system_info());

        // print some info about the processing
        fprintf(stderr, "\n");
        fprintf(stderr, "%s: processing '%s' (%d samples, %.1f sec), %d threads, %d beams + best of %d, lang = %s, task = %s, %stimestamps = %d ...\n",
                __func__, fname_inp.c_str(), int(pcmf32_all.size()), float(pcmf32_all.size())/WHISPER_SAMPLE_RATE,
                params.n_threads, params.beam_size, params.best_of,
                params.language.c_str(),
                params.translate ? "translate" : "transcribe",
                params.tinydiarize ? "tdrz = 1, " : "",
                params.no_timestamps ? 0 : 1);

        fprintf(stderr, "\n");
    }
    
    // for real-time simulation of the whisper stream processing
    int64_t pcmf32_index = 0; // Current position in pcmf32_all
    int64_t start = ggml_time_us() / 1000.0 - pcmf32_index;
    int64_t pcmf32_index_end = 0;
    int64_t now = 0;

    // whisper_streaming OnlineASRProcessor related variables
    std::vector<float> pcmf32_audio_buffer;
    HypothesisBuffer transcript_buffer;
    double buffer_time_offset = 0;
    std::vector<std::tuple<double, double, std::string>> committed;

    // record the each token latency
    std::vector<std::tuple<double, double, std::string, double, double>> latency_record;

    // main audio loop
    bool is_last_iter = false;
    while (is_running || !is_last_iter) {

        // process new audio

        if (!use_vad) {
            if (!is_running) {
                // No more new audio — this is the final iteration
                is_last_iter = true;
                fprintf(stderr, "[iter=%d] Audio ended, final inference on buffer (%.2fs)\n",
                        n_iter, (float)pcmf32_audio_buffer.size()/WHISPER_SAMPLE_RATE);
            } else {
                if (params.no_realtime) {
                    pcmf32_index_end += params.step_ms;
                } else {
                    now = ggml_time_us() / 1000.0 - start;
                    if (now < pcmf32_index_end + params.step_ms) {
                        //sleep to get the audio ingest.
                        precise_sleep((params.step_ms + pcmf32_index_end - now) / 1000.0);
                    }
                    // get the ingested data so far
                    pcmf32_index_end = ggml_time_us() / 1000.0 - start;
                }
                is_running = get_audio_chunk(pcmf32_all, pcmf32_new, pcmf32_index, pcmf32_index_end - pcmf32_index, WHISPER_SAMPLE_RATE);
                // update the start point for next audio segment
                pcmf32_index = pcmf32_index_end;

                // whisper_streaming online.insert_audio_chunk()
                pcmf32_audio_buffer.insert(pcmf32_audio_buffer.end(), pcmf32_new.begin(), pcmf32_new.end());
            }
            printf("[BUFFER_STATS] iter=%d buffer_sec=%.3f buffer_samples=%ld\n", n_iter, (float)pcmf32_audio_buffer.size()/WHISPER_SAMPLE_RATE, (long)pcmf32_audio_buffer.size());

            // Stop if buffer exceeds 30s (model can only process first 30s)
            if (pcmf32_audio_buffer.size() > 30 * WHISPER_SAMPLE_RATE) {
                fprintf(stderr, "%s: WARNING: audio buffer exceeded 30s (%.1fs), stopping.\n", __func__, (float)pcmf32_audio_buffer.size()/WHISPER_SAMPLE_RATE);
                is_running = false;
                break;
            }

            pcmf32 = pcmf32_audio_buffer;
            //const int n_samples_new = pcmf32_new.size();

            // take up to params.length_ms audio from previous iteration
            //const int n_samples_take = std::min((int) pcmf32_old.size(), std::max(0, n_samples_keep + n_samples_len - n_samples_new));

            //printf("processing: take = %d, new = %d, old = %d\n", n_samples_take, n_samples_new, (int) pcmf32_old.size());

            //pcmf32.resize(n_samples_new);

            //for (int i = 0; i < n_samples_take; i++) {
            //    pcmf32[i] = pcmf32_old[pcmf32_old.size() - n_samples_take + i];
            //}

            //memcpy(pcmf32.data() + n_samples_take, pcmf32_new.data(), n_samples_new*sizeof(float));

            //pcmf32_old = pcmf32;

            // prepare the prompt for whisper_streaming
            prompt_tokens = prompt(ctx, committed, buffer_time_offset);

        } else {
            fprintf(stderr, "The use_vad version is not implemented for now.\n");
            /*
            const auto t_now  = std::chrono::high_resolution_clock::now();
            const auto t_diff = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - t_last).count();

            if (t_diff < 2000) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));

                continue;
            }

            audio.get(2000, pcmf32_new);

            if (::vad_simple(pcmf32_new, WHISPER_SAMPLE_RATE, 1000, params.vad_thold, params.freq_thold, false)) {
                audio.get(params.length_ms, pcmf32);
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));

                continue;
            }

            t_last = t_now;
            */
        }

        // run the inference
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


            wparams.audio_ctx        = params.audio_ctx;
            int max_round_decode_tmp = int(float(pcmf32.size()) / WHISPER_SAMPLE_RATE / (params.step_ms / 1000.0)  * params.max_round_decode);
            wparams.max_round_decode = max_round_decode_tmp;
            wparams.speed_up         = params.speed_up;

            wparams.tdrz_enable      = params.tinydiarize; // [TDRZ]

            // disable temperature fallback
            //wparams.temperature_inc  = -1.0f;
            wparams.temperature_inc  = params.no_fallback ? 0.0f : wparams.temperature_inc;

            wparams.prompt_tokens    = params.no_context ? nullptr : prompt_tokens.data();
            wparams.prompt_n_tokens  = params.no_context ? 0       : prompt_tokens.size();

            if (params.audio_tag != "") {
                std::vector<float> audio_with_tag;
                audio_with_tag.reserve( pcmf32.size() + pcmf32_audio_tag.size());
                audio_with_tag.insert( audio_with_tag.end(), pcmf32.begin(), pcmf32.end());
                audio_with_tag.insert( audio_with_tag.end(), pcmf32_audio_tag.begin(), pcmf32_audio_tag.end());
                pcmf32 = audio_with_tag;
            }
            
            if (pcmf32.size() < n_samples_step) {
                pcmf32.resize(n_samples_step, 0.0f); // Pad with zeros
            }

            printf("\n");
            printf("Start new round of inference, data length %ld, buffer offset %f.\n", pcmf32.size(), buffer_time_offset);

            // Input stats — compute actual encoder input length based on audio_ctx
            int content_mel_len = (int)(pcmf32.size() / 320);  // actual content in encoder frames
            int effective_audio_ctx;
            if (params.audio_ctx == 0) {
                effective_audio_ctx = 1500;  // whisper default: always 30s
            } else if (params.audio_ctx == -1) {
                // dynamic: same as whisper.cpp GGML_PAD(mel.n_len / 2, 8)
                int mel_len = (int)(pcmf32.size() / 160);
                effective_audio_ctx = ((mel_len / 2 + 7) / 8) * 8;
            } else {
                effective_audio_ctx = params.audio_ctx;
            }
            float input_sec = (float)effective_audio_ctx / 50.0f;  // audio_ctx * 320 / 16000
            printf("[INPUT_STATS] iter=%d input_sec=%.3f input_samples=%ld content_mel_len=%d\n",
                   n_iter, input_sec, (long)pcmf32.size(), content_mel_len);

            // whisper_streaming asr.transcribe() in an iter
            if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
                fprintf(stderr, "%s: failed to process audio\n", argv[0]);
                return 6;
            }

            whisper_print_timings(ctx);

            // whisper_streaming asr.ts_words in an iter, the return value is the tsw with format [(beg,end,"word1"), ...]
            std::vector<std::tuple<double, double, std::string>> tsw = output_word_level_timestamp(ctx, params, true);
            printf("New round of transcript length: %d\n", tsw.size());
            // whisper_streaming transcript buffer management
            // transcript_buffer.insert()
            transcript_buffer.insert(tsw, buffer_time_offset);
            transcript_buffer.print_info();
            // transcript_buffer.flush()
            std::vector<std::tuple<double, double, std::string>> o = transcript_buffer.flush();
            // committed.extend(o)
            committed.insert(committed.end(), o.begin(), o.end());

            // printing debug info
            std::vector<std::tuple<double, double, std::string>> r_o = transcript_buffer.complete();
            if (is_last_iter) {
                // Final iteration: commit all remaining uncommitted tokens
                committed.insert(committed.end(), r_o.begin(), r_o.end());
            }
            std::tuple<double, double, std::string> completed = to_flush(o, buffer_time_offset);
            std::tuple<double, double, std::string> the_rest = to_flush(r_o, buffer_time_offset);
            std::string complete_transcript = std::get<2>(completed);
            std::string incomplete_transcript = std::get<2>(the_rest);
            printf("COMPLETE NOW: %s\n", complete_transcript.c_str());
            printf("INCOMPLETE: %s\n", incomplete_transcript.c_str());

            // recording the latency for each token
            double step_end = (ggml_time_us() / 1000.0 - start) / 1000.0;
            for (auto& token : o) {
                double start_time, end_time;
                std::string transcript;
                std::tie(start_time, end_time, transcript) = token;
                double latency = std::max(0.0, step_end - end_time);
                latency_record.push_back(std::make_tuple(start_time, end_time, transcript, latency, step_end));
            }

            // whisper_streaming audio_buffer management
            int64_t s = 15; // tentative buffer_trimming_sec set to be 15s
            // get the end time of each segments for chunk_completed_segment function
            std::vector<int64_t> segment_end_time = get_end_time_of_res(ctx);

            if (pcmf32_audio_buffer.size() > (s * WHISPER_SAMPLE_RATE)) {
                chunk_completed_segment(segment_end_time, committed, pcmf32_audio_buffer, transcript_buffer, buffer_time_offset);
                printf("[BUFFER_STATS_AFTER_TRIM] iter=%d buffer_sec=%.3f\n", n_iter, (float)pcmf32_audio_buffer.size()/WHISPER_SAMPLE_RATE);
            }


            // update the now time stamp after finishing execution
            now = ggml_time_us() / 1000.0 - start;

            // print result;
            {
                if (!use_vad) {
                    printf("\33[2K\r");

                    // print long empty line to clear the previous line
                    printf("%s", std::string(100, ' ').c_str());

                    printf("\33[2K\r");
                } else {
                    const int64_t t1 = (t_last - t_start).count()/1000000;
                    const int64_t t0 = std::max(0.0, t1 - pcmf32.size()*1000.0/WHISPER_SAMPLE_RATE);

                    printf("\n");
                    printf("### Transcription %d START | t0 = %d ms | t1 = %d ms\n", n_iter, (int) t0, (int) t1);
                    printf("\n");
                }

                const int n_segments = whisper_full_n_segments(ctx);
                for (int i = 0; i < n_segments; ++i) {
                    const char * text = whisper_full_get_segment_text(ctx, i);

                    if (params.no_timestamps) {
                        printf("%s", text);
                        fflush(stdout);

                        /*
                        if (params.fname_out.length() > 0) {
                            fout << text;
                        }
                        */
                    } else {
                        const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
                        const int64_t t1 = whisper_full_get_segment_t1(ctx, i);

                        std::string output = "[" + to_timestamp(t0, false) + " --> " + to_timestamp(t1, false) + "]  " + text;

                        if (whisper_full_get_segment_speaker_turn_next(ctx, i)) {
                            output += " [SPEAKER_TURN]";
                        }

                        output += "\n";

                        printf("%s", output.c_str());
                        fflush(stdout);
                        /*
                        if (params.fname_out.length() > 0) {
                            fout << output;
                        }
                        */
                    }
                }
                /*
                if (params.fname_out.length() > 0) {
                    fout << std::endl;
                }
                */
                if (use_vad) {
                    printf("\n");
                    printf("### Transcription %d END\n", n_iter);
                }
            }

            ++n_iter;
            
            /* if (!use_vad && (n_iter % n_new_line) == 0) {
                printf("\n");

                // keep part of the audio for next iteration to try to mitigate word boundary issues
                pcmf32_old = std::vector<float>(pcmf32.end() - n_samples_keep, pcmf32.end());

                // Add tokens of the last full length segment as the prompt
                if (!params.no_context) {
                    prompt_tokens.clear();

                    const int n_segments = whisper_full_n_segments(ctx);
                    for (int i = 0; i < n_segments; ++i) {
                        const int token_count = whisper_full_n_tokens(ctx, i);
                        for (int j = 0; j < token_count; ++j) {
                            prompt_tokens.push_back(whisper_full_get_token_id(ctx, i, j));
                        }
                    }
                }
                printf("prompt_tokens: ");
                for (size_t i = 0; i < prompt_tokens.size(); ++i) {
                    printf("%d ", prompt_tokens[i]);
                }
                printf("\n");

            } */
            fflush(stdout);

            if (is_last_iter) {
                break;
            }
        }
        //whisper_free(ctx);
        //struct whisper_context * ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);
        //ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);
    }

    //audio.pause();
    print_tsw(committed);
    print_tsw_with_token_latency(latency_record);
    whisper_print_timings(ctx);
    whisper_free(ctx);

    return 0;
}
