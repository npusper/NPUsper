"""
Compare Whisper-S (baseline) vs WhisperFlow (serialized/pipeline) vs SimulStreaming on LibriSpeech.

Runs C++ binaries on the same concatenated LibriSpeech audio,
parses stdout to extract committed transcripts and latency,
and computes WER, CER, and latency metrics.

Usage:
    # 2-way (baseline vs serialized):
    python run_streaming_comparison.py --num_samples 50 --model base --step 500

    # 3-way (baseline vs serialized vs pipeline):
    python run_streaming_comparison.py --num_samples 50 --model base --step 500 --pipeline_dir ../whisperflow-pipeline

    # With SimulStreaming (AlignAtt):
    python run_streaming_comparison.py --num_samples 5 --model base --step 1000 --simul-streaming

    # SimulStreaming only:
    python run_streaming_comparison.py --num_samples 5 --model base --step 1000 --simul-streaming --only simul

    # With SimulWhisper (AlignAtt + CIF):
    python run_streaming_comparison.py --num_samples 5 --model base --step 1000 --simul-whisper

    # SimulWhisper only:
    python run_streaming_comparison.py --num_samples 5 --model base --step 1000 --simul-whisper --only simul_whisper

    # With Ours (backward peak detection):
    python run_streaming_comparison.py --num_samples 5 --model base --step 1000 --ours-streaming

    # Ours only:
    python run_streaming_comparison.py --num_samples 5 --model base --step 1000 --ours-streaming --only ours

    # On S25 (direct audio input, no LibriSpeech needed):
    python run_streaming_comparison.py --audio-file test_audio_5samples.wav --ground-truth ground_truth_5samples.txt \
        --model base --step 1000 --power-monitor

    # On S25 with CMake build directory:
    python run_streaming_comparison.py --audio-file test_audio_5samples.wav --ground-truth ground_truth_5samples.txt \
        --model base --step 1000 --bin-dir build/bin
"""

import os
import sys
import re
import glob
import json
import argparse
import subprocess
import time
import threading
import numpy as np
import soundfile as sf
from datetime import datetime

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


def detect_gpu_usage(combined_output):
    """Detect GPU backend usage from C++ binary output logs.

    Returns dict with:
        backend: str - detected backend name (e.g., "CUDA", "Metal", "CPU")
        gpu_active: bool - whether GPU backend was actually used
    """
    backend = "CPU"
    gpu_active = False

    if "using CUDA backend" in combined_output:
        backend = "CUDA"
        gpu_active = True
    elif "using Metal backend" in combined_output:
        backend = "Metal"
        gpu_active = True
    elif "using SYCL backend" in combined_output:
        backend = "SYCL"
        gpu_active = True
    elif "using Vulkan backend" in combined_output:
        backend = "Vulkan"
        gpu_active = True
    elif "use gpu    = 1" in combined_output or "use gpu = 1" in combined_output:
        backend = "CUDA"
        gpu_active = True

    # Check for init failures
    if gpu_active and "failed" in combined_output:
        for kw in ["cuda_init() failed", "metal_init() failed",
                    "sycl_init() failed", "vk_init() failed"]:
            if kw in combined_output:
                gpu_active = False
                backend += " (init failed, fell back to CPU)"
                break

    return {"backend": backend, "gpu_active": gpu_active}


def load_librispeech_samples(root, url="test-clean"):
    """Load LibriSpeech samples manually (same ordering as torchaudio)."""
    base_dir = os.path.join(root, "LibriSpeech", url)
    samples = []
    trans_files = sorted(glob.glob(os.path.join(base_dir, "*", "*", "*.trans.txt")))
    for trans_file in trans_files:
        chapter_dir = os.path.dirname(trans_file)
        with open(trans_file, "r") as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    utterance_id, transcript = parts
                    flac_path = os.path.join(chapter_dir, f"{utterance_id}.flac")
                    if os.path.exists(flac_path):
                        samples.append((flac_path, transcript))
    samples.sort(key=lambda x: x[0])
    return samples


# FLEURS language code mapping (Whisper language code → FLEURS config name)
_FLEURS_LANG_MAP = {
    "ko": "ko_kr", "ja": "ja_jp", "zh": "cmn_hans_cn", "en": "en_us",
    "fr": "fr_fr", "de": "de_de", "es": "es_419", "it": "it_it",
    "pt": "pt_br", "ru": "ru_ru", "ar": "ar_eg", "hi": "hi_in",
    "vi": "vi_vn", "th": "th_th", "tr": "tr_tr", "pl": "pl_pl",
    "nl": "nl_nl", "sv": "sv_se", "da": "da_dk", "fi": "fi_fi",
}

def load_fleurs_samples(language, split="test"):
    """Load FLEURS samples via HuggingFace datasets. Returns list of (audio_array, transcript)."""
    from datasets import load_dataset
    fleurs_lang = _FLEURS_LANG_MAP.get(language, f"{language}_{language}")
    print(f"Loading FLEURS dataset: google/fleurs/{fleurs_lang} split={split}")
    ds = load_dataset("google/fleurs", fleurs_lang, split=split, trust_remote_code=True)
    samples = []
    for item in ds:
        audio_array = item["audio"]["array"]
        transcript = item["transcription"]
        samples.append((audio_array, transcript))
    return samples


def load_tedlium_samples(split="test"):
    """Load TED-LIUM 3 long-form samples. Each sample is a full TED talk.
    Returns list of (audio_array, transcript)."""
    from datasets import load_dataset
    print(f"Loading TED-LIUM long-form dataset: split={split}")
    ds = load_dataset("distil-whisper/tedlium-long-form", split=split, trust_remote_code=True)
    samples = []
    for item in ds:
        audio_array = item["audio"]["array"]
        transcript = item["text"]
        dur = len(audio_array) / item["audio"]["sampling_rate"]
        if dur < 10:  # skip very short samples
            continue
        samples.append((audio_array, transcript))
    print(f"Loaded {len(samples)} TED-LIUM samples")
    return samples


_NUM_WORD_TO_DIGIT = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000",
}

# Languages without word-separating spaces — CER is the primary metric
_NO_SPACE_LANGS = {"zh"}

def normalize_text(text, language="en"):
    """Normalize text for WER/CER: lowercase, remove punctuation, normalize numbers."""
    text = text.lower()
    text = text.replace("-", " ")
    if language in _NO_SPACE_LANGS:
        # Remove all punctuation (ASCII + CJK fullwidth + CJK symbols)
        text = re.sub(r"[^\w]", "", text)
        text = re.sub(r'[。，、！？；：""''（）【】《》—…·～\u3000\uff01-\uff5e]', '', text)
        # Insert space between every character for character-level WER/CER
        text = " ".join(text)
    else:
        text = re.sub(r"[^\w\s]", "", text)
        # Normalize English number words to digits (skip for non-English)
        if language == "en":
            words = text.split()
            words = [_NUM_WORD_TO_DIGIT.get(w, w) for w in words]
            text = " ".join(words)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_metrics(reference, hypothesis, truncate_hyp=False, language="en"):
    """Compute normalized WER and CER using jiwer.

    Truncates the reference to match the hypothesis word count
    so that missing words at the end don't unfairly penalize scores.
    If truncate_hyp=True, finds the last 3 words of ref in hyp and truncates
    hyp at that match point (useful for systems like 'ours' that may generate
    trailing repetitions after the actual content ends).
    """
    import jiwer
    ref = normalize_text(reference, language=language)
    hyp = normalize_text(hypothesis, language=language)
    if not ref:
        return {"wer": 0.0 if not hyp else 1.0, "cer": 0.0 if not hyp else 1.0}
    hyp_words = hyp.split()
    ref_words = ref.split()
    # Truncate hypothesis by matching last 3 words of ref in hyp
    if truncate_hyp and len(hyp_words) > len(ref_words):
        tail_n = min(3, len(ref_words))
        tail = ref_words[-tail_n:]
        # Search for the last occurrence of tail sequence in hyp_words
        match_end = -1
        for i in range(len(hyp_words) - tail_n, -1, -1):
            if hyp_words[i:i + tail_n] == tail:
                match_end = i + tail_n
                break
        if match_end > 0:
            hyp = " ".join(hyp_words[:match_end])
            hyp_words = hyp.split()
    # Truncate both to the shorter length (compare up to REF's last word)
    if len(hyp_words) < len(ref_words):
        ref_truncated = " ".join(ref_words[:len(hyp_words)])
        hyp_truncated = hyp
    elif len(hyp_words) > len(ref_words):
        ref_truncated = ref
        hyp_truncated = " ".join(hyp_words[:len(ref_words)])
    else:
        ref_truncated = ref
        hyp_truncated = hyp
    wer = jiwer.wer(ref_truncated, hyp_truncated)
    cer = jiwer.cer(ref_truncated, hyp_truncated)
    return {"wer": wer, "cer": cer, "ref_words": len(ref_words), "hyp_words": len(hyp_words), "ref_truncated_words": len(ref_truncated.split())}


def parse_committed_output(output_text):
    """
    Parse committed transcript lines from C++ binary stdout.

    The C++ binaries output two blocks of "Start Time:" lines:
      1. print_tsw(committed)                — without Latency
      2. print_tsw_with_token_latency(...)   — with Latency (+ optional Emission Time)
    We parse both separately. If latency lines exist, use those (to avoid
    double-counting). Otherwise fall back to non-latency lines.
    """
    # Pattern with Latency + Emission Time (new format)
    pattern_with_emission = re.compile(
        r"Start Time:\s*([\d.]+),\s*End Time:\s*([\d.]+),\s*Transcript: (.*?),\s*Latency:\s*([\d.e+-]+),\s*Emission Time:\s*([\d.e+-]+)\s*$"
    )
    # Pattern with Latency only (legacy format)
    pattern_with_lat = re.compile(
        r"Start Time:\s*([\d.]+),\s*End Time:\s*([\d.]+),\s*Transcript: (.*?),\s*Latency:\s*([\d.-]+)\s*$"
    )
    pattern_no_lat = re.compile(
        r"Start Time:\s*([\d.]+),\s*End Time:\s*([\d.]+),\s*Transcript: (.*?)$"
    )

    records_with_lat = []
    records_no_lat = []

    for line in output_text.split("\n"):
        line = line.strip()
        # Try emission time format first
        m = pattern_with_emission.match(line)
        if m:
            records_with_lat.append({
                "start": float(m.group(1)),
                "end": float(m.group(2)),
                "transcript": m.group(3),
                "latency": float(m.group(4)),
                "emission_time": float(m.group(5)),
            })
            continue
        m = pattern_with_lat.match(line)
        if m:
            records_with_lat.append({
                "start": float(m.group(1)),
                "end": float(m.group(2)),
                "transcript": m.group(3),
                "latency": float(m.group(4)),
                "emission_time": None,
            })
            continue
        m = pattern_no_lat.match(line)
        if m:
            # Skip lines that look like they have Latency (regex didn't match above)
            text = m.group(3)
            if ", Latency:" in text:
                continue
            records_no_lat.append({
                "start": float(m.group(1)),
                "end": float(m.group(2)),
                "transcript": text,
                "latency": None,
                "emission_time": None,
            })

    # Prefer latency records if available (avoids double-counting)
    return records_with_lat if records_with_lat else records_no_lat


def parse_simul_streaming_output(output_text):
    """Parse SimulStreaming output.

    SimulStreaming prints tokens directly to stdout (no Start Time: format).
    The final transcript appears between '=== Final transcript ===' markers in stderr.
    We extract both the streaming stdout and the final transcript.
    """
    # stdout contains the streamed tokens concatenated
    # stderr contains debug info and the final transcript block
    # combined = stdout + "\n" + stderr
    #
    # Extract everything from stdout (before stderr section).
    # The stdout is just concatenated token text, e.g. " He hoped there would be..."
    # Final transcript in stderr: === Final transcript === \n <text> \n ========================
    m = re.search(r"=== Final transcript ===\n(.*?)\n={20,}", output_text, re.DOTALL)
    if m:
        transcript = m.group(1).strip()
    else:
        # Fallback: use stdout directly (first section before stderr markers)
        transcript = output_text.split("\n")[0].strip() if output_text else ""

    return transcript


def process_simul_streaming_result(run_result, ground_truth, language="en"):
    """Process SimulStreaming run result into metrics dict (compatible with process_run_result format)."""
    combined = run_result["combined"]
    stdout = run_result.get("stdout", "")

    # Use stdout as the transcript (tokens printed via printf)
    # Clean up: remove leading/trailing whitespace, collapse spaces
    hypothesis = re.sub(r"\s+", " ", stdout.strip())

    # Fallback to stderr final transcript block if stdout is empty
    if not hypothesis:
        hypothesis = parse_simul_streaming_output(combined)

    metrics = compute_metrics(ground_truth, hypothesis, language=language)

    return {
        "metrics": metrics,
        "hypothesis": hypothesis,
        "records": [],
        "word_records": [],
        "avg_latency": None,
        "latency_stats": {},
        "word_latency_stats": {},
        "buffer_stats": {},
        "encode_stats": {},
        "decode_stats": {},
        "decode_per_token_stats": {},
        "wall_time": run_result["wall_time"],
        "num_tokens": len(hypothesis.split()),
        "num_words": len(hypothesis.split()),
        "buffer_overflow": False,
    }


def merge_subword_tokens(records):
    """
    Merge sub-word tokens into words.
    Tokens starting with a space are word beginnings; others are continuations.
    """
    if not records:
        return records

    merged = []
    current = None

    for r in records:
        raw_transcript = r["transcript"]
        # Token starts with space = new word
        if raw_transcript.startswith(" ") or current is None:
            if current is not None:
                merged.append(current)
            current = {
                "start": r["start"],
                "end": r["end"],
                "transcript": raw_transcript.strip(),
                "latency": r["latency"],
                "emission_time": r.get("emission_time"),
            }
        else:
            # Continuation token — append to current word
            current["transcript"] += raw_transcript
            current["end"] = r["end"]
            # Use the last sub-token's latency (when the full word is available)
            if r["latency"] is not None:
                current["latency"] = r["latency"]
            if r.get("emission_time") is not None:
                current["emission_time"] = r["emission_time"]

    if current is not None:
        merged.append(current)

    return merged



def parse_per_iteration_encode_decode(output_text):
    """Parse per-iteration encode/decode times from cumulative whisper_print_timings.

    whisper_print_timings accumulates encode/decode times across iterations.
    Whisper-S calls it once per iter (1 context), WhisperFlow calls it twice
    per iter (ctx=GPU encode, ctx_cpu=CPU decode).  We detect which mode and
    compute per-iteration values from cumulative diffs.

    Note: whisper.cpp splits decode timing by batch size:
      - n_tokens == 1  -> t_decode_us  (greedy, best_of=1)
      - n_tokens < 16  -> t_batchd_us  (batched decode, best_of>1 or beam search)
    We combine decode + batchd to get the total decoding time.

    Returns (encode_per_iter, decode_per_iter, decode_per_token,
             mel_per_iter, prefill_per_iter) — all in ms.
    """
    mel_pat = re.compile(
        r'whisper_print_timings:\s+mel time\s+=\s+([\d.]+)\s+ms'
    )
    enc_pat = re.compile(
        r'whisper_print_timings:\s+encode time\s+=\s+([\d.]+)\s+ms\s*/\s*(\d+)\s+runs'
    )
    dec_pat = re.compile(
        r'whisper_print_timings:\s+decode time\s+=\s+([\d.]+)\s+ms\s*/\s*(\d+)\s+runs'
    )
    batchd_pat = re.compile(
        r'whisper_print_timings:\s+batchd time\s+=\s+([\d.]+)\s+ms\s*/\s*(\d+)\s+runs'
    )
    prompt_pat = re.compile(
        r'whisper_print_timings:\s+prompt time\s+=\s+([\d.]+)\s+ms\s*/\s*(\d+)\s+runs'
    )

    all_mel = []          # cumulative ms
    all_encode = []       # cumulative ms
    all_decode = []       # cumulative ms
    all_decode_runs = []  # cumulative runs
    all_batchd = []       # cumulative ms
    all_batchd_runs = []  # cumulative runs
    all_prompt = []       # cumulative ms
    all_prompt_runs = []  # cumulative runs
    for line in output_text.split("\n"):
        m = mel_pat.search(line)
        if m:
            all_mel.append(float(m.group(1)))
            continue
        m = enc_pat.search(line)
        if m:
            all_encode.append(float(m.group(1)))
            continue
        m = dec_pat.search(line)
        if m:
            all_decode.append(float(m.group(1)))
            all_decode_runs.append(int(m.group(2)))
            continue
        m = batchd_pat.search(line)
        if m:
            all_batchd.append(float(m.group(1)))
            all_batchd_runs.append(int(m.group(2)))
            continue
        m = prompt_pat.search(line)
        if m:
            all_prompt.append(float(m.group(1)))
            all_prompt_runs.append(int(m.group(2)))

    if not all_encode or (not all_decode and not all_batchd):
        return [], [], [], [], []

    # Ensure all lists exist (fill missing with zeros)
    if not all_decode:
        all_decode = [0.0] * len(all_batchd) if all_batchd else [0.0] * len(all_encode)
        all_decode_runs = [0] * len(all_decode)
    if not all_batchd:
        all_batchd = [0.0] * len(all_decode)
        all_batchd_runs = [0] * len(all_decode)
    if not all_prompt:
        all_prompt = [0.0] * len(all_batchd)
        all_prompt_runs = [0] * len(all_batchd)

    # decode = t_decode_us only (token-by-token, n_tokens==1)
    # No batchd mixed in — prefill and decode are now cleanly separated.
    n_timings = len(all_decode)
    all_encode = all_encode[:n_timings]

    # Prefill = batchd + prompt (SOT batch decode, n_tokens >= 2)
    # Both are cumulative; combine them
    all_prefill = [all_batchd[i] + (all_prompt[i] if i < len(all_prompt) else (all_prompt[-1] if all_prompt else 0.0))
                   for i in range(len(all_batchd))]

    # Truncate mel/prefill to match
    all_mel = all_mel[:n_timings]
    all_prefill = all_prefill[:n_timings]

    # Detect 2-context mode: one set of alternating indices is always ~0
    def has_alternating_zeros(vals):
        if len(vals) < 4:
            return False
        odd_vals = [vals[i] for i in range(1, len(vals), 2)]
        even_vals = [vals[i] for i in range(0, len(vals), 2)]
        return all(v < 0.01 for v in odd_vals) or all(v < 0.01 for v in even_vals)

    two_ctx = has_alternating_zeros(all_encode) or has_alternating_zeros(all_decode)

    if two_ctx:
        even_enc = [all_encode[i] for i in range(0, len(all_encode), 2)]
        odd_enc  = [all_encode[i] for i in range(1, len(all_encode), 2)]
        even_dec = [all_decode[i] for i in range(0, len(all_decode), 2)]
        odd_dec  = [all_decode[i] for i in range(1, len(all_decode), 2)]
        even_dec_r = [all_decode_runs[i] for i in range(0, len(all_decode_runs), 2)]
        odd_dec_r  = [all_decode_runs[i] for i in range(1, len(all_decode_runs), 2)]
        encode_cumul = even_enc if sum(even_enc) > sum(odd_enc) else odd_enc
        if sum(odd_dec) > sum(even_dec):
            decode_cumul = odd_dec
            decode_runs_cumul = odd_dec_r
        else:
            decode_cumul = even_dec
            decode_runs_cumul = even_dec_r
        # For 2-context mode, mel/prefill follow the same split
        even_mel = [all_mel[i] for i in range(0, len(all_mel), 2)]
        odd_mel  = [all_mel[i] for i in range(1, len(all_mel), 2)]
        mel_cumul = even_mel if sum(even_mel) > sum(odd_mel) else odd_mel
        even_pf = [all_prefill[i] for i in range(0, len(all_prefill), 2)]
        odd_pf  = [all_prefill[i] for i in range(1, len(all_prefill), 2)]
        prefill_cumul = even_pf if sum(even_pf) > sum(odd_pf) else odd_pf
    else:
        encode_cumul = all_encode
        decode_cumul = all_decode
        decode_runs_cumul = all_decode_runs
        mel_cumul = all_mel
        prefill_cumul = all_prefill

    def cumul_to_per_iter(cumul):
        if not cumul:
            return []
        per = [cumul[0]]
        for i in range(1, len(cumul)):
            per.append(max(0.0, cumul[i] - cumul[i - 1]))
        return per

    def cumul_to_per_iter_int(cumul):
        if not cumul:
            return []
        per = [cumul[0]]
        for i in range(1, len(cumul)):
            per.append(max(0, cumul[i] - cumul[i - 1]))
        return per

    encode_per_iter = cumul_to_per_iter(encode_cumul)
    decode_per_iter = cumul_to_per_iter(decode_cumul)
    decode_runs_per_iter = cumul_to_per_iter_int(decode_runs_cumul)
    mel_per_iter = cumul_to_per_iter(mel_cumul)
    prefill_per_iter = cumul_to_per_iter(prefill_cumul)

    # Remove duplicate final print (diff == 0 at the end)
    if len(encode_per_iter) > 1 and encode_per_iter[-1] < 0.01:
        encode_per_iter = encode_per_iter[:-1]
    if len(decode_per_iter) > 1 and decode_per_iter[-1] < 0.01:
        decode_per_iter = decode_per_iter[:-1]
        decode_runs_per_iter = decode_runs_per_iter[:-1]
    if len(mel_per_iter) > 1 and mel_per_iter[-1] < 0.01:
        mel_per_iter = mel_per_iter[:-1]
    if len(prefill_per_iter) > 1 and prefill_per_iter[-1] < 0.01:
        prefill_per_iter = prefill_per_iter[:-1]

    # Per-token decode time = decode_time / runs for each iteration
    decode_per_token = []
    for dt, runs in zip(decode_per_iter, decode_runs_per_iter):
        if runs > 0 and dt > 0:
            decode_per_token.append(dt / runs)

    return encode_per_iter, decode_per_iter, decode_per_token, mel_per_iter, prefill_per_iter


def parse_iter_timing(output_text):
    """Parse per-iteration timing from [ITER_TIMING] lines (direct chrono measurement).

    These are emitted by simul_streaming, simul_whisper, and ours_streaming
    binaries using external chrono timing around each phase.

    Returns (mel_per_iter, encode_per_iter, prefill_per_iter,
             decode_per_iter, decode_per_token) — all in ms.
    """
    pat = re.compile(
        r'\[ITER_TIMING\]\s+iter=(\d+)\s+mel=([\d.]+)\s+ms\s+encode=([\d.]+)\s+ms\s+'
        r'prefill=([\d.]+)\s+ms\s+decode=([\d.]+)\s+ms\s+n_tokens=(\d+)'
    )
    mel = []
    encode = []
    prefill = []
    decode = []
    decode_per_token = []
    n_tokens_per_iter = []
    for line in output_text.split("\n"):
        m = pat.search(line)
        if m:
            mel.append(float(m.group(2)))
            encode.append(float(m.group(3)))
            prefill.append(float(m.group(4)))
            dec_ms = float(m.group(5))
            n_tok = int(m.group(6))
            decode.append(dec_ms)
            n_tokens_per_iter.append(n_tok)
            if n_tok > 0:
                decode_per_token.append(dec_ms / n_tok)
    return mel, encode, prefill, decode, decode_per_token, n_tokens_per_iter


def parse_cross_attn_times(output_text):
    """Parse per-iteration cross-attention extraction times.

    Expects lines like:
      [CROSS_ATTN_TIME] iter=N copy=X.XX ms calc=Y.YY ms

    Returns (copy_per_iter, calc_per_iter) — lists of ms values.
    """
    pat = re.compile(
        r'\[CROSS_ATTN_TIME\]\s+iter=(\d+)\s+copy=([\d.]+)\s+ms\s+calc=([\d.]+)\s+ms'
    )
    copy_times = []
    calc_times = []
    for line in output_text.split("\n"):
        m = pat.search(line)
        if m:
            copy_times.append(float(m.group(2)))
            calc_times.append(float(m.group(3)))
    return copy_times, calc_times


def compute_stats_ms(values):
    """Compute summary statistics for a list of ms values."""
    if not values:
        return {}
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n_iters": len(arr),
    }


def parse_avg_latency(output_text):
    """Parse 'Average latency: X' from output."""
    m = re.search(r"Average latency:\s*([\d.]+)", output_text)
    if m:
        return float(m.group(1))
    return None


def parse_buffer_stats(output_text):
    """Parse [BUFFER_STATS] lines."""
    pattern = re.compile(r'\[BUFFER_STATS\] iter=(\d+) buffer_sec=([\d.]+)')
    stats = []
    for line in output_text.split("\n"):
        m = pattern.search(line)
        if m:
            stats.append(float(m.group(2)))
    return stats


def parse_input_stats(output_text):
    """Parse [INPUT_STATS] lines (whisper input length per iteration)."""
    pattern = re.compile(r'\[INPUT_STATS\] iter=(\d+) input_sec=([\d.]+) input_samples=(\d+) content_mel_len=(\d+)')
    input_secs = []
    content_mel_lens = []
    for line in output_text.split("\n"):
        m = pattern.search(line)
        if m:
            input_secs.append(float(m.group(2)))
            content_mel_lens.append(int(m.group(4)))
    return input_secs, content_mel_lens


def parse_cif_inference_times(output_text):
    """Parse [iter=N] CIF inference: X us (content_mel_len=Y) lines from SimulWhisper stderr."""
    pattern = re.compile(r'\[iter=(\d+)\] CIF inference:\s+([\d.]+)\s+us')
    times_us = []
    for line in output_text.split("\n"):
        m = pattern.search(line)
        if m:
            times_us.append(float(m.group(2)))
    return times_us


class PowerMonitor:
    """Sample power consumption from sysfs during binary execution."""

    def __init__(self, sysfs_path, interval_ms=100):
        self.sysfs_path = sysfs_path
        self.interval = interval_ms / 1000.0
        self.samples = []
        self._stop_event = threading.Event()
        self._thread = None
        self.power_now_path = os.path.join(sysfs_path, "power_now")
        self.current_now_path = os.path.join(sysfs_path, "current_now")
        self.voltage_now_path = os.path.join(sysfs_path, "voltage_now")

    def _read_power_uw(self):
        try:
            if os.path.exists(self.power_now_path):
                with open(self.power_now_path, "r") as f:
                    return abs(int(f.read().strip()))
            elif os.path.exists(self.current_now_path) and os.path.exists(self.voltage_now_path):
                with open(self.current_now_path, "r") as f:
                    current_ua = abs(int(f.read().strip()))
                with open(self.voltage_now_path, "r") as f:
                    voltage_uv = abs(int(f.read().strip()))
                return (current_ua * voltage_uv) // 1_000_000
        except (IOError, ValueError):
            pass
        return None

    def _sample_loop(self):
        while not self._stop_event.is_set():
            power = self._read_power_uw()
            if power is not None:
                self.samples.append((time.perf_counter(), power))
            self._stop_event.wait(self.interval)

    def start(self):
        self._stop_event.clear()
        self.samples = []
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def get_stats(self):
        if not self.samples:
            return None
        powers = [s[1] for s in self.samples]
        timestamps = [s[0] for s in self.samples]
        duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
        arr = np.array(powers, dtype=np.float64)
        energy_uj = 0.0
        for i in range(1, len(self.samples)):
            dt = self.samples[i][0] - self.samples[i - 1][0]
            avg_power = (self.samples[i][1] + self.samples[i - 1][1]) / 2.0
            energy_uj += avg_power * dt
        return {
            "mean_power_mw": float(np.mean(arr)) / 1000.0,
            "max_power_mw": float(np.max(arr)) / 1000.0,
            "min_power_mw": float(np.min(arr)) / 1000.0,
            "std_power_mw": float(np.std(arr)) / 1000.0,
            "total_energy_j": energy_uj / 1_000_000.0,
            "duration_s": duration,
            "num_samples": len(self.samples),
        }


def run_binary(binary_path, model_path, audio_path, args_list, label, power_monitor=None,
               adb_device=None, adb_remote_dir=None):
    """Run a C++ binary and capture stdout + stderr."""
    if adb_device:
        # Build remote command for adb shell
        remote_bin = binary_path  # already relative to adb_remote_dir
        remote_model = model_path
        remote_audio = os.path.basename(audio_path)
        inner_args = " ".join(args_list)
        remote_cmd = (f"cd {adb_remote_dir} && "
                      f"LD_LIBRARY_PATH=/data/local/tmp "
                      f"{remote_bin} -m {remote_model} {remote_audio} {inner_args}")
        cmd = ["adb", "-s", adb_device, "shell", remote_cmd]
    else:
        cmd = [binary_path, "-m", model_path, audio_path] + args_list

    print(f"\n{'='*80}")
    print(f"Running {label}...")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*80}")

    if power_monitor:
        power_monitor.start()

    # Set LD_LIBRARY_PATH to binary's parent dir (for libwhisper.so)
    env = os.environ.copy()
    if not adb_device:
        bin_lib_dir = os.path.dirname(os.path.dirname(binary_path))  # binary is in bin/, lib is in parent
        if os.path.isdir(bin_lib_dir):
            env["LD_LIBRARY_PATH"] = bin_lib_dir + ":" + env.get("LD_LIBRARY_PATH", "")

    t0 = time.perf_counter()
    result = subprocess.run(
        cmd,
        capture_output=True,
        timeout=600,
        env=env,
    )
    wall_time = time.perf_counter() - t0

    power_stats = None
    power_samples = []
    if power_monitor:
        power_monitor.stop()
        power_stats = power_monitor.get_stats()
        # Save raw samples with time relative to start
        if power_monitor.samples:
            t_start = power_monitor.samples[0][0]
            power_samples = [(t - t_start, p) for t, p in power_monitor.samples]

    stdout = result.stdout.decode("utf-8", errors="replace")
    stderr = result.stderr.decode("utf-8", errors="replace")
    combined = stdout + "\n" + stderr
    return {
        "stdout": stdout,
        "stderr": stderr,
        "combined": combined,
        "returncode": result.returncode,
        "wall_time": wall_time,
        "power_stats": power_stats,
        "power_samples": power_samples,
    }


def recalculate_latency_with_ground_truth(result, gt_words):
    """Recalculate latency using offline ground truth timestamps.

    Matches word_records to gt_words by normalized text alignment,
    then computes: latency = emission_time - gt_end_time

    Args:
        result: dict from process_run_result (modified in-place)
        gt_words: list of {"word": str, "start": float, "end": float}
    """
    word_records = result.get("word_records", [])
    if not word_records or not gt_words:
        return

    # Normalize gt words for matching
    gt_normalized = []
    for gw in gt_words:
        norm = re.sub(r"[^\w]", "", gw["word"].lower())
        if norm:
            gt_normalized.append({"word": norm, "start": gw["start"], "end": gw["end"]})

    # Normalize hypothesis words and match sequentially
    gt_idx = 0
    matched_count = 0
    for wr in word_records:
        hyp_norm = re.sub(r"[^\w]", "", wr["transcript"].lower())
        if not hyp_norm:
            wr["gt_latency"] = None
            continue

        # Find matching gt word (allow skipping mismatches)
        found = False
        search_start = gt_idx
        search_end = min(gt_idx + 5, len(gt_normalized))  # look ahead up to 5 words
        for j in range(search_start, search_end):
            if gt_normalized[j]["word"] == hyp_norm:
                gt_end = gt_normalized[j]["end"]
                emission = wr.get("emission_time")
                if emission is not None:
                    wr["gt_latency"] = emission - gt_end
                else:
                    wr["gt_latency"] = None
                gt_idx = j + 1
                found = True
                matched_count += 1
                break

        if not found:
            wr["gt_latency"] = None

    # Recompute latency stats from gt_latency
    gt_latencies = [wr["gt_latency"] for wr in word_records if wr.get("gt_latency") is not None]
    if gt_latencies:
        lat_arr = np.array(gt_latencies)
        result["gt_word_latency_stats"] = {
            "mean": float(np.mean(lat_arr)),
            "median": float(np.median(lat_arr)),
            "std": float(np.std(lat_arr)),
            "p90": float(np.percentile(lat_arr, 90)),
            "p95": float(np.percentile(lat_arr, 95)),
            "min": float(np.min(lat_arr)),
            "max": float(np.max(lat_arr)),
        }
        result["gt_matched_words"] = matched_count
        result["gt_total_words"] = len(gt_normalized)
    else:
        result["gt_word_latency_stats"] = {}
        result["gt_matched_words"] = 0
        result["gt_total_words"] = len(gt_normalized)


def generate_chunk_summary(stderr_text, label, params_info="", gt_words=None):
    """Parse ours_streaming stderr (debug output) into a chunk-by-chunk summary."""
    import re
    lines = stderr_text.split('\n')

    chunks = []
    current_chunk = None

    for line in lines:
        # [iter=N] effective=Xs, carryover=Ys, is_last=Z, abs=A-Bs
        m = re.match(r'\[iter=(\d+)\] effective=([\d.]+)s, carryover=([\d.]+)s, is_last=(\d)(?:, abs=([\d.]+)-([\d.]+)s)?', line)
        if m:
            if current_chunk is not None:
                chunks.append(current_chunk)
            current_chunk = {
                'iter': int(m.group(1)),
                'effective': float(m.group(2)),
                'carryover_in': float(m.group(3)),
                'is_last': m.group(4) == '1',
                'abs_start': float(m.group(5)) if m.group(5) else None,
                'abs_end': float(m.group(6)) if m.group(6) else None,
                'emitted': '',
                'stop_reason': None,
                'eot_skips': [],
                'olap_dedup_skipped': 0,
                'olap_dedup_kept_both': False,
                'dtw_words': [],
                'dtw_last_word_end': None,
                'carryover_from': None,
                'carryover_dur': None,
                'prefill_text': None,
                'speculative': False,
                'replaced_speculative': 0,
                'reinfer_text': None,
                'reinfer_n_tokens': 0,
                'reinfer_eot_details': [],
            }
            continue

        if current_chunk is None:
            continue

        # [iter=N] emitted: "text"
        m = re.match(r'\[iter=\d+\] emitted: "(.*)"', line)
        if m:
            current_chunk['emitted'] = m.group(1)
            continue

        # [REPETITION] Token 'X' repeated 3+ times at N
        if '[REPETITION]' in line:
            m2 = re.search(r"\[REPETITION\] Token '([^']+)' repeated (\d+)\+ times at (\d+)", line)
            if m2:
                current_chunk['stop_reason'] = f"repetition ('{m2.group(1)}' repeated {m2.group(2)}+ times at step {m2.group(3)})"
            else:
                current_chunk['stop_reason'] = 'repetition'
            continue

        # [SKIP EOT] step N, using next best: 'X' (id=N)
        if '[SKIP EOT]' in line:
            m2 = re.search(r"\[SKIP EOT\] step (\d+), using next best: '([^']+)'", line)
            if m2:
                if not current_chunk.get('eot_skips'):
                    current_chunk['eot_skips'] = []
                current_chunk['eot_skips'].append(f"step {m2.group(1)} -> '{m2.group(2)}'")
            continue

        # [STOP EOT] step N: model signaled end
        if '[STOP EOT]' in line:
            m2 = re.search(r'\[STOP EOT\] step (\d+)', line)
            if m2:
                current_chunk['stop_reason'] = f"eot (step {m2.group(1)})"
            else:
                current_chunk['stop_reason'] = 'eot'
            continue

        # [REINFER] Re-inferred with 30s padding
        if '[REINFER]' in line:
            m2 = re.search(r'\[REINFER\] Re-inferred (\d+) tokens \(no carryover(?:, (\d+) EOT events, (\d+) skipped)?\): "(.*)"', line)
            if m2:
                current_chunk['reinfer_text'] = m2.group(4)
                current_chunk['reinfer_n_tokens'] = int(m2.group(1))
                current_chunk['reinfer_eot_events'] = int(m2.group(2) or 0)
                current_chunk['reinfer_eot_skipped'] = int(m2.group(3) or 0)
                current_chunk['stop_reason'] = 'reinfer'
            # Parse EOT event details (top5 logits)
            if 'EOT at step' in line and 'top5:' in line:
                detail = re.search(r'EOT at step (\d+)(.*?): top5: (.*)', line)
                if detail:
                    if 'reinfer_eot_details' not in current_chunk:
                        current_chunk['reinfer_eot_details'] = []
                    skipped_tag = " [SKIPPED]" if "SKIPPED" in detail.group(2) else ""
                    current_chunk['reinfer_eot_details'].append(
                        f"step {detail.group(1)}{skipped_tag}: {detail.group(3)}")
            continue

        # [STOP] Backward peak
        if '[STOP] Backward peak' in line:
            m2 = re.search(r"step (\d+) '([^']+)': max=([\d.]+)s < min=([\d.]+)s", line)
            if m2:
                current_chunk['stop_reason'] = f"backward_peak (step {m2.group(1)} '{m2.group(2)}', max={m2.group(3)}s < min={m2.group(4)}s)"
            else:
                current_chunk['stop_reason'] = 'backward_peak'
            continue

        # [PEAK MARGIN]
        if '[PEAK MARGIN]' in line:
            m2 = re.search(r"'([^']+)': peak=([\d.]+)s in margin \(([\d.]+-[\d.]+)s\)", line)
            if m2:
                current_chunk['stop_reason'] = f"peak_margin ('{m2.group(1)}', peak={m2.group(2)}s, margin={m2.group(3)}s)"
            else:
                current_chunk['stop_reason'] = 'peak_margin'
            continue

        # [DTW] word[N]='text' Xs-Ys
        m = re.match(r'\s+\[DTW\] word\[(\d+)\]=\'([^\']*)\' ([\d.]+)s-([\d.]+)s', line)
        if m:
            current_chunk['dtw_words'].append({
                'idx': int(m.group(1)),
                'text': m.group(2),
                'start': float(m.group(3)),
                'end': float(m.group(4)),
            })
            continue

        # [DTW CARRYOVER] last word end=Xs
        m = re.search(r'\[DTW CARRYOVER\] last word end=([\d.]+)s', line)
        if m:
            current_chunk['dtw_last_word_end'] = float(m.group(1))
            continue

        # [MODE1] prefill last word: "text" (N tokens)
        m = re.search(r'\[MODE\d+\] prefill last word: "(.*)" \((\d+) tokens\)', line)
        if m:
            current_chunk['prefill_text'] = m.group(1)
            continue

        # [MODE2/3] prefill (2nd-last word only): "text" (N tokens)
        m = re.search(r'\[MODE\d+\] prefill \(2nd-last word only\): "(.*)" \((\d+) tokens\)', line)
        if m:
            current_chunk['prefill_text'] = m.group(1)
            continue

        # [iter=N] CARRYOVER (...): from Xs, Ys carried
        m = re.search(r'\[iter=\d+\] CARRYOVER \(([^)]+)\): from ([\d.]+)s, ([\d.]+)s carried', line)
        if m:
            if current_chunk['carryover_from'] is None:
                current_chunk['carryover_from'] = float(m.group(2))
            current_chunk['carryover_dur'] = float(m.group(3))
            continue

        # [iter=N] CARRYOVER (no emission): entire Xs
        m = re.search(r'\[iter=\d+\] CARRYOVER \(no emission\): entire ([\d.]+)s', line)
        if m:
            current_chunk['stop_reason'] = 'no_emission'
            current_chunk['carryover_dur'] = float(m.group(1))
            continue

        # [iter=N] no tokens decoded — carryover capped: Xs -> Ys (dropped front Zs)
        m = re.search(r'\[iter=\d+\] no tokens decoded — carryover capped: ([\d.]+)s -> ([\d.]+)s \(dropped front ([\d.]+)s\)', line)
        if m:
            current_chunk['stop_reason'] = f"no_decode (capped: {m.group(1)}s -> {m.group(2)}s, dropped {m.group(3)}s)"
            current_chunk['carryover_dur'] = float(m.group(2))
            continue

        # [iter=N] no tokens decoded — carrying over entire Xs
        m = re.search(r'\[iter=\d+\] no tokens decoded — carrying over entire ([\d.]+)s', line)
        if m:
            current_chunk['stop_reason'] = 'no_decode'
            current_chunk['carryover_dur'] = float(m.group(1))
            continue

        # [MODE2/3 DEDUP] Skipped N duplicate tokens
        m = re.search(r'\[MODE\d+ DEDUP\] Skipped (\d+) duplicate tokens', line)
        if m:
            current_chunk['olap_dedup_skipped'] = int(m.group(1))
            continue

        # [MODE2/3 DEDUP] No match — keeping both
        if 'DEDUP] No match' in line:
            current_chunk['olap_dedup_kept_both'] = True
            continue

        # [SPECULATIVE] DTW unreliable
        if '[SPECULATIVE] DTW unreliable' in line:
            current_chunk['speculative'] = True
            continue

        # [SPECULATIVE] Replacing N speculative tokens
        m = re.search(r'\[SPECULATIVE\] Replacing (\d+) speculative tokens', line)
        if m:
            current_chunk['replaced_speculative'] = int(m.group(1))
            continue

        # [iter=N] last chunk — skipping emission
        if 'last chunk' in line and 'skipping' in line:
            current_chunk['stop_reason'] = 'last_chunk (skipped)'
            continue

    if current_chunk is not None:
        chunks.append(current_chunk)

    # Format summary
    out = []
    out.append("=" * 100)
    out.append("CHUNK EXPERIMENT SUMMARY (ours_streaming)")
    out.append("=" * 100)
    out.append("")
    out.append(f"Configuration: {label}")
    if params_info:
        out.append(f"  {params_info}")
    out.append(f"  Total chunks: {len(chunks)}")
    carryover_count = sum(1 for c in chunks if c['carryover_dur'] is not None)
    out.append(f"  Carryover events: {carryover_count}")
    dtw_chunks = [c for c in chunks if c['dtw_words']]
    dtw_allzero = [c for c in dtw_chunks if all(w['start'] == 0.0 and w['end'] == 0.0 for w in c['dtw_words'])]
    dtw_suspicious = [c for c in dtw_chunks
                      if not all(w['start'] == 0.0 and w['end'] == 0.0 for w in c['dtw_words'])
                      and (c['dtw_words'][-1]['end'] <= 0.0
                           or (c['effective'] > 0 and c['dtw_words'][-1]['end'] < c['effective'] * 0.3))]
    if dtw_chunks:
        out.append(f"  DTW chunks: {len(dtw_chunks)}, all-zero: {len(dtw_allzero)} ({100*len(dtw_allzero)/len(dtw_chunks):.1f}%), suspicious: {len(dtw_suspicious)} ({100*len(dtw_suspicious)/len(dtw_chunks):.1f}%)")
    speculative_count = sum(1 for c in chunks if c['speculative'])
    if speculative_count > 0:
        out.append(f"  Speculative emissions: {speculative_count} (replaced in next round)")
    out.append("")
    out.append("=" * 100)
    out.append("CHUNK LIST")
    out.append("=" * 100)
    out.append("")

    for c in chunks:
        carryover_str = f", carryover_in: {c['carryover_in']:.2f}s" if c['carryover_in'] > 0 else ""
        abs_str = f", abs: {c['abs_start']:.2f}-{c['abs_end']:.2f}s" if c['abs_start'] is not None else ""
        out.append(f"[Chunk {c['iter']:03d}] effective: {c['effective']:.2f}s{carryover_str}{abs_str}")

        # Ground truth for this chunk's absolute time range
        if gt_words and c['abs_start'] is not None and c['abs_end'] is not None:
            # Find words that overlap with this chunk's NEW audio region
            # New audio = abs_end - step_duration .. abs_end (exclude carryover region)
            chunk_gt = [w for w in gt_words
                        if w['end'] > c['abs_start'] and w['start'] < c['abs_end']]
            if chunk_gt:
                gt_text = " ".join(w['word'] for w in chunk_gt)
                out.append(f"  Ground truth: {gt_text}")

        if c['replaced_speculative'] > 0:
            out.append(f"  >>> REPLACED {c['replaced_speculative']} speculative tokens from previous round")

        if c['eot_skips']:
            out.append(f"  EOT skipped: {', '.join(c['eot_skips'])}")
        if c.get('reinfer_text'):
            eot_info = ""
            eot_count = c.get('reinfer_eot_events', 0)
            eot_skipped = c.get('reinfer_eot_skipped', 0)
            if eot_count > 0:
                eot_info = f", {eot_count} EOT events"
                if eot_skipped > 0:
                    eot_info += f" ({eot_skipped} skipped)"
            out.append(f"  >>> REINFER: 30s padding ({c['reinfer_n_tokens']} tokens{eot_info}): \"{c['reinfer_text']}\"")
            if c.get('reinfer_eot_details'):
                for detail in c['reinfer_eot_details']:
                    out.append(f"  >>> REINFER EOT: {detail}")

        if c['olap_dedup_skipped'] > 0:
            out.append(f"  >>> OLAP DEDUP: skipped {c['olap_dedup_skipped']} duplicate tokens (kept prev emission)")
        if c['olap_dedup_kept_both']:
            out.append(f"  >>> OLAP DEDUP: no match — kept both")

        if c['stop_reason']:
            out.append(f"  Stop: {c['stop_reason']}")

        if c['emitted']:
            spec_tag = " [SPECULATIVE]" if c['speculative'] else ""
            out.append(f"  Generated: {c['emitted']}{spec_tag}")

        if c['dtw_words']:
            all_zero = all(w['start'] == 0.0 and w['end'] == 0.0 for w in c['dtw_words'])
            last_end = c['dtw_words'][-1]['end']
            last_start = c['dtw_words'][-1]['start']
            eff = c['effective']
            warnings = []
            if all_zero:
                warnings.append(f"ALL-ZERO ({len(c['dtw_words'])} tokens vs {eff:.1f}s audio)")
            else:
                if last_end <= 0.0:
                    warnings.append(f"last word end={last_end:.2f}s")
                elif eff > 0 and last_end < eff * 0.3:
                    warnings.append(f"last word end={last_end:.2f}s vs audio={eff:.1f}s (too early, {last_end/eff*100:.0f}%)")
            warn_str = f" *** {', '.join(warnings)} ***" if warnings else ""
            out.append(f"  DTW words:{warn_str}")
            for w in c['dtw_words']:
                out.append(f"    [{w['idx']}] '{w['text']}' {w['start']:.2f}s-{w['end']:.2f}s")

        if c['dtw_last_word_end'] is not None:
            last_word = c['dtw_words'][-1]['text'] if c['dtw_words'] else '?'
            out.append(f"  Last word: '{last_word}' end={c['dtw_last_word_end']:.2f}s")

        if c['prefill_text'] is not None:
            out.append(f"  Prefill: \"{c['prefill_text']}\"")

        if c['carryover_from'] is not None:
            dur_str = f"{c['carryover_dur']:.2f}s carried" if c['carryover_dur'] is not None else "?"
            out.append(f"  >>> CARRYOVER from {c['carryover_from']:.2f}s ({dur_str})")

        out.append("")

    return "\n".join(out)


def process_run_result(run_result, ground_truth, truncate_hyp=False, language="en"):
    """Process a binary run result into parsed metrics."""
    combined = run_result["combined"]

    records = parse_committed_output(combined)
    avg_latency = parse_avg_latency(combined)
    buffer_stats = parse_buffer_stats(combined)
    input_secs, content_mel_lens = parse_input_stats(combined)
    ca_copy_per_iter, ca_calc_per_iter = parse_cross_attn_times(combined)

    # Try direct per-iteration chrono timing first (simul_streaming, simul_whisper, ours)
    mel_pi, enc_pi, pf_pi, dec_pi, dpt_pi, ntok_pi = parse_iter_timing(combined)
    n_tokens_per_iter = []
    if enc_pi:
        mel_per_iter = mel_pi
        encode_per_iter = enc_pi
        prefill_per_iter = pf_pi
        decode_per_iter = dec_pi
        decode_per_token = dpt_pi
        n_tokens_per_iter = ntok_pi
    else:
        # Fall back to cumulative whisper_print_timings parsing (Whisper-S, WhisperFlow)
        encode_per_iter, decode_per_iter, decode_per_token, mel_per_iter, prefill_per_iter = parse_per_iteration_encode_decode(combined)

    hypothesis = "".join(r["transcript"] for r in records).strip()
    metrics = compute_metrics(ground_truth, hypothesis, truncate_hyp=truncate_hyp, language=language)

    # Latency stats from per-token latency
    latencies = [r["latency"] for r in records if r["latency"] is not None]
    latency_stats = {}
    if latencies:
        lat_arr = np.array(latencies)
        latency_stats = {
            "mean": float(np.mean(lat_arr)),
            "median": float(np.median(lat_arr)),
            "std": float(np.std(lat_arr)),
            "p90": float(np.percentile(lat_arr, 90)),
            "p95": float(np.percentile(lat_arr, 95)),
            "min": float(np.min(lat_arr)),
            "max": float(np.max(lat_arr)),
        }

    # Per-word latency
    word_records = merge_subword_tokens(records)
    word_latencies = [r["latency"] for r in word_records if r["latency"] is not None]
    word_latency_stats = {}
    if word_latencies:
        wl_arr = np.array(word_latencies)
        word_latency_stats = {
            "mean": float(np.mean(wl_arr)),
            "median": float(np.median(wl_arr)),
            "std": float(np.std(wl_arr)),
            "p90": float(np.percentile(wl_arr, 90)),
            "p95": float(np.percentile(wl_arr, 95)),
            "min": float(np.min(wl_arr)),
            "max": float(np.max(wl_arr)),
        }

    # Buffer stats
    buf_stats = {}
    if buffer_stats:
        buf_arr = np.array(buffer_stats)
        buf_stats = {
            "mean": float(np.mean(buf_arr)),
            "max": float(np.max(buf_arr)),
            "min": float(np.min(buf_arr)),
        }

    # Input stats (whisper input length per iteration)
    input_stats = {}
    if input_secs:
        is_arr = np.array(input_secs)
        cml_arr = np.array(content_mel_lens)
        input_stats = {
            "input_sec_mean": float(np.mean(is_arr)),
            "input_sec_min": float(np.min(is_arr)),
            "input_sec_max": float(np.max(is_arr)),
            "content_mel_mean": float(np.mean(cml_arr)),
            "content_mel_min": float(np.min(cml_arr)),
            "content_mel_max": float(np.max(cml_arr)),
        }

    # TTFT: first token's emission_time (wall-clock time when first output appeared)
    ttft = None
    if word_records:
        for wr in word_records:
            if wr.get("emission_time") is not None:
                ttft = wr["emission_time"]
                break

    buffer_overflow = "audio buffer exceeded 30s" in run_result["stderr"]
    crashed = run_result["returncode"] not in (0, None)

    encode_stats = compute_stats_ms(encode_per_iter)
    decode_stats = compute_stats_ms(decode_per_iter)
    decode_per_token_stats = compute_stats_ms(decode_per_token)
    mel_stats = compute_stats_ms(mel_per_iter)
    prefill_stats = compute_stats_ms(prefill_per_iter)
    ca_copy_stats = compute_stats_ms(ca_copy_per_iter)
    ca_calc_stats = compute_stats_ms(ca_calc_per_iter)
    n_tokens_per_iter_stats = compute_stats_ms(n_tokens_per_iter) if n_tokens_per_iter else {}

    # CIF inference times (SimulWhisper only, in us → convert to ms for display)
    cif_times_us = parse_cif_inference_times(combined)
    cif_times_ms = [t / 1000.0 for t in cif_times_us]
    cif_stats = compute_stats_ms(cif_times_ms)

    return {
        "metrics": metrics,
        "hypothesis": hypothesis,
        "records": records,
        "word_records": word_records,
        "avg_latency": avg_latency,
        "latency_stats": latency_stats,
        "word_latency_stats": word_latency_stats,
        "buffer_stats": buf_stats,
        "input_stats": input_stats,
        "encode_stats": encode_stats,
        "decode_stats": decode_stats,
        "decode_per_token_stats": decode_per_token_stats,
        "mel_stats": mel_stats,
        "prefill_stats": prefill_stats,
        "ca_copy_stats": ca_copy_stats,
        "ca_calc_stats": ca_calc_stats,
        "n_tokens_per_iter_stats": n_tokens_per_iter_stats,
        "cif_stats": cif_stats,
        "ttft": ttft,
        "wall_time": run_result["wall_time"],
        "num_tokens": len(records),
        "num_words": len(word_records),
        "buffer_overflow": buffer_overflow,
        "crashed": crashed,
        "returncode": run_result["returncode"],
    }


def fmt_val(val, fmt=".3f", suffix="s"):
    """Format a value or return N/A."""
    if val is not None:
        return f"{val:{fmt}}{suffix}"
    return "N/A"


def main():
    parser = argparse.ArgumentParser(description="Compare Whisper-S vs WhisperFlow on LibriSpeech")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--model", type=str, default="base")
    parser.add_argument("--step", type=int, default=2000, help="Chunk step size in ms")
    parser.add_argument("--data_root", type=str, default="../../data")
    parser.add_argument("--dataset", type=str, choices=["librispeech", "fleurs", "tedlium"], default=None,
                        help="Force dataset: 'librispeech', 'fleurs', or 'tedlium'. Default: auto (LibriSpeech for en, FLEURS for others)")
    parser.add_argument("--talk-index", type=int, default=0,
                        help="TED-LIUM only: index of talk to use (0-10, default 0)")
    parser.add_argument("--whisperflow_dir", type=str, default="..",
                        help="Path to whisperflow root (serialized binaries and models)")
    parser.add_argument("--pipeline_dir", type=str, default=None,
                        help="Path to pipeline worktree (if set, runs 3-way comparison)")
    parser.add_argument("--output_dir", type=str, default="./comparison_results")
    parser.add_argument("--no_gpu", action="store_true", help="Disable GPU")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--audio-file", type=str, default=None,
                        help="Path to pre-cached audio WAV file (bypasses LibriSpeech loading)")
    parser.add_argument("--ground-truth", type=str, default=None,
                        help="Path to ground truth text file (required with --audio-file)")
    parser.add_argument("--bin-dir", type=str, default=None,
                        help="Subdirectory for binaries (e.g., 'build/bin' for CMake builds)")
    parser.add_argument("--bin-dir-cpu", type=str, default=None,
                        help="Separate bin dir for CPU-only builds (used by ours_cpu)")
    parser.add_argument("--adb-device", type=str, default=None,
                        help="Run binaries on remote device via adb (e.g., '192.168.31.161:5555')")
    parser.add_argument("--adb-remote-dir", type=str, default="/data/local/tmp/whisperflow",
                        help="Remote directory on adb device where binaries/models are located")
    parser.add_argument("--power-monitor", action="store_true",
                        help="Enable power consumption monitoring")
    parser.add_argument("--power-sysfs", type=str,
                        default="/sys/class/power_supply/qcom-battmgr-bat",
                        help="Sysfs path for power supply")
    parser.add_argument("--power-interval-ms", type=int, default=100,
                        help="Power sampling interval in ms")
    parser.add_argument("--offline", action="store_true",
                        help="Include offline Whisper baseline (per-sample 30s-padded inference)")
    parser.add_argument("--whisper-streaming", action="store_true",
                        help="Include Whisper-S (baseline) in comparison")
    parser.add_argument("--whisperflow", action="store_true",
                        help="Include WhisperFlow (serialized) in comparison")
    parser.add_argument("--simul-streaming", action="store_true",
                        help="Include SimulStreaming (AlignAtt) in comparison")
    parser.add_argument("--simul-whisper", action="store_true",
                        help="Include SimulWhisper (AlignAtt + CIF) in comparison")
    parser.add_argument("--ours-streaming", action="store_true",
                        help="Include Ours (backward peak detection) in comparison")
    parser.add_argument("--whisperflow-gpu-decode", action="store_true",
                        help="Include WhisperFlow with GPU decode (no KV copy) in comparison")
    parser.add_argument("--ground-truth-timestamps", type=str, default=None,
                        help="Path to ground truth timestamps JSON (from generate_ground_truth_timestamps.py). "
                             "When provided, latency is recomputed as emission_time - gt_end_time for all systems.")
    parser.add_argument("--only", type=str, nargs="+", default=None,
                        help="Run only specific system(s) (skip others). Can specify multiple.")
    parser.add_argument("--simul-frame-threshold", type=int, default=4,
                        help="SimulStreaming/SimulWhisper AlignAtt frame threshold")
    parser.add_argument("--simul-rewind-threshold", type=int, default=200,
                        help="SimulStreaming/SimulWhisper AlignAtt rewind threshold")
    # Ours streaming parameters
    parser.add_argument("--ours-smoothing", type=int, default=10,
                        help="Ours: smoothing window (default 10)")
    parser.add_argument("--ours-median-filter", type=int, default=7,
                        help="Ours: median filter window (default 7)")
    parser.add_argument("--ours-cross-attn-layer", type=int, default=-1,
                        help="Ours: decoder layer for cross-attention (-1=last)")
    parser.add_argument("--ours-peak-margin", type=float, nargs="+", default=[0.6],
                        help="Ours: peak margin seconds (default 0.6). "
                             "Multiple values create separate ours variants for comparison.")
    parser.add_argument("--ours-carryover-overlap", type=float, default=0.6,
                        help="Ours: carryover overlap seconds before peak (default 0.6)")
    parser.add_argument("--ours-prompt-prefill", type=int, default=1,
                        help="Ours: prompt prefill N tokens from prev chunk (default 1)")
    parser.add_argument("--ours-min-chunk", type=float, default=1.0,
                        help="Ours: skip round if input audio shorter than this (seconds, default 1.0)")
    parser.add_argument("--ours-carryover-mode", type=int, nargs="+", default=[0],
                        help="Ours: carryover mode(s) (0=peak, 1=pfill, 2=olap, 3=olap+reinfer, default 0). "
                             "Multiple values register variants for each mode.")
    parser.add_argument("--ours-word-end-offset", type=float, nargs="+", default=[-0.2],
                        help="Ours: offset(s) after DTW word end (default -0.2). "
                             "Multiple values create separate ours variants for comparison.")
    parser.add_argument("--no-realtime", action="store_true", default=False,
                        help="Skip real-time sleep for all systems, process chunks as fast as possible")
    parser.add_argument("--ours-no-postprocess", action="store_true", default=False,
                        help="Ours: disable dedup and prompt prefill (raw output per chunk)")
    parser.add_argument("--ours-skip-eot", action="store_true", default=False,
                        help="Ours: suppress EOT token generation (useful for Korean)")
    parser.add_argument("--ours-debug", action="store_true", default=False,
                        help="Ours: enable debug output (per-round emitted text, DTW details)")
    parser.add_argument("--zero-pad", type=float, default=0.0,
                        help="Append N seconds of silence to audio (default 0.0)")
    args = parser.parse_args()

    if args.audio_file and not args.ground_truth:
        auto_json = os.path.splitext(os.path.abspath(args.audio_file))[0] + "_ground_truth.json"
        if not os.path.exists(auto_json):
            parser.error("--ground-truth is required when --audio-file is specified and no JSON ground truth found")
    if args.ground_truth and not args.audio_file:
        parser.error("--audio-file is required when --ground-truth is specified")

    wf_dir = os.path.abspath(args.whisperflow_dir)

    if args.adb_device:
        # In adb mode, paths are relative to adb_remote_dir
        remote_dir = args.adb_remote_dir
        if args.bin_dir:
            bin_dir = args.bin_dir
        else:
            bin_dir = "build-vk/bin"
        baseline_bin = f"{bin_dir}/whisper_streaming_cpp"
        serialized_bin = f"{bin_dir}/whisper_streaming_cpp_optimized"
        simul_bin = f"{bin_dir}/simul_streaming"
        model_path = f"models/ggml-{args.model}.bin"
        onnx_model_path = f"models/{args.model}"
    else:
        # Determine binary directory (auto-detect CMake build/bin/ or build-test/bin/)
        # Prefer cmake build directories over root (root may have stale binaries)
        if args.bin_dir:
            bin_dir = os.path.join(wf_dir, args.bin_dir)
        else:
            bin_dir = wf_dir
            for candidate in ["build-test/bin", "build/bin"]:
                cmake_bin_dir = os.path.join(wf_dir, candidate)
                if os.path.exists(os.path.join(cmake_bin_dir, "whisper_streaming_cpp_optimized")):
                    bin_dir = cmake_bin_dir
                    print(f"Auto-detected CMake build directory: {bin_dir}")
                    break

        baseline_bin = os.path.join(bin_dir, "whisper_streaming_cpp")
        serialized_bin = os.path.join(bin_dir, "whisper_streaming_cpp_optimized")
        simul_bin = os.path.join(bin_dir, "simul_streaming")
        model_path = os.path.join(wf_dir, "models", f"ggml-{args.model}.bin")
        onnx_model_path = os.path.join(wf_dir, "models", args.model)

    # Also check build-test/bin/ for simul_streaming (cmake build)
    if args.adb_device:
        simul_whisper_bin = f"{bin_dir}/simul_whisper"
        ours_bin = f"{bin_dir}/ours_streaming"
        model_name = args.model.replace("-", "_")
        cif_model_path = f"models/cif_{model_name}.bin"
    else:
        if not os.path.exists(simul_bin):
            alt_simul = os.path.join(wf_dir, "build-test", "bin", "simul_streaming")
            if os.path.exists(alt_simul):
                simul_bin = alt_simul

        # SimulWhisper binary
        simul_whisper_bin = os.path.join(bin_dir, "simul_whisper")
        if not os.path.exists(simul_whisper_bin):
            alt_sw = os.path.join(wf_dir, "build-test", "bin", "simul_whisper")
            if os.path.exists(alt_sw):
                simul_whisper_bin = alt_sw

        # Ours streaming binary
        ours_bin = os.path.join(bin_dir, "ours_streaming")
        if not os.path.exists(ours_bin):
            alt_ours = os.path.join(wf_dir, "build-test", "bin", "ours_streaming")
            if os.path.exists(alt_ours):
                ours_bin = alt_ours

        # CIF model path for SimulWhisper
        model_name = args.model.replace("-", "_")  # e.g., large-v2 -> large_v2
        cif_model_path = os.path.join(wf_dir, "models", f"cif_{model_name}.bin")

    has_dtw_modes = any(m in (1, 2, 3) for m in args.ours_carryover_mode)

    # Map of system key -> (label, binary_path, source_dir, kind)
    # kind is used to select args and output parser
    all_systems = {}
    if args.whisper_streaming:
        all_systems["baseline"] = ("Whisper-Streaming", baseline_bin, wf_dir, "baseline")
    if args.whisperflow or args.whisperflow_gpu_decode or args.pipeline_dir:
        all_systems["serialized"] = ("WhisperFlow", serialized_bin, wf_dir, "whisperflow")

    if args.pipeline_dir:
        pipeline_dir = os.path.abspath(args.pipeline_dir)
        if args.bin_dir:
            pipeline_bin_dir = os.path.join(pipeline_dir, args.bin_dir)
        else:
            pipeline_bin_dir = pipeline_dir
            cmake_pipeline_dir = os.path.join(pipeline_dir, "build", "bin")
            if (not os.path.exists(os.path.join(pipeline_dir, "whisper_streaming_cpp_optimized"))
                    and os.path.exists(os.path.join(cmake_pipeline_dir, "whisper_streaming_cpp_optimized"))):
                pipeline_bin_dir = cmake_pipeline_dir
        pipeline_bin = os.path.join(pipeline_bin_dir, "whisper_streaming_cpp_optimized")
        if not os.path.exists(pipeline_bin):
            print(f"ERROR: pipeline binary not found at {pipeline_bin}")
            sys.exit(1)
        all_systems["pipeline"] = ("WhisperFlow (pipeline)", pipeline_bin, pipeline_dir, "whisperflow")

    if args.simul_streaming:
        all_systems["simul"] = ("SimulStreaming", simul_bin, wf_dir, "simul")

    if args.simul_whisper:
        if not args.adb_device and not os.path.exists(cif_model_path):
            print(f"ERROR: CIF model not found at {cif_model_path}")
            print(f"  Run convert_cif_weights.py first to generate CIF binary weights.")
            sys.exit(1)
        all_systems["simul_whisper"] = ("SimulWhisper", simul_whisper_bin, wf_dir, "simul_whisper")

    if args.ours_streaming:
        # When any carryover_mode is 1, 2, or 3, register variants
        if has_dtw_modes:
            # Register mode 0 (default) only if explicitly requested
            if 0 in args.ours_carryover_mode:
                key = "ours_default"
                label = "Ours (default)"
                all_systems[key] = (label, ours_bin, wf_dir, key)
            # Register mode grid (peak_margin x offset) for each DTW mode
            mode_prefixes = {1: "pfill", 2: "olap", 3: "olap_reinfer"}
            for mode in args.ours_carryover_mode:
                if mode not in mode_prefixes:
                    continue
                mode_prefix = mode_prefixes[mode]
                for margin in args.ours_peak_margin:
                    for offset in args.ours_word_end_offset:
                        key = f"ours_{mode_prefix}_m{margin:.1f}_off{offset:+.2f}"
                        label = f"Ours ({mode_prefix}_m{margin:.1f}_off{offset:+.2f})"
                        all_systems[key] = (label, ours_bin, wf_dir, key)
        else:
            all_systems["ours"] = ("Ours (BackwardPeak)", ours_bin, wf_dir, "ours")
            # Only register ours_cpu for single-variant mode
            if args.adb_device:
                if args.bin_dir_cpu:
                    ours_cpu_bin = f"{args.bin_dir_cpu}/ours_streaming"
                else:
                    ours_cpu_bin = ours_bin
            else:
                if args.bin_dir_cpu:
                    ours_cpu_bin = os.path.join(wf_dir, args.bin_dir_cpu, "ours_streaming")
                else:
                    ours_cpu_bin = ours_bin
            all_systems["ours_cpu"] = ("Ours (BackwardPeak, CPU)", ours_cpu_bin, wf_dir, "ours_cpu")

    if args.whisperflow_gpu_decode:
        all_systems["serialized_gpu"] = ("WhisperFlow (GPU decode)", serialized_bin, wf_dir, "whisperflow_gpu")

    # Apply --only filter
    if args.only:
        for o in args.only:
            if o not in all_systems:
                print(f"ERROR: --only {o} not available. Available: {list(all_systems.keys())}")
                sys.exit(1)
        all_systems = {o: all_systems[o] for o in args.only}

    # Build runs list: (label, binary_path, source_dir, kind)
    runs = list(all_systems.values())

    # Check essential files exist (skip for adb mode — files are on remote device)
    if not args.adb_device:
        check_files = []
        if args.ours_streaming:
            check_files.append((onnx_model_path, "ONNX model directory"))
        has_non_onnx = args.whisper_streaming or args.whisperflow or args.simul_streaming
        if args.simul_whisper:
            check_files.append((onnx_model_path, "ONNX model directory (simul_whisper)"))
        if has_non_onnx:
            check_files.append((model_path, "model file"))
        for label, binary, source_dir, kind in runs:
            check_files.append((binary, f"{label} binary"))
        for path, name in check_files:
            if not os.path.exists(path):
                print(f"ERROR: {name} not found at {path}")
                sys.exit(1)

    # Output directory
    # When using --audio-file, try to extract sample count from filename (e.g., "test_audio_5samples.wav")
    total = args.num_samples
    if args.audio_file:
        m = re.search(r'(\d+)samples', os.path.basename(args.audio_file))
        if m:
            total = int(m.group(1))
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_way = len(runs)
    device_tag = f"_adb_{args.adb_device.replace(':', '_')}" if args.adb_device else ""
    lang_tag = f"_{args.language}" if args.language != "en" else ""
    ds_tag = f"_{args.dataset}" if args.dataset else ""
    talk_tag = f"_talk{args.talk_index}" if args.dataset == "tedlium" else ""
    result_dir = os.path.join(
        args.output_dir,
        f"comparison_{date_str}_{args.model}_{total}samples_step{args.step}{ds_tag}{talk_tag}{lang_tag}{device_tag}",
    )
    os.makedirs(result_dir, exist_ok=True)

    # Load audio and ground truth
    if args.audio_file:
        wav_path = os.path.abspath(args.audio_file)
        if not os.path.exists(wav_path):
            print(f"ERROR: audio file not found: {wav_path}")
            sys.exit(1)
        print(f"Using provided audio file: {wav_path}")
        audio_data, sr = sf.read(wav_path, dtype="float32")
        total_duration = len(audio_data) / 16000.0
        if args.ground_truth:
            gt_file = os.path.abspath(args.ground_truth)
            if not os.path.exists(gt_file):
                print(f"ERROR: ground truth file not found: {gt_file}")
                sys.exit(1)
            with open(gt_file, "r") as f:
                ground_truth = f.read().strip()
        else:
            # Auto-extract from JSON ground truth timestamps
            auto_json = os.path.splitext(wav_path)[0] + "_ground_truth.json"
            if os.path.exists(auto_json):
                with open(auto_json, "r") as f:
                    gt_data = json.load(f)
                ground_truth = " ".join(w["word"] for w in gt_data["words"])
                print(f"Auto-extracted ground truth from: {auto_json}")
            else:
                print(f"ERROR: no ground truth found. Provide --ground-truth or place JSON at {auto_json}")
                sys.exit(1)
    else:
        if args.dataset:
            use_fleurs = args.dataset == "fleurs"
            use_tedlium = args.dataset == "tedlium"
        else:
            use_fleurs = args.language != "en"
            use_tedlium = False

        if use_tedlium:
            dataset = load_tedlium_samples()
            is_audio_array = True
        elif use_fleurs:
            # FLEURS dataset (non-English languages)
            dataset = load_fleurs_samples(args.language)
            print(f"Loaded {len(dataset)} FLEURS samples")
            is_audio_array = True  # FLEURS returns (audio_array, transcript)
        else:
            # LibriSpeech (English)
            if not HAS_LIBROSA:
                print("ERROR: librosa is required for LibriSpeech loading.")
                print("  Use --audio-file and --ground-truth to bypass LibriSpeech.")
                sys.exit(1)
            print(f"Loading LibriSpeech dataset from {args.data_root}...")
            dataset = load_librispeech_samples(root=args.data_root)
            print(f"Loaded {len(dataset)} total samples")
            is_audio_array = False  # LibriSpeech returns (flac_path, transcript)

        # TED-LIUM: each sample is a full talk, use talk_index to select
        if use_tedlium:
            talk_idx = min(args.talk_index, len(dataset) - 1)
            total = 1  # single talk

            cache_dir = os.path.join(args.output_dir, "cached_audio")
            os.makedirs(cache_dir, exist_ok=True)
            ds_tag = "_tedlium"
            wav_path = os.path.join(cache_dir, f"test_audio_talk{talk_idx}{ds_tag}.wav")
            gt_path = os.path.join(cache_dir, f"ground_truth_talk{talk_idx}{ds_tag}.txt")

            if os.path.exists(wav_path) and os.path.exists(gt_path):
                print(f"Reusing cached audio: {wav_path}")
                audio_data, sr = sf.read(wav_path, dtype="float32")
                total_duration = len(audio_data) / 16000.0
                with open(gt_path, "r") as f:
                    ground_truth = f.read().strip()
            else:
                audio_array, transcript = dataset[talk_idx]
                concatenated = np.array(audio_array, dtype=np.float32)
                # Append 1s silence at the end
                concatenated = np.concatenate([concatenated, np.zeros(16000, dtype=np.float32)])
                total_duration = len(concatenated) / 16000.0
                ground_truth = transcript
                print(f"Using TED-LIUM talk {talk_idx}: {total_duration:.1f}s ({total_duration/60:.1f}min)")

                sf.write(wav_path, concatenated, 16000)
                with open(gt_path, "w") as f:
                    f.write(ground_truth)
                print(f"Saved test audio to {wav_path}")
        else:
            total = min(args.num_samples, len(dataset))

            cache_dir = os.path.join(args.output_dir, "cached_audio")
            os.makedirs(cache_dir, exist_ok=True)
            lang_tag = f"_{args.language}" if args.language != "en" else ""
            ds_tag = f"_{args.dataset}" if args.dataset else ""
            pad_tag = f"_{int(args.zero_pad)}spad" if args.zero_pad > 0 else "_1spad"
            wav_path = os.path.join(cache_dir, f"test_audio_{total}samples{lang_tag}{ds_tag}{pad_tag}.wav")
            gt_path = os.path.join(cache_dir, f"ground_truth_{total}samples{lang_tag}{ds_tag}.txt")

            if os.path.exists(wav_path) and os.path.exists(gt_path):
                print(f"Reusing cached audio: {wav_path}")
                audio_data, sr = sf.read(wav_path, dtype="float32")
                total_duration = len(audio_data) / 16000.0
                with open(gt_path, "r") as f:
                    ground_truth = f.read().strip()
            else:
                print(f"Concatenating {total} samples...")
                all_audio = []
                ground_truth_parts = []
                inter_pad = args.zero_pad if args.zero_pad > 0 else (1.0 if use_fleurs else 0.0)
                for i in range(total):
                    if is_audio_array:
                        audio_array, transcript = dataset[i]
                        all_audio.append(np.array(audio_array, dtype=np.float32))
                    else:
                        flac_path, transcript = dataset[i]
                        audio, _ = librosa.load(flac_path, sr=16000, dtype=np.float32)
                        all_audio.append(audio)
                    ground_truth_parts.append(transcript)
                    if inter_pad > 0 and i < total - 1:
                        all_audio.append(np.zeros(int(16000 * inter_pad), dtype=np.float32))

                # Append 1s silence at the end (always)
                all_audio.append(np.zeros(int(16000 * 1.0), dtype=np.float32))
                concatenated = np.concatenate(all_audio).astype(np.float32)
                total_duration = len(concatenated) / 16000.0
                ground_truth = " ".join(ground_truth_parts)

                sf.write(wav_path, concatenated, 16000)
                with open(gt_path, "w") as f:
                    f.write(ground_truth)
                print(f"Saved test audio to {wav_path}")

    print(f"Total audio duration: {total_duration:.2f}s")

    # Auto-generate ground truth timestamps if not provided
    gt_ts_path = None
    if args.ground_truth_timestamps:
        gt_ts_path = os.path.abspath(args.ground_truth_timestamps)
        if not os.path.exists(gt_ts_path):
            print(f"WARNING: ground truth timestamps file not found: {gt_ts_path}")
            gt_ts_path = None
    else:
        auto_ts_path = os.path.splitext(wav_path)[0] + "_ground_truth.json"
        if not os.path.exists(auto_ts_path):
            print(f"\nGround truth timestamps not found. Auto-generating: {auto_ts_path}")
            gen_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "generate_ground_truth_timestamps.py")
            if os.path.exists(gen_script):
                gen_cmd = [sys.executable, gen_script,
                           "--audio", wav_path,
                           "--model", args.model,
                           "--language", args.language,
                           "--output", auto_ts_path]
                if args.no_gpu:
                    gen_cmd += ["--device", "cpu"]
                print(f"  Running: {' '.join(gen_cmd)}")
                gen_result = subprocess.run(gen_cmd, capture_output=True, text=True)
                if gen_result.returncode != 0:
                    print(f"  WARNING: ground truth generation failed:")
                    print(f"  {gen_result.stderr[-500:]}")
                else:
                    print(f"  {gen_result.stdout.strip()}")
            else:
                print(f"  WARNING: generate_ground_truth_timestamps.py not found at {gen_script}")
        if os.path.exists(auto_ts_path):
            gt_ts_path = auto_ts_path

    # Load ground truth word timestamps early (needed for chunk summary)
    gt_words = None
    if gt_ts_path and os.path.exists(gt_ts_path):
        with open(gt_ts_path, "r") as f:
            gt_data = json.load(f)
        gt_words = gt_data.get("words", [])
        print(f"Ground truth timestamps: {gt_ts_path} ({len(gt_words)} words)")
    elif gt_ts_path:
        print(f"Ground truth timestamps: {gt_ts_path}")

    # Common args for baseline/whisperflow
    common_args = ["-l", args.language, "--step", str(args.step), "-kc", "-dtw", args.model]
    if args.no_gpu:
        common_args.append("-ng")
    if args.no_realtime:
        common_args.append("--no-realtime")

    # SimulStreaming args (different binary, different flags)
    simul_args = ["-l", args.language, "--step", str(args.step), "-dtw", args.model,
                  "-ft", str(args.simul_frame_threshold), "-rt", str(args.simul_rewind_threshold)]
    if args.no_gpu:
        simul_args.append("-ng")
    if args.no_realtime:
        simul_args.append("--no-realtime")

    # SimulWhisper args = simul_args + CIF model
    simul_whisper_args = simul_args + ["-cf", cif_model_path]

    # Ours streaming args
    # peak_margin and word_end_offset are added per-variant when carryover_mode==1/2/3
    ours_args = ["-l", args.language, "--step", str(args.step),
                 "--smoothing", str(args.ours_smoothing),
                 "--median-filter", str(args.ours_median_filter),
                 "--cross-attn-layer", str(args.ours_cross_attn_layer)]
    # ONNX ours_streaming auto-detects alignment heads from model path (no -dtw needed)
    if args.ours_carryover_mode == [0]:
        # Mode 0 only: single peak_margin value
        if args.ours_peak_margin[0] > 0:
            ours_args += ["--peak-margin", str(args.ours_peak_margin[0])]
    if args.ours_carryover_overlap > 0:
        ours_args += ["--carryover-overlap", str(args.ours_carryover_overlap)]
    if args.ours_prompt_prefill > 0:
        ours_args += ["--prompt-prefill", str(args.ours_prompt_prefill)]
    if args.ours_min_chunk > 0:
        ours_args += ["--min-chunk", str(args.ours_min_chunk)]
    # Don't add --carryover-mode to base ours_args; it's added per-variant
    if args.no_realtime:
        ours_args.append("--no-realtime")
    if args.ours_no_postprocess:
        ours_args.append("--no-postprocess")
    if args.ours_skip_eot:
        ours_args.append("--skip-eot")
    if args.ours_debug:
        ours_args.append("--debug")
    if args.no_gpu:
        ours_args.append("-ng")

    # Build per-variant args dict for carryover_mode==1/2/3 (peak_margin x offset grid)
    ours_variant_args = {}
    if has_dtw_modes:
        # Mode 0 baselines (peak-based, no DTW) — one per peak_margin value
        # Build base args without --carryover-mode
        ours_base_args = ["-l", args.language, "--step", str(args.step),
                          "--smoothing", str(args.ours_smoothing),
                          "--median-filter", str(args.ours_median_filter),
                          "--cross-attn-layer", str(args.ours_cross_attn_layer)]
        if args.ours_carryover_overlap > 0:
            ours_base_args += ["--carryover-overlap", str(args.ours_carryover_overlap)]
        if args.ours_prompt_prefill > 0:
            ours_base_args += ["--prompt-prefill", str(args.ours_prompt_prefill)]
        if args.ours_min_chunk > 0:
            ours_base_args += ["--min-chunk", str(args.ours_min_chunk)]
        if args.no_realtime:
            ours_base_args.append("--no-realtime")
        if args.ours_no_postprocess:
            ours_base_args.append("--no-postprocess")
        if args.ours_skip_eot:
            ours_base_args.append("--skip-eot")
        if args.ours_debug:
            ours_base_args.append("--debug")
        # Default mode 0 (with postprocess: dedup + prefill)
        ours_default_args = ["-l", args.language, "--step", str(args.step),
                             "--smoothing", str(args.ours_smoothing),
                             "--median-filter", str(args.ours_median_filter),
                             "--cross-attn-layer", str(args.ours_cross_attn_layer)]
        if args.ours_carryover_overlap > 0:
            ours_default_args += ["--carryover-overlap", str(args.ours_carryover_overlap)]
        if args.ours_prompt_prefill > 0:
            ours_default_args += ["--prompt-prefill", str(args.ours_prompt_prefill)]
        if args.ours_min_chunk > 0:
            ours_default_args += ["--min-chunk", str(args.ours_min_chunk)]
        if args.no_realtime:
            ours_default_args.append("--no-realtime")
        if args.ours_debug:
            ours_default_args.append("--debug")
        default_variant = ours_default_args.copy()
        if args.ours_peak_margin[0] > 0:
            default_variant += ["--peak-margin", str(args.ours_peak_margin[0])]
        ours_variant_args["ours_default"] = default_variant

        # Mode 0 without postprocess (fair comparison with DTW variants)
        for margin in args.ours_peak_margin:
            key = f"ours_peak_m{margin:.1f}"
            variant_args = ours_base_args.copy()
            if margin > 0:
                variant_args += ["--peak-margin", str(margin)]
            ours_variant_args[key] = variant_args

        # Mode 1/2/3 variants — each mode gets its own grid
        mode_prefixes = {1: "pfill", 2: "olap", 3: "olap_reinfer"}
        for mode in args.ours_carryover_mode:
            if mode not in mode_prefixes:
                continue
            mode_prefix = mode_prefixes[mode]
            for margin in args.ours_peak_margin:
                for offset in args.ours_word_end_offset:
                    key = f"ours_{mode_prefix}_m{margin:.1f}_off{offset:+.2f}"
                    variant_args = ours_args.copy() + ["--carryover-mode", str(mode)]
                    if margin > 0:
                        variant_args += ["--peak-margin", str(margin)]
                    if offset != 0:
                        variant_args += ["--word-end-offset", str(offset)]
                    ours_variant_args[key] = variant_args

    # Ours CPU args: same as ours_args but always with -ng
    ours_cpu_args = ours_args.copy()
    if "-ng" not in ours_cpu_args:
        ours_cpu_args.append("-ng")

    # WhisperFlow-specific args: audio tag (hush word) + audio context
    # Use audio_tag from the source_dir of each run
    def get_whisperflow_extra_args(source_dir):
        audio_tag_path = os.path.join(source_dir, "audio_tag", f"{args.model}_0.5s_avg.csv")
        if os.path.exists(audio_tag_path):
            return ["-ac", "-1", "-at", audio_tag_path]
        else:
            print(f"ERROR: audio tag file not found at {audio_tag_path}")
            print(f"  WhisperFlow requires audio tag. Copy from whisperflow/audio_tag/")
            sys.exit(1)

    # Power monitoring setup
    power_enabled = args.power_monitor and os.path.exists(args.power_sysfs)
    if args.power_monitor:
        if power_enabled:
            print(f"Power monitoring enabled: {args.power_sysfs}")
        else:
            print(f"WARNING: Power sysfs path not found: {args.power_sysfs}")
            print("  Power monitoring disabled.")

    # Run offline baseline (per-sample 30s-padded Whisper inference)
    results = {}
    if args.offline and not args.audio_file:
        print(f"\n{'='*80}")
        print(f"Running Offline (per-sample, 30s padded)...")
        print(f"{'='*80}")
        import whisper as whisper_offline
        import time as _time
        offline_model = whisper_offline.load_model(args.model,
                                                   device="cpu" if args.no_gpu else "cuda")
        offline_hyps = []
        t_offline_start = _time.perf_counter()
        for i in range(total):
            if is_audio_array:
                audio_arr, _ = dataset[i]
                audio_np = np.array(audio_arr, dtype=np.float32)
            else:
                flac_path, _ = dataset[i]
                audio_np, _ = librosa.load(flac_path, sr=16000, dtype=np.float32)
            # Pad to 30s and run single-pass decode (no long-form chunking)
            import whisper as _w
            mel = _w.log_mel_spectrogram(_w.pad_or_trim(audio_np)).to(offline_model.device)
            if not args.no_gpu:
                mel = mel.half()
            options = _w.DecodingOptions(language=args.language, task="transcribe",
                                         without_timestamps=True, fp16=(not args.no_gpu))
            result = _w.decode(offline_model, mel, options)
            offline_hyps.append(result.text.strip())
        t_offline_end = _time.perf_counter()
        offline_hyp = " ".join(offline_hyps)
        offline_wall = t_offline_end - t_offline_start
        offline_metrics = compute_metrics(ground_truth, offline_hyp, language=args.language)
        results["Offline"] = {
            "metrics": offline_metrics,
            "hypothesis": offline_hyp,
            "records": [],
            "word_records": [],
            "wall_time": offline_wall,
            "num_tokens": len(offline_hyp.split()),
            "buffer_overflow": False,
            "avg_latency": None,
            "latency_stats": {},
            "word_latency_stats": {},
            "encode_stats": {},
            "decode_stats": {},
            "decode_per_token_stats": {},
            "mel_stats": {},
            "prefill_stats": {},
            "ca_copy_stats": {},
            "ca_calc_stats": {},
            "num_words": len(offline_hyp.split()),
            "n_tokens_per_iter": [],
            "buffer_stats": {},
            "power_stats": None,
            "gpu_info": {"backend": "CUDA" if not args.no_gpu else "CPU",
                         "gpu_active": not args.no_gpu},
        }
        print(f"  Offline: CER={offline_metrics['cer']*100:.2f}%, wall={offline_wall:.2f}s")
        # Save raw offline output
        safe_name = "offline"
        with open(os.path.join(result_dir, f"raw_log_{safe_name}.txt"), "w") as f:
            f.write(f"Offline hypothesis:\n{offline_hyp}\n")
        del offline_model

    # Run all binaries
    for label, binary, source_dir, kind in runs:
        # Build args based on system kind
        if kind == "simul":
            run_args = simul_args
        elif kind == "simul_whisper":
            run_args = simul_whisper_args
        elif kind in ours_variant_args:
            run_args = ours_variant_args[kind]
        elif kind == "ours":
            run_args = ours_args
        elif kind == "ours_cpu":
            run_args = ours_cpu_args
        elif kind == "baseline":
            run_args = common_args
        elif kind == "whisperflow_gpu":
            run_args = common_args + get_whisperflow_extra_args(source_dir) + ["--gpu-decode"]
        else:  # whisperflow
            run_args = common_args + get_whisperflow_extra_args(source_dir)

        pmon = PowerMonitor(args.power_sysfs, args.power_interval_ms) if power_enabled else None
        # ours_streaming uses ONNX model directory, others use ggml model file
        is_onnx_kind = kind in ("ours", "ours_cpu", "simul_whisper") or kind in ours_variant_args
        effective_model_path = onnx_model_path if is_onnx_kind else model_path
        run_result = run_binary(binary, effective_model_path, wav_path, run_args, label, power_monitor=pmon,
                                adb_device=args.adb_device, adb_remote_dir=args.adb_remote_dir)

        if run_result["returncode"] != 0:
            print(f"WARNING: {label} exited with code {run_result['returncode']}")
            print(f"  stderr (last 500 chars): ...{run_result['stderr'][-500:]}")

        # All systems now use the same output format (Start Time: ...)
        is_ours = kind in ("ours", "ours_cpu") or kind in ours_variant_args
        results[label] = process_run_result(run_result, ground_truth,
                                            truncate_hyp=is_ours, language=args.language)
        results[label]["power_stats"] = run_result.get("power_stats")
        results[label]["gpu_info"] = detect_gpu_usage(run_result["combined"])

        # Save raw log
        safe_name = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
        with open(os.path.join(result_dir, f"raw_log_{safe_name}.txt"), "w") as f:
            f.write(run_result["combined"])

        # Save chunk summary for ours variants (when debug is enabled)
        if is_ours and args.ours_debug and run_result.get("stderr"):
            summary = generate_chunk_summary(run_result["stderr"], label, gt_words=gt_words)
            with open(os.path.join(result_dir, f"chunk_summary_{safe_name}.txt"), "w") as f:
                f.write(summary)

        # Save raw power samples to CSV
        if run_result.get("power_samples"):
            with open(os.path.join(result_dir, f"power_samples_{safe_name}.csv"), "w") as f:
                f.write("time_s,power_uw\n")
                for t, p in run_result["power_samples"]:
                    f.write(f"{t:.6f},{p}\n")

    # Recalculate latency with ground truth timestamps
    if gt_words:
        # Recalculate latency for all systems
        for label in results:
            recalculate_latency_with_ground_truth(results[label], gt_words)
            matched = results[label].get("gt_matched_words", 0)
            total = results[label].get("gt_total_words", 0)
            print(f"  {label}: matched {matched}/{total} words")

    # Print and save comparison
    labels = list(results.keys())
    col_width = 25
    comparison_file = os.path.join(result_dir, "comparison.txt")
    with open(comparison_file, "w") as f:
        def write(line=""):
            f.write(line + "\n")

        def write_row(metric, values):
            """Write a row with metric name and values for each system."""
            row = f"  {metric:<25}"
            for v in values:
                row += f" {v:>{col_width}}"
            write(row)

        title = " vs ".join(labels)
        write(f"{'='*120}")
        write(title)
        write(f"{'='*120}")
        write()
        write(f"Configuration:")
        write(f"  Model: ggml-{args.model}.bin")
        write(f"  Language: {args.language}")
        write(f"  Step size: {args.step}ms")
        write(f"  Num samples: {args.num_samples}")
        write(f"  Audio duration: {total_duration:.2f}s")
        write(f"  GPU: {'disabled (--no_gpu)' if args.no_gpu else 'enabled'}")
        if args.whisperflow or args.whisperflow_gpu_decode or args.pipeline_dir:
            hush_status = "enabled" if any(get_whisperflow_extra_args(wf_dir)) else "disabled"
            write(f"  Hush word (WhisperFlow): {hush_status}")
        if args.pipeline_dir:
            write(f"  Pipeline dir: {args.pipeline_dir}")
        only_set = set(args.only) if args.only else set()
        if args.simul_streaming or "simul" in only_set:
            write(f"  SimulStreaming: enabled (frame_threshold={args.simul_frame_threshold}, rewind_threshold={args.simul_rewind_threshold})")
        if args.simul_whisper or "simul_whisper" in only_set:
            write(f"  SimulWhisper: enabled (frame_threshold={args.simul_frame_threshold}, rewind_threshold={args.simul_rewind_threshold}, cif_model={cif_model_path})")
        if args.ours_streaming or "ours" in only_set or "ours_cpu" in only_set:
            write(f"  Ours: enabled (smoothing={args.ours_smoothing}, median_filter={args.ours_median_filter}, "
                  f"cross_attn_layer={args.ours_cross_attn_layer}, "
                  f"carryover_overlap={args.ours_carryover_overlap}, prompt_prefill={args.ours_prompt_prefill}, "
                  f"carryover_mode={args.ours_carryover_mode})")
            if has_dtw_modes:
                dtw_mode_count = sum(1 for m in args.ours_carryover_mode if m in (1, 2, 3))
                grid_size = len(args.ours_peak_margin) * len(args.ours_word_end_offset) * dtw_mode_count
                write(f"    peak_margin variants: {args.ours_peak_margin}")
                write(f"    word_end_offset variants: {args.ours_word_end_offset}")
                write(f"    grid: {len(args.ours_peak_margin)} x {len(args.ours_word_end_offset)} x {dtw_mode_count} modes = {grid_size} runs")
            else:
                write(f"    peak_margin={args.ours_peak_margin[0]}")
        if args.whisperflow_gpu_decode or "serialized_gpu" in only_set:
            write(f"  WhisperFlow GPU decode: enabled (no KV cache copy, decode on same GPU context)")
        write(f"  Systems: {', '.join(labels)}")
        write()

        # GPU verification per system
        write(f"  GPU Status (per system):")
        any_gpu_mismatch = False
        for l in labels:
            gpu_info = results[l].get("gpu_info", {})
            backend = gpu_info.get("backend", "unknown")
            gpu_active = gpu_info.get("gpu_active", False)
            expected_gpu = not args.no_gpu
            status = "OK" if gpu_active == expected_gpu else "MISMATCH"
            if status == "MISMATCH":
                any_gpu_mismatch = True
            write(f"    {l}: backend={backend}, active={gpu_active} [{status}]")
        if any_gpu_mismatch:
            write(f"  *** WARNING: GPU usage does not match --no_gpu setting! ***")
            write(f"  *** Binary may not be built with CUDA support. Rebuild with -DWHISPER_CUDA=ON ***")
        write()

        # Metrics comparison table
        write(f"{'='*120}")
        write(f"METRICS COMPARISON")
        write(f"{'='*120}")
        write()
        write_row("Metric", labels)
        write(f"  {'-'*(25 + (col_width + 1) * len(labels))}")

        write_row("WER", [f"{results[l]['metrics']['wer']*100:.2f}%" for l in labels])
        write_row("CER", [f"{results[l]['metrics']['cer']*100:.2f}%" for l in labels])
        write_row("GT words (total)", [str(results[l]['metrics']['ref_words']) for l in labels])
        write_row("GT words (truncated)", [str(results[l]['metrics']['ref_truncated_words']) for l in labels])
        write_row("Hyp words", [str(results[l]['metrics']['hyp_words']) for l in labels])
        write_row("Wall time (s)", [f"{results[l]['wall_time']:.2f}s" for l in labels])
        write_row("RTF", [f"{results[l]['wall_time']/total_duration:.3f}" for l in labels])
        write_row("TTFT (s)", [fmt_val(results[l].get("ttft"), ".3f", "s") for l in labels])
        write_row("Committed tokens", [str(results[l]['num_tokens']) for l in labels])
        def _buffer_status(l):
            if results[l].get('crashed'):
                return f"CRASHED (exit {results[l].get('returncode', '?')})"
            if results[l]['buffer_overflow']:
                return "STOPPED (>30s)"
            return "OK"
        write_row("Buffer status", [_buffer_status(l) for l in labels])
        write()

        # Ranking
        if len(labels) > 1:
            write(f"{'='*120}")
            if args.language in _NO_SPACE_LANGS or args.language == "ko":
                write(f"RANKING (by CER)")
                write(f"{'='*120}")
                write()
                ranked = sorted(labels, key=lambda l: results[l]['metrics']['cer'])
                write(f"  {'Rank':<6} {'System':<35} {'WER':>10} {'CER':>10}")
                write(f"  {'-'*6} {'-'*35} {'-'*10} {'-'*10}")
                for i, l in enumerate(ranked):
                    wer = results[l]['metrics']['wer'] * 100
                    cer = results[l]['metrics']['cer'] * 100
                    write(f"  {i+1:<6} {l:<35} {wer:>9.2f}% {cer:>9.2f}%")
            else:
                write(f"RANKING (by WER, then CER)")
                write(f"{'='*120}")
                write()
                ranked = sorted(labels, key=lambda l: (results[l]['metrics']['wer'], results[l]['metrics']['cer']))
                write(f"  {'Rank':<6} {'System':<35} {'WER':>10} {'CER':>10}")
                write(f"  {'-'*6} {'-'*35} {'-'*10} {'-'*10}")
                for i, l in enumerate(ranked):
                    wer = results[l]['metrics']['wer'] * 100
                    cer = results[l]['metrics']['cer'] * 100
                    write(f"  {i+1:<6} {l:<35} {wer:>9.2f}% {cer:>9.2f}%")
            write()

        # Latency comparison (per token)
        write(f"{'='*120}")
        write(f"LATENCY COMPARISON (per token)")
        write(f"{'='*120}")
        write()

        has_any_avg = any(results[l].get("avg_latency") is not None for l in labels)
        if has_any_avg:
            write_row("Avg latency (reported)", [fmt_val(results[l].get("avg_latency")) for l in labels])

        for stat_name in ["mean", "median", "std", "p90", "p95", "min", "max"]:
            write_row(stat_name, [fmt_val(results[l]["latency_stats"].get(stat_name)) for l in labels])
        write()

        # Per-word latency comparison
        write(f"{'='*120}")
        write(f"LATENCY COMPARISON (per word)")
        write(f"{'='*120}")
        write()
        write_row("Committed words", [str(results[l]['num_words']) for l in labels])
        for stat_name in ["mean", "median", "std", "p90", "p95", "min", "max"]:
            write_row(stat_name, [fmt_val(results[l]["word_latency_stats"].get(stat_name)) for l in labels])
        write()

        # Ground truth latency comparison (if timestamps provided)
        has_gt_latency = any(results[l].get("gt_word_latency_stats") for l in labels)
        if has_gt_latency:
            write(f"{'='*120}")
            write(f"LATENCY COMPARISON — GROUND TRUTH (per word, emission_time - gt_end_time)")
            write(f"{'='*120}")
            write()
            write_row("Matched words", [
                f"{results[l].get('gt_matched_words', 'N/A')}/{results[l].get('gt_total_words', 'N/A')}"
                for l in labels
            ])
            for stat_name in ["mean", "median", "std", "p90", "p95", "min", "max"]:
                write_row(stat_name, [
                    fmt_val(results[l].get("gt_word_latency_stats", {}).get(stat_name))
                    for l in labels
                ])
            write()

        # Encode/Decode per-iteration stats
        has_enc_dec = any(results[l]["encode_stats"] or results[l]["decode_stats"] for l in labels)
        if has_enc_dec:
            write(f"{'='*120}")
            write(f"ENCODE / DECODE PER-ITERATION STATISTICS (ms)")
            write(f"{'='*120}")
            write()
            write_row("Iterations", [
                str(results[l]["encode_stats"].get("n_iters", "N/A")) for l in labels
            ])
            write()
            write(f"  --- Encode (per iteration) ---")
            for stat_name in ["mean", "median", "std", "p90", "p95", "min", "max"]:
                write_row(f"encode {stat_name}", [
                    fmt_val(results[l]["encode_stats"].get(stat_name), ".2f", "ms") for l in labels
                ])
            write()
            write(f"  --- Decode (per iteration, token-by-token only) ---")
            for stat_name in ["mean", "median", "std", "p90", "p95", "min", "max"]:
                write_row(f"decode {stat_name}", [
                    fmt_val(results[l]["decode_stats"].get(stat_name), ".2f", "ms") for l in labels
                ])
            write()
            write(f"  --- Decode (per token) ---")
            for stat_name in ["mean", "median", "std", "p90", "p95", "min", "max"]:
                write_row(f"per-token {stat_name}", [
                    fmt_val(results[l]["decode_per_token_stats"].get(stat_name), ".3f", "ms") for l in labels
                ])
            write()

            # Timing breakdown: mel, prefill, cross-attention
            has_mel = any(results[l].get("mel_stats") for l in labels)
            if has_mel:
                write(f"  --- Mel (per iteration) ---")
                for stat_name in ["mean", "median", "std", "min", "max"]:
                    write_row(f"mel {stat_name}", [
                        fmt_val(results[l].get("mel_stats", {}).get(stat_name), ".2f", "ms") for l in labels
                    ])
                write()

            has_prefill = any(results[l].get("prefill_stats") for l in labels)
            if has_prefill:
                write(f"  --- Prefill / SOT decode (per iteration) ---")
                for stat_name in ["mean", "median", "std", "min", "max"]:
                    write_row(f"prefill {stat_name}", [
                        fmt_val(results[l].get("prefill_stats", {}).get(stat_name), ".2f", "ms") for l in labels
                    ])
                write()

            has_ntok = any(results[l].get("n_tokens_per_iter_stats") for l in labels)
            if has_ntok:
                write(f"  --- Tokens generated (per iteration) ---")
                for stat_name in ["mean", "median", "std", "min", "max"]:
                    write_row(f"n_tokens {stat_name}", [
                        fmt_val(results[l].get("n_tokens_per_iter_stats", {}).get(stat_name), ".1f", "") for l in labels
                    ])
                write()

            has_ca = any(results[l].get("ca_copy_stats") for l in labels)
            if has_ca:
                write(f"  --- Cross-attention: GPU->CPU copy (per iteration) ---")
                for stat_name in ["mean", "median", "std", "min", "max"]:
                    write_row(f"ca_copy {stat_name}", [
                        fmt_val(results[l].get("ca_copy_stats", {}).get(stat_name), ".3f", "ms") for l in labels
                    ])
                write()
                write(f"  --- Cross-attention: head averaging (per iteration) ---")
                for stat_name in ["mean", "median", "std", "min", "max"]:
                    write_row(f"ca_calc {stat_name}", [
                        fmt_val(results[l].get("ca_calc_stats", {}).get(stat_name), ".3f", "ms") for l in labels
                    ])
                write()

        # CIF inference stats (SimulWhisper only)
        has_cif = any(results[l].get("cif_stats") for l in labels)
        if has_cif:
            write(f"{'='*120}")
            write(f"CIF INFERENCE STATISTICS (ms, per iteration)")
            write(f"{'='*120}")
            write()
            write_row("CIF iterations", [
                str(results[l].get("cif_stats", {}).get("n_iters", "N/A")) if results[l].get("cif_stats") else "N/A"
                for l in labels
            ])
            for stat_name in ["mean", "median", "std", "p90", "p95", "min", "max"]:
                write_row(f"CIF {stat_name}", [
                    fmt_val(results[l].get("cif_stats", {}).get(stat_name), ".3f", "ms") if results[l].get("cif_stats") else "N/A"
                    for l in labels
                ])
            write()

        # Buffer stats
        has_any_buf = any(results[l]["buffer_stats"] for l in labels)
        if has_any_buf:
            write(f"{'='*120}")
            write(f"AUDIO BUFFER STATISTICS")
            write(f"{'='*120}")
            write()
            for stat_name in ["mean", "min", "max"]:
                write_row(f"Buffer {stat_name}", [fmt_val(results[l]["buffer_stats"].get(stat_name)) for l in labels])
            write()

        # Input stats (whisper input length per iteration)
        has_any_input = any(results[l].get("input_stats") for l in labels)
        if has_any_input:
            write(f"{'='*120}")
            write(f"WHISPER INPUT STATISTICS (per iteration)")
            write(f"{'='*120}")
            write()
            write_row("Input sec (mean)", [fmt_val(results[l].get("input_stats", {}).get("input_sec_mean")) for l in labels])
            write_row("Input sec (min)", [fmt_val(results[l].get("input_stats", {}).get("input_sec_min")) for l in labels])
            write_row("Input sec (max)", [fmt_val(results[l].get("input_stats", {}).get("input_sec_max")) for l in labels])
            write_row("Content mel (mean)", [fmt_val(results[l].get("input_stats", {}).get("content_mel_mean"), ".1f", "") for l in labels])
            write_row("Content mel (min)", [fmt_val(results[l].get("input_stats", {}).get("content_mel_min"), ".0f", "") for l in labels])
            write_row("Content mel (max)", [fmt_val(results[l].get("input_stats", {}).get("content_mel_max"), ".0f", "") for l in labels])
            write()

        # Power consumption
        has_power = any(results[l].get("power_stats") is not None for l in labels)
        if has_power:
            write(f"{'='*120}")
            write(f"POWER CONSUMPTION")
            write(f"{'='*120}")
            write()
            write_row("Mean power (mW)", [
                f"{results[l]['power_stats']['mean_power_mw']:.1f}" if results[l].get('power_stats') else "N/A"
                for l in labels
            ])
            write_row("Max power (mW)", [
                f"{results[l]['power_stats']['max_power_mw']:.1f}" if results[l].get('power_stats') else "N/A"
                for l in labels
            ])
            write_row("Min power (mW)", [
                f"{results[l]['power_stats']['min_power_mw']:.1f}" if results[l].get('power_stats') else "N/A"
                for l in labels
            ])
            write_row("Std power (mW)", [
                f"{results[l]['power_stats']['std_power_mw']:.1f}" if results[l].get('power_stats') else "N/A"
                for l in labels
            ])
            write_row("Total energy (J)", [
                f"{results[l]['power_stats']['total_energy_j']:.3f}" if results[l].get('power_stats') else "N/A"
                for l in labels
            ])
            write_row("Power samples", [
                str(results[l]['power_stats']['num_samples']) if results[l].get('power_stats') else "N/A"
                for l in labels
            ])
            write()

        # Ground truth vs hypothesis
        write(f"{'='*120}")
        write(f"GROUND TRUTH")
        write(f"{'='*120}")
        write(normalize_text(ground_truth))
        write()

        for label in labels:
            write(f"{'='*120}")
            write(f"HYPOTHESIS — {label}")
            write(f"{'='*120}")
            write(results[label]["hypothesis"].strip())
            write()
            # Aligned word diff: REF and HYP shown side by side with * for missing words
            gt_norm = normalize_text(ground_truth, language=args.language)
            hyp_norm = normalize_text(results[label]["hypothesis"].strip(), language=args.language)
            import difflib

            if args.language == "ko" or args.language in _NO_SPACE_LANGS:
                import unicodedata
                import jiwer
                def _charw(c):
                    """Display width: 2 for fullwidth/wide (CJK), 1 otherwise."""
                    eaw = unicodedata.east_asian_width(c)
                    return 2 if eaw in ('W', 'F') else 1

                out = jiwer.process_characters(gt_norm, hyp_norm)
                ref_chars = out.references[0]
                hyp_chars = out.hypotheses[0]

                write(f"  --- Char Diff (GT vs {label}) ---")
                write(f"  Legend: S=substitution, D=deletion, I=insertion")
                write()
                # Build aligned columns from jiwer alignments
                columns = []  # (ref_char, hyp_char, op_label, display_width)
                for chunk in out.alignments[0]:
                    rslice = ref_chars[chunk.ref_start_idx:chunk.ref_end_idx]
                    hslice = hyp_chars[chunk.hyp_start_idx:chunk.hyp_end_idx]
                    if chunk.type == 'equal':
                        for c in rslice:
                            columns.append((c, c, ' ', _charw(c)))
                    elif chunk.type == 'substitute':
                        for r, h in zip(rslice, hslice):
                            w = max(_charw(r), _charw(h))
                            columns.append((r, h, 'S', w))
                    elif chunk.type == 'delete':
                        for c in rslice:
                            columns.append((c, '', 'D', _charw(c)))
                    elif chunk.type == 'insert':
                        for c in hslice:
                            columns.append(('', c, 'I', _charw(c)))
                # Render with width-aware padding
                def _pad(ch, w):
                    actual = _charw(ch) if ch else 0
                    return ch + ' ' * (w - actual)
                max_line_w = 80
                idx = 0
                while idx < len(columns):
                    ref_line = []
                    hyp_line = []
                    op_line = []
                    cur_w = 0
                    while idx < len(columns) and cur_w + columns[idx][3] <= max_line_w:
                        r, h, o, w = columns[idx]
                        ref_line.append(_pad(r, w))
                        hyp_line.append(_pad(h, w))
                        op_line.append(o + ' ' * (w - 1))
                        cur_w += w
                        idx += 1
                    write(f"  REF: {''.join(ref_line)}")
                    write(f"  HYP: {''.join(hyp_line)}")
                    op_str = ''.join(op_line)
                    if any(c != ' ' for c in op_str):
                        write(f"  OP:  {op_str}")
                    write()
            else:
                # Word-level diff for English
                gt_words = gt_norm.split()
                hyp_words = hyp_norm.split()
                sm = difflib.SequenceMatcher(None, gt_words, hyp_words)
                ref_aligned = []
                hyp_aligned = []
                for op, i1, i2, j1, j2 in sm.get_opcodes():
                    if op == 'equal':
                        for w in gt_words[i1:i2]:
                            ref_aligned.append(w)
                            hyp_aligned.append(w)
                    elif op == 'replace':
                        gt_chunk = gt_words[i1:i2]
                        hyp_chunk = hyp_words[j1:j2]
                        max_len = max(len(gt_chunk), len(hyp_chunk))
                        gt_chunk += ['*'] * (max_len - len(gt_chunk))
                        hyp_chunk += ['*'] * (max_len - len(hyp_chunk))
                        ref_aligned.extend(gt_chunk)
                        hyp_aligned.extend(hyp_chunk)
                    elif op == 'delete':
                        for w in gt_words[i1:i2]:
                            ref_aligned.append(w)
                            hyp_aligned.append('*')
                    elif op == 'insert':
                        for w in hyp_words[j1:j2]:
                            ref_aligned.append('*')
                            hyp_aligned.append(w)
                # Format aligned pairs into fixed-width lines
                write(f"  --- Word Diff (GT vs {label}) ---")
                line_width = 100
                ref_line = []
                hyp_line = []
                cur_len = 0
                for r, h in zip(ref_aligned, hyp_aligned):
                    col_w = max(len(r), len(h)) + 1
                    if cur_len + col_w > line_width and ref_line:
                        write("  REF: " + " ".join(f"{w:<{max(len(w), len(h))}}" for w, h in zip(ref_line, hyp_line)))
                        write("  HYP: " + " ".join(f"{h:<{max(len(w), len(h))}}" for w, h in zip(ref_line, hyp_line)))
                        write()
                        ref_line = []
                        hyp_line = []
                        cur_len = 0
                    ref_line.append(r)
                    hyp_line.append(h)
                    cur_len += col_w
                if ref_line:
                    write("  REF: " + " ".join(f"{w:<{max(len(w), len(h))}}" for w, h in zip(ref_line, hyp_line)))
                    write("  HYP: " + " ".join(f"{h:<{max(len(w), len(h))}}" for w, h in zip(ref_line, hyp_line)))
                    write()
            write()

        # Per-token latency details
        for label in labels:
            recs = results[label]["records"]
            has_latency = any(r["latency"] is not None for r in recs)
            if not has_latency:
                continue
            write(f"{'='*120}")
            write(f"PER-TOKEN LATENCY — {label}")
            write(f"{'='*120}")
            write()
            write(f"  {'Start(s)':<10} {'End(s)':<10} {'Latency(s)':<12} {'Transcript'}")
            write(f"  {'-'*80}")
            for r in recs:
                lat_str = f"{r['latency']:.3f}" if r["latency"] is not None else "N/A"
                write(f"  {r['start']:<10.2f} {r['end']:<10.2f} {lat_str:<12} {r['transcript']}")
            write()

    print(f"\nResults saved to: {result_dir}/")
    print(f"  comparison.txt — side-by-side metrics")
    for label in labels:
        safe_name = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
        print(f"  raw_log_{safe_name}.txt — full {label} output")


if __name__ == "__main__":
    main()
