"""
Verify WER difference between backends by running non-streaming inference.

Compares:
  1. Python Whisper (reference) — per-sample 30s-padded decode
  2. ONNX C++ whisper_full     — concatenated audio, single pass
  3. GGML C++ whisper_full     — concatenated audio, single pass (if binary available)

Usage:
  python verify_backend_wer.py --model base --num_samples 10
  python verify_backend_wer.py --model base --num_samples 100 --no_gpu
"""

import argparse
import os
import sys
import subprocess
import tempfile
import re
import json
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

# Reuse loader from run_streaming_comparison
from run_streaming_comparison import (
    load_librispeech_samples, compute_metrics
)


def run_python_whisper(samples, model_name, language, use_gpu):
    """Run Python Whisper per-sample (reference)."""
    import whisper
    device = "cuda" if use_gpu else "cpu"
    model = whisper.load_model(model_name, device=device)
    hyps = []
    for i, (audio_source, transcript) in enumerate(samples):
        if isinstance(audio_source, str):
            audio_np, _ = librosa.load(audio_source, sr=16000, dtype=np.float32)
        else:
            audio_np = np.array(audio_source, dtype=np.float32)
        mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio_np)).to(device)
        if use_gpu:
            mel = mel.half()
        options = whisper.DecodingOptions(language=language, task="transcribe",
                                          without_timestamps=True, fp16=use_gpu)
        result = whisper.decode(model, mel, options)
        hyps.append(result.text.strip())
        if (i + 1) % 20 == 0:
            print(f"  Python Whisper: {i+1}/{len(samples)} samples done")
    return " ".join(hyps)


def run_verify_cpp(binary_path, model_path, samples, language, no_gpu, cache_dir):
    """Run verify_cpp binary per-sample (non-streaming, 30s padded greedy decode).
    Parses "Transcript: '...'" from stdout."""
    pattern = re.compile(r"Transcript: '(.*)'")

    hyps = []
    for i, (path_or_arr, transcript) in enumerate(samples):
        if isinstance(path_or_arr, str):
            audio, _ = librosa.load(path_or_arr, sr=16000, dtype=np.float32)
        else:
            audio = np.array(path_or_arr, dtype=np.float32)

        tmp_wav = os.path.join(cache_dir, f"_verify_sample_{i}.wav")
        sf.write(tmp_wav, audio, 16000)

        cmd = [binary_path, "-m", model_path, "-a", tmp_wav, "-l", language]
        if no_gpu:
            cmd.append("-ng")

        result = subprocess.run(cmd, capture_output=True, timeout=120)
        stdout = result.stdout.decode("utf-8", errors="replace")

        hyp = ""
        for line in stdout.split("\n"):
            m = pattern.match(line.strip())
            if m:
                hyp = m.group(1).strip()
                break
        hyps.append(hyp)

        os.remove(tmp_wav)

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(samples)} samples done")

    return " ".join(hyps)


def main():
    parser = argparse.ArgumentParser(description="Verify backend WER differences (non-streaming)")
    parser.add_argument("--model", type=str, default="base")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--data_root", type=str, default="../../data")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--onnx-bin", type=str, default=None,
                        help="Path to ONNX verify_cpp binary")
    parser.add_argument("--ggml-bin", type=str, default=None,
                        help="Path to GGML verify_cpp binary (not available yet)")
    parser.add_argument("--onnx-model", type=str, default=None,
                        help="ONNX model directory (e.g. ../models/base)")
    parser.add_argument("--ggml-model", type=str, default=None,
                        help="GGML model file (e.g. ../../whisper-ggml/models/ggml-base.bin)")
    args = parser.parse_args()

    # Load dataset
    print(f"Loading LibriSpeech from {args.data_root}...")
    dataset = load_librispeech_samples(root=args.data_root)
    total = min(args.num_samples, len(dataset))
    samples = dataset[:total]
    print(f"Using {total} samples")

    # Ground truth
    ground_truth = " ".join(t for _, t in samples)

    # Cache dir for temporary per-sample WAVs
    cache_dir = os.path.join(".", "comparison_results", "cached_audio")
    os.makedirs(cache_dir, exist_ok=True)

    results = {}

    # 1. Python Whisper (reference)
    print(f"\n{'='*60}")
    print("1. Python Whisper (reference, per-sample 30s padded)")
    print(f"{'='*60}")
    py_hyp = run_python_whisper(samples, args.model, args.language, not args.no_gpu)
    py_metrics = compute_metrics(ground_truth, py_hyp, language=args.language)
    results["Python Whisper"] = py_metrics
    print(f"  WER: {py_metrics['wer']*100:.2f}%, CER: {py_metrics['cer']*100:.2f}%")

    # 2. ONNX C++ (if binary available)
    if args.onnx_bin:
        onnx_model = args.onnx_model or os.path.join("..", "models", args.model)
        print(f"\n{'='*60}")
        print(f"2. ONNX C++ (per-sample 30s padded, model={onnx_model})")
        print(f"{'='*60}")
        onnx_hyp = run_verify_cpp(args.onnx_bin, onnx_model, samples,
                                         args.language, args.no_gpu, cache_dir)
        if onnx_hyp:
            onnx_metrics = compute_metrics(ground_truth, onnx_hyp, language=args.language)
            results["ONNX C++"] = onnx_metrics
            print(f"  WER: {onnx_metrics['wer']*100:.2f}%, CER: {onnx_metrics['cer']*100:.2f}%")
        else:
            print(f"  FAILED (empty hypothesis)")

    # 3. GGML C++ (if binary available)
    if args.ggml_bin:
        ggml_model = args.ggml_model or os.path.join("..", "..", "whisper-ggml", "models", f"ggml-{args.model}.bin")
        print(f"\n{'='*60}")
        print(f"3. GGML C++ (per-sample 30s padded, model={ggml_model})")
        print(f"{'='*60}")
        ggml_hyp = run_verify_cpp(args.ggml_bin, ggml_model, samples,
                                         args.language, args.no_gpu, cache_dir)
        if ggml_hyp:
            ggml_metrics = compute_metrics(ground_truth, ggml_hyp, language=args.language)
            results["GGML C++"] = ggml_metrics
            print(f"  WER: {ggml_metrics['wer']*100:.2f}%, CER: {ggml_metrics['cer']*100:.2f}%")
        else:
            print(f"  FAILED (empty hypothesis)")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY (Non-streaming WER comparison)")
    print(f"{'='*60}")
    print(f"  {'Backend':<20} {'WER':>8} {'CER':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8}")
    for name, m in results.items():
        print(f"  {name:<20} {m['wer']*100:>7.2f}% {m['cer']*100:>7.2f}%")


if __name__ == "__main__":
    main()
