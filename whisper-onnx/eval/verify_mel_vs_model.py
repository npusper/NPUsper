"""
Isolate WER difference: mel spectrogram vs model weights.

Compares 4 configurations on the same audio:
  A. Python Whisper (mel + model)           — reference
  B. Python mel → ONNX model               — isolates model weight difference
  C. ONNX C++ mel → ONNX model (verify_cpp) — current ONNX pipeline
  D. Python mel → Python model (= A)       — sanity check

If B ≈ A: ONNX model weights are fine, mel is the problem.
If B ≈ C: mel difference is small, model weights are the problem.
If B is between A and C: both contribute.

Usage:
  python verify_mel_vs_model.py --model base --num_samples 100 --no_gpu
  python verify_mel_vs_model.py --model base --num_samples 10
"""

import argparse
import os
import sys
import subprocess
import re
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))

try:
    import librosa
except ImportError:
    print("ERROR: librosa required"); sys.exit(1)

from run_streaming_comparison import load_librispeech_samples, compute_metrics


def run_python_whisper(samples, model_name, language, use_gpu):
    """A. Full Python Whisper pipeline (reference)."""
    import whisper
    device = "cuda" if use_gpu else "cpu"
    model = whisper.load_model(model_name, device=device)
    hyps = []
    for i, (src, _) in enumerate(samples):
        audio = librosa.load(src, sr=16000, dtype=np.float32)[0] if isinstance(src, str) else np.array(src, dtype=np.float32)
        mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio)).to(device)
        if use_gpu:
            mel = mel.half()
        opts = whisper.DecodingOptions(language=language, task="transcribe",
                                       without_timestamps=True, fp16=use_gpu)
        result = whisper.decode(model, mel, opts)
        hyps.append(result.text.strip())
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(samples)}")
    return " ".join(hyps)


def run_python_mel_onnx_model(samples, model_dir, language, use_gpu):
    """B. Python mel → ONNX Runtime encoder/decoder."""
    import whisper
    import onnxruntime as ort

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]

    encoder_sess = ort.InferenceSession(os.path.join(model_dir, "encoder.onnx"), providers=providers)
    prefill_sess = ort.InferenceSession(os.path.join(model_dir, "decoder_prefill.onnx"), providers=providers)
    step_sess = ort.InferenceSession(os.path.join(model_dir, "decoder_step.onnx"), providers=providers)

    # Load vocab
    vocab = []
    with open(os.path.join(model_dir, "vocab.txt"), "r", encoding="utf-8") as f:
        for line in f:
            vocab.append(line.rstrip("\n").replace("\\n", "\n").replace("\\r", "\r"))

    # Load dims
    dims = {}
    with open(os.path.join(model_dir, "dims.txt"), "r") as f:
        for line in f:
            k, v = line.strip().split("=")
            dims[k] = int(v)

    # Token IDs (same as whisper tokenizer)
    is_multi = bool(dims.get("is_multilingual", 1))
    tokenizer = whisper.tokenizer.get_tokenizer(multilingual=is_multi, language=language)
    eot = tokenizer.eot

    hyps = []
    for i, (src, _) in enumerate(samples):
        audio = librosa.load(src, sr=16000, dtype=np.float32)[0] if isinstance(src, str) else np.array(src, dtype=np.float32)

        # Use Python whisper mel (identical to reference)
        mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio))
        mel_np = mel.unsqueeze(0).numpy()  # [1, 80, 3000]

        # Encode
        enc_out = encoder_sess.run(None, {"mel": mel_np})[0]  # [1, 1500, 512]

        # Build SOT sequence: sot_sequence + no_timestamps
        sot_tokens = list(tokenizer.sot_sequence) + [tokenizer.no_timestamps]
        tokens_np = np.array([sot_tokens], dtype=np.int64)

        # Prefill
        prefill_out = prefill_sess.run(None, {
            "tokens": tokens_np,
            "encoder_output": enc_out,
        })
        logits = prefill_out[0]      # [1, n_tokens, n_vocab]
        self_kv = prefill_out[1]     # [n_layers, 2, 1, n_head, n_tokens, head_dim]
        cross_kv = prefill_out[2]    # [n_layers, 2, 1, n_head, n_audio_ctx, head_dim]

        # Greedy decode
        decoded = []
        for step in range(224):
            next_logits = logits[0, -1, :]  # [n_vocab]
            next_token = int(np.argmax(next_logits))
            if next_token >= eot:
                break
            decoded.append(next_token)

            token_np = np.array([[next_token]], dtype=np.int64)
            step_out = step_sess.run(None, {
                "token": token_np,
                "past_self_kv": self_kv,
                "cross_kv": cross_kv,
            })
            logits = step_out[0]     # [1, 1, n_vocab]
            self_kv = step_out[1]    # [n_layers, 2, 1, n_head, new_len, head_dim]

        hyp = "".join(vocab[t] for t in decoded if t < len(vocab)).strip()
        hyps.append(hyp)

        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(samples)}")

    return " ".join(hyps)


def run_verify_cpp(binary_path, model_path, samples, language, no_gpu, cache_dir):
    """C. Full ONNX C++ pipeline (C++ mel + ONNX model)."""
    pattern = re.compile(r"Transcript: '(.*)'")
    hyps = []
    for i, (src, _) in enumerate(samples):
        audio = librosa.load(src, sr=16000, dtype=np.float32)[0] if isinstance(src, str) else np.array(src, dtype=np.float32)
        tmp_wav = os.path.join(cache_dir, f"_tmp_{i}.wav")
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
            print(f"    {i+1}/{len(samples)}")

    return " ".join(hyps)


def main():
    parser = argparse.ArgumentParser(description="Isolate mel vs model weight WER differences")
    parser.add_argument("--model", type=str, default="base")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--data_root", type=str, default="../../data")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--onnx-model", type=str, default=None)
    parser.add_argument("--verify-cpp-bin", type=str, default=None,
                        help="Path to ONNX verify_cpp binary (for C++ mel test)")
    args = parser.parse_args()

    onnx_model = args.onnx_model or os.path.join("..", "models", args.model)
    use_gpu = not args.no_gpu

    print(f"Loading LibriSpeech from {args.data_root}...")
    dataset = load_librispeech_samples(root=args.data_root)
    total = min(args.num_samples, len(dataset))
    samples = dataset[:total]
    ground_truth = " ".join(t for _, t in samples)
    print(f"Using {total} samples\n")

    cache_dir = os.path.join(".", "comparison_results", "cached_audio")
    os.makedirs(cache_dir, exist_ok=True)

    results = {}

    # A. Python Whisper (reference)
    print(f"{'='*60}")
    print("A. Python Whisper (Python mel + Python model) — reference")
    print(f"{'='*60}")
    hyp_a = run_python_whisper(samples, args.model, args.language, use_gpu)
    m_a = compute_metrics(ground_truth, hyp_a, language=args.language)
    results["A. Python Whisper"] = m_a
    print(f"  WER: {m_a['wer']*100:.2f}%, CER: {m_a['cer']*100:.2f}%\n")

    # B. Python mel → ONNX model
    print(f"{'='*60}")
    print("B. Python mel → ONNX model (isolates model weight diff)")
    print(f"{'='*60}")
    hyp_b = run_python_mel_onnx_model(samples, onnx_model, args.language, use_gpu)
    m_b = compute_metrics(ground_truth, hyp_b, language=args.language)
    results["B. Py mel + ONNX model"] = m_b
    print(f"  WER: {m_b['wer']*100:.2f}%, CER: {m_b['cer']*100:.2f}%\n")

    # C. ONNX C++ (C++ mel + ONNX model) — if binary available
    if args.verify_cpp_bin:
        print(f"{'='*60}")
        print("C. C++ mel → ONNX model (current ONNX pipeline)")
        print(f"{'='*60}")
        hyp_c = run_verify_cpp(args.verify_cpp_bin, onnx_model, samples,
                                args.language, args.no_gpu, cache_dir)
        if hyp_c:
            m_c = compute_metrics(ground_truth, hyp_c, language=args.language)
            results["C. C++ mel + ONNX model"] = m_c
            print(f"  WER: {m_c['wer']*100:.2f}%, CER: {m_c['cer']*100:.2f}%\n")

    # Summary
    print(f"{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Config':<30} {'WER':>8} {'CER':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*8}")
    for name, m in results.items():
        print(f"  {name:<30} {m['wer']*100:>7.2f}% {m['cer']*100:>7.2f}%")

    print(f"\n{'='*60}")
    print("INTERPRETATION")
    print(f"{'='*60}")
    wer_a = m_a['wer'] * 100
    wer_b = m_b['wer'] * 100
    if "C. C++ mel + ONNX model" in results:
        wer_c = results["C. C++ mel + ONNX model"]['wer'] * 100
        model_diff = abs(wer_b - wer_a)
        mel_diff = abs(wer_c - wer_b)
        print(f"  Model weight diff (B-A):  {wer_b - wer_a:+.2f}%p")
        print(f"  Mel diff (C-B):           {wer_c - wer_b:+.2f}%p")
        print(f"  Total diff (C-A):         {wer_c - wer_a:+.2f}%p")
        if model_diff < 0.3 and mel_diff > 0.5:
            print(f"  → Mel is the main cause. Replacing ONNX mel with GGML mel should help.")
        elif mel_diff < 0.3 and model_diff > 0.5:
            print(f"  → Model weights / ONNX export is the main cause.")
        else:
            print(f"  → Both mel and model contribute to the difference.")
    else:
        print(f"  Model weight diff (B-A): {wer_b - wer_a:+.2f}%p")
        if abs(wer_b - wer_a) < 0.3:
            print(f"  → ONNX model weights are fine. Mel is likely the problem.")
        else:
            print(f"  → ONNX model weights contribute to WER difference.")


if __name__ == "__main__":
    main()
