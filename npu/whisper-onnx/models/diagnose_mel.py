"""
Diagnose mel computation differences between Python and C++ implementations.
Tests whether C++-style mel computation (zero padding, HTK filters) gives correct
ONNX inference output.
"""
import sys
import os
import numpy as np
import soundfile as sf

WHISPER_REPO = os.environ.get("WHISPER_REPO")
if WHISPER_REPO:
    sys.path.insert(0, WHISPER_REPO)
import whisper
import onnxruntime as ort

MODEL_DIR = os.path.join(os.path.dirname(__file__), "base")
WAV_PATH = os.environ.get("NPUSPER_DIAG_AUDIO", os.path.join(os.path.dirname(__file__), "sample.wav"))

WHISPER_SAMPLE_RATE = 16000
N_FFT      = 400
HOP_LENGTH = 160
N_MELS     = 80
N_SAMPLES  = WHISPER_SAMPLE_RATE * 30  # 480000


def make_session(path):
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(path, opts)


# ── Python-style mel (what verify_onnx.py uses) ───────────────────────────────
def python_mel(audio_path):
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio, n_mels=N_MELS)
    return mel.unsqueeze(0).numpy()  # [1, 80, 3000]


# ── C++-style mel: zero padding + simple triangular HTK filterbank ────────────
def hann_window(n):
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n) / n))


def htk_mel_filterbank(n_mels, n_fft, sr):
    hz_to_mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    mel_to_hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    n_bins = n_fft // 2 + 1
    f_max = sr / 2.0
    mel_min, mel_max = hz_to_mel(0.0), hz_to_mel(f_max)
    mel_pts = mel_to_hz(mel_min + (mel_max - mel_min) * np.arange(n_mels + 2) / (n_mels + 1))
    freq = np.arange(n_bins) * float(sr) / n_fft
    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for m in range(n_mels):
        f_lo, f_ctr, f_hi = mel_pts[m], mel_pts[m+1], mel_pts[m+2]
        for k in range(n_bins):
            f = freq[k]
            if f >= f_lo and f <= f_ctr:
                fb[m, k] = (f - f_lo) / (f_ctr - f_lo)
            elif f > f_ctr and f <= f_hi:
                fb[m, k] = (f_hi - f) / (f_hi - f_ctr)
    return fb


def cpp_style_mel(audio_path, use_reflect=False):
    """Compute mel spectrogram as the C++ code does."""
    # Read WAV
    audio, sr = sf.read(audio_path, dtype='float32')
    assert sr == WHISPER_SAMPLE_RATE, f"Expected 16kHz, got {sr}"
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Pad to 30 seconds
    if len(audio) < N_SAMPLES:
        audio = np.concatenate([audio, np.zeros(N_SAMPLES - len(audio), dtype=np.float32)])
    else:
        audio = audio[:N_SAMPLES]

    # Zero-pad for STFT (C++ style: 200 zeros on each side)
    pad = N_FFT // 2  # 200
    if use_reflect:
        padded = np.pad(audio, pad, mode='reflect')
    else:
        padded = np.concatenate([np.zeros(pad, dtype=np.float32),
                                  audio,
                                  np.zeros(pad, dtype=np.float32)])

    n_frames = (len(padded) - N_FFT) // HOP_LENGTH  # = 3000
    win = hann_window(N_FFT).astype(np.float32)
    fb  = htk_mel_filterbank(N_MELS, N_FFT, WHISPER_SAMPLE_RATE)

    mel = np.zeros((N_MELS, n_frames), dtype=np.float32)
    for i in range(n_frames):
        frame = padded[i*HOP_LENGTH : i*HOP_LENGTH + N_FFT] * win
        spec  = np.abs(np.fft.rfft(frame, n=N_FFT)) ** 2  # 201 bins
        for m in range(N_MELS):
            s = 1e-10 + np.dot(fb[m], spec)
            mel[m, i] = np.log10(s)

    # Normalize: clip to max-8, scale to [-1,1]
    max_val = mel.max()
    mel = np.maximum(mel, max_val - 8.0)
    mel = (mel + 4.0) / 4.0

    return mel[np.newaxis]  # [1, 80, 3000]


def greedy_decode(enc_sess, prefill_sess, step_sess, mel_np, tokenizer, max_tokens=80):
    """Full greedy decode given mel [1, 80, n_frames]."""
    enc_out = enc_sess.run(["encoder_output"], {"mel": mel_np})[0]

    sot    = tokenizer.sot
    lang   = tokenizer.sot_sequence[1]
    task   = tokenizer.transcribe
    notsp  = tokenizer.no_timestamps
    prompt = np.array([[sot, lang, task, notsp]], dtype=np.int64)

    logits, self_kv, cross_kv = prefill_sess.run(
        ["logits", "self_kv", "cross_kv"],
        {"tokens": prompt, "encoder_output": enc_out})

    tokens = list(prompt[0])
    for _ in range(max_tokens):
        next_token = int(np.argmax(logits[0, -1]))
        tokens.append(next_token)
        if next_token == tokenizer.eot:
            break
        logits, self_kv, _ = step_sess.run(
            ["logits", "new_self_kv", "cross_attn_weights"],
            {"token": np.array([[next_token]], dtype=np.int64),
             "past_self_kv": self_kv, "cross_kv": cross_kv})

    gen = tokens[len(prompt[0]):]
    if gen and gen[-1] == tokenizer.eot:
        gen = gen[:-1]
    return tokenizer.decode(gen), tokens


def main():
    print(f"Model dir : {MODEL_DIR}")
    print(f"Audio     : {WAV_PATH}")

    enc_sess     = make_session(f"{MODEL_DIR}/encoder.onnx")
    prefill_sess = make_session(f"{MODEL_DIR}/decoder_prefill.onnx")
    step_sess    = make_session(f"{MODEL_DIR}/decoder_step.onnx")

    model     = whisper.load_model("base", device="cpu").eval()
    tokenizer = whisper.tokenizer.get_tokenizer(multilingual=model.is_multilingual)

    print("\n── Test 1: Python mel (whisper.log_mel_spectrogram) ──────────────")
    mel1 = python_mel(WAV_PATH)
    print(f"  mel shape: {mel1.shape}, min={mel1.min():.4f}, max={mel1.max():.4f}")
    text1, _ = greedy_decode(enc_sess, prefill_sess, step_sess, mel1, tokenizer)
    print(f"  decoded: '{text1[:100]}'")

    print("\n── Test 2: C++-style mel (zero padding, HTK filterbank) ───────────")
    mel2 = cpp_style_mel(WAV_PATH, use_reflect=False)
    print(f"  mel shape: {mel2.shape}, min={mel2.min():.4f}, max={mel2.max():.4f}")
    print(f"  mel diff from Python: max={np.abs(mel1 - mel2).max():.4f}, mean={np.abs(mel1 - mel2).mean():.6f}")
    text2, _ = greedy_decode(enc_sess, prefill_sess, step_sess, mel2, tokenizer)
    print(f"  decoded: '{text2[:100]}'")

    print("\n── Test 3: C++-style mel (reflect padding, HTK filterbank) ─────────")
    mel3 = cpp_style_mel(WAV_PATH, use_reflect=True)
    print(f"  mel shape: {mel3.shape}, min={mel3.min():.4f}, max={mel3.max():.4f}")
    print(f"  mel diff from Python: max={np.abs(mel1 - mel3).max():.4f}, mean={np.abs(mel1 - mel3).mean():.6f}")
    text3, _ = greedy_decode(enc_sess, prefill_sess, step_sess, mel3, tokenizer)
    print(f"  decoded: '{text3[:100]}'")

    print("\n── Summary ─────────────────────────────────────────────────────────")
    print(f"  Python mel → '{text1[:80]}'")
    print(f"  C++ mel (zero) → '{text2[:80]}'")
    print(f"  C++ mel (reflect) → '{text3[:80]}'")


if __name__ == "__main__":
    main()
