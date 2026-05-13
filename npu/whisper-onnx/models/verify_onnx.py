"""
Verify exported ONNX models against PyTorch reference.

Checks:
  1. encoder.onnx        : mel → encoder_output
  2. decoder_prefill.onnx: tokens + encoder_output → logits, self_kv, cross_kv
  3. decoder_step.onnx   : token + past_self_kv + cross_kv → logits, new_self_kv, cross_attn_weights
  4. Full greedy decode  : PyTorch vs ONNX token-by-token match

Usage:
    python verify_onnx.py --model_dir ./base
    python verify_onnx.py --model_dir ./base --audio /path/to/audio.wav
"""

import sys
import os
import argparse
import numpy as np

WHISPER_REPO = os.environ.get("WHISPER_REPO")
if WHISPER_REPO:
    sys.path.insert(0, WHISPER_REPO)

import torch
import whisper
from whisper.model import MultiHeadAttention
import onnxruntime as ort


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_session(path):
    opts = ort.SessionOptions()
    opts.log_severity_level = 3   # suppress ORT warnings
    return ort.InferenceSession(path, opts)


def check(name, a, b, atol=5e-4):
    a = np.array(a)
    b = np.array(b)
    max_diff = np.abs(a - b).max()
    ok = max_diff < atol
    status = "✅" if ok else "❌"
    print(f"  {status}  {name:40s}  max_diff={max_diff:.2e}  shape={a.shape}")
    return ok


# ── 1. Encoder ────────────────────────────────────────────────────────────────

def verify_encoder(model, model_dir, n_frames=1000):
    print("── Encoder ─────────────────────────────────────────────────────────")
    sess = make_session(f"{model_dir}/encoder.onnx")

    mel = torch.randn(1, model.dims.n_mels, n_frames)

    with torch.no_grad():
        ref = model.encoder(mel).numpy()

    out = sess.run(["encoder_output"], {"mel": mel.numpy()})[0]
    check("encoder_output", ref, out)


# ── 2. Decoder prefill ────────────────────────────────────────────────────────

def verify_decoder_prefill(model, model_dir):
    print("── Decoder prefill ─────────────────────────────────────────────────")
    sess = make_session(f"{model_dir}/decoder_prefill.onnx")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from export_whisper_onnx import DecoderPrefillWrapper

    MultiHeadAttention.use_sdpa = False
    wrapper = DecoderPrefillWrapper(model.decoder).eval()

    d = model.dims
    tokens  = torch.randint(0, d.n_vocab, (1, 5))
    enc_out = torch.randn(1, d.n_audio_ctx, d.n_audio_state)

    with torch.no_grad():
        ref_logits, ref_self_kv, ref_cross_kv = wrapper(tokens, enc_out)

    ort_logits, ort_self_kv, ort_cross_kv = sess.run(
        ["logits", "self_kv", "cross_kv"],
        {"tokens": tokens.numpy(), "encoder_output": enc_out.numpy()}
    )

    check("logits",   ref_logits.numpy(),   ort_logits)
    check("self_kv",  ref_self_kv.numpy(),  ort_self_kv)
    check("cross_kv", ref_cross_kv.numpy(), ort_cross_kv)

    return ref_self_kv.numpy(), ref_cross_kv.numpy(), ort_self_kv, ort_cross_kv


# ── 3. Decoder step ───────────────────────────────────────────────────────────

def verify_decoder_step(model, model_dir, ref_self_kv, ref_cross_kv, ort_self_kv, ort_cross_kv):
    print("── Decoder step ────────────────────────────────────────────────────")
    sess = make_session(f"{model_dir}/decoder_step.onnx")

    from export_whisper_onnx import DecoderStepWrapper
    MultiHeadAttention.use_sdpa = False
    wrapper = DecoderStepWrapper(model.decoder).eval()

    d = model.dims
    token = torch.randint(0, d.n_vocab, (1, 1))

    ref_self_kv_t  = torch.from_numpy(ref_self_kv)
    ref_cross_kv_t = torch.from_numpy(ref_cross_kv)

    with torch.no_grad():
        ref_logits, ref_new_kv, ref_cross_attn_w = wrapper(token, ref_self_kv_t, ref_cross_kv_t)

    ort_logits, ort_new_kv, ort_cross_attn_w = sess.run(
        ["logits", "new_self_kv", "cross_attn_weights"],
        {"token": token.numpy(), "past_self_kv": ort_self_kv, "cross_kv": ort_cross_kv}
    )

    check("logits",     ref_logits.numpy(),  ort_logits)
    check("new_self_kv", ref_new_kv.numpy(), ort_new_kv)
    print(f"  ℹ  cross_attn_weights  shape={ort_cross_attn_w.shape}  "
          f"sum={ort_cross_attn_w.sum():.4f}  max={ort_cross_attn_w.max():.4f}")


# ── 4. Full greedy decode: PyTorch vs ONNX ────────────────────────────────────

def verify_greedy_decode(model, model_dir, audio_path=None, n_frames=1000, max_tokens=80):
    print("── Full greedy decode (PyTorch vs ONNX) ────────────────────────────")

    from export_whisper_onnx import DecoderPrefillWrapper, DecoderStepWrapper
    MultiHeadAttention.use_sdpa = False

    enc_sess      = make_session(f"{model_dir}/encoder.onnx")
    prefill_sess  = make_session(f"{model_dir}/decoder_prefill.onnx")
    step_sess     = make_session(f"{model_dir}/decoder_step.onnx")

    d = model.dims
    if audio_path:
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, n_mels=d.n_mels).unsqueeze(0)
        print(f"  audio: {audio_path}")
    else:
        mel = torch.randn(1, d.n_mels, n_frames)

    # ── ONNX greedy ──────────────────────────────────────────────────────────
    enc_out = enc_sess.run(["encoder_output"], {"mel": mel.numpy()})[0]

    # Initial prompt: SOT + language (English) + task (transcribe) + no_timestamps
    tokenizer = whisper.tokenizer.get_tokenizer(multilingual=model.is_multilingual)
    sot   = tokenizer.sot
    lang  = tokenizer.sot_sequence[1]   # <|en|>
    task  = tokenizer.transcribe        # <|transcribe|>
    notsp = tokenizer.no_timestamps
    prompt = np.array([[sot, lang, task, notsp]], dtype=np.int64)

    logits, self_kv, cross_kv = prefill_sess.run(
        ["logits", "self_kv", "cross_kv"],
        {"tokens": prompt, "encoder_output": enc_out}
    )

    onnx_tokens = list(prompt[0])
    for _ in range(max_tokens):
        next_token = int(np.argmax(logits[0, -1]))
        onnx_tokens.append(next_token)
        if next_token == tokenizer.eot:
            break
        logits, self_kv, _ = step_sess.run(
            ["logits", "new_self_kv", "cross_attn_weights"],
            {"token": np.array([[next_token]], dtype=np.int64),
             "past_self_kv": self_kv,
             "cross_kv": cross_kv}
        )

    # ── PyTorch greedy ───────────────────────────────────────────────────────
    prefill_wrap = DecoderPrefillWrapper(model.decoder).eval()
    step_wrap    = DecoderStepWrapper(model.decoder).eval()
    mel_t        = mel
    prompt_t     = torch.tensor(prompt)
    enc_out_t    = model.encoder(mel_t)

    with torch.no_grad():
        logits_t, self_kv_t, cross_kv_t = prefill_wrap(prompt_t, enc_out_t)

    pt_tokens = list(prompt[0])
    for _ in range(max_tokens):
        next_token = int(logits_t[0, -1].argmax())
        pt_tokens.append(next_token)
        if next_token == tokenizer.eot:
            break
        with torch.no_grad():
            logits_t, self_kv_t, _ = step_wrap(
                torch.tensor([[next_token]]), self_kv_t, cross_kv_t
            )

    match = onnx_tokens == pt_tokens
    status = "✅" if match else "❌"
    print(f"  {status}  token sequence match")
    if not match:
        print(f"  PyTorch : {pt_tokens}")
        print(f"  ONNX    : {onnx_tokens}")
    else:
        # Decode just the generated part (skip prompt)
        gen = onnx_tokens[len(prompt[0]):]
        if gen and gen[-1] == tokenizer.eot:
            gen = gen[:-1]
        print(f"  tokens  : {onnx_tokens}")
        decoded = tokenizer.decode(gen)
        print(f"  decoded : '{decoded}'")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="./base")
    parser.add_argument("--model",     default="base")
    parser.add_argument("--audio",     default=None, help="Path to audio file for real decode test")
    args = parser.parse_args()

    print(f"Loading whisper '{args.model}' ...")
    model = whisper.load_model(args.model, device="cpu").eval()
    MultiHeadAttention.use_sdpa = False

    print()
    verify_encoder(model, args.model_dir)
    print()
    ref_self_kv, ref_cross_kv, ort_self_kv, ort_cross_kv = \
        verify_decoder_prefill(model, args.model_dir)
    print()
    verify_decoder_step(model, args.model_dir,
                        ref_self_kv, ref_cross_kv, ort_self_kv, ort_cross_kv)
    print()
    verify_greedy_decode(model, args.model_dir, audio_path=args.audio)


if __name__ == "__main__":
    main()
