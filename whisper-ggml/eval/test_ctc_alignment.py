"""
Test CTC forced alignment (MMS-FA) vs Whisper DTW for Korean/Chinese/English.

Usage:
    python test_ctc_alignment.py --language ko --num_samples 3
    python test_ctc_alignment.py --language en --num_samples 3
    python test_ctc_alignment.py --language zh --num_samples 3
"""

import argparse
import os
import time
import numpy as np
import torch
import torchaudio
import whisper
import soundfile as sf
from datasets import load_dataset

FLEURS_MAP = {"en": "en_us", "ko": "ko_kr", "zh": "cmn_hans_cn"}


def load_samples(language, num_samples):
    fleurs_id = FLEURS_MAP.get(language)
    if not fleurs_id:
        raise ValueError(f"Unsupported language: {language}")
    ds = load_dataset("google/fleurs", fleurs_id, split="test", streaming=True)
    samples = []
    for i, sample in enumerate(ds):
        if i >= num_samples:
            break
        audio = np.array(sample["audio"]["array"], dtype=np.float32)
        sr = sample["audio"]["sampling_rate"]
        text = sample["transcription"]
        samples.append({"audio": audio, "sr": sr, "text": text, "id": sample.get("id", i)})
    return samples


def whisper_dtw_alignment(model, audio, sr, language):
    """Get word-level timestamps using Whisper's built-in DTW."""
    # Resample to 16kHz if needed
    if sr != 16000:
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)
        audio_tensor = torchaudio.functional.resample(audio_tensor, sr, 16000).squeeze(0)
        audio = audio_tensor.numpy()

    t0 = time.time()
    result = model.transcribe(
        audio,
        language=language,
        word_timestamps=True,
        no_speech_threshold=0.5,
    )
    elapsed = time.time() - t0

    words = []
    for seg in result["segments"]:
        for w in seg.get("words", []):
            words.append({
                "word": w["word"],
                "start": w["start"],
                "end": w["end"],
            })
    return words, result["text"], elapsed


def mms_ctc_alignment(audio, sr, transcript, language):
    """Get character-level timestamps using MMS forced alignment."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = torchaudio.pipelines.MMS_FA
    model_fa = bundle.get_model().to(device)
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()

    # Resample
    if sr != bundle.sample_rate:
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)
        audio_tensor = torchaudio.functional.resample(audio_tensor, sr, bundle.sample_rate)
    else:
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)

    audio_tensor = audio_tensor.to(device)

    # Normalize transcript for tokenization
    # MMS-FA expects romanized/normalized text for non-Latin scripts
    # For Korean, we use the raw text (MMS supports Korean script)
    transcript_clean = transcript.strip()

    t0 = time.time()
    with torch.inference_mode():
        emission, _ = model_fa(audio_tensor)

    # Tokenize
    tokens = tokenizer([transcript_clean])

    # Align
    token_spans = aligner(emission[0], tokens[0])
    elapsed = time.time() - t0

    # Convert to time
    num_frames = emission.shape[1]
    duration = audio_tensor.shape[1] / bundle.sample_rate
    frame_duration = duration / num_frames

    chars = []
    for span in token_spans[0]:
        chars.append({
            "char": transcript_clean[span.token] if span.token < len(transcript_clean) else "?",
            "start": span.start * frame_duration,
            "end": span.end * frame_duration,
            "score": span.score,
        })

    return chars, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, default="ko", choices=["en", "ko", "zh"])
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--model", type=str, default="base")
    args = parser.parse_args()

    print(f"Loading Whisper {args.model} model...")
    whisper_model = whisper.load_model(args.model)

    print(f"Loading FLEURS {args.language} samples...")
    samples = load_samples(args.language, args.num_samples)

    for i, sample in enumerate(samples):
        print(f"\n{'='*80}")
        print(f"Sample {i}: {sample['text'][:80]}...")
        print(f"Audio: {len(sample['audio'])/sample['sr']:.1f}s")
        print(f"{'='*80}")

        # Whisper DTW
        print(f"\n--- Whisper DTW ---")
        dtw_words, dtw_text, dtw_time = whisper_dtw_alignment(
            whisper_model, sample["audio"], sample["sr"], args.language)
        print(f"  Transcription: {dtw_text[:80]}")
        print(f"  Time: {dtw_time:.2f}s")
        for w in dtw_words:
            print(f"  {w['start']:6.2f}-{w['end']:6.2f}  {w['word']}")

        # MMS CTC Forced Alignment
        print(f"\n--- MMS CTC Forced Alignment ---")
        try:
            ctc_chars, ctc_time = mms_ctc_alignment(
                sample["audio"], sample["sr"], sample["text"], args.language)
            print(f"  Reference: {sample['text'][:80]}")
            print(f"  Time: {ctc_time:.2f}s")
            # Group chars into words (by space for ko/en, by char for zh)
            if args.language == "zh":
                for c in ctc_chars:
                    print(f"  {c['start']:6.2f}-{c['end']:6.2f}  {c['char']}  (score={c['score']:.2f})")
            else:
                # Group by spaces
                current_word = ""
                word_start = None
                word_end = None
                min_score = 1.0
                for c in ctc_chars:
                    if c["char"] == " " or c["char"] == "|":
                        if current_word:
                            print(f"  {word_start:6.2f}-{word_end:6.2f}  {current_word}  (min_score={min_score:.2f})")
                            current_word = ""
                            word_start = None
                            min_score = 1.0
                    else:
                        if word_start is None:
                            word_start = c["start"]
                        word_end = c["end"]
                        current_word += c["char"]
                        min_score = min(min_score, c["score"])
                if current_word:
                    print(f"  {word_start:6.2f}-{word_end:6.2f}  {current_word}  (min_score={min_score:.2f})")
        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
