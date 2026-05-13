"""
Generate ground truth word-level timestamps using Whisper DTW (offline).

Runs standard Whisper transcription with word_timestamps=True on the full audio
to produce per-word start/end times. These timestamps serve as the ground truth
for computing latency across all streaming methods uniformly.

Latency formula: latency = emission_time - ground_truth_end_time
  (a word can't be recognized until it's fully spoken, so we use end_time)

Usage:
    python generate_ground_truth_timestamps.py \
        --audio test_audio.wav \
        --model base \
        --output ground_truth_timestamps.json

    # With specific language:
    python generate_ground_truth_timestamps.py \
        --audio test_audio.wav \
        --model base \
        --language en \
        --output ground_truth_timestamps.json

Output JSON format:
    {
        "audio_file": "test_audio.wav",
        "model": "base",
        "words": [
            {"word": "hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.6, "end": 1.1},
            ...
        ]
    }
"""

import argparse
import json
import os
import re
import sys


def normalize_word(word):
    """Normalize a word for matching: lowercase, strip punctuation."""
    word = word.lower().strip()
    word = re.sub(r"[^\w]", "", word)
    return word


def main():
    parser = argparse.ArgumentParser(
        description="Generate ground truth word-level timestamps using Whisper DTW"
    )
    parser.add_argument("--audio", type=str, required=True,
                        help="Path to audio WAV file")
    parser.add_argument("--model", type=str, default="base",
                        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
                        help="Whisper model size (default: base)")
    parser.add_argument("--language", type=str, default="en",
                        help="Language code (default: en)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path (default: <audio_basename>_ground_truth.json)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (default: cuda)")
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"ERROR: audio file not found: {args.audio}")
        sys.exit(1)

    # Default output path
    if args.output is None:
        base = os.path.splitext(os.path.basename(args.audio))[0]
        args.output = os.path.join(os.path.dirname(args.audio) or ".", f"{base}_ground_truth.json")

    # Import whisper
    try:
        import whisper
    except ImportError:
        print("ERROR: openai-whisper package not found.")
        print("  Install with: pip install openai-whisper")
        sys.exit(1)

    print(f"Loading Whisper model: {args.model}")
    model = whisper.load_model(args.model, device=args.device)

    print(f"Transcribing: {args.audio}")
    result = model.transcribe(
        args.audio,
        language=args.language,
        word_timestamps=True,
        verbose=False,
    )

    # Extract word-level timestamps from segments
    words = []
    for segment in result["segments"]:
        if "words" not in segment:
            continue
        for w in segment["words"]:
            word_text = w["word"].strip()
            if not word_text:
                continue
            words.append({
                "word": word_text,
                "start": round(w["start"], 3),
                "end": round(w["end"], 3),
            })

    # Build output
    output = {
        "audio_file": os.path.basename(args.audio),
        "model": args.model,
        "language": args.language,
        "num_words": len(words),
        "words": words,
    }

    # Also include the full transcript for reference
    full_transcript = " ".join(w["word"] for w in words)
    output["transcript"] = full_transcript

    # Save
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(words)} word timestamps")
    print(f"Transcript: {full_transcript[:200]}{'...' if len(full_transcript) > 200 else ''}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
