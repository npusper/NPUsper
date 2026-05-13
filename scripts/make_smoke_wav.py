"""Generate a tiny WAV file for runtime smoke tests.

The generated tone is not an ASR accuracy sample. It only verifies that the
runtime can open a 16 kHz mono WAV file and execute the model path end to end.
"""

from __future__ import annotations

import argparse
import math
import struct
import wave
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a 16 kHz mono smoke-test WAV.")
    parser.add_argument("output", type=Path, help="Output WAV path")
    parser.add_argument("--seconds", type=float, default=2.0, help="Duration in seconds")
    parser.add_argument("--freq", type=float, default=440.0, help="Tone frequency in Hz")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Sample rate in Hz")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n_samples = int(args.seconds * args.sample_rate)
    amplitude = 0.2 * 32767

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.output), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(args.sample_rate)
        for i in range(n_samples):
            t = i / args.sample_rate
            sample = int(amplitude * math.sin(2.0 * math.pi * args.freq * t))
            wav.writeframesraw(struct.pack("<h", sample))


if __name__ == "__main__":
    main()
