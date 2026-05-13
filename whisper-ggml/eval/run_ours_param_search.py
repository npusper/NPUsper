#!/usr/bin/env python
"""
Parameter search for Ours (BackwardPeak) streaming across different step lengths.

Runs run_streaming_comparison.py with --only ours for each combination of
step length and parameter values, then summarizes WER and latency.

Usage:
    python run_ours_param_search.py
    python run_ours_param_search.py --num_samples 30 --model base
    python run_ours_param_search.py --audio-file path/to/audio.wav --ground-truth path/to/gt.txt
"""

import os
import sys
import re
import subprocess
import argparse
from datetime import datetime
from itertools import product


def parse_comparison_results(result_dir):
    """Parse comparison.txt to extract WER, latency, TTFT for Ours."""
    comp_file = os.path.join(result_dir, "comparison.txt")
    if not os.path.exists(comp_file):
        return None

    with open(comp_file, "r") as f:
        text = f.read()

    result = {}

    # WER
    m = re.search(r"WER\s+.*?([\d.]+)%", text)
    if m:
        result["wer"] = float(m.group(1))

    # Wall time
    m = re.search(r"Wall time.*?([\d.]+)s", text)
    if m:
        result["wall_time"] = float(m.group(1))

    # RTF
    m = re.search(r"RTF\s+([\d.]+)", text)
    if m:
        result["rtf"] = float(m.group(1))

    # TTFT
    m = re.search(r"TTFT.*?([\d.]+)s", text)
    if m:
        result["ttft"] = float(m.group(1))

    # GT word latency (mean)
    gt_lat_section = text.find("GROUND TRUTH")
    if gt_lat_section >= 0:
        after = text[gt_lat_section:]
        m = re.search(r"mean\s+([\d.]+)s", after)
        if m:
            result["gt_word_lat_mean"] = float(m.group(1))
        m = re.search(r"median\s+([\d.]+)s", after)
        if m:
            result["gt_word_lat_median"] = float(m.group(1))
        m = re.search(r"Matched words\s+(\d+)/(\d+)", after)
        if m:
            result["matched"] = int(m.group(1))
            result["total"] = int(m.group(2))

    return result


def main():
    parser = argparse.ArgumentParser(description="Parameter search for Ours streaming")
    parser.add_argument("--model", type=str, default="base")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--audio-file", type=str, default=None)
    parser.add_argument("--ground-truth", type=str, default=None)
    parser.add_argument("--ground-truth-timestamps", type=str, default=None)
    parser.add_argument("--bin-dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./param_search_results")
    parser.add_argument("--no_gpu", action="store_true")
    args = parser.parse_args()

    # Parameter grid
    steps = [500, 1000, 2000]
    peak_margins = [0.3, 0.6, 1.0]
    carryover_overlaps = [0.3, 0.6, 1.0]
    min_chunks = [0.5, 1.0, 1.5]
    prompt_prefills = [1, 3]

    # Fixed params
    smoothing = 10
    median_filter = 7
    cross_attn_layer = -1

    configs = list(product(steps, peak_margins, carryover_overlaps, min_chunks, prompt_prefills))
    total = len(configs)

    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"param_search_{date_str}")
    os.makedirs(output_dir, exist_ok=True)

    results = []

    for idx, (step, pm, co, mc, pp) in enumerate(configs):
        label = f"step{step}_pm{pm}_co{co}_mc{mc}_pp{pp}"
        print(f"\n{'='*80}")
        print(f"[{idx+1}/{total}] {label}")
        print(f"{'='*80}")

        # Build command
        cmd = [
            sys.executable, "run_streaming_comparison.py",
            "--model", args.model,
            "--step", str(step),
            "--ours-streaming",
            "--only", "ours",
            "--ours-smoothing", str(smoothing),
            "--ours-median-filter", str(median_filter),
            "--ours-cross-attn-layer", str(cross_attn_layer),
            "--ours-peak-margin", str(pm),
            "--ours-carryover-overlap", str(co),
            "--ours-min-chunk", str(mc),
            "--ours-prompt-prefill", str(pp),
        ]

        if args.audio_file:
            cmd += ["--audio-file", args.audio_file]
        if args.ground_truth:
            cmd += ["--ground-truth", args.ground_truth]
        if args.ground_truth_timestamps:
            cmd += ["--ground-truth-timestamps", args.ground_truth_timestamps]
        if args.num_samples:
            cmd += ["--num_samples", str(args.num_samples)]
        if args.bin_dir:
            cmd += ["--bin-dir", args.bin_dir]
        if args.no_gpu:
            cmd += ["--no_gpu"]

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT")
            results.append({"label": label, "step": step, "pm": pm, "co": co, "mc": mc, "pp": pp, "error": "timeout"})
            continue

        # Find the result directory from output
        m = re.search(r"Results saved to: (.+)/", proc.stdout)
        if not m:
            print(f"  FAILED to find result dir")
            results.append({"label": label, "step": step, "pm": pm, "co": co, "mc": mc, "pp": pp, "error": "no_output"})
            continue

        result_dir = m.group(1)
        parsed = parse_comparison_results(result_dir)
        if parsed:
            parsed.update({"label": label, "step": step, "pm": pm, "co": co, "mc": mc, "pp": pp})
            results.append(parsed)
            print(f"  WER={parsed.get('wer', '?')}%  "
                  f"GT_WordLat={parsed.get('gt_word_lat_mean', '?')}s  "
                  f"TTFT={parsed.get('ttft', '?')}s  "
                  f"Matched={parsed.get('matched', '?')}/{parsed.get('total', '?')}")
        else:
            results.append({"label": label, "step": step, "pm": pm, "co": co, "mc": mc, "pp": pp, "error": "parse_fail"})

    # Write summary
    summary_path = os.path.join(output_dir, "param_search_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"OURS (BackwardPeak) PARAMETER SEARCH\n")
        f.write(f"{'='*120}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Fixed: smoothing={smoothing}, median_filter={median_filter}, cross_attn_layer={cross_attn_layer}\n")
        f.write(f"Grid: steps={steps}, peak_margins={peak_margins}, carryover_overlaps={carryover_overlaps}, "
                f"min_chunks={min_chunks}, prompt_prefills={prompt_prefills}\n")
        f.write(f"Total configs: {total}\n\n")

        # Sort by step, then WER
        valid = [r for r in results if "error" not in r]
        valid.sort(key=lambda r: (r["step"], r.get("wer", 999)))

        header = f"{'Step':>5s}  {'PM':>4s}  {'CO':>4s}  {'MC':>4s}  {'PP':>3s}  {'WER%':>7s}  {'WordLat':>8s}  {'TTFT':>7s}  {'Match':>9s}  {'RTF':>6s}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        for r in valid:
            f.write(f"{r['step']:>5d}  {r['pm']:>4.1f}  {r['co']:>4.1f}  {r['mc']:>4.1f}  {r['pp']:>3d}  "
                    f"{r.get('wer', -1):>6.2f}%  "
                    f"{r.get('gt_word_lat_mean', -1):>7.3f}s  "
                    f"{r.get('ttft', -1):>6.3f}s  "
                    f"{r.get('matched', '?'):>4s}/{r.get('total', '?'):<4s}  "
                    f"{r.get('rtf', -1):>5.3f}\n")

        # Errors
        errors = [r for r in results if "error" in r]
        if errors:
            f.write(f"\nFailed configs ({len(errors)}):\n")
            for r in errors:
                f.write(f"  {r['label']}: {r['error']}\n")

        # Best per step
        f.write(f"\n{'='*120}\n")
        f.write(f"BEST CONFIG PER STEP (lowest WER)\n")
        f.write(f"{'='*120}\n\n")
        for step in steps:
            step_results = [r for r in valid if r["step"] == step]
            if step_results:
                best = min(step_results, key=lambda r: r.get("wer", 999))
                f.write(f"  Step {step}ms: WER={best.get('wer', '?')}%  "
                        f"pm={best['pm']} co={best['co']} mc={best['mc']} pp={best['pp']}  "
                        f"WordLat={best.get('gt_word_lat_mean', '?')}s  TTFT={best.get('ttft', '?')}s\n")

    print(f"\n{'='*80}")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
