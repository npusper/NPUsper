"""
Visualize comparison results from run_streaming_comparison.py.

Parses comparison.txt and raw_log files to generate publication-ready charts.

Usage:
    python visualize_comparison.py comparison_results/comparison_2way_base_50samples_step500_20260228_120000/

    # Specify output format
    python visualize_comparison.py <result_dir> --format png --dpi 300

    # Compare multiple experiments
    python visualize_comparison.py <dir1> <dir2> --multi
"""

import os
import re
import sys
import argparse
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError:
    print("ERROR: matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)


# ── Parsing ──────────────────────────────────────────────────────────────

def parse_comparison_file(path):
    """Parse comparison.txt into a structured dict."""
    with open(path, "r") as f:
        text = f.read()

    data = {"config": {}, "labels": [], "metrics": {}}

    # Parse config
    for key in ["Model", "Language", "Step size", "Num samples", "Audio duration", "GPU"]:
        m = re.search(rf"{key}:\s*(.+)", text)
        if m:
            data["config"][key] = m.group(1).strip()

    # Detect labels from the title line (first non-separator, non-empty line)
    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        if line and not line.startswith("=") and " vs " in line:
            data["labels"] = [l.strip() for l in line.split(" vs ")]
            break

    if not data["labels"]:
        print("ERROR: Could not detect system labels from comparison.txt")
        sys.exit(1)

    n = len(data["labels"])

    def parse_row(section_text, metric_name):
        """Parse a metric row value for each system."""
        pattern = rf"^\s*{re.escape(metric_name)}\s+(.+)$"
        m = re.search(pattern, section_text, re.MULTILINE)
        if not m:
            return [None] * n
        raw = m.group(1).strip()
        parts = raw.split()
        vals = []
        for p in parts:
            # Strip trailing units: %, ms, mW, s, J
            p = re.sub(r'(ms|mW|%)$', '', p)
            p = re.sub(r'([0-9])s$', r'\1', p)
            p = re.sub(r'([0-9])J$', r'\1', p)
            try:
                vals.append(float(p))
            except ValueError:
                vals.append(None)
        return vals[:n]

    # Extract sections: pattern is ===\nSECTION NAME\n===\ncontent...
    sections = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Look for separator line
        if line.startswith("===") and len(line) > 20:
            # Next non-empty line is section name
            if i + 1 < len(lines) and lines[i + 1].strip() and not lines[i + 1].strip().startswith("="):
                section_name = lines[i + 1].strip()
                # Skip section name and next separator
                i += 2
                if i < len(lines) and lines[i].strip().startswith("==="):
                    i += 1
                # Collect content until next separator
                content_lines = []
                while i < len(lines) and not (lines[i].strip().startswith("===") and len(lines[i].strip()) > 20):
                    content_lines.append(lines[i])
                    i += 1
                sections[section_name] = "\n".join(content_lines)
                continue
        i += 1

    metrics_text = sections.get("METRICS COMPARISON", "")
    lat_token_text = sections.get("LATENCY COMPARISON (per token)", "")
    lat_word_text = sections.get("LATENCY COMPARISON (per word)", "")
    enc_dec_text = sections.get("ENCODE / DECODE PER-ITERATION STATISTICS (ms)", "")
    power_text = sections.get("POWER CONSUMPTION", "")

    # Parse metrics
    data["metrics"]["wer"] = parse_row(metrics_text, "WER")
    data["metrics"]["cer"] = parse_row(metrics_text, "CER")
    data["metrics"]["wall_time"] = parse_row(metrics_text, "Wall time (s)")
    data["metrics"]["rtf"] = parse_row(metrics_text, "RTF")

    # Token latency
    data["metrics"]["token_lat_mean"] = parse_row(lat_token_text, "mean")
    data["metrics"]["token_lat_median"] = parse_row(lat_token_text, "median")
    data["metrics"]["token_lat_p90"] = parse_row(lat_token_text, "p90")
    data["metrics"]["token_lat_p95"] = parse_row(lat_token_text, "p95")

    # Word latency
    data["metrics"]["word_lat_mean"] = parse_row(lat_word_text, "mean")
    data["metrics"]["word_lat_median"] = parse_row(lat_word_text, "median")
    data["metrics"]["word_lat_p90"] = parse_row(lat_word_text, "p90")
    data["metrics"]["word_lat_p95"] = parse_row(lat_word_text, "p95")

    # Encode/Decode
    data["metrics"]["encode_mean"] = parse_row(enc_dec_text, "encode mean")
    data["metrics"]["encode_std"] = parse_row(enc_dec_text, "encode std")
    data["metrics"]["encode_p95"] = parse_row(enc_dec_text, "encode p95")
    data["metrics"]["decode_mean"] = parse_row(enc_dec_text, "decode mean")
    data["metrics"]["decode_std"] = parse_row(enc_dec_text, "decode std")
    data["metrics"]["decode_p95"] = parse_row(enc_dec_text, "decode p95")
    data["metrics"]["per_token_decode_mean"] = parse_row(enc_dec_text, "per-token mean")
    data["metrics"]["per_token_decode_std"] = parse_row(enc_dec_text, "per-token std")
    data["metrics"]["per_token_decode_p95"] = parse_row(enc_dec_text, "per-token p95")

    # Timing breakdown
    data["metrics"]["mel_mean"] = parse_row(enc_dec_text, "mel mean")
    data["metrics"]["prefill_mean"] = parse_row(enc_dec_text, "prefill mean")
    data["metrics"]["prefill_std"] = parse_row(enc_dec_text, "prefill std")
    data["metrics"]["ca_copy_mean"] = parse_row(enc_dec_text, "ca_copy mean")
    data["metrics"]["ca_calc_mean"] = parse_row(enc_dec_text, "ca_calc mean")
    data["metrics"]["n_tokens_mean"] = parse_row(enc_dec_text, "n_tokens mean")
    data["metrics"]["n_tokens_std"] = parse_row(enc_dec_text, "n_tokens std")

    # GT Latency
    gt_lat_text = sections.get("LATENCY COMPARISON \u2014 GROUND TRUTH (per word, emission_time - gt_end_time)", "")
    if not gt_lat_text:
        # Try alternate key without em-dash
        for key in sections:
            if "GROUND TRUTH" in key and "LATENCY" in key:
                gt_lat_text = sections[key]
                break
    data["metrics"]["gt_lat_mean"] = parse_row(gt_lat_text, "mean")
    data["metrics"]["gt_lat_std"] = parse_row(gt_lat_text, "std")
    data["metrics"]["gt_lat_median"] = parse_row(gt_lat_text, "median")
    data["metrics"]["gt_lat_p90"] = parse_row(gt_lat_text, "p90")
    data["metrics"]["gt_lat_p95"] = parse_row(gt_lat_text, "p95")

    # Buffer stats
    buffer_text = sections.get("AUDIO BUFFER STATISTICS", "")
    data["metrics"]["buffer_mean"] = parse_row(buffer_text, "Buffer mean")
    data["metrics"]["buffer_min"] = parse_row(buffer_text, "Buffer min")
    data["metrics"]["buffer_max"] = parse_row(buffer_text, "Buffer max")

    # CIF stats
    cif_text = sections.get("CIF INFERENCE STATISTICS (ms, per iteration)", "")
    data["metrics"]["cif_mean"] = parse_row(cif_text, "CIF mean")

    # Power
    data["metrics"]["power_mean"] = parse_row(power_text, "Mean power (mW)")
    data["metrics"]["power_max"] = parse_row(power_text, "Max power (mW)")
    data["metrics"]["energy_total"] = parse_row(power_text, "Total energy (J)")

    # Input stats
    input_text = sections.get("WHISPER INPUT STATISTICS (per iteration)", "")
    data["metrics"]["input_sec_mean"] = parse_row(input_text, "Input sec (mean)")
    data["metrics"]["input_sec_min"] = parse_row(input_text, "Input sec (min)")
    data["metrics"]["input_sec_max"] = parse_row(input_text, "Input sec (max)")
    data["metrics"]["content_mel_mean"] = parse_row(input_text, "Content mel (mean)")

    # Iterations
    data["metrics"]["iterations"] = parse_row(enc_dec_text, "Iterations")

    return data


def parse_per_token_latencies_from_log(log_path):
    """Parse per-token latency values from a raw log file."""
    if not os.path.exists(log_path):
        return []
    with open(log_path, "r") as f:
        text = f.read()

    pattern = re.compile(
        r"Start Time:\s*([\d.]+),\s*End Time:\s*([\d.]+),\s*Transcript: .*?,\s*Latency:\s*([\d.-]+)"
    )
    latencies = []
    for m in pattern.finditer(text):
        latencies.append(float(m.group(3)))
    return latencies


# ── Plotting ─────────────────────────────────────────────────────────────

COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]
SHORT_LABELS = {
    "Whisper-S": "Whisper-S",
    "WhisperFlow (serialized)": "WhisperFlow",
    "WhisperFlow (pipeline)": "WF-Pipeline",
    "SimulStreaming (AlignAtt)": "SimulStream",
    "SimulWhisper (CIF)": "SimulWhisper",
    "Ours (BackwardPeak)": "Ours",
}


def short_label(label):
    return SHORT_LABELS.get(label, label)


def bar_chart(ax, labels, values, title, ylabel, fmt=".2f", suffix="", color_list=None):
    """Generic grouped bar chart."""
    if color_list is None:
        color_list = COLORS
    x = np.arange(len(labels))
    valid = [(i, v) for i, v in enumerate(values) if v is not None]
    if not valid:
        ax.set_visible(False)
        return
    bars = ax.bar(
        [x[i] for i, _ in valid],
        [v for _, v in valid],
        color=[color_list[i % len(color_list)] for i, _ in valid],
        width=0.5,
        edgecolor="white",
        linewidth=0.5,
    )
    for bar, (_, v) in zip(bars, valid):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:{fmt}}{suffix}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([short_label(l) for l in labels], fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def grouped_bar_chart(ax, labels, groups, title, ylabel, group_names, fmt=".2f", suffix=""):
    """Grouped bar chart with multiple metrics per system."""
    x = np.arange(len(labels))
    n_groups = len(groups)
    width = 0.8 / n_groups

    for i, (vals, name) in enumerate(zip(groups, group_names)):
        offsets = x - 0.4 + width * (i + 0.5)
        valid_vals = [v if v is not None else 0 for v in vals]
        bars = ax.bar(offsets, valid_vals, width, label=name,
                      color=COLORS[i % len(COLORS)], edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v is not None:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{v:{fmt}}{suffix}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([short_label(l) for l in labels], fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def latency_distribution_plot(ax, labels, all_latencies, title):
    """Box/violin plot of per-token latencies."""
    plot_data = []
    plot_labels = []
    for label, lats in zip(labels, all_latencies):
        if lats:
            plot_data.append(lats)
            plot_labels.append(short_label(label))

    if not plot_data:
        ax.set_visible(False)
        return

    bp = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True, widths=0.5,
                    showfliers=True, flierprops=dict(marker=".", markersize=2, alpha=0.3))
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(COLORS[i % len(COLORS)])
        patch.set_alpha(0.7)

    ax.set_ylabel("Latency (s)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def latency_timeline_plot(ax, labels, all_latencies, title):
    """Plot latency over token index (timeline)."""
    for i, (label, lats) in enumerate(zip(labels, all_latencies)):
        if lats:
            ax.plot(range(len(lats)), lats, color=COLORS[i % len(COLORS)],
                    alpha=0.6, linewidth=0.8, label=short_label(label))
    ax.set_xlabel("Token index", fontsize=10)
    ax.set_ylabel("Latency (s)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def generate_plots(data, result_dir, fmt="png", dpi=200):
    """Generate all comparison plots."""
    labels = data["labels"]
    m = data["metrics"]
    out_dir = os.path.join(result_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    title_suffix = f"({data['config'].get('Model', '')}, step={data['config'].get('Step size', '')})"

    # ── 1. Overview dashboard (2×3) ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle(f"Streaming Comparison Overview {title_suffix}", fontsize=14, fontweight="bold")

    bar_chart(axes[0, 0], labels, m["wer"], "Word Error Rate", "WER (%)", ".1f", "%")
    bar_chart(axes[0, 1], labels, m["rtf"], "Real-Time Factor", "RTF", ".3f")
    bar_chart(axes[0, 2], labels, m["wall_time"], "Wall Time", "Time (s)", ".1f", "s")

    gt_mean = m.get("gt_lat_mean", [None] * len(labels))
    bar_chart(axes[1, 0], labels, gt_mean, "GT Latency (mean)", "Latency (s)", ".3f", "s")
    bar_chart(axes[1, 1], labels, m["encode_mean"], "Encode Time (mean)", "Time (ms)", ".0f", "ms")
    bar_chart(axes[1, 2], labels, m["energy_total"], "Total Energy", "Energy (J)", ".0f", "J")

    plt.tight_layout()
    path = os.path.join(out_dir, f"01_overview.{fmt}")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # ── 2. WER / CER ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Quality Metrics {title_suffix}", fontsize=14, fontweight="bold")
    bar_chart(axes[0], labels, m["wer"], "Word Error Rate", "WER (%)", ".1f", "%")
    bar_chart(axes[1], labels, m["cer"], "Character Error Rate", "CER (%)", ".2f", "%")
    plt.tight_layout()
    path = os.path.join(out_dir, f"02_wer_cer.{fmt}")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # ── 3. GT Latency (mean bar + std error bar) ──
    has_gt_lat = any(v is not None for v in m.get("gt_lat_mean", []))
    if has_gt_lat:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        fig.suptitle(f"Ground Truth Latency {title_suffix}", fontsize=14, fontweight="bold")

        means = m["gt_lat_mean"]
        stds = m.get("gt_lat_std", [None] * len(labels))
        x = np.arange(len(labels))
        valid = [(i, means[i], stds[i] if stds[i] is not None else 0)
                 for i in range(len(labels)) if means[i] is not None]

        bar_x = [x[i] for i, _, _ in valid]
        bar_vals = [v for _, v, _ in valid]
        bar_stds = [s for _, _, s in valid]
        bar_colors = [COLORS[i % len(COLORS)] for i, _, _ in valid]

        bars = ax.bar(bar_x, bar_vals, color=bar_colors, width=0.5,
                      edgecolor="white", linewidth=0.5)
        ax.errorbar(bar_x, bar_vals, yerr=bar_stds,
                    fmt="none", ecolor="black", capsize=5, capthick=1.5, linewidth=1.5)

        for bar, (_, mean_v, std_v) in zip(bars, valid):
            ax.text(bar.get_x() + bar.get_width() / 2, mean_v - 0.3,
                    f"{mean_v:.1f}s", ha="center", va="top", fontsize=9,
                    fontweight="bold", color="white")
            ax.text(bar.get_x() + bar.get_width() / 2, mean_v + std_v + 0.2,
                    f"\u00b1{std_v:.1f}s", ha="center", va="bottom", fontsize=8,
                    color="gray")

        ax.set_xticks(x)
        ax.set_xticklabels([short_label(l) for l in labels], fontsize=10)
        ax.set_ylabel("Latency (s)", fontsize=10)
        ax.set_title("Mean latency (bar) \u00b1 Std (error bar)", fontsize=12, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        path = os.path.join(out_dir, f"03_gt_latency.{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

    # ── 4. Encode/Decode breakdown (mean bar + std error bar, 2×2 grid) ──
    has_enc_dec = any(v is not None for v in m["encode_mean"])
    if has_enc_dec:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Encode / Decode Performance {title_suffix}", fontsize=14, fontweight="bold")

        enc_dec_configs = [
            (axes[0, 0], m["encode_mean"], m.get("encode_std", [None]*len(labels)),
             "Encode Time (per iter)", ".1f"),
            (axes[0, 1], m.get("prefill_mean", [None]*len(labels)), m.get("prefill_std", [None]*len(labels)),
             "Prefill Time (per iter)", ".1f"),
            (axes[1, 0], m["decode_mean"], m.get("decode_std", [None]*len(labels)),
             "Decode Time (per iter)", ".1f"),
            (axes[1, 1], m["per_token_decode_mean"], m.get("per_token_decode_std", [None]*len(labels)),
             "Decode Time (per token)", ".2f"),
        ]

        for ax, means, stds, title, val_fmt in enc_dec_configs:
            x = np.arange(len(labels))
            valid = [(i, means[i], stds[i] if stds[i] is not None else 0)
                     for i in range(len(labels)) if means[i] is not None]
            if not valid:
                ax.set_visible(False)
                continue

            bar_x = [x[i] for i, _, _ in valid]
            bar_vals = [v for _, v, _ in valid]
            bar_stds = [s for _, _, s in valid]
            bar_colors = [COLORS[i % len(COLORS)] for i, _, _ in valid]

            bars = ax.bar(bar_x, bar_vals, color=bar_colors, width=0.5,
                          edgecolor="white", linewidth=0.5)
            ax.errorbar(bar_x, bar_vals, yerr=bar_stds,
                        fmt="none", ecolor="black", capsize=5, capthick=1.5, linewidth=1.5)

            for bar, (_, mean_v, std_v) in zip(bars, valid):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{mean_v:{val_fmt}}", ha="center", va="bottom", fontsize=9)

            ax.set_xticks(x)
            ax.set_xticklabels([short_label(l) for l in labels], fontsize=10)
            ax.set_ylabel("Time (ms)", fontsize=10)
            ax.set_title(f"{title} (mean \u00b1 std)", fontsize=12, fontweight="bold")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        plt.tight_layout()
        path = os.path.join(out_dir, f"04_encode_decode.{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

    # ── 4b. Timing breakdown (stacked bar) ──
    has_mel = any(v is not None for v in m.get("mel_mean", []))
    if has_mel and has_enc_dec:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        fig.suptitle(f"Per-Iteration Timing Breakdown {title_suffix}", fontsize=14, fontweight="bold")

        x = np.arange(len(labels))
        width = 0.5

        # Stack order: mel, encode, prefill, decode (token-by-token), ca_copy, ca_calc
        # Prefill = SOT batch decode (context + special tokens)
        # Decode  = token-by-token autoregressive generation
        stack_config = [
            ("Mel",             m.get("mel_mean", [None]*len(labels)),     "#B3E5FC"),
            ("Encode",          m.get("encode_mean", [None]*len(labels)),  "#4FC3F7"),
            ("Prefill",         m.get("prefill_mean", [None]*len(labels)), "#E53935"),
            ("Decode",           m.get("decode_mean", [None]*len(labels)), "#01579B"),
            ("CA Copy",         m.get("ca_copy_mean", [None]*len(labels)), "#FFB74D"),
            ("CA Calc",         m.get("ca_calc_mean", [None]*len(labels)), "#FF8A65"),
        ]

        bottoms = np.zeros(len(labels))
        for name, vals, color in stack_config:
            safe_vals = np.array([v if v is not None else 0 for v in vals])
            if np.sum(safe_vals) < 0.01:
                continue
            bars = ax.bar(x, safe_vals, bottom=bottoms, width=width,
                          label=name, color=color, edgecolor="white", linewidth=0.5)
            # Label segments > 5ms
            for j, (bar, v) in enumerate(zip(bars, vals)):
                if v is not None and v > 5:
                    mid = bottoms[j] + v / 2
                    ax.text(bar.get_x() + bar.get_width() / 2, mid,
                            f"{v:.0f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
            bottoms += safe_vals

        # Total label on top
        for j in range(len(labels)):
            if bottoms[j] > 0:
                ax.text(x[j], bottoms[j] + bottoms.max() * 0.01,
                        f"{bottoms[j]:.0f}ms", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([short_label(l) for l in labels], fontsize=10)
        ax.set_ylabel("Time (ms)", fontsize=11)
        ax.set_title("Mean Per-Iteration Time (stacked)", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        path = os.path.join(out_dir, f"04b_timing_breakdown.{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

    # ── 4c. Tokens per iteration ──
    has_ntok = any(v is not None for v in m.get("n_tokens_mean", []))
    if has_ntok:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        fig.suptitle(f"Tokens Generated Per Iteration {title_suffix}", fontsize=14, fontweight="bold")

        x = np.arange(len(labels))
        means = [v if v is not None else 0 for v in m["n_tokens_mean"]]
        stds = [v if v is not None else 0 for v in m.get("n_tokens_std", [0]*len(labels))]
        bars = ax.bar(x, means, yerr=stds, capsize=4, width=0.5,
                       color="#4FC3F7", edgecolor="white", linewidth=0.5)
        for j, bar in enumerate(bars):
            if means[j] > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + stds[j] + 0.5,
                        f"{means[j]:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([short_label(l) for l in labels], fontsize=10)
        ax.set_ylabel("Tokens", fontsize=11)
        ax.set_title("Mean tokens generated per iteration (±std)", fontsize=12, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        path = os.path.join(out_dir, f"04c_tokens_per_iter.{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

    # ── 5. Buffer Statistics ──
    has_buffer = any(v is not None for v in m.get("buffer_mean", []))
    if has_buffer:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        fig.suptitle(f"Audio Buffer Statistics {title_suffix}", fontsize=14, fontweight="bold")
        bar_chart(ax, labels, m["buffer_mean"],
                  "Buffer Size (per iteration, mean)", "Time (s)", ".1f", "s")
        plt.tight_layout()
        path = os.path.join(out_dir, f"05_buffer_stats.{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

    # ── 6. Whisper Input Statistics ──
    has_input = any(v is not None for v in m.get("input_sec_mean", []))
    if has_input:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        fig.suptitle(f"Whisper Input Statistics {title_suffix}", fontsize=14, fontweight="bold")
        bar_chart(ax, labels, m["input_sec_mean"],
                  "Input Audio Length (per iter, mean)", "Time (s)", ".1f", "s")
        plt.tight_layout()
        path = os.path.join(out_dir, f"06_input_stats.{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

    # ── 7. Power consumption ──
    has_power = any(v is not None for v in m["power_mean"])
    if has_power:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Power Consumption", fontsize=14, fontweight="bold")

        grouped_bar_chart(
            axes[0], labels,
            [m["power_mean"], m["power_max"]],
            "Power Draw", "Power (mW)",
            ["Mean", "Max"], ".0f", "mW"
        )

        bar_chart(axes[1], labels, m["energy_total"], "Total Energy", "Energy (J)", ".3f", "J")

        plt.tight_layout()
        path = os.path.join(out_dir, f"07_power.{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

    # ── 7b. Power consumption time-series ──
    power_csv_data = {}
    for label in labels:
        safe_name = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
        csv_path = os.path.join(result_dir, f"power_samples_{safe_name}.csv")
        if os.path.exists(csv_path):
            times, powers = [], []
            with open(csv_path, "r") as f:
                next(f)  # skip header
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) == 2:
                        times.append(float(parts[0]))
                        powers.append(float(parts[1]) / 1000.0)  # uW -> mW
            if times:
                power_csv_data[label] = (np.array(times), np.array(powers))

    if power_csv_data:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        fig.suptitle("Power Consumption Over Time", fontsize=14, fontweight="bold")

        # Top: raw power time-series
        for i, label in enumerate(labels):
            if label in power_csv_data:
                t, p = power_csv_data[label]
                axes[0].plot(t, p, color=COLORS[i % len(COLORS)],
                             alpha=0.7, linewidth=0.8, label=short_label(label))
        axes[0].set_xlabel("Time (s)", fontsize=10)
        axes[0].set_ylabel("Power (mW)", fontsize=10)
        axes[0].set_title("Power Draw Over Time", fontsize=12, fontweight="bold")
        axes[0].legend(fontsize=9)
        axes[0].spines["top"].set_visible(False)
        axes[0].spines["right"].set_visible(False)

        # Bottom: smoothed (rolling average)
        window = 10  # number of samples for rolling average
        for i, label in enumerate(labels):
            if label in power_csv_data:
                t, p = power_csv_data[label]
                if len(p) >= window:
                    kernel = np.ones(window) / window
                    p_smooth = np.convolve(p, kernel, mode="valid")
                    t_smooth = t[:len(p_smooth)]
                else:
                    t_smooth, p_smooth = t, p
                axes[1].plot(t_smooth, p_smooth, color=COLORS[i % len(COLORS)],
                             linewidth=1.5, label=short_label(label))
        axes[1].set_xlabel("Time (s)", fontsize=10)
        axes[1].set_ylabel("Power (mW)", fontsize=10)
        axes[1].set_title(f"Power Draw (smoothed, window={window})", fontsize=12, fontweight="bold")
        axes[1].legend(fontsize=9)
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)

        plt.tight_layout()
        path = os.path.join(out_dir, f"07b_power_timeline.{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

    # ── 8. Summary table ──
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    col_labels = [short_label(l) for l in labels]
    row_labels = ["WER (%)", "RTF", "Wall Time (s)"]
    metric_list = [
        (m["wer"], ".1f"), (m["rtf"], ".3f"), (m["wall_time"], ".1f"),
    ]

    # GT latency
    if any(v is not None for v in m.get("gt_lat_mean", [])):
        row_labels.append("GT Latency (s)")
        metric_list.append((m["gt_lat_mean"], ".3f"))

    row_labels.append("Encode (ms)")
    metric_list.append((m["encode_mean"], ".0f"))

    cell_data = []

    for metric_vals, fmt_str in metric_list:
        row = []
        for v in metric_vals:
            row.append(f"{v:{fmt_str}}" if v is not None else "N/A")
        cell_data.append(row)

    # Add power if available
    if has_power:
        row_labels.append("Mean Power (mW)")
        row = []
        for v in m["power_mean"]:
            row.append(f"{v:.0f}" if v is not None else "N/A")
        cell_data.append(row)

        row_labels.append("Energy (J)")
        row = []
        for v in m["energy_total"]:
            row.append(f"{v:.3f}" if v is not None else "N/A")
        cell_data.append(row)

    table = ax.table(
        cellText=cell_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Color headers
    for j in range(len(col_labels)):
        table[0, j].set_facecolor(COLORS[j % len(COLORS)])
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight best values (lower is better for all metrics here)
    for i, row in enumerate(cell_data):
        vals = []
        for v in row:
            try:
                vals.append(float(v))
            except ValueError:
                vals.append(float("inf"))
        if vals:
            best_idx = np.argmin(vals)
            table[i + 1, best_idx].set_facecolor("#E8F5E9")

    ax.set_title("Summary Table", fontsize=14, fontweight="bold", pad=20)
    path = os.path.join(out_dir, f"08_summary_table.{fmt}")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Visualize comparison results")
    parser.add_argument("result_dirs", nargs="+", help="Path(s) to comparison result directory")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"],
                        help="Output image format")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for raster formats")
    args = parser.parse_args()

    for result_dir in args.result_dirs:
        comparison_file = os.path.join(result_dir, "comparison.txt")
        if not os.path.exists(comparison_file):
            print(f"ERROR: comparison.txt not found in {result_dir}")
            continue

        print(f"\nProcessing: {result_dir}")
        data = parse_comparison_file(comparison_file)
        print(f"  Systems: {', '.join(data['labels'])}")
        print(f"  Config: {data['config']}")
        print()

        out_dir = generate_plots(data, result_dir, fmt=args.format, dpi=args.dpi)
        print(f"\nAll plots saved to: {out_dir}/")


if __name__ == "__main__":
    main()
