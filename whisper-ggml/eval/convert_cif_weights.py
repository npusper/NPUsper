"""Convert CIF model weights from PyTorch .pt to simple binary format for C++.

Binary format: [int32 n_state] [float32 weight × n_state] [float32 bias × 1]

Usage:
    python convert_cif_weights.py
    python convert_cif_weights.py --model base
    python convert_cif_weights.py --model all
"""

import os
import sys
import struct
import argparse

def convert_cif_weights(pt_path, bin_path):
    import torch
    ckpt = torch.load(pt_path, map_location="cpu")
    weight = ckpt["weight"].squeeze(0).numpy()  # (n_state,)
    bias = ckpt["bias"].squeeze(0).item()        # scalar
    n_state = len(weight)

    with open(bin_path, "wb") as f:
        f.write(struct.pack("<i", n_state))
        for w in weight:
            f.write(struct.pack("<f", float(w)))
        f.write(struct.pack("<f", bias))

    print(f"  {pt_path} -> {bin_path}")
    print(f"  n_state={n_state}, bias={bias:.6f}, file_size={os.path.getsize(bin_path)} bytes")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all",
                        choices=["base", "small", "medium", "large-v2", "all"])
    parser.add_argument("--cif-dir", type=str,
                        default="../../simul_whisper/cif_models")
    parser.add_argument("--output-dir", type=str,
                        default="../models")
    args = parser.parse_args()

    cif_dir = os.path.abspath(args.cif_dir)
    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    if args.model == "all":
        models = ["base", "small", "medium"]
        # Add large-v2 if exists
        if os.path.exists(os.path.join(cif_dir, "large-v2.pt")):
            models.append("large-v2")
    else:
        models = [args.model]

    for model in models:
        pt_path = os.path.join(cif_dir, f"{model}.pt")
        bin_path = os.path.join(out_dir, f"cif_{model.replace('-', '_')}.bin")
        if not os.path.exists(pt_path):
            print(f"  WARNING: {pt_path} not found, skipping")
            continue
        convert_cif_weights(pt_path, bin_path)

    print("Done.")

if __name__ == "__main__":
    main()
