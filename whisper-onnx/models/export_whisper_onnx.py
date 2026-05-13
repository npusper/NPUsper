"""
Export OpenAI Whisper encoder and decoder (with KV cache) to ONNX.

Three ONNX files:

  encoder.onnx
    in : mel             [1, 80, n_frames]          (n_frames dynamic)
    out: encoder_output  [1, n_audio_ctx, d_model]  (n_audio_ctx dynamic)

  decoder_prefill.onnx   -- run once for initial prompt tokens
    in : tokens          [1, n_tokens]              (n_tokens dynamic)
         encoder_output  [1, n_audio_ctx, d_model]
    out: logits          [1, n_tokens, n_vocab]
         self_kv         [n_layers, 2, 1, n_head, n_tokens, head_dim]
         cross_kv        [n_layers, 2, 1, n_head, n_audio_ctx, head_dim]

  decoder_step.onnx      -- run once per new token during greedy decode
    in : token           [1, 1]
         past_self_kv    [n_layers, 2, 1, n_head, past_len, head_dim]  (past_len dynamic)
         cross_kv        [n_layers, 2, 1, n_head, n_audio_ctx, head_dim]
    out: logits          [1, 1, n_vocab]
         new_self_kv     [n_layers, 2, 1, n_head, past_len+1, head_dim]

KV cache shape convention (same as Moonshine):
  dim 0 = layer index
  dim 1 = 0 for K, 1 for V
  dim 2 = batch (always 1 for streaming)
  dim 3 = n_head
  dim 4 = sequence length  (dynamic)
  dim 5 = head_dim

Usage:
    python export_whisper_onnx.py --model base   --output ./base
    python export_whisper_onnx.py --model base --checkpoint /path/to/base.pt --output ./base
"""

import sys
import argparse
import os
from typing import Tuple

sys.path.insert(0, '../../whisper')

import torch
import torch.nn as nn
import torch.nn.functional as F
import whisper


# ── Encoder wrapper (unchanged) ───────────────────────────────────────────────

class EncoderWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.encoder(mel)


# ── Attention helpers ─────────────────────────────────────────────────────────

def _qkv_split(linear_q, linear_k, linear_v, x, n_head):
    """Project x → Q, K, V and reshape to [B, H, T, D]."""
    B, T, _ = x.shape
    D = linear_q.out_features // n_head
    q = linear_q(x).view(B, T, n_head, D).permute(0, 2, 1, 3)
    k = linear_k(x).view(B, T, n_head, D).permute(0, 2, 1, 3)
    v = linear_v(x).view(B, T, n_head, D).permute(0, 2, 1, 3)
    return q, k, v


def _attn(q, k, v, mask=None, return_weights=False):
    """Scaled dot-product attention. Returns [B, T_q, d_model] and optionally weights [B, H, T_q, T_k]."""
    D = q.shape[-1]
    scale = D ** -0.25
    qk = (q * scale) @ (k * scale).transpose(-1, -2)   # [B, H, T_q, T_k]
    if mask is not None:
        qk = qk + mask
    w = F.softmax(qk.float(), dim=-1).to(q.dtype)
    out = (w @ v).permute(0, 2, 1, 3)                  # [B, T_q, H, D]
    out = out.flatten(start_dim=2)                      # [B, T_q, d_model]
    if return_weights:
        return out, w
    return out


# ── Decoder prefill wrapper ───────────────────────────────────────────────────

class DecoderPrefillWrapper(nn.Module):
    """
    Process initial prompt tokens.
    Returns logits, self_kv, and cross_kv for all layers.
    self_kv / cross_kv shape: [n_layers, 2, 1, n_head, seq_len, head_dim]
    """
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(
        self,
        tokens: torch.Tensor,          # [1, n_tokens]
        encoder_output: torch.Tensor,  # [1, n_audio_ctx, d_model]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dec = self.decoder
        B, T = tokens.shape
        n_audio_ctx = encoder_output.shape[1]

        x = dec.token_embedding(tokens) + dec.positional_embedding[:T]
        x = x.to(encoder_output.dtype)

        # Causal mask for self-attention
        mask = dec.mask[:T, :T]  # [T, T]

        self_ks, self_vs = [], []
        cross_ks, cross_vs = [], []

        for block in dec.blocks:
            attn = block.attn
            H = attn.n_head
            D = attn.query.out_features // H

            # ── Self-attention ──────────────────────────────────────────────
            normed = block.attn_ln(x)
            q, k, v = _qkv_split(attn.query, attn.key, attn.value, normed, H)
            # k, v: [1, H, T, D]
            self_ks.append(k)
            self_vs.append(v)
            out = _attn(q, k, v, mask)   # [1, T, d_model]
            x = x + attn.out(out)

            # ── Cross-attention ─────────────────────────────────────────────
            cattn = block.cross_attn
            normed = block.cross_attn_ln(x)
            q_c = cattn.query(normed).view(B, T, H, D).permute(0, 2, 1, 3)
            k_c = cattn.key(encoder_output).view(B, n_audio_ctx, H, D).permute(0, 2, 1, 3)
            v_c = cattn.value(encoder_output).view(B, n_audio_ctx, H, D).permute(0, 2, 1, 3)
            cross_ks.append(k_c)
            cross_vs.append(v_c)
            out = _attn(q_c, k_c, v_c)   # no mask for cross-attn
            x = x + cattn.out(out)

            # ── FFN ─────────────────────────────────────────────────────────
            x = x + block.mlp(block.mlp_ln(x))

        x = dec.ln(x)
        logits = (x @ dec.token_embedding.weight.to(x.dtype).T).float()

        # Stack: [n_layers, 2, 1, H, seq, D]
        self_kv  = torch.stack([torch.stack(self_ks),  torch.stack(self_vs)],  dim=1)
        cross_kv = torch.stack([torch.stack(cross_ks), torch.stack(cross_vs)], dim=1)

        return logits, self_kv, cross_kv


# ── Decoder step wrapper ──────────────────────────────────────────────────────

class DecoderStepWrapper(nn.Module):
    """
    Single-token decode step with KV cache.
    past_self_kv shape: [n_layers, 2, 1, n_head, past_len, head_dim]
    cross_kv shape:     [n_layers, 2, 1, n_head, n_audio_ctx, head_dim]
    Returns logits [1, 1, n_vocab], new_self_kv [n_layers, 2, 1, n_head, past_len+1, head_dim],
    and cross_attn_weights [1, n_layers, n_head, n_audio_ctx].
    """
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(
        self,
        token: torch.Tensor,           # [1, 1]
        past_self_kv: torch.Tensor,    # [n_layers, 2, 1, n_head, past_len, head_dim]
        cross_kv: torch.Tensor,        # [n_layers, 2, 1, n_head, n_audio_ctx, head_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dec = self.decoder
        B = token.shape[0]
        past_len = past_self_kv.shape[4]

        x = dec.token_embedding(token) + dec.positional_embedding[past_len : past_len + 1]
        x = x.to(cross_kv.dtype)

        new_ks, new_vs = [], []
        cross_attn_ws = []   # per-layer cross-attn weights [B, H, n_audio_ctx]

        for i, block in enumerate(dec.blocks):
            attn = block.attn
            H = attn.n_head
            D = attn.query.out_features // H

            # ── Self-attention with KV cache ────────────────────────────────
            normed = block.attn_ln(x)
            q = attn.query(normed).view(B, 1, H, D).permute(0, 2, 1, 3)   # [1, H, 1, D]
            k_new = attn.key(normed).view(B, 1, H, D).permute(0, 2, 1, 3)
            v_new = attn.value(normed).view(B, 1, H, D).permute(0, 2, 1, 3)

            past_k = past_self_kv[i, 0]   # [1, H, past_len, D]
            past_v = past_self_kv[i, 1]

            k = torch.cat([past_k, k_new], dim=2)  # [1, H, past_len+1, D]
            v = torch.cat([past_v, v_new], dim=2)

            new_ks.append(k)
            new_vs.append(v)

            out = _attn(q, k, v)           # no causal mask — single query token
            x = x + attn.out(out)

            # ── Cross-attention (reuse cached KV, no recompute) ─────────────
            cattn = block.cross_attn
            normed = block.cross_attn_ln(x)
            q_c = cattn.query(normed).view(B, 1, H, D).permute(0, 2, 1, 3)
            c_k = cross_kv[i, 0]           # [1, H, n_audio_ctx, D]
            c_v = cross_kv[i, 1]

            out, w = _attn(q_c, c_k, c_v, return_weights=True)  # w: [B, H, 1, n_audio_ctx]
            cross_attn_ws.append(w[:, :, 0, :])   # [B, H, n_audio_ctx]
            x = x + cattn.out(out)

            # ── FFN ─────────────────────────────────────────────────────────
            x = x + block.mlp(block.mlp_ln(x))

        x = dec.ln(x)
        logits = (x @ dec.token_embedding.weight.to(x.dtype).T).float()   # [1, 1, n_vocab]

        # [n_layers, 2, 1, H, past_len+1, D]
        new_self_kv = torch.stack([torch.stack(new_ks), torch.stack(new_vs)], dim=1)

        # cross_attn_weights: [1, n_layers, n_head, n_audio_ctx]
        cross_attn_weights = torch.stack(cross_attn_ws, dim=1)

        return logits, new_self_kv, cross_attn_weights


# ── Export functions ──────────────────────────────────────────────────────────

def export_encoder(model, output_dir: str, opset: int, fp16: bool = False):
    encoder = EncoderWrapper(model.encoder).eval()
    # Use a small dummy (dtype/rank/fixed-dims only); actual n_frames is dynamic.
    dummy_mel = torch.zeros(1, model.dims.n_mels, 200)

    # n_frames can be anything from 1 chunk (~200 frames / 2s) up to 3000 (30s).
    # After the stride-2 conv2, n_audio_ctx = n_frames // 2.
    # Whisper's positional_embedding[:x.shape[1]] slice becomes a dynamic Slice op
    # in the ONNX graph when exported with dynamo=True.
    n_frames_dim = torch.export.Dim("n_frames", min=2, max=3000)

    path = os.path.join(output_dir, "encoder.onnx")
    torch.onnx.export(
        encoder, dummy_mel, path,
        opset_version=opset,
        dynamo=True,
        input_names=["mel"],
        output_names=["encoder_output"],
        dynamic_shapes={"mel": {2: n_frames_dim}},
    )
    print(f"[encoder]         saved → {path}")
    print(f"  mel [1, {model.dims.n_mels}, n_frames*] → encoder_output [1, n_frames//2, {model.dims.n_audio_state}]")
    print(f"  * n_frames is dynamic: any even value in [2, 3000]")


def export_decoder_prefill(model, output_dir: str, opset: int, fp16: bool = False):
    d = model.dims
    n_layers  = d.n_text_layer
    n_head    = d.n_text_head
    head_dim  = d.n_text_state // n_head

    wrapper = DecoderPrefillWrapper(model.decoder).eval()

    dummy_tokens  = torch.zeros(1, 5, dtype=torch.long)
    dummy_enc_out = torch.zeros(1, d.n_audio_ctx, d.n_audio_state)

    path = os.path.join(output_dir, "decoder_prefill.onnx")
    torch.onnx.export(
        wrapper,
        (dummy_tokens, dummy_enc_out),
        path,
        opset_version=opset, dynamo=False,
        input_names=["tokens", "encoder_output"],
        output_names=["logits", "self_kv", "cross_kv"],
        dynamic_axes={
            "tokens":         {0: "batch", 1: "n_tokens"},
            "encoder_output": {0: "batch", 1: "n_audio_ctx"},
            "logits":         {0: "batch", 1: "n_tokens"},
            "self_kv":        {2: "batch", 4: "n_tokens"},
            "cross_kv":       {2: "batch", 4: "n_audio_ctx"},
        },
        do_constant_folding=True,
    )
    print(f"[decoder_prefill] saved → {path}")
    print(f"  in : tokens [batch, n_tokens], encoder_output [batch, n_audio_ctx, {d.n_audio_state}]")
    print(f"  out: logits [batch, n_tokens, {d.n_vocab}]")
    print(f"       self_kv  [{n_layers}, 2, batch, {n_head}, n_tokens,    {head_dim}]")
    print(f"       cross_kv [{n_layers}, 2, batch, {n_head}, n_audio_ctx, {head_dim}]")


def export_decoder_step(model, output_dir: str, opset: int, fp16: bool = False):
    d = model.dims
    n_layers  = d.n_text_layer
    n_head    = d.n_text_head
    head_dim  = d.n_text_state // n_head
    past_len  = 5   # dummy past length for tracing

    wrapper = DecoderStepWrapper(model.decoder).eval()

    dummy_token        = torch.zeros(1, 1, dtype=torch.long)
    dummy_past_self_kv = torch.zeros(n_layers, 2, 1, n_head, past_len, head_dim)
    dummy_cross_kv     = torch.zeros(n_layers, 2, 1, n_head, d.n_audio_ctx, head_dim)

    path = os.path.join(output_dir, "decoder_step.onnx")
    torch.onnx.export(
        wrapper,
        (dummy_token, dummy_past_self_kv, dummy_cross_kv),
        path,
        opset_version=opset, dynamo=False,
        input_names=["token", "past_self_kv", "cross_kv"],
        output_names=["logits", "new_self_kv", "cross_attn_weights"],
        dynamic_axes={
            "token":               {0: "batch"},
            "past_self_kv":        {2: "batch", 4: "past_len"},
            "cross_kv":            {2: "batch", 4: "n_audio_ctx"},
            "logits":              {0: "batch"},
            "new_self_kv":         {2: "batch", 4: "new_len"},
            "cross_attn_weights":  {0: "batch", 3: "n_audio_ctx"},
        },
        do_constant_folding=True,
    )
    print(f"[decoder_step]    saved → {path}")
    print(f"  in : token [batch, 1], past_self_kv [{n_layers}, 2, batch, {n_head}, past_len, {head_dim}]")
    print(f"       cross_kv [{n_layers}, 2, batch, {n_head}, n_audio_ctx, {head_dim}]")
    print(f"  out: logits [batch, 1, {d.n_vocab}], new_self_kv [{n_layers}, 2, batch, {n_head}, past_len+1, {head_dim}]")
    print(f"       cross_attn_weights [batch, {n_layers}, {n_head}, n_audio_ctx]")


def export_vocab(model, output_dir: str):
    tokenizer = whisper.tokenizer.get_tokenizer(multilingual=model.is_multilingual)
    n_vocab = model.dims.n_vocab
    lines = []
    for i in range(n_vocab):
        try:
            s = tokenizer.decode([i])
        except Exception:
            s = f"<unk_{i}>"
        lines.append(s.replace("\n", "\\n").replace("\r", "\\r"))
    path = os.path.join(output_dir, "vocab.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[vocab]           saved → {path}  ({n_vocab} tokens)")


def export_mel_filters(output_dir: str, n_mels: int = 80):
    """Export mel filterbank weights as binary file (matches whisper.cpp format)."""
    import numpy as np
    filters = whisper.audio.mel_filters("cpu", n_mels)
    data = filters.numpy().astype(np.float32)
    path = os.path.join(output_dir, "mel_filters.bin")
    # Header: n_mel (int32), n_fft (int32), then data [n_mel x n_fft] float32
    import struct
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", data.shape[0], data.shape[1]))
        f.write(data.tobytes())
    print(f"[mel_filters]     saved → {path}  ({data.shape[0]} x {data.shape[1]}, {os.path.getsize(path)} bytes)")


def export_dims(model, output_dir: str):
    d = model.dims
    lines = [
        f"n_mels={d.n_mels}",
        f"n_audio_ctx={d.n_audio_ctx}",
        f"n_audio_state={d.n_audio_state}",
        f"n_audio_head={d.n_audio_head}",
        f"n_audio_layer={d.n_audio_layer}",
        f"n_vocab={d.n_vocab}",
        f"n_text_ctx={d.n_text_ctx}",
        f"n_text_state={d.n_text_state}",
        f"n_text_head={d.n_text_head}",
        f"n_text_layer={d.n_text_layer}",
        f"head_dim={d.n_text_state // d.n_text_head}",
        f"is_multilingual={int(model.is_multilingual)}",
    ]
    path = os.path.join(output_dir, "dims.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[dims]            saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="base",
                        choices=whisper.available_models())
    parser.add_argument("--checkpoint", default=None,
                        help="Path to local .pt checkpoint")
    parser.add_argument("--output",     default=None,
                        help="Output directory (default: ./<model>)")
    parser.add_argument("--opset",      type=int, default=17)
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--fp16",       action="store_true",
                        help="Export model in float16 (to match ggml fp16 weights)")
    args = parser.parse_args()

    output_dir = args.output or os.path.join(os.path.dirname(__file__), args.model)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading whisper '{args.model}' on {args.device} ...")
    if args.checkpoint:
        model = whisper.load_model(args.checkpoint, device=args.device)
    else:
        model = whisper.load_model(args.model, device=args.device)
    model.eval()

    # fp16 conversion is done post-export at ONNX level (see below)

    # Disable SDPA: 'is_causal' becomes SymBool during tracing → breaks export
    from whisper.model import MultiHeadAttention
    MultiHeadAttention.use_sdpa = False

    print(f"Model dims: {model.dims}")
    print(f"Output dir: {output_dir}\n")

    with torch.no_grad():
        export_encoder(model, output_dir, args.opset)
        export_decoder_prefill(model, output_dir, args.opset)
        export_decoder_step(model, output_dir, args.opset)
        export_vocab(model, output_dir)
        export_dims(model, output_dir)
        export_mel_filters(output_dir, model.dims.n_mels)

    # Post-export fp16 conversion at ONNX level
    # Converts float weights to float16 while keeping certain ops in fp32 for stability
    if args.fp16:
        from onnxruntime.transformers.float16 import convert_float_to_float16
        import onnx
        print("\nConverting ONNX models to float16 ...")
        for fname in ["encoder.onnx", "decoder_prefill.onnx", "decoder_step.onnx"]:
            path = os.path.join(output_dir, fname)
            if not os.path.exists(path):
                continue
            m = onnx.load(path)
            m_fp16 = convert_float_to_float16(m, keep_io_types=True)
            onnx.save(m_fp16, path)
            print(f"  {fname} → fp16")

    print("\nDone. Files written:")
    for f in sorted(os.listdir(output_dir)):
        size = os.path.getsize(os.path.join(output_dir, f))
        print(f"  {f:30s}  {size/1e6:.1f} MB" if size > 1e5 else f"  {f}")


if __name__ == "__main__":
    main()
