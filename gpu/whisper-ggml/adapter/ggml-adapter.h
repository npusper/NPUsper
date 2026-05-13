#pragma once

// Whisperflow GGML Adapter Layer
// Bridges whisperflow's whisper.cpp (v1.6.2 ggml API) to vendored ggml (v0.9.7 from whisper.cpp latest)
//
// Changes handled:
// 1. CPU functions moved from ggml-backend.h to ggml-cpu.h
// 2. ggml_flash_attn_ext gained logit_softcap parameter (7-arg → 8-arg)
// 3. GGML_CALL macro deprecated
// 4. ggml_flash_ff removed (only used behind WHISPER_USE_FLASH_FF ifdef)

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ggml_concat compatibility wrapper
// Old (v1.6.2): ggml_concat(ctx, a, b) — always concat on dim 2
// New:          ggml_concat(ctx, a, b, dim)
static inline struct ggml_tensor * whisperflow_concat(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b) {
    return ggml_concat(ctx, a, b, 2);
}

// ggml_flash_attn_ext compatibility wrapper
// Old (v1.6.2): ggml_flash_attn_ext(ctx, q, k, v, mask, scale, max_bias)
// New:          ggml_flash_attn_ext(ctx, q, k, v, mask, scale, max_bias, logit_softcap)
// Whisper always uses logit_softcap=0.0f
static inline struct ggml_tensor * whisperflow_flash_attn_ext(
        struct ggml_context * ctx,
        struct ggml_tensor  * q,
        struct ggml_tensor  * k,
        struct ggml_tensor  * v,
        struct ggml_tensor  * mask,
        float                 scale,
        float                 max_bias) {
    return ggml_flash_attn_ext(ctx, q, k, v, mask, scale, max_bias, 0.0f);
}

#ifdef __cplusplus
}
#endif

// GGML_CALL macro - removed in latest ggml, define as empty if not defined
#ifndef GGML_CALL
#define GGML_CALL
#endif

// GGML_KQ_MASK_PAD - removed in latest ggml, was used for attention mask padding
#ifndef GGML_KQ_MASK_PAD
#define GGML_KQ_MASK_PAD 32
#endif
