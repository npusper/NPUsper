from __future__ import annotations

from dataclasses import dataclass
import os


def _normalize_model_key(model: str) -> str:
    model = model.strip()
    if model.startswith("openai/whisper-"):
        model = model[len("openai/whisper-"):]
    if model.startswith("whisper-"):
        model = model[len("whisper-"):]
    if model in {"large", "large-v1", "large-v2", "large-v3"}:
        return model
    return model.lower()


def sanitize_name(name: str) -> str:
    return name.replace(".", "_").replace("-", "_")


@dataclass(frozen=True)
class WhisperModelProfile:
    key: str
    hf_repo: str
    num_blocks: int
    num_heads: int
    attention_dim: int
    vocab_size: int
    num_mel_bins: int
    alignment_heads: str
    aheads_preset: str

    @property
    def head_dim(self) -> int:
        return self.attention_dim // self.num_heads

    @property
    def deploy_model_name(self) -> str:
        return f"whisper-{self.key}"

    @property
    def safe_name(self) -> str:
        return sanitize_name(self.key)

    @property
    def n_alignment_heads(self) -> int:
        if not self.alignment_heads:
            return 0
        return len([x for x in self.alignment_heads.split(";") if x.strip()])


@dataclass(frozen=True)
class TargetProfile:
    key: str
    artifact_suffix: str
    device_name: str
    target_device: str
    soc_model: int
    dsp_arch: str
    bucket_emb_overrides: dict[float, int]

    @property
    def safe_name(self) -> str:
        return sanitize_name(self.key)


WHISPER_MODEL_PROFILES = {
    "tiny.en": WhisperModelProfile(
        key="tiny.en",
        hf_repo="openai/whisper-tiny.en",
        num_blocks=4,
        num_heads=6,
        attention_dim=384,
        vocab_size=51865,
        num_mel_bins=80,
        alignment_heads="1,0;2,0;2,5;3,0;3,1;3,2;3,3;3,4",
        aheads_preset="tiny.en",
    ),
    "tiny": WhisperModelProfile(
        key="tiny",
        hf_repo="openai/whisper-tiny",
        num_blocks=4,
        num_heads=6,
        attention_dim=384,
        vocab_size=51865,
        num_mel_bins=80,
        alignment_heads="2,2;3,0;3,2;3,3;3,4;3,5",
        aheads_preset="tiny",
    ),
    "base.en": WhisperModelProfile(
        key="base.en",
        hf_repo="openai/whisper-base.en",
        num_blocks=6,
        num_heads=8,
        attention_dim=512,
        vocab_size=51865,
        num_mel_bins=80,
        alignment_heads="3,3;4,7;5,1;5,5;5,7",
        aheads_preset="base.en",
    ),
    "base": WhisperModelProfile(
        key="base",
        hf_repo="openai/whisper-base",
        num_blocks=6,
        num_heads=8,
        attention_dim=512,
        vocab_size=51865,
        num_mel_bins=80,
        alignment_heads="3,1;4,2;4,3;4,7;5,1;5,2;5,4;5,6",
        aheads_preset="base",
    ),
    "small.en": WhisperModelProfile(
        key="small.en",
        hf_repo="openai/whisper-small.en",
        num_blocks=12,
        num_heads=12,
        attention_dim=768,
        vocab_size=51865,
        num_mel_bins=80,
        alignment_heads="6,6;7,0;7,3;7,8;8,2;8,5;8,7;9,0;9,4;9,8;9,10;10,0;10,1;10,2;10,3;10,6;10,11;11,2;11,4",
        aheads_preset="small.en",
    ),
    "small": WhisperModelProfile(
        key="small",
        hf_repo="openai/whisper-small",
        num_blocks=12,
        num_heads=12,
        attention_dim=768,
        vocab_size=51865,
        num_mel_bins=80,
        alignment_heads="5,3;5,9;8,0;8,4;8,7;8,8;9,0;9,7;9,9;10,5",
        aheads_preset="small",
    ),
    "medium.en": WhisperModelProfile(
        key="medium.en",
        hf_repo="openai/whisper-medium.en",
        num_blocks=24,
        num_heads=16,
        attention_dim=1024,
        vocab_size=51865,
        num_mel_bins=80,
        alignment_heads="11,4;14,1;14,12;14,14;15,4;16,0;16,4;16,9;17,12;17,14;18,7;18,10;18,15;20,0;20,3;20,9;20,14;21,12",
        aheads_preset="medium.en",
    ),
    "medium": WhisperModelProfile(
        key="medium",
        hf_repo="openai/whisper-medium",
        num_blocks=24,
        num_heads=16,
        attention_dim=1024,
        vocab_size=51865,
        num_mel_bins=80,
        alignment_heads="13,15;15,4;15,15;16,1;20,0;23,4",
        aheads_preset="medium",
    ),
    "large-v1": WhisperModelProfile(
        key="large-v1",
        hf_repo="openai/whisper-large-v1",
        num_blocks=32,
        num_heads=20,
        attention_dim=1280,
        vocab_size=51865,
        num_mel_bins=80,
        alignment_heads="9,19;11,2;11,4;11,17;22,7;22,11;22,17;23,2;23,15",
        aheads_preset="large-v1",
    ),
    "large-v2": WhisperModelProfile(
        key="large-v2",
        hf_repo="openai/whisper-large-v2",
        num_blocks=32,
        num_heads=20,
        attention_dim=1280,
        vocab_size=51865,
        num_mel_bins=80,
        alignment_heads="10,12;13,17;16,11;16,12;16,13;17,15;17,16;18,4;18,11;18,19;19,11;21,2;21,3;22,3;22,9;22,12;23,5;23,7;23,13;25,5;26,1;26,12;27,15",
        aheads_preset="large-v2",
    ),
    "large-v3": WhisperModelProfile(
        key="large-v3",
        hf_repo="openai/whisper-large-v3",
        num_blocks=32,
        num_heads=20,
        attention_dim=1280,
        vocab_size=51866,
        num_mel_bins=128,
        alignment_heads="7,0;10,17;12,18;13,12;16,1;17,14;19,11;21,4;24,1;25,6",
        aheads_preset="large-v3",
    ),
    "large": WhisperModelProfile(
        key="large",
        hf_repo="openai/whisper-large",
        num_blocks=32,
        num_heads=20,
        attention_dim=1280,
        vocab_size=51865,
        num_mel_bins=80,
        alignment_heads="9,19;11,2;11,4;11,17;22,7;22,11;22,17;23,2;23,15",
        aheads_preset="large-v1",
    ),
}


TARGET_PROFILES = {
    "xplus": TargetProfile(
        key="xplus",
        artifact_suffix="xplus",
        device_name="Snapdragon X Plus 8-Core CRD",
        target_device="Snapdragon X Plus 8-Core CRD",
        soc_model=60,
        dsp_arch="v73",
        bucket_emb_overrides={2.0: 100, 30.0: 1500},
    ),
    "s25": TargetProfile(
        key="s25",
        artifact_suffix="s25",
        device_name="Samsung Galaxy S25",
        target_device="Samsung Galaxy S25",
        soc_model=69,
        dsp_arch="v79",
        bucket_emb_overrides={30.0: 1500},
    ),
    "s25_xplus_compat": TargetProfile(
        key="s25_xplus_compat",
        artifact_suffix="xplus",
        device_name="Samsung Galaxy S25",
        target_device="Samsung Galaxy S25",
        soc_model=69,
        dsp_arch="v79",
        bucket_emb_overrides={2.0: 100, 30.0: 1500},
    ),
}


def resolve_whisper_model_profile(model: str) -> WhisperModelProfile:
    key = _normalize_model_key(model)
    if key not in WHISPER_MODEL_PROFILES:
        raise KeyError(f"Unsupported Whisper model '{model}'")
    return WHISPER_MODEL_PROFILES[key]


def resolve_target_profile(target: str) -> TargetProfile:
    key = target.strip().lower().replace("snapdragon-x-plus", "xplus")
    aliases = {
        "snapdragon x plus 8-core crd": "xplus",
        "samsung galaxy s25": "s25",
    }
    key = aliases.get(key, key)
    if key not in TARGET_PROFILES:
        raise KeyError(f"Unsupported target device '{target}'")
    return TARGET_PROFILES[key]


def bucket_audio_emb_len(audio_sec: float, target: TargetProfile) -> int:
    for sec, emb in target.bucket_emb_overrides.items():
        if abs(audio_sec - sec) < 1e-6:
            return emb
    raw = int(audio_sec * 50)
    return ((raw + 7) // 8) * 8


def default_build_dir(kind: str, model: WhisperModelProfile, target: TargetProfile) -> str:
    legacy = {
        ("encoder", "base", "xplus"): "xplus_build",
        ("decoder_1step", "base", "xplus"): "xplus_build_1step",
        ("decoder_1step_bucket", "base", "xplus"): "xplus_build_1step_{sec}s",
        ("unroll_k", "base", "xplus"): "xplus_build_unroll_k",
        ("nk_full", "base", "xplus"): "xplus_build_nk_full",
    }
    legacy_key = (kind, model.key, target.key)
    if legacy_key in legacy:
        return legacy[legacy_key]
    generic = {
        "encoder": f"{target.safe_name}_build_encoder_{model.safe_name}",
        "decoder_1step": f"{target.safe_name}_build_1step_{model.safe_name}",
        "decoder_1step_bucket": f"{target.safe_name}_build_1step_{model.safe_name}_{{sec}}s",
        "unroll_k": f"{target.safe_name}_build_unroll_k_{model.safe_name}",
        "nk_full": f"{target.safe_name}_build_nk_full_{model.safe_name}",
    }
    if kind not in generic:
        raise KeyError(f"Unsupported build dir kind '{kind}'")
    return generic[kind]


def resolve_hf_model_source(model: WhisperModelProfile) -> str:
    offline = os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    if not offline:
        return model.hf_repo

    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(model.hf_repo, local_files_only=True)
    except Exception:
        return model.hf_repo
