"""QAIRT/QNN SDK path resolution shared by compile scripts."""

import os
from pathlib import Path

QAIRT_VERSION = "2.37.1.250807"
HOST_TRIPLE = "x86_64-linux-clang"


def _npu_root() -> Path:
    return Path(__file__).resolve().parent


def qairt_converter_path(root: Path) -> Path:
    return root / "bin" / HOST_TRIPLE / "qairt-converter"


def qairt_context_binary_generator_path(root: Path) -> Path:
    return root / "bin" / HOST_TRIPLE / "qnn-context-binary-generator"


def qairt_htp_lib_path(root: Path) -> Path:
    return root / "lib" / HOST_TRIPLE / "libQnnHtp.so"


def qairt_htp_ext_lib_path(root: Path) -> Path:
    return root / "lib" / HOST_TRIPLE / "libQnnHtpNetRunExtensions.so"


def candidate_qnn_sdk_roots() -> list[Path]:
    candidates: list[Path] = []

    env = os.environ.get("QNN_SDK_ROOT")
    if env:
        candidates.append(Path(env).expanduser())

    npu_root = _npu_root()
    candidates.extend([
        npu_root / "qairt" / QAIRT_VERSION,
        Path.home() / "qairt" / QAIRT_VERSION,
        Path("/opt/qairt") / QAIRT_VERSION,
        Path("/opt/qualcomm/qairt") / QAIRT_VERSION,
    ])

    # Preserve order while removing duplicates.
    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def resolve_qnn_sdk_root() -> Path:
    for root in candidate_qnn_sdk_roots():
        if qairt_converter_path(root).exists() and qairt_context_binary_generator_path(root).exists():
            return root

    env = os.environ.get("QNN_SDK_ROOT")
    if env:
        return Path(env).expanduser()

    return _npu_root() / "qairt" / QAIRT_VERSION


def qairt_missing_message(root: Path) -> str:
    candidates = "\n".join(f"    - {p}" for p in candidate_qnn_sdk_roots())
    return (
        f"QAIRT/QNN SDK tools were not found under: {root}\n"
        f"Expected converter: {qairt_converter_path(root)}\n"
        "Install Qualcomm AI Runtime SDK 2.37.1.250807 and either set QNN_SDK_ROOT "
        "or place it at npu/qairt/2.37.1.250807.\n"
        "Searched:\n"
        f"{candidates}"
    )
