"""Runtime gate that pins desktop app startup to the Clyde image code."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Mapping

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
CLYDE_CODE_ENV_VAR = "QUANTSHIELD_CLYDE_CODE"
PINNED_CLYDE_CODE = "QdQVKK276ugizr9d0RMRTmHgOQ11saOl"


def base62_from_bytes(data: bytes, length: int = 32) -> str:
    n = int.from_bytes(data, "big")

    chars: list[str] = []
    while n:
        n, remainder = divmod(n, 62)
        chars.append(ALPHABET[remainder])

    code = "".join(reversed(chars)).rjust(length, "0")

    return code[:length]


def image_kernel_bytes(image_path: str | Path, size: int = 128) -> bytes:
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:  # pragma: no cover - depends on optional runtime deps
        raise SystemExit(
            "The desktop app requires Pillow for Clyde image verification. "
            "Install it with `pip install -r requirements-app.txt` or `pip install -e .[app]`."
        ) from exc

    import numpy as np

    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img).convert("RGB")

    width, height = img.size

    small = img.resize((size, size), Image.Resampling.LANCZOS)

    rgb = np.asarray(small, dtype=np.uint8)
    gray = np.asarray(small.convert("L"), dtype=np.float32)

    block_size = size // 16

    luminance_blocks = gray.reshape(16, block_size, 16, block_size).mean(axis=(1, 3))
    luminance_kernel = np.clip(np.rint(luminance_blocks), 0, 255).astype(np.uint8)

    gx = np.diff(gray, axis=1, append=gray[:, -1:])
    gy = np.diff(gray, axis=0, append=gray[-1:, :])
    gradient_magnitude = np.sqrt(gx * gx + gy * gy)

    gradient_blocks = gradient_magnitude.reshape(16, block_size, 16, block_size).mean(axis=(1, 3))
    gradient_kernel = np.clip(np.rint(gradient_blocks), 0, 255).astype(np.uint8)

    color_kernel_parts: list[bytes] = []
    for channel in range(3):
        hist, _ = np.histogram(rgb[:, :, channel], bins=16, range=(0, 256))
        hist = hist / hist.sum()
        hist = np.clip(np.rint(hist * 255), 0, 255).astype(np.uint8)
        color_kernel_parts.append(hist.tobytes())

    with Path(image_path).open("rb") as image_file:
        exact_digest = hashlib.sha256(image_file.read()).digest()[:16]

    header = f"KIC1|{width}x{height}|RGB|{size}|".encode("ascii")

    return (
        header
        + luminance_kernel.tobytes()
        + gradient_kernel.tobytes()
        + b"".join(color_kernel_parts)
        + exact_digest
    )


def image_code(image_path: str | Path) -> str:
    kernel = image_kernel_bytes(image_path)
    digest = hashlib.sha256(kernel).digest()

    return base62_from_bytes(digest, length=32)


def resolved_clyde_image_path() -> Path:
    return Path(__file__).resolve().parents[2] / "assets" / "clyde.jpg"


def validate_clyde_runtime_environment(
    env: Mapping[str, str] | None = None,
    image_path: str | Path | None = None,
) -> str:
    clyde_image_path = Path(image_path) if image_path is not None else resolved_clyde_image_path()
    if not clyde_image_path.is_file():
        raise SystemExit(
            f"The desktop app requires `{clyde_image_path}` to exist so its Clyde access code can be verified."
        )

    computed_code = image_code(clyde_image_path)
    if computed_code != PINNED_CLYDE_CODE:
        raise SystemExit(
            "The desktop app is pinned to the Clyde access code "
            f"`{PINNED_CLYDE_CODE}`, but `{clyde_image_path}` currently resolves to `{computed_code}`."
        )

    environment = os.environ if env is None else env
    provided_code = environment.get(CLYDE_CODE_ENV_VAR)
    if provided_code != PINNED_CLYDE_CODE:
        raise SystemExit(
            "Set the Clyde access code in the environment before launching the desktop app: "
            f"`export {CLYDE_CODE_ENV_VAR}={PINNED_CLYDE_CODE}`."
        )

    return computed_code
