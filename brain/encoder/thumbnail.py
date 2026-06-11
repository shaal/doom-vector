"""Cheap, scenario-agnostic encoder: screen buffer -> small grayscale thumbnail.

Block-mean pools the frame to `size x size` and flattens to a [0,1] vector of
length `size*size`. Pure NumPy (no cv2/PIL), so it runs anywhere including the
Pi. Robust default for the Phase 0 spikes; the structured encoder (game vars +
labels) is the lighter Pi-optimized alternative once a scenario is fixed.
"""
from __future__ import annotations

import numpy as np


def encode_thumbnail(screen_buffer, size: int = 16) -> np.ndarray:
    """Encode a ViZDoom screen buffer into a length-`size*size` float32 vector.

    Accepts GRAY8 (H, W), RGB24 (H, W, 3), or CRCGCB (3, H, W) buffers.
    """
    img = np.asarray(screen_buffer, dtype=np.float32)
    if img.ndim == 3:
        # collapse channels: CRCGCB is (3,H,W); RGB24 is (H,W,3)
        ch_axis = 0 if img.shape[0] in (1, 3) else 2
        img = img.mean(axis=ch_axis)

    h, w = img.shape
    hs, ws = h // size, w // size
    if hs == 0 or ws == 0:
        raise ValueError(f"frame {img.shape} too small to pool to {size}x{size}")
    img = img[: hs * size, : ws * size]
    pooled = img.reshape(size, hs, size, ws).mean(axis=(1, 3))
    return (pooled.reshape(-1) / 255.0).astype(np.float32)


THUMBNAIL_DIM = lambda size=16: size * size  # noqa: E731 - convenience
