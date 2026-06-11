"""Structured encoder: game variables + labeled-object features -> vector.

This is the Pi-optimized path: it skips pixel processing entirely and encodes
the engine's own state (health/ammo/etc. via `game_variables`, plus the
relative position/size of visible labeled objects). Requires the labels buffer
to be enabled on the game (`make_basic_game(labels=True)`).

The vector is fixed-length: `n_game_vars` normalized variables followed by
`max_objects * 4` features (dx, dy, size, type-hash) for the nearest objects,
zero-padded. Tune the normalization constants per scenario.
"""
from __future__ import annotations

import numpy as np

# Rough normalizers so values land in ~[-1, 1]. Adjust per scenario.
_VAR_SCALE = 100.0
_POS_SCALE = 320.0  # screen-ish coordinate scale


def structured_dim(n_game_vars: int, max_objects: int = 8) -> int:
    return n_game_vars + max_objects * 4


def encode_structured(state, n_game_vars: int, max_objects: int = 8) -> np.ndarray:
    """Encode a ViZDoom `GameState` into a fixed-length float32 vector."""
    gv = np.asarray(state.game_variables, dtype=np.float32)
    gv = np.resize(gv, n_game_vars) / _VAR_SCALE

    obj_feats = np.zeros((max_objects, 4), dtype=np.float32)
    labels = getattr(state, "labels", None) or []
    # nearest-by-screen-area first (bigger == closer, cheap heuristic)
    labels = sorted(labels, key=lambda l: -(l.width * l.height))[:max_objects]
    for i, lab in enumerate(labels):
        cx = (lab.x + lab.width / 2.0) / _POS_SCALE - 1.0
        cy = (lab.y + lab.height / 2.0) / _POS_SCALE - 1.0
        area = (lab.width * lab.height) / (_POS_SCALE * _POS_SCALE)
        type_hash = (lab.object_id % 97) / 97.0
        obj_feats[i] = (cx, cy, area, type_hash)

    return np.concatenate([gv, obj_feats.reshape(-1)]).astype(np.float32)
