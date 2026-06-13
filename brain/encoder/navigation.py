"""Navigation encoder: agent pose + nearest visible object -> small vector.

The right recall key for path prediction: "from a spot (and facing) like this,
which trajectory worked?" Encodes normalized POSITION_X/Y, heading as
(sin, cos) (no wraparound discontinuity), and the screen-relative bearing/size
of the nearest visible labeled object (e.g. a health kit / goal). Tiny and
cheap — ideal for the Pi.

Requires the game created with `position=True` (and `labels=True` for the
object features). Reads pose via `get_game_variable` so it is independent of
the game-variable ordering in the .cfg.
"""
from __future__ import annotations

import math

import numpy as np

_POS_SCALE = 1000.0  # Doom map units -> ~O(1); only relative scale matters for k-NN
_HEALTH_SCALE = 100.0
_SCREEN_W = 320.0
_SCREEN_H = 240.0


def navigation_dim(max_objects: int = 1) -> int:
    # px, py, sin, cos, health + (dx, dy, area) per object
    return 5 + max_objects * 3


def make_nav_encoder(game, *, max_objects: int = 1, pos_scale: float = _POS_SCALE):
    # Imported lazily so the encoder package is importable (and unit-testable)
    # on machines without the ViZDoom engine; only nav recall needs it at runtime.
    import vizdoom as vzd

    def enc(state):
        px = game.get_game_variable(vzd.GameVariable.POSITION_X) / pos_scale
        py = game.get_game_variable(vzd.GameVariable.POSITION_Y) / pos_scale
        ang = math.radians(game.get_game_variable(vzd.GameVariable.ANGLE))
        # HEALTH is the decisive variable in survival tasks (health_gathering);
        # without it, states at the same pose but different health collide.
        health = game.get_game_variable(vzd.GameVariable.HEALTH) / _HEALTH_SCALE
        feats = [px, py, math.sin(ang), math.cos(ang), health]

        objs = np.zeros((max_objects, 3), dtype=np.float32)
        labels = sorted(
            getattr(state, "labels", None) or [],
            key=lambda l: -(l.width * l.height),
        )[:max_objects]
        for i, lab in enumerate(labels):
            cx = (lab.x + lab.width / 2.0) / _SCREEN_W - 1.0
            cy = (lab.y + lab.height / 2.0) / _SCREEN_H - 1.0
            area = (lab.width * lab.height) / (_SCREEN_W * _SCREEN_H)
            objs[i] = (cx, cy, area)

        return np.concatenate([np.asarray(feats, dtype=np.float32), objs.reshape(-1)]).astype(
            np.float32
        )

    return enc, navigation_dim(max_objects)
