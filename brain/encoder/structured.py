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
_HEALTH_SCALE = 100.0  # Doom full health -> ~1.0 (matches navigation encoder)

# A label with this name is the agent itself, not a target. ViZDoom tags the
# player's own body in the labels buffer; it must be excluded from "enemies".
_SELF_NAME = "DoomPlayer"

# Substrings that mark a label as an incoming *projectile* (the thing to dodge),
# as opposed to a stationary monster. Doom projectiles are named e.g.
# "DoomImpBall", "CacodemonBall", "Rocket", "PlasmaBall". In take_cover the only
# projectile is "DoomImpBall"; dodging keys on these, not the wall of imps that
# fire them (a monster's screen offset is the wrong thing to strafe away from).
_PROJECTILE_KEYS = ("ball", "rocket", "plasma", "missile")

# Number of extra dims appended in aim mode: (enemy_dx, enemy_size, enemy_visible).
AIM_DIMS = 3

# Number of extra dims appended in threat (dodge) mode:
# (threat_dx, threat_size, threat_visible, dhealth). Track 3.
THREAT_DIMS = 4


def structured_dim(
    n_game_vars: int, max_objects: int = 8, *, aim: bool = False, threat: bool = False
) -> int:
    return (
        n_game_vars
        + max_objects * 4
        + (AIM_DIMS if aim else 0)
        + (THREAT_DIMS if threat else 0)
    )


def _enemies(labels, self_name: str = _SELF_NAME):
    """Visible enemy labels (everything except the agent), nearest-first by
    screen area. ViZDoom labels expose `object_name`; older states without it
    are treated as enemies (fail-open — better to aim than to ignore a threat)."""
    enemies = [l for l in labels if getattr(l, "object_name", "") != self_name]
    return sorted(enemies, key=lambda l: -(l.width * l.height))


def enemy_visible(state, self_name: str = _SELF_NAME) -> float:
    """1.0 if any non-self target is on screen, else 0.0. This is both an encoder
    dimension and the recall filter key (`{"enemy_visible": 1.0}`), so it lives
    here as the single source of truth shared by the encoder and the train loop."""
    labels = getattr(state, "labels", None) or []
    return 1.0 if _enemies(labels, self_name) else 0.0


def _projectiles(labels):
    """Visible incoming projectiles (labels whose name marks them a missile/ball),
    nearest-first by screen area (bigger ⇒ closer ⇒ more urgent to dodge). A label
    with no `object_name` can't be classified as a projectile, so it's excluded —
    the dodge fallback then simply doesn't fire (fail-closed) rather than strafe
    away from a wall monster."""
    proj = [
        l for l in labels
        if any(key in getattr(l, "object_name", "").lower() for key in _PROJECTILE_KEYS)
    ]
    return sorted(proj, key=lambda l: -(l.width * l.height))


def threat_visible(state) -> float:
    """1.0 if any incoming projectile is on screen, else 0.0 — the dodge recall
    filter key (`{"threat_visible": 1.0}`) and threat encoder flag. Single source
    of truth shared by the encoder and the train loop (Track 3)."""
    labels = getattr(state, "labels", None) or []
    return 1.0 if _projectiles(labels) else 0.0


def aim_features(state, screen_w: float = _POS_SCALE, self_name: str = _SELF_NAME):
    """Explicit aim signals for the nearest enemy: recall can't learn "pull the
    trigger when lined up" unless alignment is a dimension.

    Returns (enemy_dx, enemy_size, enemy_visible):
      - enemy_dx:  horizontal offset of the nearest enemy from screen centre,
                   normalized to ~[-1, 1] (≈0 ⇒ on target).
      - enemy_size: screen-area fraction of that enemy — a monotonic closeness
                    proxy (bigger ⇒ nearer) standing in for true distance.
      - enemy_visible: 1.0 if any enemy is on screen, else 0.0.
    """
    labels = getattr(state, "labels", None) or []
    enemies = _enemies(labels, self_name)
    if not enemies:
        return (0.0, 0.0, 0.0)
    lab = enemies[0]
    w = screen_w or 1.0  # guard a degenerate 0-width config (ViZDoom never 0)
    cx = lab.x + lab.width / 2.0
    half = w / 2.0
    enemy_dx = (cx - half) / half
    enemy_size = (lab.width * lab.height) / (w * w)
    return (float(enemy_dx), float(enemy_size), 1.0)


def threat_features(
    state,
    dhealth: float = 0.0,
    *,
    screen_w: float = _POS_SCALE,
    health_scale: float = _HEALTH_SCALE,
):
    """Explicit threat signals for dodging (Track 3): where is the nearest
    incoming hazard, and did I just get hit.

    Returns (threat_dx, threat_size, threat_visible, dhealth_norm):
      - threat_dx / threat_size / threat_visible: the nearest *projectile's*
        horizontal offset, screen-area (closeness proxy) and presence flag. Unlike
        aim (which centres on any enemy), dodge keys on the incoming projectile —
        a wall monster's offset is the wrong thing to strafe away from.
      - dhealth_norm: damage taken since the previous step (>=0), normalized.
        Prior phases proved HEALTH is decisive for survival; the *change* in it is
        the dodge signal — "that approach just got me hit." The caller threads the
        previous health in (the encoder is otherwise stateless); negatives (pickups
        / episode reset to full health) are clamped to 0 — not damage.
    """
    labels = getattr(state, "labels", None) or []
    projectiles = _projectiles(labels)
    dh = max(0.0, float(dhealth)) / health_scale
    if not projectiles:
        return (0.0, 0.0, 0.0, dh)
    lab = projectiles[0]
    w = screen_w or 1.0  # guard a degenerate 0-width config (ViZDoom never 0)
    cx = lab.x + lab.width / 2.0
    half = w / 2.0
    return (float((cx - half) / half), float((lab.width * lab.height) / (w * w)), 1.0, dh)


def encode_structured(
    state,
    n_game_vars: int,
    max_objects: int = 8,
    *,
    aim: bool = False,
    threat: bool = False,
    dhealth: float = 0.0,
    screen_w: float = _POS_SCALE,
) -> np.ndarray:
    """Encode a ViZDoom `GameState` into a fixed-length float32 vector.

    With `aim=True`, three explicit aim dims (dx-to-target, target size, target
    visible) are appended — the alignment signal the aim policy votes on, and the
    `enemy_visible` flag used as the recall filter key (Track 1).

    With `threat=True`, four dodge dims (threat dx/size/visible + damage-taken-
    this-step) are appended — the threat-awareness signal the dodge policy votes
    on, and the `threat_visible` flag used as its recall filter key (Track 3).
    `dhealth` is the previous-minus-current HEALTH the caller threads in."""
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

    parts = [gv, obj_feats.reshape(-1)]
    if aim:
        parts.append(np.asarray(aim_features(state, screen_w), dtype=np.float32))
    if threat:
        parts.append(
            np.asarray(threat_features(state, dhealth, screen_w=screen_w), dtype=np.float32)
        )
    return np.concatenate(parts).astype(np.float32)
