"""Threat (dodge) encoder features: nearest-threat geometry, damage-taken dim,
and dim/shape math. Uses duck-typed fakes for ViZDoom's Label/GameState so it
runs with no engine (Track 3)."""
import numpy as np

from brain.encoder.structured import (
    AIM_DIMS,
    THREAT_DIMS,
    encode_structured,
    structured_dim,
    threat_features,
    threat_visible,
)

# Real take_cover label names: the incoming projectile vs. the wall monster.
BALL, IMP = "DoomImpBall", "DoomImp"


class FakeLabel:
    def __init__(self, x, y, w, h, name=BALL, oid=1):
        self.x, self.y, self.width, self.height = x, y, w, h
        self.object_name, self.object_id = name, oid


class FakeState:
    def __init__(self, labels, game_variables=(100.0,)):
        self.labels = labels
        self.game_variables = list(game_variables)


def test_threat_dims_constant_is_four():
    # threat block = (threat_dx, threat_size, threat_visible, dhealth)
    assert THREAT_DIMS == 4


def test_threat_features_geometry_matches_nearest_projectile():
    # a fireball whose centre is at screen-centre -> dx≈0, visible, and zero
    # damage taken when no health was lost.
    state = FakeState([FakeLabel(70, 50, 20, 20)])  # centre x = 80 on a 160px screen
    dx, size, vis, dh = threat_features(state, dhealth=0.0, screen_w=160.0)
    assert abs(dx) < 1e-6
    assert size > 0.0
    assert vis == 1.0
    assert dh == 0.0


def test_threat_features_dx_sign_tracks_projectile_side():
    left = threat_features(FakeState([FakeLabel(0, 50, 10, 10)]), screen_w=160.0)
    right = threat_features(FakeState([FakeLabel(150, 50, 10, 10)]), screen_w=160.0)
    assert left[0] < 0 < right[0]  # ball on the left -> negative dx (flee right)


def test_threat_targets_projectile_not_monster():
    # A big wall monster (DoomImp) on the left and a small incoming ball on the
    # right: the threat must be the *ball*, even though the monster is bigger.
    state = FakeState([FakeLabel(0, 50, 40, 40, name=IMP), FakeLabel(150, 50, 8, 8, name=BALL)])
    dx, _, vis, _ = threat_features(state, screen_w=160.0)
    assert vis == 1.0 and dx > 0  # ball is on the right
    assert threat_visible(state) == 1.0


def test_no_projectile_means_no_threat_even_with_monsters():
    # Only wall monsters on screen -> nothing to dodge yet.
    monsters_only = FakeState([FakeLabel(80, 60, 40, 40, name=IMP)])
    assert threat_visible(monsters_only) == 0.0
    dx, size, vis, _ = threat_features(monsters_only, screen_w=160.0)
    assert (dx, size, vis) == (0.0, 0.0, 0.0)


def test_threat_features_no_labels_is_zero_geometry():
    dx, size, vis, dh = threat_features(FakeState([]), dhealth=0.0, screen_w=160.0)
    assert (dx, size, vis) == (0.0, 0.0, 0.0)


def test_threat_features_dhealth_normalized_and_clamped():
    s = FakeState([FakeLabel(80, 60, 10, 10)])
    # 25 HP lost over a 100-scale -> 0.25
    assert threat_features(s, dhealth=25.0, screen_w=160.0)[3] == 0.25
    # negative dhealth (a pickup / episode reset to full) is not damage -> 0
    assert threat_features(s, dhealth=-40.0, screen_w=160.0)[3] == 0.0


def test_structured_dim_adds_four_for_threat():
    assert structured_dim(1, 8, threat=True) == structured_dim(1, 8) + THREAT_DIMS


def test_structured_dim_aim_and_threat_compose():
    base = structured_dim(1, 8)
    assert structured_dim(1, 8, aim=True, threat=True) == base + AIM_DIMS + THREAT_DIMS


def test_encode_structured_threat_shape_and_tail():
    state = FakeState([FakeLabel(70, 50, 20, 20)], game_variables=(100.0,))
    base = encode_structured(state, n_game_vars=1, max_objects=8)
    threat = encode_structured(
        state, n_game_vars=1, max_objects=8, threat=True, dhealth=10.0, screen_w=160.0
    )
    assert threat.shape[0] == base.shape[0] + THREAT_DIMS
    assert threat.dtype == np.float32
    # the threat tail is (dx, size, visible, dhealth_norm); last dim is damage taken
    assert threat[-1] == 0.1  # 10 HP / 100 scale
    assert threat[-2] == 1.0  # threat visible flag

    empty = encode_structured(
        FakeState([], game_variables=(100.0,)),
        n_game_vars=1, max_objects=8, threat=True, dhealth=0.0, screen_w=160.0,
    )
    assert list(empty[-THREAT_DIMS:]) == [0.0, 0.0, 0.0, 0.0]
