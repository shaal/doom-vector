"""Aim encoder features: enemy detection, dx-to-target, visibility, dim/shape.

Uses duck-typed fakes for ViZDoom's Label/GameState so it runs with no engine.
"""
import numpy as np

from brain.encoder.structured import (
    AIM_DIMS,
    aim_features,
    encode_structured,
    enemy_visible,
    structured_dim,
)


class FakeLabel:
    def __init__(self, x, y, w, h, name="Cacodemon", oid=1):
        self.x, self.y, self.width, self.height = x, y, w, h
        self.object_name, self.object_id = name, oid


class FakeState:
    def __init__(self, labels, game_variables=(0.0, 0.0)):
        self.labels = labels
        self.game_variables = list(game_variables)


def test_enemy_visible_excludes_self():
    only_self = FakeState([FakeLabel(80, 60, 10, 10, name="DoomPlayer")])
    assert enemy_visible(only_self) == 0.0
    with_enemy = FakeState([FakeLabel(80, 60, 10, 10, name="DoomPlayer"), FakeLabel(80, 60, 20, 20)])
    assert enemy_visible(with_enemy) == 1.0
    assert enemy_visible(FakeState([])) == 0.0


def test_aim_features_centered_enemy_has_zero_offset():
    # screen width 160 -> centre at x=80; an enemy whose centre is at 80 -> dx≈0
    state = FakeState([FakeLabel(70, 50, 20, 20)])  # centre x = 70+10 = 80
    dx, size, vis = aim_features(state, screen_w=160.0)
    assert abs(dx) < 1e-6
    assert vis == 1.0
    assert size > 0.0


def test_aim_features_sign_tracks_horizontal_side():
    left = aim_features(FakeState([FakeLabel(0, 50, 10, 10)]), screen_w=160.0)
    right = aim_features(FakeState([FakeLabel(150, 50, 10, 10)]), screen_w=160.0)
    assert left[0] < 0 < right[0]  # enemy on the left -> negative dx, right -> positive


def test_aim_features_picks_nearest_by_area():
    # bigger (closer) enemy on the right should win over a small one on the left
    state = FakeState([FakeLabel(0, 50, 4, 4, oid=1), FakeLabel(150, 50, 30, 30, oid=2)])
    dx, _, vis = aim_features(state, screen_w=160.0)
    assert vis == 1.0 and dx > 0


def test_aim_features_no_enemy_is_all_zero():
    assert aim_features(FakeState([]), screen_w=160.0) == (0.0, 0.0, 0.0)


def test_aim_features_tolerates_degenerate_screen_width():
    # A 0-width config must not ZeroDivisionError (guarded to 1.0).
    dx, size, vis = aim_features(FakeState([FakeLabel(0, 0, 1, 1)]), screen_w=0.0)
    assert vis == 1.0
    assert all(np.isfinite([dx, size]))


def test_aim_features_handles_label_without_object_name():
    # Older ViZDoom states may lack object_name -> treated as an enemy (fail-open).
    class Bare:
        x, y, width, height = 80, 60, 20, 20

    assert enemy_visible(FakeState([Bare()])) == 1.0


def test_structured_dim_adds_three_for_aim():
    assert structured_dim(2, 8, aim=True) == structured_dim(2, 8) + AIM_DIMS


def test_encode_structured_aim_shape_and_visibility_tail():
    state = FakeState([FakeLabel(70, 50, 20, 20)], game_variables=(50.0, 30.0))
    base = encode_structured(state, n_game_vars=2, max_objects=8)
    aimed = encode_structured(state, n_game_vars=2, max_objects=8, aim=True, screen_w=160.0)
    assert aimed.shape[0] == base.shape[0] + AIM_DIMS
    assert aimed.dtype == np.float32
    # the final dim is the enemy_visible flag (the recall filter key)
    assert aimed[-1] == 1.0
    # no-enemy state -> tail is all zeros
    empty = encode_structured(FakeState([], game_variables=(50.0, 30.0)),
                              n_game_vars=2, max_objects=8, aim=True, screen_w=160.0)
    assert list(empty[-AIM_DIMS:]) == [0.0, 0.0, 0.0]
