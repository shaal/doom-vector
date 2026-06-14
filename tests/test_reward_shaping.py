"""Reward shaping: DAMAGECOUNT delta + additive hit bonus (Track 1) and the
symmetric HEALTH delta + subtractive dodge penalty (Track 3)."""
from brain.policy.reward import (
    damage_delta,
    dodge_shaped_reward,
    health_delta,
    hit_shaped_reward,
)


def test_damage_delta_positive():
    assert damage_delta(10.0, 25.0) == 15.0


def test_damage_delta_clamps_episode_reset():
    # new_episode resets cumulative DAMAGECOUNT to 0; carried-over prev must not
    # produce a spurious negative "hit".
    assert damage_delta(40.0, 0.0) == 0.0


def test_damage_delta_no_change_is_zero():
    assert damage_delta(7.0, 7.0) == 0.0


def test_hit_shaped_reward_adds_bonus():
    assert hit_shaped_reward(1.0, 20.0, 0.01) == 1.2


def test_hit_shaped_reward_zero_delta_is_noop():
    assert hit_shaped_reward(-0.5, 0.0, 0.01) == -0.5


def test_health_delta_counts_damage_taken():
    assert health_delta(100.0, 75.0) == 25.0


def test_health_delta_clamps_pickup_and_reset():
    # a health pickup raises HEALTH; new_episode resets it to full. Neither is
    # damage taken -> 0, mirroring damage_delta's episode-reset clamp.
    assert health_delta(40.0, 60.0) == 0.0
    assert health_delta(20.0, 100.0) == 0.0


def test_health_delta_no_change_is_zero():
    assert health_delta(50.0, 50.0) == 0.0


def test_dodge_shaped_reward_subtracts_penalty():
    # living_reward 1.0, lost 20 HP, penalty 0.05/HP -> 1.0 - 1.0 = 0.0
    assert dodge_shaped_reward(1.0, 20.0, 0.05) == 0.0


def test_dodge_shaped_reward_zero_loss_is_noop():
    assert dodge_shaped_reward(1.0, 0.0, 0.05) == 1.0
