"""Reward shaping: DAMAGECOUNT delta (clamped) and the additive hit bonus."""
from brain.policy.reward import damage_delta, hit_shaped_reward


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
