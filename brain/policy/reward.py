"""Reward shaping helpers — small, dense bonuses layered on the sparse scenario
reward so episodic recall has something to learn from many tics before a kill.

Kept free of any ViZDoom import so it's unit-testable off-device and reusable:
Track 1 (Aim) uses the hit bonus; Track 3 (Dodge) will add a symmetric health
penalty. The shaping is deliberately *additive and small* — evaluation always
reports the unshaped scenario score (`game.get_total_reward()`), so a too-eager
shaping that games the bonus shows up as flat eval, not inflated success.
"""
from __future__ import annotations


def damage_delta(prev: float, now: float) -> float:
    """Damage dealt *this step*, from ViZDoom's cumulative DAMAGECOUNT.

    Clamped to ≥0: DAMAGECOUNT is monotonic within an episode, but new_episode
    resets it to 0, so the first step of a new episode (prev carried over) could
    otherwise read negative. Returning 0 there is correct — no hit happened."""
    return max(0.0, now - prev)


def hit_shaped_reward(reward: float, dmg_delta: float, bonus: float) -> float:
    """Add a small per-step bonus for damage just dealt.

    `bonus` is per unit of DAMAGECOUNT; keep it small relative to the scenario
    reward so it shapes aim without encouraging spray-and-pray (the eval on the
    unshaped score is the guard against that)."""
    return reward + bonus * dmg_delta


def health_delta(prev: float, now: float) -> float:
    """Damage *taken* this step (>=0), from ViZDoom's HEALTH variable — the
    symmetric counterpart to `damage_delta`.

    Clamped to >=0: HEALTH drops on damage but rises on a pickup, and new_episode
    resets it to full, so `prev - now` can go negative. Returning 0 there is
    correct — gaining or resetting health is not 'damage taken'."""
    return max(0.0, prev - now)


def dodge_shaped_reward(reward: float, health_loss: float, penalty: float) -> float:
    """Subtract a small per-step penalty for damage taken (Track 3, Dodge).

    Dodging is continuous and needs continuous feedback — death alone is too
    sparse to shape evasion. `penalty` is per unit of HEALTH lost; keep it small
    relative to the scenario reward so it shapes dodging without teaching the
    agent to freeze in a corner (the eval on the unshaped survival score is the
    guard against that)."""
    return reward - penalty * health_loss
