"""Synthetic integration test: does the dodge mechanism actually reduce damage?

ViZDoom isn't available off-device, so we exercise the *exact mechanism* train.py
uses for Track 3 — `ExperienceStore.search(filter=..., diversify=...)`,
`recall_uncertainty`, `choose_action_safe` with an evasive default,
`discounted_returns`, `dodge_shaped_reward` — against a toy dodge MDP. Two claims
are tested:

  1. **Learning** — the full dodge loop (filtered/diversified value recall + the
     health-penalty shaped return) learns to strafe away, taking far less damage
     than a random policy.
  2. **The uncertainty-gated safe fallback measurably reduces damage vs. the same
     policy without it** — the Track-3 success criterion. We show it in the
     cold-start regime the fallback is built for: with an unfamiliar (here, empty)
     store, recognizing the uncertainty and evading takes far less damage than
     blindly voting (which degenerates to a random guess).
"""
import random

import numpy as np

from brain.memory.experience_store import ExperienceStore
from brain.policy.episodic import choose_action, choose_action_safe, discounted_returns
from brain.policy.reward import dodge_shaped_reward, health_delta

# 4 actions mirroring take_cover's {noop, move_right, move_left, both}.
NOOP, MOVE_RIGHT, MOVE_LEFT, BOTH = 0, 1, 2, 3
N_ACTIONS = 4
LEFT, RIGHT = -0.6, 0.6


def _evade(threat_dx: float) -> int:
    """The safe default the policy falls back on: strafe *away* from the threat —
    the same rule train.make_dodge_fallback applies (dx>0 -> MOVE_LEFT)."""
    return MOVE_LEFT if threat_dx > 0 else MOVE_RIGHT


class DodgeWorld:
    """Each step a threat appears on the left or the right; the agent must strafe
    away to avoid it. Moving away -> no damage; anything else -> 1 HP lost. State
    is [threat_dx, threat_visible]; the threat is always visible here so dodging
    is isolated from detection. Fixed-length episodes so 'damage taken' is the
    clean metric (not confounded by survival length)."""

    def __init__(self, rng, steps=24):
        self.rng = rng
        self.steps = steps

    def reset(self):
        self.t = 0
        self._spawn()
        return self._state()

    def _spawn(self):
        self.dx = self.rng.choice((LEFT, RIGHT))

    def _state(self):
        return np.array([self.dx, 1.0], dtype=np.float32)

    def step(self, a):
        dodged = a == _evade(self.dx)
        health_loss = 0.0 if dodged else 1.0
        self.t += 1
        done = self.t >= self.steps
        self._spawn()
        return self._state(), health_loss, done


def _run(world, store, rng, *, epsilon, learn, use_fallback, k=12, penalty=0.1):
    """One episode of the real Track-3 loop. Returns total damage (HP lost)."""
    s = world.reset()
    traj = []
    damage = 0.0
    done = False
    while not done:
        dx = float(s[0])
        filt = {"threat_visible": 1.0}  # threat always visible in this world
        nb = store.search(s, k=k, filter=filt, diversify=True)
        if use_fallback:
            a = choose_action_safe(
                nb, N_ACTIONS, epsilon=epsilon, rng=rng,
                evade_action=_evade(dx), evade_threshold=0.6,
            )
        else:
            a = choose_action(nb, N_ACTIONS, epsilon=epsilon, rng=rng)
        s2, health_loss, done = world.step(a)
        # living_reward 1.0 each tic, penalized by HEALTH lost (health_delta is
        # exercised here on the raw before/after health to mirror train.py).
        shaped = dodge_shaped_reward(1.0, health_delta(100.0, 100.0 - health_loss), penalty)
        traj.append((s.copy(), a, shaped))
        damage += health_loss
        s = s2
    if learn:
        rets = discounted_returns([t[2] for t in traj], gamma=0.9)
        for (vec, a, _), g in zip(traj, rets):
            store.insert(vec, {"action_idx": float(a), "return": float(g), "threat_visible": 1.0})
    return damage


def _mean_damage(world, store, rng, *, use_fallback, episodes=40):
    return sum(
        _run(world, store, rng, epsilon=0.0, learn=False, use_fallback=use_fallback)
        for _ in range(episodes)
    ) / episodes


def test_dodge_loop_learns_to_avoid_damage():
    rng = random.Random(0)
    world = DodgeWorld(rng)
    store = ExperienceStore(dim=2, backend="numpy", capacity=5000)

    random_damage = _mean_damage(world, store, rng, use_fallback=False)  # empty -> random

    episodes = 200
    for ep in range(episodes):
        eps = max(0.05, 1.0 - ep / (episodes * 0.7))
        _run(world, store, rng, epsilon=eps, learn=True, use_fallback=True)

    learned_damage = _mean_damage(world, store, rng, use_fallback=True)

    # Random guessing dodges ~1/4 of the time -> ~18 HP / 24-step episode. The
    # learned policy must take clearly less than half that.
    assert learned_damage < random_damage * 0.5, (
        f"no learning: random={random_damage:.1f} learned={learned_damage:.1f}"
    )
    # And the learned greedy policy must strafe correctly on canonical states.
    for dx, away in ((RIGHT, MOVE_LEFT), (LEFT, MOVE_RIGHT)):
        nb = store.search(np.array([dx, 1.0], dtype=np.float32), k=12,
                          filter={"threat_visible": 1.0}, diversify=True)
        assert choose_action(nb, N_ACTIONS, epsilon=0.0) == away


def test_uncertainty_fallback_reduces_damage_vs_no_fallback():
    """The Track-3 ablation: same (unfamiliar) store, fallback on vs off.

    On an empty store every state is unfamiliar -> recall_uncertainty is maximal.
    The fallback turns that into a correct evade; without it the vote degenerates
    to a random guess. The gap is the measured value of the safety signal."""
    rng = random.Random(1)
    world = DodgeWorld(rng)
    empty = ExperienceStore(dim=2, backend="numpy", capacity=5000)

    no_fallback = _mean_damage(world, empty, rng, use_fallback=False)  # random guesses
    with_fallback = _mean_damage(world, empty, rng, use_fallback=True)  # evades

    # Evading should take near-zero damage; random ~3/4 of 24 steps.
    assert with_fallback < no_fallback * 0.25, (
        f"fallback did not help: no_fallback={no_fallback:.1f} with_fallback={with_fallback:.1f}"
    )
    assert with_fallback < 1.0  # essentially damage-free under uncertainty
