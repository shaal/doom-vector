"""Synthetic integration test: does filtered + diversified value-recall actually
learn the aim behaviour?

ViZDoom isn't available off-device, so we exercise the *exact mechanism*
train.py uses — `ExperienceStore.search(filter=..., diversify=...)`,
`choose_action`, `discounted_returns`, enemy_visible metadata — against a toy
aim MDP. The agent has no turn action; the only skill to learn is *when to pull
the trigger*: shoot when an enemy is lined up, hold otherwise. If the recall
machinery works, greedy eval reward must rise clearly above the random baseline,
and the learned greedy policy must shoot aligned enemies and hold on empty
frames.
"""
import random

import numpy as np

from brain.memory.experience_store import ExperienceStore
from brain.policy.episodic import choose_action, discounted_returns

NOOP, ATTACK = 0, 1
ALIGNED = 0.0


class AimWorld:
    """Each step shows an enemy at a random horizontal offset (or no enemy).
    Reward: +1 for shooting an aligned enemy, -0.2 for a wasted shot (no enemy),
    0 otherwise. State = [dx, enemy_visible]."""

    OFFSETS = (-0.6, ALIGNED, 0.6)

    def __init__(self, rng, steps=16):
        self.rng = rng
        self.steps = steps

    def reset(self):
        self.t = 0
        self._spawn()
        return self._state()

    def _spawn(self):
        self.vis = 1.0 if self.rng.random() < 0.6 else 0.0
        self.dx = self.rng.choice(self.OFFSETS) if self.vis else 0.0

    def _state(self):
        return np.array([self.dx, self.vis], dtype=np.float32)

    def step(self, a):
        if a == ATTACK and self.vis and abs(self.dx) <= 0.15:
            r = 1.0
        elif a == ATTACK and not self.vis:
            r = -0.2
        else:
            r = 0.0
        self.t += 1
        done = self.t >= self.steps
        self._spawn()
        return self._state(), r, done


def _episode(world, store, rng, *, epsilon, learn, k=12):
    s = world.reset()
    traj = []
    total = 0.0
    done = False
    while not done:
        vis = float(s[1])
        filt = {"enemy_visible": 1.0} if vis else None
        nb = store.search(s, k=k, filter=filt, diversify=True)
        a = choose_action(nb, 2, epsilon=epsilon, rng=rng)
        s2, r, done = world.step(a)
        traj.append((s.copy(), a, r, vis))
        total += r
        s = s2
    if learn:
        rets = discounted_returns([t[2] for t in traj], gamma=0.9)
        for (vec, a, _, vis), g in zip(traj, rets):
            store.insert(vec, {"action_idx": float(a), "return": float(g), "enemy_visible": vis})
    return total


def _greedy_mean(world, store, rng, episodes=40):
    return sum(_episode(world, store, rng, epsilon=0.0, learn=False) for _ in range(episodes)) / episodes


def test_filtered_diversified_recall_learns_to_aim():
    rng = random.Random(0)
    world = AimWorld(rng)
    store = ExperienceStore(dim=2, backend="numpy", capacity=5000)

    # Baseline: empty store -> choose_action falls back to random.
    baseline = _greedy_mean(world, store, rng)

    # Train with annealed exploration.
    episodes = 250
    for ep in range(episodes):
        eps = max(0.05, 1.0 - ep / (episodes * 0.7))
        _episode(world, store, rng, epsilon=eps, learn=True)

    learned = _greedy_mean(world, store, rng)

    # The mechanism must produce a clear, not marginal, improvement.
    assert learned > baseline + 0.8, f"no learning: baseline={baseline:.2f} learned={learned:.2f}"

    # And the learned greedy policy must be sensible on canonical states:
    # aligned enemy -> shoot; empty frame -> hold.
    aligned = np.array([ALIGNED, 1.0], dtype=np.float32)
    nb = store.search(aligned, k=12, filter={"enemy_visible": 1.0}, diversify=True)
    assert choose_action(nb, 2, epsilon=0.0) == ATTACK

    empty = np.array([0.0, 0.0], dtype=np.float32)
    nb = store.search(empty, k=12, diversify=True)
    assert choose_action(nb, 2, epsilon=0.0) == NOOP
