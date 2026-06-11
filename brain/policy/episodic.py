"""Episodic-control policy: act by reward-weighted recall, no trained network.

Given the current state vector, recall the k nearest past experiences and pick
the action whose neighbors had the highest similarity-weighted return. With
probability epsilon, explore a random action instead. "Learning" happens by
writing better experiences back into the store, not by gradient descent.
"""
from __future__ import annotations

import random
from collections import defaultdict


def choose_action(
    neighbors: list[tuple[str, float, dict]],
    n_actions: int,
    *,
    epsilon: float = 0.1,
    rng: random.Random | None = None,
) -> int:
    """Pick an action index from recalled neighbors.

    Each neighbor's metadata is expected to carry 'action_idx' and 'return'.
    Scores are used as similarity weights (shifted to be non-negative).
    """
    rng = rng or random
    if not neighbors or rng.random() < epsilon:
        return rng.randrange(n_actions)

    # shift scores so the least-similar neighbor has weight ~0
    min_score = min(s for _, s, _ in neighbors)
    weighted = defaultdict(float)
    seen = defaultdict(float)
    for _id, score, meta in neighbors:
        if "action_idx" not in meta:
            continue
        a = int(meta["action_idx"])
        w = (score - min_score) + 1e-6
        weighted[a] += w * float(meta.get("return", 0.0))
        seen[a] += w

    if not seen:
        return rng.randrange(n_actions)
    # average weighted return per action; argmax
    best = max(seen, key=lambda a: weighted[a] / seen[a])
    return best


def linear_epsilon(
    episode: int, *, eps_start: float = 1.0, eps_end: float = 0.05, decay_episodes: int = 150
) -> float:
    """Linearly anneal exploration from eps_start to eps_end over decay_episodes."""
    if episode >= decay_episodes:
        return eps_end
    frac = episode / max(1, decay_episodes)
    return eps_start + (eps_end - eps_start) * frac


def discounted_returns(rewards: list[float], gamma: float = 0.99) -> list[float]:
    """Backfill discounted returns G_t for a finished episode trajectory."""
    out = [0.0] * len(rewards)
    g = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        g = rewards[t] + gamma * g
        out[t] = g
    return out
