"""Episodic-control policy: act by reward-weighted recall, no trained network.

Given the current state vector, recall the k nearest past experiences and pick
the action whose neighbors had the highest similarity-weighted return. With
probability epsilon, explore a random action instead. "Learning" happens by
writing better experiences back into the store, not by gradient descent.
"""
from __future__ import annotations

import math
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


def recall_uncertainty(neighbors: list[tuple[str, float, dict]], n_actions: int) -> float:
    """How unreliable is the recalled advice, in [0, 1]? (Track 3 safety signal.)

    1.0 = maximally uncertain (the agent is in unfamiliar/conflicted territory and
    should fall back to a safe default); 0.0 = a confident consensus.

    Two failure modes collapse to "uncertain":
      - **Empty recall** — nothing similar was found (or the filter pruned the pool
        to nothing) -> 1.0. The store has never seen anything like this state.
      - **Disagreement** — the recalled neighbours voted across many different
        actions. We take the normalized Shannon entropy of the similarity-weighted
        action distribution: one action dominating -> ~0 (agreement); the vote
        spread evenly across actions -> ~1 (conflict).

    Entropy of the *action* distribution (not a raw distance threshold) is used on
    purpose: it depends only on the relative neighbour weights. Combined with the
    store normalizing native/numpy scores to the same sign convention
    (`ExperienceStore.search`), the same neighbour structure yields the same
    uncertainty on both backends — genuinely backend-agnostic."""
    if n_actions <= 1:
        return 0.0
    if not neighbors:
        return 1.0

    min_score = min(s for _, s, _ in neighbors)
    weighted: dict[int, float] = defaultdict(float)
    for _id, score, meta in neighbors:
        if "action_idx" not in meta:
            continue
        a = int(meta["action_idx"])
        weighted[a] += (score - min_score) + 1e-6  # same shift as choose_action

    total = sum(weighted.values())
    if total <= 0.0:
        return 1.0  # neighbours carried no usable action metadata

    entropy = 0.0
    for w in weighted.values():
        p = w / total
        if p > 0.0:
            entropy -= p * math.log(p)
    # Normalize to [0, 1] by the max entropy; clamp guards corrupt metadata where
    # more distinct action_idx values appear than n_actions (entropy > log n).
    return min(1.0, entropy / math.log(n_actions))


def choose_action_safe(
    neighbors: list[tuple[str, float, dict]],
    n_actions: int,
    *,
    epsilon: float = 0.1,
    rng: random.Random | None = None,
    evade_action: int | None = None,
    evade_threshold: float = 0.6,
) -> int:
    """Uncertainty-gated action choice (Track 3, Dodge).

    When the recall is reliable, defer to the ordinary value vote
    (`choose_action`). When `recall_uncertainty` meets `evade_threshold` *and* the
    caller supplied an `evade_action` (a precomputed safe default — e.g. strafe
    away from the nearest threat), take that instead: under uncertainty the right
    default in a dodge task is to evade, not to trust a weak vote or freeze.

    Returns the value vote unchanged when `evade_action is None` (no safe default
    available, e.g. no threat on screen), so this is a strict superset of
    `choose_action` and the ablation "with vs. without the fallback" is just
    `evade_action` present vs. forced None."""
    if evade_action is not None and recall_uncertainty(neighbors, n_actions) >= evade_threshold:
        return evade_action
    return choose_action(neighbors, n_actions, epsilon=epsilon, rng=rng)


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
