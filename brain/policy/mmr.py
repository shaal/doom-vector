"""Maximal Marginal Relevance (MMR) re-ranking over recalled candidates.

The value vote (`episodic.choose_action`) is only as good as the *spread* of
neighbours it sees. A raw top-k recall on `defend_the_center` is often 16
near-identical "enemy dead-ahead" frames that all voted the same way — the one
neighbour that tried a different action gets drowned out. MMR fixes that: it
greedily picks results that are relevant to the query *and* dissimilar to what's
already chosen, so the vote sees a diverse set instead of an echo chamber.

This mirrors `ruvector-core`'s `advanced_features::mmr::MMRSearch.rerank` but is
pure post-processing over candidates we already fetched (§1.5 of the skills
plan), so the first cut lives in Python — no bridge round-trip. It composes over
the over-fetched pool the Track-1 filter already builds.

Candidates are `(id, score, vector, metadata)` tuples; `vector` is the stored
embedding (the native bridge returns it only when `with_vectors=True`). If any
candidate is missing its vector, diversity can't be measured and we fall back to
the relevance order (`candidates[:k]`).
"""
from __future__ import annotations

import numpy as np


def _unit_rows(mat: np.ndarray) -> np.ndarray:
    """Row-normalize so dot products are cosine similarities. Zero rows stay zero
    (their similarity to everything is 0 — they neither attract nor repel)."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return mat / norms


def mmr_rerank(query, candidates, k, *, lam: float = 0.5):
    """Re-rank `candidates` to a diverse top-`k` by Maximal Marginal Relevance.

    Args:
        query: the query vector (the current encoded state).
        candidates: list of (id, score, vector, metadata). Order is assumed to be
            by relevance already (nearest-first) so the fallback is sensible.
        k: number of results to keep.
        lam: relevance/diversity trade-off in [0, 1]. lam=1 is pure relevance
            (== top-k), lam=0 is pure diversity. 0.5 balances both.

    Returns:
        A list of the selected candidates (same tuple shape), length min(k, len).
    """
    if k <= 0 or not candidates:
        return []
    if len(candidates) <= k:
        return list(candidates)

    vecs = [c[2] for c in candidates]
    if any(v is None for v in vecs):
        # No vectors -> can't measure diversity; keep the relevance order.
        return list(candidates[:k])

    q = np.asarray(query, dtype=np.float32).reshape(-1)
    qn = q / (np.linalg.norm(q) or 1.0)
    cand = _unit_rows(np.asarray(vecs, dtype=np.float32))

    relevance = cand @ qn  # cosine(query, candidate), one per candidate
    selected: list[int] = []
    remaining = list(range(len(candidates)))

    while remaining and len(selected) < k:
        if not selected:
            # Seed with the most relevant; MMR's first pick has no diversity term.
            best = max(remaining, key=lambda i: relevance[i])
        else:
            sel = cand[selected]  # (s, d)
            # max similarity of each remaining candidate to anything chosen
            sims = cand[remaining] @ sel.T  # (r, s)
            max_sim = sims.max(axis=1)  # (r,)
            scores = lam * relevance[remaining] - (1.0 - lam) * max_sim
            best = remaining[int(np.argmax(scores))]
        selected.append(best)
        remaining.remove(best)

    return [candidates[i] for i in selected]
