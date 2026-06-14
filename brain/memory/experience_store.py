"""Experience store: RuVector-backed k-NN memory with a NumPy fallback.

The agent's "learning" lives here. Each experience is a state vector tagged
with the action taken and the discounted return that followed. Recall = k-NN
search; the policy then picks the action whose neighbors had the best returns.

Backends:
  - "native": the PyO3 binding `ruvector_py` -> ruvector-core (used on the Pi).
  - "numpy":  brute-force fallback so the loop runs before the binding is built.
Backend "auto" prefers native and silently falls back to numpy.

Bounded capacity: when `capacity` is set, the lowest-return experience is
evicted once the store exceeds it. For the native backend this calls
ruvector-core's `delete`; note HNSW deletes may be tombstoned (recall stays
correct, but RAM is fully reclaimed only on a compaction/rebuild — a Phase 2
concern for very long Pi runs).
"""
from __future__ import annotations

import numpy as np


def _nonzero(vector) -> np.ndarray:
    """Return a float32 copy guaranteed to have non-zero norm.

    A zero-norm vector makes cosine distance 0/0 = NaN, which trips an
    assertion deep in hnsw_rs (seen only with the scalar distance fallback used
    on 32-bit ARM, where simsimd is disabled). Degenerate all-zero states carry
    no recall signal anyway, so we nudge one component.
    """
    v = np.asarray(vector, dtype=np.float32)
    if not v.any():
        v = v.copy()
        v[0] = 1.0
    return v


def _meta_matches(meta: dict, filter: dict) -> bool:
    """All filter keys present in `meta` with equal value — mirrors native
    ruvector-core's exact `serde_json::Value` equality post-filter. Float values
    come from the same source on both sides (e.g. 1.0), so `==` is faithful."""
    return all(meta.get(key) == value for key, value in filter.items())


class ExperienceStore:
    def __init__(
        self,
        dim: int,
        *,
        backend: str = "auto",
        storage_path: str = "./ruvector_store",
        capacity: int | None = None,
    ):
        self.dim = dim
        self.backend = backend
        self.capacity = capacity
        self._impl = None
        # (id, value) per live entry, used for value-based eviction (native).
        self._index: list[tuple[str, float]] = []

        if backend in ("auto", "native"):
            try:
                import ruvector_py  # type: ignore

                self._impl = ruvector_py.RuVectorMemory(dim, storage_path)
                self.backend = "native"
            except Exception as exc:  # not built / import failure
                if backend == "native":
                    raise
                self.backend = "numpy"
                self._note = f"native unavailable ({exc!s}); using numpy fallback"

        if self.backend == "numpy":
            self._vecs: list[np.ndarray] = []
            self._meta: list[dict] = []
            self._vals: list[float] = []

    # --- write ---------------------------------------------------------------
    def insert(self, vector, metadata: dict | None = None) -> None:
        v = _nonzero(vector)
        md = metadata or {}
        val = float(md.get("return", 0.0))
        if self.backend == "native":
            vid = self._impl.insert(v.tolist(), None, md)
            self._index.append((vid, val))
            self._evict_native()
        else:
            self._vecs.append(v)
            self._meta.append(md)
            self._vals.append(val)
            self._evict_numpy()

    def _evict_native(self) -> None:
        if self.capacity and len(self._index) > self.capacity:
            i = min(range(len(self._index)), key=lambda j: self._index[j][1])
            vid, _ = self._index.pop(i)
            try:
                self._impl.delete(vid)
            except Exception:
                pass

    def _evict_numpy(self) -> None:
        if self.capacity and len(self._vecs) > self.capacity:
            i = int(np.argmin(self._vals))
            del self._vecs[i]
            del self._meta[i]
            del self._vals[i]

    # --- read ----------------------------------------------------------------
    def search(
        self,
        vector,
        k: int = 8,
        *,
        filter: dict | None = None,
        over_fetch: int = 4,
        diversify: bool = False,
        mmr_lambda: float = 0.5,
    ) -> list[tuple[str, float, dict]]:
        """Return up to k (id, score, metadata) tuples, nearest first.

        Score sign is normalized to "bigger score == more similar" on *both*
        backends: ruvector-core's native ``search`` returns raw L2 *distance*
        (smaller == closer), so we negate it here; the numpy fallback already
        returns negative L2. Without this, `choose_action` / `recall_uncertainty`
        — which weight by ``score - min_score`` — would weight the *farthest*
        neighbour most on the native (Pi) backend. After normalization the two
        backends return identical scores for identical data, so the value vote
        and the uncertainty gate are genuinely backend-agnostic.

        Track 1 (Aim) adds three composable knobs, all encapsulated here so the
        public contract stays a 3-tuple (`choose_action` is unchanged):

        - ``filter``: exact-match metadata filter (e.g. ``{"enemy_visible": 1.0}``)
          so the aim vote is advised only by relevant frames. ruvector-core
          applies it as a *post-filter* over the k-NN hits, so it returns ≤ k.
        - ``over_fetch``: when filtering or diversifying, search a larger
          ``k_raw = k * over_fetch`` so the post-filter / MMR has a pool to work
          with instead of starving the vote.
        - ``diversify``: MMR-rerank the over-fetched pool down to a *diverse* k
          (see ``brain.policy.mmr``), so a cluster of near-duplicate neighbours
          can't dominate the vote.
        """
        v = _nonzero(vector)
        need_pool = bool(filter) or diversify
        k_raw = max(k, k * over_fetch) if need_pool else k

        if self.backend == "native":
            # The native bridge applies `filter` server-side and returns the
            # candidate vector only when we ask (with_vectors), keeping the
            # no-MMR path cheap on the Pi.
            raw = self._impl.search(v.tolist(), k_raw, filter, with_vectors=diversify)
            # Negate native L2 distance -> "bigger == more similar" (see docstring).
            cands = [(r[0], -r[1], r[2], r[3]) for r in raw]
        else:
            cands = self._search_numpy(v, k_raw, filter, with_vectors=diversify)

        if diversify and len(cands) > k:
            from brain.policy.mmr import mmr_rerank

            cands = mmr_rerank(v, cands, k, lam=mmr_lambda)
        else:
            cands = cands[:k]

        # Strip the (internal-only) candidate vector back to the public 3-tuple.
        return [(c[0], c[1], c[3]) for c in cands]

    def _search_numpy(self, v, k_raw, filter, with_vectors):
        """Numpy fallback that mirrors native ordering: k-NN to k_raw first, then
        post-filter (so it returns ≤ k_raw, possibly fewer — exactly like
        ruvector-core's `results.retain`). Returns 4-tuples (id, score, vector,
        metadata); `vector` is None unless `with_vectors`."""
        if not self._vecs:
            return []
        mat = np.stack(self._vecs)
        d = np.linalg.norm(mat - v, axis=1)
        idx = np.argsort(d)[:k_raw]
        out = []
        for i in idx:
            meta = self._meta[i]
            if filter and not _meta_matches(meta, filter):
                continue
            vec = self._vecs[i].tolist() if with_vectors else None
            out.append((str(i), float(-d[i]), vec, meta))
        return out

    def __len__(self) -> int:
        return len(self._index) if self.backend == "native" else len(self._vecs)
