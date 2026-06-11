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
    def search(self, vector, k: int = 8) -> list[tuple[str, float, dict]]:
        """Return up to k (id, score, metadata) tuples, nearest first.

        Native scores come from ruvector-core's distance metric. The numpy
        fallback returns negative L2 distance (higher = closer) so both
        backends agree on "bigger score == more similar".
        """
        v = _nonzero(vector)
        if self.backend == "native":
            return [(r[0], r[1], r[2]) for r in self._impl.search(v.tolist(), k)]

        if not self._vecs:
            return []
        mat = np.stack(self._vecs)
        d = np.linalg.norm(mat - v, axis=1)
        idx = np.argsort(d)[:k]
        return [(str(i), float(-d[i]), self._meta[i]) for i in idx]

    def __len__(self) -> int:
        return len(self._index) if self.backend == "native" else len(self._vecs)
