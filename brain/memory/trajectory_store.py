"""Trajectory store for path prediction (Phase 2, Option A).

Each past episode is a trajectory: a sequence of state vectors + the actions
taken. We index every step's state vector in RuVector (metadata links it back
to its trajectory id, step index, and discounted return-to-go), while the
action sequences live in a Python dict so we can *follow* a recalled path.

Recall returns, for states near the query, which (trajectory, step) they belong
to and the value (return-to-go) from that step — the planner then follows that
trajectory's subsequent actions. Bounded by `max_trajectories`; the lowest
total-return trajectory is evicted whole when full (curating toward good paths
is the learning signal).

Backends mirror ExperienceStore: native `ruvector_py`, else a NumPy fallback.
"""
from __future__ import annotations

import numpy as np


class TrajectoryStore:
    def __init__(
        self,
        dim: int,
        *,
        backend: str = "auto",
        storage_path: str = "./traj_store",
        max_trajectories: int = 400,
    ):
        self.dim = dim
        self.max_trajectories = max_trajectories
        self.backend = backend
        self._impl = None
        self.actions: dict[int, list[int]] = {}
        self.returns: dict[int, float] = {}  # episode total, used for eviction
        self._step_ids: dict[int, list[str]] = {}
        self._next_id = 0

        if backend in ("auto", "native"):
            try:
                import ruvector_py  # type: ignore

                self._impl = ruvector_py.RuVectorMemory(dim, storage_path)
                self.backend = "native"
            except Exception:
                if backend == "native":
                    raise
                self.backend = "numpy"
        if self.backend == "numpy":
            self._vecs: list[np.ndarray] = []
            self._md: list[dict] = []

    def add_trajectory(self, states, actions, step_values, total_return: float) -> int:
        tid = self._next_id
        self._next_id += 1
        self.actions[tid] = [int(a) for a in actions]
        self.returns[tid] = float(total_return)
        ids: list[str] = []
        for i, vec in enumerate(states):
            md = {"traj_id": float(tid), "step_idx": float(i), "step_value": float(step_values[i])}
            v = np.asarray(vec, dtype=np.float32)
            if self.backend == "native":
                ids.append(self._impl.insert(v.tolist(), None, md))
            else:
                self._vecs.append(v)
                self._md.append(md)
        self._step_ids[tid] = ids
        self._evict()
        return tid

    def recall(self, vector, k: int = 16) -> list[tuple[int, int, float, float]]:
        """Return (traj_id, step_idx, step_value, score) for the k nearest states."""
        v = np.asarray(vector, dtype=np.float32)
        out: list[tuple[int, int, float, float]] = []
        if self.backend == "native":
            # The bridge returns (id, score, vector, metadata); the vector is
            # None here (with_vectors defaults False) and unused by trajectories.
            for _id, score, _vec, md in self._impl.search(v.tolist(), k):
                out.append((int(md["traj_id"]), int(md["step_idx"]), md.get("step_value", 0.0), score))
            return out
        if not self._vecs:
            return []
        mat = np.stack(self._vecs)
        d = np.linalg.norm(mat - v, axis=1)
        for i in np.argsort(d)[:k]:
            md = self._md[i]
            out.append((int(md["traj_id"]), int(md["step_idx"]), md.get("step_value", 0.0), float(-d[i])))
        return out

    def _evict(self) -> None:
        while len(self.actions) > self.max_trajectories:
            tid = min(self.returns, key=lambda t: self.returns[t])
            if self.backend == "native":
                for vid in self._step_ids.get(tid, []):
                    try:
                        self._impl.delete(vid)
                    except Exception:
                        pass
            else:
                kept = [(v, m) for v, m in zip(self._vecs, self._md) if int(m["traj_id"]) != tid]
                self._vecs = [v for v, _ in kept]
                self._md = [m for _, m in kept]
            self._step_ids.pop(tid, None)
            self.actions.pop(tid, None)
            self.returns.pop(tid, None)

    def __len__(self) -> int:
        return len(self.actions)
