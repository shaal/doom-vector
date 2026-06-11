"""Retrieve-and-follow path planner (Phase 2, Option A).

To act: recall the highest-value trajectory passing near the current state and
*commit to following its action sequence* for up to `replan_every` steps, then
re-query. This is the literal "predict a path, play according to it": the plan
is a recalled past trajectory; we replan as the world drifts. Exploration
(epsilon) breaks the current plan and takes a random action so new paths are
discovered.

Contrast with the per-step episodic policy (`episodic.choose_action`), which
re-decides every single step; here we follow a multi-step path.
"""
from __future__ import annotations

import random


class TrajectoryPlanner:
    def __init__(
        self,
        store,
        n_actions: int,
        *,
        k: int = 16,
        replan_every: int = 8,
        epsilon: float = 0.1,
        min_score: float | None = None,
        rng: random.Random | None = None,
    ):
        self.store = store
        self.n = n_actions
        self.k = k
        self.replan_every = replan_every
        self.epsilon = epsilon
        self.min_score = min_score
        self.rng = rng or random
        self.reset()

    def reset(self) -> None:
        self._tid: int | None = None
        self._cursor = 0
        self._since_replan = 0

    def act(self, vector) -> int:
        if self.rng.random() < self.epsilon:
            self._tid = None  # abandon the plan; explore
            return self.rng.randrange(self.n)

        plan = self.store.actions.get(self._tid) if self._tid is not None else None
        if plan is None or self._cursor >= len(plan) or self._since_replan >= self.replan_every:
            self._replan(vector)
            plan = self.store.actions.get(self._tid) if self._tid is not None else None

        if plan is None or self._cursor >= len(plan):
            return self.rng.randrange(self.n)

        a = plan[self._cursor]
        self._cursor += 1
        self._since_replan += 1
        return a

    def _replan(self, vector) -> None:
        best = None  # (value, tid, step)
        for tid, step, value, score in self.store.recall(vector, self.k):
            seq = self.store.actions.get(tid)
            if seq is None or step + 1 >= len(seq):
                continue  # no next action to follow
            if self.min_score is not None and score < self.min_score:
                continue
            if best is None or value > best[0]:
                best = (value, tid, step)
        if best is None:
            self._tid = None
            return
        _, tid, step = best
        self._tid = tid
        self._cursor = step + 1
        self._since_replan = 0
