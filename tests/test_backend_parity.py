"""Native (ruvector_py) vs. numpy backend parity for the recall contract.

Regression guard for the score-sign normalization: ruvector-core's native
`search` returns raw L2 *distance* (smaller == closer) while the numpy fallback
returns negative L2. `ExperienceStore.search` negates the native score so both
report "bigger == more similar"; without that, the similarity-weighted value vote
and the uncertainty gate are inverted on the Pi's native backend. Every other
test pins the numpy backend, so this is the only check that exercises native.

Skips when ruvector_py isn't built (e.g. off-device CI)."""
import os
import tempfile

import numpy as np
import pytest

from brain.memory.experience_store import ExperienceStore
from brain.policy.episodic import choose_action, recall_uncertainty

native_available = False
try:
    import ruvector_py  # noqa: F401

    native_available = True
except Exception:  # pragma: no cover - depends on build env
    pass

pytestmark = pytest.mark.skipif(not native_available, reason="native ruvector_py not built")


# A spread of states with different actions and returns, plus a clear query.
_DATA = [
    ([1.0, 0.0, 0.0], 0, 5.0),   # identical to query -> nearest
    ([0.9, 0.1, 0.0], 1, 1.0),   # near
    ([0.2, 0.9, 0.0], 2, 9.0),   # far, but best return
    ([-1.0, 0.0, 0.0], 3, 0.0),  # opposite -> farthest
]
_QUERY = [1.0, 0.0, 0.0]
_N_ACTIONS = 4


def _fill(store):
    for vec, act, ret in _DATA:
        store.insert(vec, {"action_idx": float(act), "return": float(ret)})
    return store


def _native_store():
    # A fresh dir per store: the native redb backend persists and *reloads* from
    # its path, so a shared path would accumulate inserts across tests.
    sp = os.path.join(tempfile.mkdtemp(prefix="dv_parity_"), "store.rvf")
    s = ExperienceStore(dim=3, backend="native", storage_path=sp)
    assert s.backend == "native"
    return s


def test_native_score_sign_is_bigger_means_closer():
    s = _fill(_native_store())
    out = s.search(_QUERY, k=4)
    # nearest-first AND the nearest must have the *largest* (least-negative) score
    scores = [score for _, score, _ in out]
    assert scores == sorted(scores, reverse=True), f"not bigger-is-closer: {scores}"
    assert out[0][2]["action_idx"] == 0.0  # the identical vector is rank 0


def test_native_matches_numpy_scores_and_ordering():
    native = _fill(_native_store())
    numpy = _fill(ExperienceStore(dim=3, backend="numpy"))
    n_out = native.search(_QUERY, k=4)
    m_out = numpy.search(_QUERY, k=4)
    # Same L2 metric on both -> identical (score, action) sequence after sign norm.
    n_seq = [(round(s, 4), md["action_idx"]) for _, s, md in n_out]
    m_seq = [(round(s, 4), md["action_idx"]) for _, s, md in m_out]
    assert n_seq == m_seq, f"native {n_seq} != numpy {m_seq}"


def test_native_and_numpy_agree_on_vote_and_uncertainty():
    native = _fill(_native_store())
    numpy = _fill(ExperienceStore(dim=3, backend="numpy"))
    nb_n = native.search(_QUERY, k=4)
    nb_m = numpy.search(_QUERY, k=4)
    # The value vote must pick the same action on both backends...
    assert choose_action(nb_n, _N_ACTIONS, epsilon=0.0) == choose_action(nb_m, _N_ACTIONS, epsilon=0.0)
    # ...and the uncertainty (hence the evade gate) must match within float noise.
    assert abs(recall_uncertainty(nb_n, _N_ACTIONS) - recall_uncertainty(nb_m, _N_ACTIONS)) < 1e-6


def test_native_vote_prefers_best_return_neighbor():
    # With correct (non-inverted) weighting, action 2 (return 9.0) should win the
    # value vote over the merely-nearest action 0 (return 5.0): all four are in the
    # recalled set and the vote is by weighted average return.
    s = _fill(_native_store())
    assert choose_action(s.search(_QUERY, k=4), _N_ACTIONS, epsilon=0.0) == 2
