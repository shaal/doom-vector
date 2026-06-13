"""ExperienceStore search: filter post-filter, over-fetch, diversify, contract.

Forces the numpy backend, which mirrors native ruvector-core ordering (k-NN to
k_raw, then post-filter) so these assertions reflect on-Pi behavior too.
"""
import numpy as np

from brain.memory.experience_store import ExperienceStore, _meta_matches


def _store(dim=2):
    s = ExperienceStore(dim=dim, backend="numpy")
    assert s.backend == "numpy"
    return s


def test_meta_matches_exact_equality():
    assert _meta_matches({"enemy_visible": 1.0, "x": 2.0}, {"enemy_visible": 1.0})
    assert not _meta_matches({"enemy_visible": 0.0}, {"enemy_visible": 1.0})
    assert not _meta_matches({}, {"enemy_visible": 1.0})  # missing key -> no match


def test_plain_search_returns_k_nearest_3tuples():
    s = _store()
    for i in range(5):
        s.insert([float(i), 0.0], {"action_idx": float(i), "return": float(i)})
    out = s.search([0.0, 0.0], k=3)
    assert len(out) == 3
    assert all(len(t) == 3 for t in out)  # public contract is (id, score, meta)
    # nearest-first: closest to [0,0] is the i=0 vector
    assert out[0][2]["action_idx"] == 0.0


def test_filter_returns_only_matching_metadata():
    s = _store()
    # interleave enemy-visible and not, all near the query
    for i in range(10):
        vis = float(i % 2)
        s.insert([0.01 * i, 0.0], {"action_idx": float(i), "return": 0.0, "enemy_visible": vis})
    out = s.search([0.0, 0.0], k=10, filter={"enemy_visible": 1.0})
    assert out, "filter pruned everything"
    assert all(t[2]["enemy_visible"] == 1.0 for t in out)


# A tight non-matching cluster right on the query, with the lone matching entry
# parked farther away at rank ~7. (All vectors non-zero so the store's zero-norm
# nudge doesn't perturb the geometry.)
_Q = [0.5, 0.5]


def _seed_starve_layout(s):
    for i in range(6):  # closest cluster, none match
        s.insert([0.5 + 0.001 * i, 0.5], {"action_idx": 99.0, "return": 0.0, "enemy_visible": 0.0})
    s.insert([0.7, 0.5], {"action_idx": 7.0, "return": 1.0, "enemy_visible": 1.0})  # farther, matches


def test_over_fetch_reaches_matches_beyond_top_k():
    # over_fetch=4 (k_raw=8) pulls the rank-7 match into the pool so the filter
    # can keep it.
    s = _store()
    _seed_starve_layout(s)
    out = s.search(_Q, k=2, filter={"enemy_visible": 1.0}, over_fetch=4)
    assert len(out) == 1
    assert out[0][2]["action_idx"] == 7.0


def test_under_fetch_would_starve_the_vote():
    # over_fetch=1 (k_raw=2): the post-filter sees only the non-matching nearest
    # cluster and returns nothing -> demonstrates *why* over-fetch matters.
    s = _store()
    _seed_starve_layout(s)
    out = s.search(_Q, k=2, filter={"enemy_visible": 1.0}, over_fetch=1)
    assert out == []


def test_diversify_keeps_k_and_spans_clusters():
    s = _store()
    # dense cluster of near-duplicates near [1,0], plus a few near [0,1]
    for _ in range(12):
        s.insert([1.0, 0.0], {"action_idx": 1.0, "return": 0.0})
    for _ in range(3):
        s.insert([0.0, 1.0], {"action_idx": 2.0, "return": 0.0})
    plain = s.search([0.9, 0.1], k=4, diversify=False)
    diverse = s.search([0.9, 0.1], k=4, diversify=True)
    assert len(diverse) == 4
    actions_plain = {t[2]["action_idx"] for t in plain}
    actions_diverse = {t[2]["action_idx"] for t in diverse}
    # raw top-k is swamped by the dense cluster; diversify surfaces the other action
    assert actions_plain == {1.0}
    assert 2.0 in actions_diverse


def test_no_filter_no_diversify_is_unchanged_behavior():
    s = _store()
    for i in range(4):
        s.insert([float(i), 0.0], {"action_idx": float(i), "return": 0.0})
    out = s.search([0.0, 0.0], k=2)
    assert [t[2]["action_idx"] for t in out] == [0.0, 1.0]
