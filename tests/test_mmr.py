"""MMR re-ranking: relevance/diversity trade-off, k limit, vector-less fallback."""
import numpy as np

from brain.policy.mmr import mmr_rerank


def _cand(id_, vec, score=0.0, meta=None):
    return (id_, score, list(vec), meta or {})


def test_lambda_one_is_pure_relevance_order():
    # lam=1 ignores diversity -> must equal candidates[:k] (input is relevance-ordered).
    q = [1.0, 0.0]
    cands = [_cand("a", [1.0, 0.0]), _cand("b", [0.9, 0.1]), _cand("c", [0.0, 1.0])]
    out = mmr_rerank(q, cands, k=2, lam=1.0)
    assert [c[0] for c in out] == ["a", "b"]


def test_diversity_breaks_a_near_duplicate_cluster():
    # Three near-identical "dead-ahead" candidates + one different direction.
    # A diverse top-2 must include the outlier, not two clones.
    q = [1.0, 0.0]
    cands = [
        _cand("clone1", [1.0, 0.02]),
        _cand("clone2", [1.0, 0.01]),
        _cand("clone3", [1.0, 0.0]),
        _cand("other", [0.0, 1.0]),
    ]
    # lam<0.5 leans on diversity; the orthogonal outlier must displace a clone.
    out = mmr_rerank(q, cands, k=2, lam=0.2)
    ids = {c[0] for c in out}
    assert "other" in ids, f"diverse rerank dropped the outlier: {ids}"
    assert len(ids) == 2


def test_returns_all_when_k_exceeds_pool():
    q = [1.0, 0.0]
    cands = [_cand("a", [1.0, 0.0]), _cand("b", [0.0, 1.0])]
    out = mmr_rerank(q, cands, k=5, lam=0.5)
    assert len(out) == 2


def test_missing_vectors_fall_back_to_relevance_order():
    q = [1.0, 0.0]
    cands = [("a", 0.0, None, {}), ("b", 0.0, None, {}), ("c", 0.0, None, {})]
    out = mmr_rerank(q, cands, k=2, lam=0.5)
    assert [c[0] for c in out] == ["a", "b"]


def test_empty_and_zero_k():
    assert mmr_rerank([1.0, 0.0], [], k=3) == []
    assert mmr_rerank([1.0, 0.0], [_cand("a", [1.0, 0.0])], k=0) == []


def test_first_pick_is_most_relevant():
    # MMR seeds with the single most-relevant candidate regardless of input order.
    # k<len so the rerank actually runs (k>=len short-circuits to the input order).
    q = [1.0, 0.0]
    cands = [_cand("far", [0.2, 1.0]), _cand("near", [1.0, 0.0]), _cand("mid", [0.7, 0.7])]
    out = mmr_rerank(q, cands, k=2, lam=0.5)
    assert out[0][0] == "near"
