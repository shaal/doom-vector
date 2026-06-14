"""Uncertainty signal + uncertainty-gated evasive fallback (Track 3).

`recall_uncertainty` must be high when recall is empty or the neighbours disagree,
low when they agree; `choose_action_safe` must take the evasive default exactly
when uncertainty crosses the threshold (and a default is available)."""
import math
import random

from brain.policy.episodic import choose_action, choose_action_safe, recall_uncertainty


def nb(action, score, ret=1.0):
    """A neighbour tuple (id, score, metadata) for an action vote."""
    return ("x", score, {"action_idx": float(action), "return": float(ret)})


def test_uncertainty_empty_recall_is_max():
    assert recall_uncertainty([], n_actions=4) == 1.0


def test_uncertainty_single_action_is_minimal():
    # only one possible action -> nothing to be uncertain about
    assert recall_uncertainty([nb(0, 1.0)], n_actions=1) == 0.0


def test_uncertainty_consensus_is_low():
    # all neighbours took the same action -> ~0 entropy
    consensus = [nb(0, 1.0), nb(0, 0.9), nb(0, 0.8)]
    assert recall_uncertainty(consensus, n_actions=4) < 0.05


def test_uncertainty_even_split_is_maximal():
    # equal weight across all 4 actions -> normalized entropy == 1.0
    even = [nb(a, 1.0) for a in range(4)]
    assert abs(recall_uncertainty(even, n_actions=4) - 1.0) < 1e-9


def test_uncertainty_partial_disagreement_is_between():
    half = [nb(0, 1.0), nb(0, 1.0), nb(1, 1.0), nb(1, 1.0)]
    u = recall_uncertainty(half, n_actions=4)
    # two equally-weighted actions out of four -> ln(2)/ln(4) = 0.5
    assert abs(u - 0.5) < 1e-9


def test_uncertainty_is_backend_agnostic_to_score_offset():
    # native cosine vs numpy negative-L2 differ by an additive/scale offset; the
    # entropy of the *action* distribution must be insensitive to a constant shift.
    base = [nb(0, 0.9), nb(1, 0.8), nb(0, 0.7)]
    shifted = [nb(0, -5.1), nb(1, -5.2), nb(0, -5.3)]
    assert abs(recall_uncertainty(base, 4) - recall_uncertainty(shifted, 4)) < 1e-9


def test_choose_action_safe_evades_when_uncertain():
    # empty recall -> uncertainty 1.0 >= threshold -> take the evasive default
    assert choose_action_safe([], 4, epsilon=0.0, evade_action=2, evade_threshold=0.6) == 2


def test_choose_action_safe_trusts_vote_when_confident():
    # strong consensus on action 0 -> low uncertainty -> defer to the value vote,
    # not the evasive default.
    consensus = [nb(0, 1.0, ret=5.0), nb(0, 0.9, ret=5.0), nb(0, 0.8, ret=5.0)]
    chosen = choose_action_safe(consensus, 4, epsilon=0.0, evade_action=2, evade_threshold=0.6)
    assert chosen == 0
    assert chosen == choose_action(consensus, 4, epsilon=0.0)


def test_choose_action_safe_without_default_is_plain_vote():
    # no evasive default available (e.g. no threat on screen) -> identical to
    # choose_action even under high uncertainty. Makes the ablation a clean toggle.
    rng_a, rng_b = random.Random(7), random.Random(7)
    even = [nb(a, 1.0) for a in range(4)]
    safe = choose_action_safe(even, 4, epsilon=0.0, rng=rng_a, evade_action=None)
    plain = choose_action(even, 4, epsilon=0.0, rng=rng_b)
    assert safe == plain


def test_choose_action_safe_threshold_gates_the_evade():
    half = [nb(0, 1.0), nb(0, 1.0), nb(1, 1.0), nb(1, 1.0)]  # uncertainty == 0.5
    # threshold above 0.5 -> trust the vote; below -> evade.
    assert choose_action_safe(half, 4, epsilon=0.0, evade_action=2, evade_threshold=0.6) == 0
    assert choose_action_safe(half, 4, epsilon=0.0, evade_action=2, evade_threshold=0.4) == 2
