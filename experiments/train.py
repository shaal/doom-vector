"""Phase 1: train the episodic-control agent and watch it learn.

Trains with epsilon-decayed exploration; periodically runs greedy eval episodes
and prints the mean eval reward so the learning curve is visible. Memory is
bounded by --capacity (value-based eviction). Optionally records a final greedy
episode to a .lmp for replay (see experiments/replay.py).

    python experiments/train.py --scenario basic --encoder structured \
        --episodes 300 --eval-every 25 --capacity 20000

"learning" = writing reward-weighted experiences into RuVector; no gradients.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import vizdoom as vzd  # noqa: E402

from brain.encoder import make_encoder  # noqa: E402
from brain.encoder.structured import (  # noqa: E402
    enemy_visible,
    threat_features,
    threat_visible,
)
from brain.memory.experience_store import ExperienceStore  # noqa: E402
from brain.policy.episodic import (  # noqa: E402
    choose_action,
    choose_action_safe,
    discounted_returns,
    linear_epsilon,
)
from brain.policy.reward import (  # noqa: E402
    damage_delta,
    dodge_shaped_reward,
    health_delta,
    hit_shaped_reward,
)
from envs.basic import discrete_actions, make_game  # noqa: E402


def rss_mb() -> float:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    except FileNotFoundError:
        pass
    return float("nan")


def make_dodge_fallback(game, actions, screen_w):
    """Build `fallback(state) -> evade_action_idx | None` for Track 3.

    The episodic policy is scenario-agnostic, so the "strafe away from the
    nearest projectile" default is computed here, where the button layout is
    known: find the single-button MOVE_LEFT / MOVE_RIGHT actions, then steer
    *away* from the nearest incoming projectile's horizontal offset. Returns None
    when no projectile is on screen or the scenario lacks both strafe buttons (so
    the policy keeps voting and the fallback never fires spuriously)."""
    buttons = game.get_available_buttons()

    def single(btn):
        if btn not in buttons:
            return None
        combo = [1 if b == btn else 0 for b in buttons]
        return actions.index(combo) if combo in actions else None

    left = single(vzd.Button.MOVE_LEFT)
    right = single(vzd.Button.MOVE_RIGHT)

    def fallback(state):
        if left is None or right is None or not threat_visible(state):
            return None
        dx = threat_features(state, screen_w=screen_w)[0]  # >0: ball on the right -> flee left
        return left if dx > 0 else right

    return fallback


def run_episode(
    game, actions, enc, store, *, epsilon, k, frameskip, learn, gamma, rng,
    aim=False, hit_bonus=0.0, dodge=False, health_penalty=0.0, evade_threshold=0.6,
    dodge_fallback=None, record=None,
):
    """Run one episode.

    With `aim=True` (Track 1): recall is *filtered* to enemy-visible memories
    when an enemy is on screen and *MMR-diversified*, and — when `hit_bonus` is
    set and we're learning — the per-step reward is shaped by DAMAGECOUNT delta
    so aiming gets dense feedback.

    With `dodge=True` (Track 3): recall is filtered to threat-visible memories and
    MMR-diversified the same way, the action is chosen through the
    *uncertainty-gated safe fallback* (`choose_action_safe` strafes away from the
    nearest threat when recall is unreliable), and — when `health_penalty` is set
    and we're learning — the per-step reward is penalized by HEALTH lost so
    dodging gets dense feedback.

    Both modes touch only the *learned* return; `game.get_total_reward()` (what
    eval reports) stays the unshaped scenario score.
    """
    game.new_episode(record) if record else game.new_episode()
    traj = []
    prev_damage = 0.0
    prev_health = game.get_game_variable(vzd.GameVariable.HEALTH) if dodge else 0.0
    while not game.is_episode_finished():
        state = game.get_state()
        vec = enc(state)
        if aim:
            vis = enemy_visible(state)
        elif dodge:
            vis = threat_visible(state)  # dodge keys on incoming projectiles
        else:
            vis = 0.0
        if aim and vis:
            filt = {"enemy_visible": 1.0}
        elif dodge and vis:
            filt = {"threat_visible": 1.0}
        else:
            filt = None
        # MMR re-ranks the over-fetched *filtered* pool, so it's coupled to the
        # filter: diversify only when we filtered (threat visible). This also
        # spares the Pi the k_raw vector marshalling on empty frames.
        neighbors = store.search(vec, k=k, filter=filt, diversify=filt is not None)
        if dodge:
            ev = dodge_fallback(state) if dodge_fallback else None
            a = choose_action_safe(
                neighbors, len(actions), epsilon=epsilon, rng=rng,
                evade_action=ev, evade_threshold=evade_threshold,
            )
        else:
            a = choose_action(neighbors, len(actions), epsilon=epsilon, rng=rng)
        r = game.make_action(actions[a], frameskip)
        if aim and hit_bonus:
            now = game.get_game_variable(vzd.GameVariable.DAMAGECOUNT)
            r = hit_shaped_reward(r, damage_delta(prev_damage, now), hit_bonus)
            prev_damage = now
        if dodge and health_penalty:
            now = game.get_game_variable(vzd.GameVariable.HEALTH)
            r = dodge_shaped_reward(r, health_delta(prev_health, now), health_penalty)
            prev_health = now
        traj.append((vec, a, r, vis))

    if learn and traj:
        rets = discounted_returns([t[2] for t in traj], gamma)
        for (vec, a, _, vis), g in zip(traj, rets):
            md = {"action_idx": float(a), "return": float(g)}
            if aim:
                md["enemy_visible"] = float(vis)
            if dodge:
                md["threat_visible"] = float(vis)
            store.insert(vec, md)
    return game.get_total_reward()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="basic")
    ap.add_argument("--encoder", choices=["thumbnail", "structured", "navigation"], default="structured")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--eps-start", type=float, default=1.0)
    ap.add_argument("--eps-end", type=float, default=0.05)
    ap.add_argument("--eps-decay-episodes", type=int, default=None, help="default: 70%% of --episodes")
    ap.add_argument("--capacity", type=int, default=20000)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--frameskip", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--store", default=None, help="RuVector store path; default = fresh temp path")
    ap.add_argument("--record", default=None, help="record a final greedy episode to this .lmp")
    ap.add_argument(
        "--aim", action="store_true",
        help="Track 1: aim encoder dims + enemy-visible filtered/MMR recall + hit-bonus shaping "
        "(structured encoder only; e.g. --scenario defend_the_center --encoder structured --aim)",
    )
    ap.add_argument(
        "--hit-bonus", type=float, default=0.01,
        help="per-DAMAGECOUNT reward bonus when --aim (small; eval is on the unshaped score)",
    )
    ap.add_argument(
        "--dodge", action="store_true",
        help="Track 3: threat encoder dims + threat-visible filtered/MMR recall + "
        "uncertainty-gated evasive fallback + health-loss reward penalty "
        "(structured encoder only; e.g. --scenario take_cover --encoder structured --dodge)",
    )
    ap.add_argument(
        "--health-penalty", type=float, default=0.05,
        help="per-HEALTH-lost reward penalty when --dodge (small; eval is on the unshaped score)",
    )
    ap.add_argument(
        "--evade-threshold", type=float, default=0.6,
        help="recall-uncertainty in [0,1] at/above which --dodge strafes away instead of "
        "voting (higher = trust the learned recall more; a deterministic 3-training-seed "
        "paired A/B on take_cover found 0.5-0.7 all beat pure voting, 0.6 robustly)",
    )
    args = ap.parse_args()

    if args.aim and args.encoder != "structured":
        ap.error("--aim requires --encoder structured (it appends aim dims to the structured encoder)")
    if args.dodge and args.encoder != "structured":
        ap.error("--dodge requires --encoder structured (it appends threat dims to the structured encoder)")

    rng = random.Random(args.seed)
    decay = args.eps_decay_episodes or int(args.episodes * 0.7)
    use_labels = args.encoder in ("structured", "navigation")
    use_position = args.encoder == "navigation"
    store_path = args.store or os.path.join(tempfile.gettempdir(), f"dv_train_{os.getpid()}.rvf")

    game = make_game(args.scenario, labels=use_labels, position=use_position)
    actions = discrete_actions(game)
    enc, dim = make_encoder(args.encoder, game, aim=args.aim, dodge=args.dodge)
    store = ExperienceStore(dim=dim, storage_path=store_path, capacity=args.capacity)
    # Track 3: the "strafe away from the nearest threat" safe default, built where
    # the button layout is known so the episodic policy stays scenario-agnostic.
    dodge_fallback = (
        make_dodge_fallback(game, actions, float(game.get_screen_width())) if args.dodge else None
    )

    where = store_path if store.backend == "native" else "in-memory"
    print(
        f"scenario={args.scenario} encoder={args.encoder} dim={dim} actions={len(actions)} "
        f"backend={store.backend} store={where} cap={args.capacity}"
        + (f" aim=on hit_bonus={args.hit_bonus}" if args.aim else "")
        + (f" dodge=on health_penalty={args.health_penalty} evade_thr={args.evade_threshold}"
           if args.dodge else "")
    )

    def evaluate(*, evade_threshold: float | None = None, seed_base: int | None = None) -> float:
        # eval uses the real policy (filter + dodge fallback) but no reward
        # shaping; `evade_threshold` can be raised to disable the fallback for the
        # with/without-fallback ablation. `seed_base`, when set, seeds ViZDoom per
        # episode so two variants face *identical* episodes — a paired A/B that
        # cancels take_cover's high episode-to-episode variance.
        thr = args.evade_threshold if evade_threshold is None else evade_threshold
        totals = []
        for i in range(args.eval_episodes):
            if seed_base is not None:
                game.set_seed(seed_base + i)
            totals.append(
                run_episode(
                    game, actions, enc, store,
                    epsilon=0.0, k=args.k, frameskip=args.frameskip,
                    learn=False, gamma=args.gamma, rng=rng,
                    aim=args.aim, hit_bonus=0.0,
                    dodge=args.dodge, health_penalty=0.0,
                    evade_threshold=thr, dodge_fallback=dodge_fallback,
                )
            )
        return sum(totals) / len(totals)

    t0 = time.time()
    print(f"[eval @0] mean={evaluate():+.1f} (random baseline)")
    for ep in range(1, args.episodes + 1):
        eps = linear_epsilon(ep, eps_start=args.eps_start, eps_end=args.eps_end, decay_episodes=decay)
        run_episode(
            game, actions, enc, store,
            epsilon=eps, k=args.k, frameskip=args.frameskip,
            learn=True, gamma=args.gamma, rng=rng,
            aim=args.aim, hit_bonus=args.hit_bonus,
            dodge=args.dodge, health_penalty=args.health_penalty,
            evade_threshold=args.evade_threshold, dodge_fallback=dodge_fallback,
        )
        if ep % args.eval_every == 0:
            print(
                f"[eval @{ep}] mean={evaluate():+.1f} eps={eps:.2f} mem={len(store)} "
                f"rss={rss_mb():.1f}MiB t={time.time() - t0:.0f}s"
            )

    if args.dodge:
        # Ablation (Track 3 success criterion): does the uncertainty-gated evasive
        # fallback measurably help vs. the same store with the fallback disabled
        # (evade_threshold > 1.0 can never be reached, so it's pure value vote)?
        # Paired/seeded so both face identical episodes — without that, take_cover's
        # variance swamps the effect.
        seed_base = 70000
        vote_only = evaluate(evade_threshold=2.0, seed_base=seed_base)
        vote_evade = evaluate(seed_base=seed_base)
        print(
            f"[dodge ablation, paired n={args.eval_episodes}] "
            f"vote_only={vote_only:+.1f} vote+evade={vote_evade:+.1f}"
        )

    if args.record:
        total = run_episode(
            game, actions, enc, store,
            epsilon=0.0, k=args.k, frameskip=args.frameskip,
            learn=False, gamma=args.gamma, rng=rng,
            aim=args.aim, hit_bonus=0.0,
            dodge=args.dodge, health_penalty=0.0,
            evade_threshold=args.evade_threshold, dodge_fallback=dodge_fallback,
            record=args.record,
        )
        print(f"recorded greedy episode -> {args.record} (reward={total:+.1f})")

    game.close()


if __name__ == "__main__":
    main()
