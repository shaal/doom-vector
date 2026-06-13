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
from brain.encoder.structured import enemy_visible  # noqa: E402
from brain.memory.experience_store import ExperienceStore  # noqa: E402
from brain.policy.episodic import (  # noqa: E402
    choose_action,
    discounted_returns,
    linear_epsilon,
)
from brain.policy.reward import damage_delta, hit_shaped_reward  # noqa: E402
from envs.basic import discrete_actions, make_game  # noqa: E402


def rss_mb() -> float:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    except FileNotFoundError:
        pass
    return float("nan")


def run_episode(
    game, actions, enc, store, *, epsilon, k, frameskip, learn, gamma, rng,
    aim=False, hit_bonus=0.0, record=None,
):
    """Run one episode.

    With `aim=True` (Track 1): recall is *filtered* to enemy-visible memories
    when an enemy is on screen and *MMR-diversified*, and — when `hit_bonus` is
    set and we're learning — the per-step reward is shaped by DAMAGECOUNT delta
    so aiming gets dense feedback. Shaping touches only the *learned* return;
    `game.get_total_reward()` (what eval reports) stays the unshaped score.
    """
    game.new_episode(record) if record else game.new_episode()
    traj = []
    prev_damage = 0.0
    while not game.is_episode_finished():
        state = game.get_state()
        vec = enc(state)
        vis = enemy_visible(state) if aim else 0.0
        filt = {"enemy_visible": 1.0} if (aim and vis) else None
        # MMR re-ranks the over-fetched *filtered* pool, so it's coupled to the
        # filter: diversify only when we filtered (enemy visible). This also
        # spares the Pi the k_raw vector marshalling on empty frames.
        neighbors = store.search(vec, k=k, filter=filt, diversify=filt is not None)
        a = choose_action(neighbors, len(actions), epsilon=epsilon, rng=rng)
        r = game.make_action(actions[a], frameskip)
        if aim and hit_bonus:
            now = game.get_game_variable(vzd.GameVariable.DAMAGECOUNT)
            r = hit_shaped_reward(r, damage_delta(prev_damage, now), hit_bonus)
            prev_damage = now
        traj.append((vec, a, r, vis))

    if learn and traj:
        rets = discounted_returns([t[2] for t in traj], gamma)
        for (vec, a, _, vis), g in zip(traj, rets):
            md = {"action_idx": float(a), "return": float(g)}
            if aim:
                md["enemy_visible"] = float(vis)
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
    args = ap.parse_args()

    if args.aim and args.encoder != "structured":
        ap.error("--aim requires --encoder structured (it appends aim dims to the structured encoder)")

    rng = random.Random(args.seed)
    decay = args.eps_decay_episodes or int(args.episodes * 0.7)
    use_labels = args.encoder in ("structured", "navigation")
    use_position = args.encoder == "navigation"
    store_path = args.store or os.path.join(tempfile.gettempdir(), f"dv_train_{os.getpid()}.rvf")

    game = make_game(args.scenario, labels=use_labels, position=use_position)
    actions = discrete_actions(game)
    enc, dim = make_encoder(args.encoder, game, aim=args.aim)
    store = ExperienceStore(dim=dim, storage_path=store_path, capacity=args.capacity)

    where = store_path if store.backend == "native" else "in-memory"
    print(
        f"scenario={args.scenario} encoder={args.encoder} dim={dim} actions={len(actions)} "
        f"backend={store.backend} store={where} cap={args.capacity}"
        + (f" aim=on hit_bonus={args.hit_bonus}" if args.aim else "")
    )

    def evaluate() -> float:
        totals = [
            run_episode(
                game, actions, enc, store,
                epsilon=0.0, k=args.k, frameskip=args.frameskip,
                learn=False, gamma=args.gamma, rng=rng,
                aim=args.aim, hit_bonus=0.0,  # eval uses the real (filtered) policy; no shaping
            )
            for _ in range(args.eval_episodes)
        ]
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
        )
        if ep % args.eval_every == 0:
            print(
                f"[eval @{ep}] mean={evaluate():+.1f} eps={eps:.2f} mem={len(store)} "
                f"rss={rss_mb():.1f}MiB t={time.time() - t0:.0f}s"
            )

    if args.record:
        total = run_episode(
            game, actions, enc, store,
            epsilon=0.0, k=args.k, frameskip=args.frameskip,
            learn=False, gamma=args.gamma, rng=rng,
            aim=args.aim, hit_bonus=0.0, record=args.record,
        )
        print(f"recorded greedy episode -> {args.record} (reward={total:+.1f})")

    game.close()


if __name__ == "__main__":
    main()
