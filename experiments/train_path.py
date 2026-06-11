"""Phase 2 (Option A): learn to navigate by predicting & following paths.

The agent recalls the best-value past trajectory near its current pose and
follows that path (replanning periodically). Each episode's trajectory is
stored; eviction keeps the best, so the memory curates toward good paths —
that's the learning. Periodic greedy eval prints the curve.

Default scenario `health_gathering`: dense reward (survive on a damaging
floor by walking to health kits), small action set, navigation-driven — a
scenario where "path" actually matters.

    python experiments/train_path.py --scenario health_gathering --episodes 300
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

from brain.encoder import make_encoder  # noqa: E402
from brain.memory.trajectory_store import TrajectoryStore  # noqa: E402
from brain.policy.episodic import discounted_returns, linear_epsilon  # noqa: E402
from brain.policy.trajectory_follow import TrajectoryPlanner  # noqa: E402
from envs.basic import discrete_actions, make_game  # noqa: E402


def rss_mb() -> float:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    except FileNotFoundError:
        pass
    return float("nan")


def run_episode(game, actions, enc, planner, store, *, learn, gamma, frameskip):
    planner.reset()
    game.new_episode()
    states, acts, rewards = [], [], []
    while not game.is_episode_finished():
        state = game.get_state()
        vec = enc(state)
        a = planner.act(vec)
        r = game.make_action(actions[a], frameskip)
        states.append(vec)
        acts.append(a)
        rewards.append(r)
    total = game.get_total_reward()
    if learn and states:
        values = discounted_returns(rewards, gamma)
        store.add_trajectory(states, acts, values, total)
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="health_gathering")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--eps-start", type=float, default=1.0)
    ap.add_argument("--eps-end", type=float, default=0.1)
    ap.add_argument("--eps-decay-episodes", type=int, default=None, help="default: 70%% of --episodes")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--replan-every", type=int, default=8)
    ap.add_argument("--max-trajectories", type=int, default=400)
    ap.add_argument("--frameskip", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--store", default=None, help="RuVector store path; default = fresh temp path")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    decay = args.eps_decay_episodes or int(args.episodes * 0.7)
    store_path = args.store or os.path.join(tempfile.gettempdir(), f"dv_path_{os.getpid()}.rvf")

    game = make_game(args.scenario, labels=True, position=True)
    actions = discrete_actions(game)
    enc, dim = make_encoder("navigation", game)
    store = TrajectoryStore(dim, storage_path=store_path, max_trajectories=args.max_trajectories)
    planner = TrajectoryPlanner(store, len(actions), k=args.k, replan_every=args.replan_every, rng=rng)

    where = store_path if store.backend == "native" else "in-memory"
    print(
        f"scenario={args.scenario} encoder=navigation dim={dim} actions={len(actions)} "
        f"backend={store.backend} store={where} max_traj={args.max_trajectories}"
    )

    def evaluate() -> float:
        planner.epsilon = 0.0
        totals = [
            run_episode(game, actions, enc, planner, store, learn=False, gamma=args.gamma, frameskip=args.frameskip)
            for _ in range(args.eval_episodes)
        ]
        return sum(totals) / len(totals)

    t0 = time.time()
    print(f"[eval @0] mean={evaluate():+.1f} (random baseline)")
    for ep in range(1, args.episodes + 1):
        planner.epsilon = linear_epsilon(
            ep, eps_start=args.eps_start, eps_end=args.eps_end, decay_episodes=decay
        )
        run_episode(game, actions, enc, planner, store, learn=True, gamma=args.gamma, frameskip=args.frameskip)
        if ep % args.eval_every == 0:
            print(
                f"[eval @{ep}] mean={evaluate():+.1f} eps={planner.epsilon:.2f} "
                f"trajs={len(store)} rss={rss_mb():.1f}MiB t={time.time() - t0:.0f}s"
            )

    game.close()


if __name__ == "__main__":
    main()
