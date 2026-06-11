"""Phase 2 (Option B): model-based planning with a RuVector world model.

Learns a k-NN forward model (one RuVector index per action: state -> next_state,
reward) and acts by short rollouts. Unlike Option A's open-loop trajectory
replay, this REPLANS every step from the current state, so it doesn't commit to
a stale action sequence when the world diverges. The rollout loop runs entirely
in Rust (`ruvector_py.WorldModel.plan`) — the ~n_actions^2 * horizon searches
per decision never cross the Python boundary.

Requires the native binding rebuilt with WorldModel:
    maturin develop --release -m bridge/ruvector_py/Cargo.toml

    python experiments/train_world_model.py --scenario health_gathering --episodes 300
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

import ruvector_py  # noqa: E402  native binding (must include WorldModel)

from brain.encoder import make_encoder  # noqa: E402
from brain.policy.episodic import linear_epsilon  # noqa: E402
from envs.basic import discrete_actions, make_game  # noqa: E402


def rss_mb() -> float:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    except FileNotFoundError:
        pass
    return float("nan")


def run_episode(game, actions, enc, model, *, epsilon, learn, frameskip, rng):
    game.new_episode()
    vecs, acts, rews = [], [], []
    while not game.is_episode_finished():
        state = game.get_state()
        vec = enc(state)
        if rng.random() < epsilon:
            a = rng.randrange(len(actions))
        else:
            a = model.plan(vec.tolist())
        r = game.make_action(actions[a], frameskip)
        vecs.append(vec)
        acts.append(a)
        rews.append(r)

    if learn:
        for t in range(len(vecs) - 1):  # transition t -> t+1
            model.observe(acts[t], vecs[t].tolist(), float(rews[t]), vecs[t + 1].tolist())
    return game.get_total_reward()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="health_gathering")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--eps-start", type=float, default=1.0)
    ap.add_argument("--eps-end", type=float, default=0.1)
    ap.add_argument("--eps-decay-episodes", type=int, default=None, help="default: 70%% of --episodes")
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--unknown-dist", type=float, default=None, help="score above which the model is 'ignorant'")
    ap.add_argument("--frameskip", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--store", default=None, help="RuVector store prefix; default = fresh temp path")
    args = ap.parse_args()

    if not hasattr(ruvector_py, "WorldModel"):
        sys.exit(
            "ruvector_py.WorldModel missing — rebuild the binding:\n"
            "  maturin develop --release -m bridge/ruvector_py/Cargo.toml"
        )

    rng = random.Random(args.seed)
    decay = args.eps_decay_episodes or int(args.episodes * 0.7)
    prefix = args.store or os.path.join(tempfile.gettempdir(), f"dv_wm_{os.getpid()}")

    game = make_game(args.scenario, labels=True, position=True)
    actions = discrete_actions(game)
    enc, dim = make_encoder("navigation", game)
    model = ruvector_py.WorldModel(
        len(actions), dim, prefix, args.gamma, args.horizon, args.unknown_dist
    )
    print(
        f"scenario={args.scenario} encoder=navigation dim={dim} actions={len(actions)} "
        f"horizon={args.horizon} backend=native (Rust rollout)"
    )

    def evaluate() -> float:
        totals = [
            run_episode(game, actions, enc, model, epsilon=0.0, learn=False, frameskip=args.frameskip, rng=rng)
            for _ in range(args.eval_episodes)
        ]
        return sum(totals) / len(totals)

    t0 = time.time()
    print(f"[eval @0] mean={evaluate():+.1f} (random baseline)")
    for ep in range(1, args.episodes + 1):
        eps = linear_epsilon(ep, eps_start=args.eps_start, eps_end=args.eps_end, decay_episodes=decay)
        run_episode(game, actions, enc, model, epsilon=eps, learn=True, frameskip=args.frameskip, rng=rng)
        if ep % args.eval_every == 0:
            print(
                f"[eval @{ep}] mean={evaluate():+.1f} eps={eps:.2f} "
                f"rss={rss_mb():.1f}MiB t={time.time() - t0:.0f}s"
            )

    game.close()


if __name__ == "__main__":
    main()
