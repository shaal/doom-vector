"""Phase 2 (Option B v2): model-based planning with a value-bootstrapped
RuVector world model.

v1's reward-summing rollout collapsed to no-op on dense-reward scenarios. v2
scores each action by a 1-step Bellman backup over a LEARNED VALUE (discounted
return-to-go), so it has a gradient even when immediate reward is constant:
    score(a) = predicted_reward(s,a) + gamma * V(predicted_next_state)
The backup runs entirely in Rust (`ruvector_py.WorldModel.plan`). Unlike the
reactive value-vote, this is model-based: it predicts the next state and
bootstraps its value.

Requires the binding rebuilt with the v2 WorldModel:
    maturin develop --release --uv -m bridge/ruvector_py/Cargo.toml

    python experiments/train_world_model.py --scenario health_gathering --episodes 250
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

import ruvector_py  # noqa: E402  native binding (must include the v2 WorldModel)

from brain.encoder import make_encoder  # noqa: E402
from brain.policy.episodic import discounted_returns, linear_epsilon  # noqa: E402
from envs.basic import discrete_actions, make_game  # noqa: E402


def rss_mb() -> float:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    except FileNotFoundError:
        pass
    return float("nan")


def run_episode(game, actions, enc, model, *, epsilon, learn, frameskip, gamma, rng):
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

    if learn and len(vecs) > 1:
        values = discounted_returns(rews, gamma)  # return-to-go = the learned value
        for t in range(len(vecs) - 1):  # transition t -> t+1
            model.observe(acts[t], vecs[t].tolist(), float(rews[t]), vecs[t + 1].tolist(), float(values[t]))
    return game.get_total_reward()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="health_gathering")
    ap.add_argument("--episodes", type=int, default=250)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-episodes", type=int, default=8)
    ap.add_argument("--eps-start", type=float, default=1.0)
    ap.add_argument("--eps-end", type=float, default=0.1)
    ap.add_argument("--eps-decay-episodes", type=int, default=None, help="default: 70%% of --episodes")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--frameskip", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--store", default=None, help="RuVector store prefix; default = fresh temp path")
    args = ap.parse_args()

    if not hasattr(ruvector_py, "WorldModel"):
        sys.exit(
            "ruvector_py.WorldModel missing — rebuild the binding:\n"
            "  maturin develop --release --uv -m bridge/ruvector_py/Cargo.toml"
        )

    rng = random.Random(args.seed)
    decay = args.eps_decay_episodes or int(args.episodes * 0.7)
    prefix = args.store or os.path.join(tempfile.gettempdir(), f"dv_wm_{os.getpid()}")

    game = make_game(args.scenario, labels=True, position=True)
    actions = discrete_actions(game)
    enc, dim = make_encoder("navigation", game)
    model = ruvector_py.WorldModel(len(actions), dim, prefix, args.gamma)
    print(
        f"scenario={args.scenario} encoder=navigation dim={dim} actions={len(actions)} "
        f"backend=native (Rust value-bootstrapped backup)"
    )

    def evaluate() -> float:
        totals = [
            run_episode(game, actions, enc, model, epsilon=0.0, learn=False, frameskip=args.frameskip, gamma=args.gamma, rng=rng)
            for _ in range(args.eval_episodes)
        ]
        return sum(totals) / len(totals)

    t0 = time.time()
    print(f"[eval @0] mean={evaluate():+.1f} (random baseline)")
    for ep in range(1, args.episodes + 1):
        eps = linear_epsilon(ep, eps_start=args.eps_start, eps_end=args.eps_end, decay_episodes=decay)
        run_episode(game, actions, enc, model, epsilon=eps, learn=True, frameskip=args.frameskip, gamma=args.gamma, rng=rng)
        if ep % args.eval_every == 0:
            print(
                f"[eval @{ep}] mean={evaluate():+.1f} eps={eps:.2f} "
                f"rss={rss_mb():.1f}MiB t={time.time() - t0:.0f}s"
            )

    game.close()


if __name__ == "__main__":
    main()
