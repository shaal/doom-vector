"""Record a watchable clip of the trained reactive value-vote agent.

Trains on a scenario (default deadly_corridor) with the navigation encoder, then
runs greedy episodes at a higher resolution, grabbing RGB frames headlessly
(no display needed — the nav encoder uses game variables, not pixels, so we can
render at a nice resolution just for the video) and saves the best episode as an
animated GIF.

    python experiments/record_demo.py --scenario deadly_corridor --episodes 150 --out demo.gif
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import vizdoom as vzd  # noqa: E402

from brain.encoder import make_encoder  # noqa: E402
from brain.memory.experience_store import ExperienceStore  # noqa: E402
from brain.policy.episodic import choose_action, discounted_returns, linear_epsilon  # noqa: E402
from envs.basic import discrete_actions, make_game  # noqa: E402


def to_hwc(buf) -> np.ndarray:
    img = np.asarray(buf)
    if img.ndim == 3 and img.shape[0] == 3:  # CHW -> HWC
        img = img.transpose(1, 2, 0)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="deadly_corridor")
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--record-episodes", type=int, default=4)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--frameskip", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--capacity", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="demo.gif")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--downscale", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=300)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    game = make_game(
        args.scenario,
        resolution=vzd.ScreenResolution.RES_320X240,
        grayscale=False,
        labels=True,
        position=True,
    )
    actions = discrete_actions(game)
    enc, dim = make_encoder("navigation", game)
    store = ExperienceStore(
        dim=dim,
        storage_path=os.path.join(tempfile.gettempdir(), f"demo_{os.getpid()}.rvf"),
        capacity=args.capacity,
    )
    decay = int(args.episodes * 0.7)

    # ---- train (reactive value-vote) ----
    for ep in range(1, args.episodes + 1):
        eps = linear_epsilon(ep, eps_start=1.0, eps_end=0.05, decay_episodes=decay)
        game.new_episode()
        traj = []
        while not game.is_episode_finished():
            s = game.get_state()
            v = enc(s)
            a = choose_action(store.search(v, args.k), len(actions), epsilon=eps, rng=rng)
            r = game.make_action(actions[a], args.frameskip)
            traj.append((v, a, r))
        rets = discounted_returns([r for _, _, r in traj], args.gamma)
        for (v, a, _), g in zip(traj, rets):
            store.insert(v, {"action_idx": float(a), "return": float(g)})
        if ep % 25 == 0:
            print(f"trained {ep}/{args.episodes}", flush=True)

    # ---- record greedy episodes, keep the best ----
    best = None
    for i in range(args.record_episodes):
        game.new_episode()
        frames = []
        while not game.is_episode_finished():
            s = game.get_state()
            if s is None:
                break
            if len(frames) < args.max_frames:
                img = to_hwc(s.screen_buffer)
                frames.append(img[:: args.downscale, :: args.downscale])
            a = choose_action(store.search(enc(s), args.k), len(actions), epsilon=0.0, rng=rng)
            game.make_action(actions[a], args.frameskip)
        total = game.get_total_reward()
        print(f"greedy ep {i}: reward={total:+.1f} frames={len(frames)}", flush=True)
        if best is None or total > best[0]:
            best = (total, frames)
    game.close()

    total, frames = best
    if not frames:
        print("no frames captured")
        return
    imageio.mimsave(args.out, frames, fps=args.fps)
    print(f"saved {args.out} ({os.path.getsize(args.out)/1024:.0f} KiB, {len(frames)} frames, reward={total:+.1f})", flush=True)


if __name__ == "__main__":
    main()
