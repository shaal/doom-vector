"""Tier 0/2 spike: the full episodic-control loop on ViZDoom `basic`.

perceive -> encode -> recall -> act -> (episode end) store returns. Uses the
RuVector native backend if `ruvector_py` is importable, else a NumPy fallback,
so it runs before the binding is built. This is the loop we measure on the Pi.

    python experiments/spike_memory_loop.py [--episodes N] [--epsilon E]
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain.encoder.thumbnail import encode_thumbnail  # noqa: E402
from brain.memory.experience_store import ExperienceStore  # noqa: E402
from brain.policy.episodic import choose_action, discounted_returns  # noqa: E402
from envs.basic import discrete_actions, make_basic_game  # noqa: E402

THUMB = 16  # thumbnail size -> dim = THUMB*THUMB


def rss_mb() -> float:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    except FileNotFoundError:
        pass
    return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--epsilon", type=float, default=0.3)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--frameskip", type=int, default=4)
    ap.add_argument(
        "--store",
        default=None,
        help="RuVector store path (native backend). Default: a fresh per-run temp "
        "path so spikes start clean. Pass a stable path (e.g. ./ruvector_store) to "
        "keep learning across runs.",
    )
    args = ap.parse_args()

    # The native backend persists to a redb file at this path (relative paths are
    # CWD-relative). Default to a unique temp path so repeated spike runs don't
    # silently reuse or collide on an old store.
    store_path = args.store or os.path.join(tempfile.gettempdir(), f"dv_store_{os.getpid()}.rvf")

    game = make_basic_game()
    actions = discrete_actions(game)
    store = ExperienceStore(dim=THUMB * THUMB, storage_path=store_path)
    where = store_path if store.backend == "native" else "in-memory"
    print(f"backend={store.backend} store={where} actions={len(actions)} rss={rss_mb():.1f}MiB")

    for ep in range(args.episodes):
        game.new_episode()
        traj: list[tuple] = []  # (vector, action_idx, reward)
        while not game.is_episode_finished():
            state = game.get_state()
            vec = encode_thumbnail(state.screen_buffer, size=THUMB)
            neighbors = store.search(vec, k=args.k)
            a = choose_action(neighbors, len(actions), epsilon=args.epsilon)
            r = game.make_action(actions[a], args.frameskip)
            traj.append((vec, a, r))

        # backfill discounted returns and write the episode into memory
        rets = discounted_returns([r for _, _, r in traj])
        for (vec, a, _), g in zip(traj, rets):
            store.insert(vec, {"action_idx": float(a), "return": float(g)})

        print(f"ep {ep}: reward={game.get_total_reward():+.1f} mem={len(traj)}+ rss={rss_mb():.1f}MiB")

    game.close()


if __name__ == "__main__":
    main()
