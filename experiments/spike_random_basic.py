"""Tier 0 spike: ViZDoom `basic` headless with a random agent.

Proves the environment installs and runs, that we can read frames + reward, and
reports peak RSS so we have a desktop baseline before the Pi. Run from repo root:

    python experiments/spike_random_basic.py [--episodes N]
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.basic import discrete_actions, make_basic_game  # noqa: E402


def rss_mb() -> float:
    """Resident set size of this process, in MiB (Linux /proc)."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    except FileNotFoundError:
        pass
    return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--frameskip", type=int, default=4)
    args = ap.parse_args()

    game = make_basic_game()
    actions = discrete_actions(game)
    print(f"buttons={game.get_available_buttons_size()} actions={len(actions)} rss={rss_mb():.1f}MiB")

    for ep in range(args.episodes):
        game.new_episode()
        steps = 0
        while not game.is_episode_finished():
            _ = game.get_state()  # screen_buffer, game_variables, ...
            game.make_action(random.choice(actions), args.frameskip)
            steps += 1
        print(f"ep {ep}: reward={game.get_total_reward():+.1f} steps={steps} rss={rss_mb():.1f}MiB")

    game.close()


if __name__ == "__main__":
    main()
