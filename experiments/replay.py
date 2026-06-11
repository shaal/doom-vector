"""Replay a recorded episode (.lmp) in a visible window — the watchable demo.

Record on the Pi (headless) with `train.py --record demo.lmp`, copy the .lmp
back, and replay it here with the engine rendering a window:

    python experiments/replay.py demo.lmp --scenario basic

The scenario must match the one the episode was recorded on.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.basic import make_game  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("lmp", help="path to the recorded .lmp file")
    ap.add_argument("--scenario", default="basic")
    args = ap.parse_args()

    game = make_game(args.scenario, visible=True, grayscale=False)
    game.replay_episode(args.lmp)
    while not game.is_episode_finished():
        game.advance_action()
    print(f"replay reward={game.get_total_reward():+.1f}")
    game.close()


if __name__ == "__main__":
    main()
