"""Headless, low-res ViZDoom environment factories.

Defaults are tuned for the Raspberry Pi Zero 2 W: lowest resolution (160x120),
single-channel grayscale, no window. The `basic` scenario is the simplest
ViZDoom task (3 buttons: move left / move right / attack; reward for hitting a
stationary monster) and is our Phase 0 proving ground.
"""
from __future__ import annotations

import os

import vizdoom as vzd


def make_game(
    scenario: str = "basic",
    *,
    visible: bool = False,
    resolution: "vzd.ScreenResolution" = vzd.ScreenResolution.RES_160X120,
    grayscale: bool = True,
    labels: bool = False,
    position: bool = False,
) -> "vzd.DoomGame":
    """Create and `init()` a configured DoomGame for a built-in scenario.

    Args:
        scenario: ships-with-vizdoom scenario name (basic, defend_the_center,
            health_gathering, deadly_corridor, my_way_home, ...).
        visible: render a window (False = headless, required on the Pi).
        resolution: screen resolution; 160x120 is the lightest.
        grayscale: GRAY8 (1 channel) instead of RGB24 (3 channels).
        labels: enable the semantic labels buffer (needed by the structured/nav encoders).
        position: expose POSITION_X/Y + ANGLE (needed by the navigation encoder).
    """
    game = vzd.DoomGame()
    # Sets buttons, living/death reward, the .wad, etc.
    game.load_config(os.path.join(vzd.scenarios_path, f"{scenario}.cfg"))
    game.set_window_visible(visible)
    game.set_screen_resolution(resolution)
    game.set_screen_format(
        vzd.ScreenFormat.GRAY8 if grayscale else vzd.ScreenFormat.RGB24
    )
    game.set_mode(vzd.Mode.PLAYER)  # synchronous: engine waits for the agent
    if labels:
        game.set_labels_buffer_enabled(True)
    if position:
        for gv in (vzd.GameVariable.POSITION_X, vzd.GameVariable.POSITION_Y, vzd.GameVariable.ANGLE):
            game.add_available_game_variable(gv)
    game.init()
    return game


def make_basic_game(**kwargs) -> "vzd.DoomGame":
    """Backwards-compatible shortcut for the `basic` scenario."""
    return make_game("basic", **kwargs)


def discrete_actions(game: "vzd.DoomGame") -> list[list[int]]:
    """All binary button combinations as a discrete action set."""
    import itertools

    n = game.get_available_buttons_size()
    return [list(combo) for combo in itertools.product([0, 1], repeat=n)]
