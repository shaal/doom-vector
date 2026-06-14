"""Encoder factory: build a (encode_fn, dim) pair for a given game.

  - "thumbnail": cheap, scenario-agnostic grayscale thumbnail (pixels).
  - "structured": game variables + labeled-object features (no pixels).
    Requires the labels buffer enabled on the game; lighter and usually a
    stronger recall signal for combat scenarios like `basic`.
  - "navigation": agent pose (position + heading) + nearest object. The right
    key for path prediction; requires the game created with position=True
    (and labels=True for object features).
"""
from __future__ import annotations

from .navigation import make_nav_encoder
from .structured import encode_structured, structured_dim
from .thumbnail import encode_thumbnail


def make_encoder(
    kind: str, game, *, thumb: int = 16, max_objects: int = 8, aim: bool = False, dodge: bool = False
):
    if kind == "navigation":
        return make_nav_encoder(game, max_objects=max(1, max_objects // 8))

    if kind == "thumbnail":
        dim = thumb * thumb

        def enc(state):
            return encode_thumbnail(state.screen_buffer, size=thumb)

        return enc, dim

    if kind == "structured":
        n_vars = game.get_available_game_variables_size()
        dim = structured_dim(n_vars, max_objects, aim=aim, threat=dodge)
        # Capture the true screen width so the aim/threat dx-to-target is centred
        # at 0 for this resolution (the structured block's own normalizer is a
        # fixed 320 "screen-ish" constant; aim/threat need the real centre).
        screen_w = float(game.get_screen_width()) if (aim or dodge) else 320.0

        if not dodge:
            def enc(state):
                return encode_structured(state, n_vars, max_objects, aim=aim, screen_w=screen_w)

            return enc, dim

        # Dodge mode needs the *change* in health (damage taken this step), which a
        # single GameState can't carry — so the encoder is made stateful: it reads
        # HEALTH via the game handle (index-independent, like the nav encoder) and
        # remembers it across calls. The clamp in threat_features turns the episode
        # reset (health jumps back to full) into dhealth=0, so no per-episode reset
        # bookkeeping is needed here. Must be called once per step, in order.
        import vizdoom as vzd  # lazy: only the runtime path needs the engine

        prev = {"health": None}

        def enc(state):
            now = game.get_game_variable(vzd.GameVariable.HEALTH)
            last = prev["health"]
            dhealth = 0.0 if last is None else (last - now)
            prev["health"] = now
            return encode_structured(
                state, n_vars, max_objects, aim=aim, threat=True, dhealth=dhealth, screen_w=screen_w
            )

        return enc, dim

    raise ValueError(f"unknown encoder: {kind!r} (use 'thumbnail' or 'structured')")
