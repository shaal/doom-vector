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


def make_encoder(kind: str, game, *, thumb: int = 16, max_objects: int = 8):
    if kind == "navigation":
        return make_nav_encoder(game, max_objects=max(1, max_objects // 8))

    if kind == "thumbnail":
        dim = thumb * thumb

        def enc(state):
            return encode_thumbnail(state.screen_buffer, size=thumb)

        return enc, dim

    if kind == "structured":
        n_vars = game.get_available_game_variables_size()
        dim = structured_dim(n_vars, max_objects)

        def enc(state):
            return encode_structured(state, n_vars, max_objects)

        return enc, dim

    raise ValueError(f"unknown encoder: {kind!r} (use 'thumbnail' or 'structured')")
