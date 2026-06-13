# Doom-Vector: Teaching Skills & Showcasing RuVector

**Status:** Plan / queued for implementation
**Date:** 2026-06-12
**Goal:** Push the agent from "learns toy scenarios" toward real Doom competence — **aim, dodge, and (north star) finish a level** — while turning the repo into a **plain-language explainer of what RuVector can do**. Each work track teaches the agent one skill *and* lights up one RuVector feature we currently leave on the table.

This plan covers four tracks the maintainer selected, sequenced by dependency:

1. **Aim** — learn to shoot on `defend_the_center` (showcases *reward shaping* + *metadata-filtered recall*).
2. **Capacity** — wire *quantization* into the bridge so the Pi holds 4–32× more experience (the foundational "maximize" lever).
3. **Dodge** — learn to avoid damage on `take_cover` (showcases *recall-uncertainty as a safety signal*).
4. **Explainer** — a repo doc that uses the agent to teach each RuVector feature in everyday language.

The north star past these four is **whole-level completion**, which needs everything here *plus* hierarchy/waypoints and progress shaping — scoped at the end as the follow-on, not built in this plan.

---

## 0. Where we are (the substrate these tracks build on)

The winning recipe from §9 of the [self-learning plan](doom-vector-self-learning-agent.md) is **reactive value-weighted per-step recall**: encode the frame → k-NN recall from RuVector → vote for the action whose recalled neighbours had the best return-to-go → store the outcome. No gradients. It learns `basic`, `health_gathering`, and `deadly_corridor`, on the real 32-bit Pi Zero 2 W, within ~51 MiB.

What the bridge (`bridge/ruvector_py/src/lib.rs`) exposes today: `insert`, `search`, `delete`, plus `WorldModel.observe/plan`. Two things to notice for the tracks below:

- `search` already constructs `SearchQuery { vector, k, filter: None, ef_search: None }` (`lib.rs:72`). **`filter` and `ef_search` are present in the 2.2.0 API and merely hardcoded off** — Track 1 turns `filter` on.
- Metadata is stored as `serde_json::Value` (`lib.rs:61`) but only floats are read back. Filtering on float metadata is therefore feasible without changing the write path.

The honest constraint from prior phases still rules everything: **recall is only as good as the encoder.** `basic` worked because the structured encoder carried the decisive variable (monster geometry); `health_gathering` only worked once HEALTH was added. So every skill track below leads with "what must the agent perceive," not "what algorithm."

---

## 1. RuVector feature → Doom capability (the explainer seed)

This table is both the design key for the tracks and the spine of the Track-4 doc.

| RuVector feature | Plain meaning | Doom payoff | Status |
|---|---|---|---|
| HNSW k-NN `search` | "What did I do last time it looked like this?" | The whole policy | **used** |
| Float metadata (`action_idx`, `return`) | "...and how did that turn out, all things considered?" | Credit assignment | **used** |
| `delete` eviction | "Forget my worst outcomes when memory's full" | Bounded RAM on the Pi | **used** |
| `WorldModel.plan` | "Imagine one move ahead" | Stepping stone to planning | **used** |
| `SearchQuery.filter` | "Only recall *relevant* memories" | Aim: recall only enemy-visible moments | **Track 1** |
| Quantization (int8/int4/binary) | "Compress memories, keep 4–32× more" | Coverage for big maps | **Track 2** |
| Recall distance/spread → uncertainty | "Know when I'm somewhere unfamiliar" | Dodge: when unsure, play safe | **Track 3** |
| `ef_search` knob | "Trade recall quality vs. speed" | Free tuning dial | Track 1 (incidental) |
| MMR / diverse recall | "Get a spread of advice, not 16 echoes" | Robust votes, better eviction | follow-on |
| Hybrid dense+sparse | "Match on looks AND on tags" | Symbolic + visual recall | follow-on |

---

## 2. Track 1 — Aim (`defend_the_center`)

**Why this scenario:** the agent stands still and enemies close in from all sides. It isolates aiming from navigation — the cleanest place to prove "learns to shoot."

**RuVector feature showcased:** *metadata-filtered recall.* The aim policy should be advised only by moments where an enemy was actually on screen; corridor-empty memories shouldn't dilute the trigger decision.

**What the agent must perceive (encoder).** Extend the structured encoder (`brain/encoder/structured.py`) with explicit aim signals — recall can't learn "pull the trigger when lined up" if alignment isn't a dimension:
- horizontal offset of the nearest enemy from screen centre (≈0 ⇒ on target),
- nearest-enemy distance,
- enemy-visible flag (also used as the filter key, below).

**What counts as good (reward).** Stop relying on kill-only reward; it's too sparse to shape aiming. ViZDoom exposes `HITCOUNT` / `DAMAGECOUNT` game variables — add a small per-step bonus for damage just dealt, in `experiments/train.py` (the reward post-processing already lives there). Dense "you hit something" feedback arrives many tics before a kill.

**The RuVector change (bridge).** Expose `filter` (and, free alongside, `ef_search`) on `search`:
- `bridge/ruvector_py/src/lib.rs:71-72` — accept an optional `filter: Option<HashMap<String, f64>>` and `ef_search: Option<usize>`, pass them into `SearchQuery` instead of `None`.
- Confirm the 2.2.0 `SearchQuery.filter` value type and equality semantics first (`cargo doc` on the installed crate) — the survey says HashMap-equality, but verify before wiring.
- `brain/memory/experience_store.py` — thread an optional `filter=` through `search`.
- `brain/policy/episodic.py` — when an enemy is visible, recall with `filter={"enemy_visible": 1.0}`; otherwise recall unfiltered.

**Success:** greedy-eval kill count on `defend_the_center` climbs clearly above random, and a recorded GIF shows the agent tracking and dropping enemies. Compare filtered vs. unfiltered recall as a mini-ablation (this becomes a figure in the Track-4 doc).

**Pi note:** filter + `ef_search` are pure index-side; no RAM cost. Keep `k` small.

---

## 3. Track 2 — Capacity via quantization

**Why now:** this is the "how do I maximize it" lever, and it's the prerequisite for the full-level north star. A single room fits in 20k experiences; a whole map needs far more state coverage. Quantization buys that coverage *in the same 512 MB*: int8 ≈ 4×, int4 ≈ 8×, binary ≈ 32×, at the cost of approximate distances.

**RuVector feature showcased:** *vector quantization* (scalar/int4/product/binary), reportedly present in `ruvector-core` 2.2.0 but **not yet reachable through our bridge**.

**First step is a feasibility gate (honest unknown).** Confirm quantization is actually configurable via `DbOptions`/`VectorDB` in the installed 2.2.0 — `cargo doc` + a Rust smoke test — *before* planning around it. Prior phases were burned by assuming 0.x/2.2.0 surface; treat this the same way. If 2.2.0 doesn't expose it cleanly, record that as a negative result and fall back to a smaller embedding dim / tighter cap.

**If reachable — the change:**
- `bridge/ruvector_py/src/lib.rs:34-44` — add a `quantization` option to `DbOptions` in `RuVectorMemory::new` (and `WorldModel`).
- Plumb a `quantization=` kwarg through `brain/memory/experience_store.py`.
- **Benchmark on the Seed** (the only authoritative judge, per the doc's Tier-3 discipline): for each scheme, measure RSS-per-experience, recall latency, and — critically — whether learning still converges on `deadly_corridor`/`health_gathering` (approximate distance can degrade the value vote). Report the capacity ceiling each scheme unlocks under 512 MB. Watch the dim-8 nav encoder: at very low dim, int4/binary may distort the vote more than they save.

**Success:** a table (mirroring §9's RSS/throughput tables) showing experiences-held and recall-latency per scheme on the A53, and confirmation that at least int8 preserves learning. That table is the headline evidence for "we can scale to a full level."

---

## 4. Track 3 — Dodge (`take_cover`)

**Why this scenario:** `take_cover` gives the agent no weapon and incoming fireballs — it isolates "avoid getting shot" from everything else.

**RuVector feature showcased:** *recall uncertainty as a safety signal.* We already get neighbour **distances** back from `search`; we don't have to add native conformal prediction to benefit. When the nearest neighbours are far away or their action-votes disagree, the agent is in unfamiliar territory — and in a dodge task, the right default under uncertainty is *evade*, not freeze.

**What the agent must perceive (encoder).** A threat-aware encoder (new `brain/encoder/threat.py` or an extension of navigation):
- nearest projectile/enemy relative position and distance (ViZDoom labels fireballs in this scenario),
- **Δhealth** — damage taken this step. Prior phases proved HEALTH is decisive for survival; the *change* in it is the dodge signal.

**What counts as good (reward).** Penalize health loss per step in `experiments/train.py`, not just death. Dodging is continuous and needs continuous feedback.

**The mechanism (policy, mostly Python — cheap and honest).** In `brain/policy/episodic.py`, derive an uncertainty score from the recall already returned (e.g. mean neighbour distance, or vote entropy across actions). When uncertainty is high: bias toward an evasive default (strafe away from the nearest threat) and raise exploration locally. No bridge change required for the first cut.
- *Stretch:* expose `ruvector-core`'s native conformal-prediction path through the bridge for a calibrated uncertainty, and compare against the cheap distance heuristic. Only if the heuristic proves too noisy.

**Success:** greedy-eval survival time on `take_cover` rises above random; a GIF shows the agent sidestepping fireballs; the uncertainty-gated safe-fallback measurably reduces deaths vs. the same policy without it (ablation → Track-4 figure).

---

## 5. Track 4 — The explainer doc

**Goal:** make the repo teach RuVector. One short doc (`docs/ruvector-by-example.md`, linked from the README) that walks the §1 table top to bottom, and for each feature gives: the everyday-language meaning, the one-line Doom payoff, the exact call site in this repo, and — where Tracks 1–3 produced one — a GIF or ablation figure as proof.

**Structure:**
- *The one idea:* the agent has no neural net; its "brain" is a memory of past moments, and RuVector is that memory. (Reuse the framing from §1 of the self-learning plan.)
- *Per feature:* recall → metadata → eviction → filtered recall → quantization → uncertainty → (teasers for MMR/hybrid). Each anchored to a real call site (`experience_store.py`, `episodic.py`, `lib.rs`) so a reader can jump from concept to code.
- *Per scenario as a showcase:* `basic` = recall+metadata; `defend_the_center` = filtered recall; `take_cover` = uncertainty; `deadly_corridor`/`health_gathering` = eviction at scale; quantization bench = capacity.

**Why last:** it harvests the GIFs, ablations, and benchmark tables that Tracks 1–3 generate, so it ends up evidence-backed rather than aspirational. Draft the skeleton early, fill figures as each track lands.

---

## 6. Sequencing & dependencies

```
Track 1 (Aim)        ── bridge: expose filter/ef_search ─┐
                                                          ├─► Track 4 (Explainer: harvests figures)
Track 2 (Capacity)   ── bridge: expose quantization ─────┤
                                                          │
Track 3 (Dodge)      ── policy: uncertainty (no bridge) ─┘
```

- **Track 1 first** — smallest change, most visible payoff, and it establishes the filtered-recall + reward-shaping pattern the others reuse.
- **Track 2 second** — independent infra; gated on the quantization-feasibility check; unblocks the full-level north star.
- **Track 3 third** — mostly Python; reuses Track 1's encoder/reward scaffolding.
- **Track 4 throughout** — skeleton early, figures as they arrive.

Each track is a curriculum rung: `basic` ✓ → **`defend_the_center`** → **`take_cover`** → `deadly_corridor` ✓ → `my_way_home` ✓ → **full level**.

---

## 7. Risks & unknowns (tracked, in the doc's tradition)

| Risk | Why it matters | Mitigation |
|---|---|---|
| `SearchQuery.filter` value type/semantics in 2.2.0 unconfirmed | Track 1 bridge change depends on it | `cargo doc` + Rust smoke test before wiring; fall back to Python-side post-filter of `k`-NN results if the native filter is awkward |
| Quantization may not be reachable via `DbOptions` in 2.2.0 | Track 2 hinges on it | Treat as a gate; record a negative result and fall back to smaller dim / tighter cap if absent |
| Approximate distance can break the value vote | int4/binary may distort recall at dim 8 | Benchmark convergence per scheme on the Seed; keep int8 as the conservative default |
| Reward shaping can be gamed | Hit-bonus might encourage spray, damage-penalty might encourage cowering | Keep shaping small relative to scenario reward; eval on the *unshaped* scenario score |
| Encoder bloat raises dim/RAM | More features → bigger vectors on a 512 MB device | Keep aim/threat additions to a handful of dims; lean on Track-2 quantization |
| Full-level credit assignment | Episodic control is weak over long horizons | Out of scope here; the north-star section flags hierarchy + progress shaping as the next plan |

---

## 8. North star (follow-on, not in this plan): finish a level

These four tracks make the agent a competent reactive fighter that perceives threats and scales its memory. A *whole level* (find key → cross map → reach exit) additionally needs:
- **Capacity** (Track 2 delivers the substrate),
- **Hierarchy** — a coarse waypoint/subgoal memory over the fine reactive memory, graduating `WorldModel.plan` from 1-step to subgoal-chaining,
- **Progress shaping** — reward new-area-discovered / distance-to-exit so credit propagates over thousands of steps,
- **Curriculum** to a full `freedoom`/`doom2.wad` map via a custom `.cfg`.

That's the next plan. This one earns the right to attempt it.
