# Doom-Vector: North Star — Finish Full Levels, and *Show* the Self-Improvement

**Status:** Plan / queued for implementation

**Goal:** Take the agent from "wins toy scenarios" to **playing and finishing a full Doom level** — starting with **Freedoom `MAP01`** — and, just as importantly, **make the improvement legible**: a dashboard that shows the agent getting *faster, taking less damage, killing more, finding more secrets, and dying less* across training. The harness is **map-agnostic from day one** (map01 is the first target, not a hardcode), so any WAD/level can be dropped in later.

This plan harvests the §8 north-star teaser of [the skills & features plan](doom-vector-skills-and-features.md) and the architecture of [the self-learning plan](doom-vector-self-learning-agent.md), and turns the sketch into shippable tracks. Track 1 (the dashboard) ships **first and standalone** — it's a visible win even before full-level mastery, and it becomes the measuring stick every later track is judged by.

---

## 0. Where we are (the substrate this builds on)

The agent is a reactive episodic-control policy: encode frame → k-NN recall from RuVector → value-vote → store (`brain/policy/episodic.py`). Tracks 1–3 added filtered recall + MMR (aim), eviction-at-scale (capacity), and uncertainty-gated evade (dodge). It learns `basic`, `defend_the_center`, `take_cover`, `deadly_corridor`, `health_gathering` on a real Pi Zero 2 W.

Two gaps stand between that and a full level:

1. **No map awareness or planning.** The encoder has no notion of *where* it is or *where the exit is*; `WorldModel.plan` (`bridge/ruvector_py/src/world_model.rs:121`) looks exactly **one** step ahead. Reactive recall is weak over the thousands of steps a level takes.
2. **No full-level harness or metrics.** Eval measures **reward only** — `evaluate()` (`experiments/train.py:238-259`) averages `game.get_total_reward()` (`train.py:161`) over greedy episodes. There is no completion signal, no time/damage/kills/secrets/deaths tracking, nothing to *show*.

The honest constraint from every prior phase still rules: **recall is only as good as the encoder.** So the full-level tracks lead with "what must the agent perceive," not "what algorithm."

---

## 0.5 Feasibility gate (run FIRST — the repo's verify-before-building tradition)

Like §0.5 of the skills plan, settle these against the real env/crate before committing track designs. Each gate amends the tracks below.

**Gate A — Metrics are actually populated.** ViZDoom exposes the metrics we want as `GameVariable`s, but stock scenario `.cfg`s only *declare* a few. Confirm, on both a toy scenario and Freedoom `MAP01`, that these read non-trivially after we add them to `available_game_variables`: `KILLCOUNT`, `ITEMCOUNT`, `SECRETCOUNT`, `DAMAGECOUNT` (damage *dealt*), `DAMAGE_TAKEN`, `DEATHCOUNT`, `ARMOR`, `HEALTH`. Access is `vzd.GameVariable.X` (the env already uses `HEALTH`/`DAMAGECOUNT`/`POSITION_*` — `envs/basic.py:48-54`, `train.py:113,143`). Decide where total-secrets / total-items per map come from (engine doesn't report the denominator — likely a per-map constant captured once).

**Gate B — Freedoom MAP01 loads and signals completion.** No WAD ships in the repo today; `make_game` loads `vzd.scenarios_path/{scenario}.cfg` (`envs/basic.py:37`). Acquire **`freedoom2.wad`** (freely distributable; `MAP01` is its first map), write a custom `cfg/freedoom_map01.cfg` (set `doom_game_path`/`doom_scenario_path`, `doom_map = MAP01`, a generous `episode_timeout`, `available_game_variables`, `screen_*`), and teach `make_game` to accept a `--wad`/`--map`/`--cfg` path (not just a built-in scenario name). Critically, confirm **how "reached the exit" surfaces** — typically `is_episode_finished()` with `HEALTH > 0` and `time < timeout` = completed; death or timeout = not. Verify the exit actually ends the episode on this map.

**Gate C — Capacity for a whole map.** Track 2 (§3.1 of the skills plan) measured ~20k–50k experiences for a *single room* at dim 8 (≈1.5 KiB RSS/vector). A full map needs far more state coverage. Estimate the experience budget for `MAP01` and confirm it fits the Pi's 512 MB with the load-bearing levers (dim, `max_elements`, eviction) — quantization is inert in 2.2.0, so there is no 4–32× safety net.

**Gate D — Waypoint API reachability.** Confirm `graph_rag::KnowledgeGraph` (`get_neighbors`, `local/global_search`) in ruvector-core 2.2.0 is usable from the bridge (or reimplementable Python-side, like MMR was). If it's stubbed like quantization, Track 3 falls back to a hand-rolled place-graph in Python.

---

## 1. The metric map (what "self-improvement" means here)

This table is the spine of Track 1 and the scoreboard for the whole plan.

| Metric | Plain meaning | Source | Better = |
|---|---|---|---|
| **Completion** | did it reach the exit? | derived: episode-finished ∧ `HEALTH>0` ∧ not-timeout (Gate B) | more |
| **Time-to-exit** | how fast | episode length in tics → seconds (÷35), vs a reference par | lower |
| **Damage taken** | how much it got hurt | `DAMAGE_TAKEN` (or Δ`HEALTH`+Δ`ARMOR`) | lower |
| **Kills** | enemies cleared | `KILLCOUNT` | higher |
| **Secrets found** | exploration | `SECRETCOUNT` (denominator from Gate A) | higher |
| **Items** | pickups | `ITEMCOUNT` | higher |
| **Deaths** | failure rate | `DEATHCOUNT` / death-on-episode | lower |
| **Progress** | distance-to-exit / new area | derived from `POSITION_*` + a visited-grid (Track 2) | higher |

The **self-improvement story** is each of these plotted **across training episodes**: episode 1 vs episode N. That curve — not a single number — is the deliverable.

---

## 2. Track 1 — Full-level harness & self-improvement dashboard  *(ships FIRST, standalone)*

**Why first:** it's the lowest-risk piece, it's independent of the hard encoder/planning work, and it gives an immediately *showable* artifact — even on today's toy scenarios, before any full-level mastery exists. Everything after is measured by it.

**Build:**
- **Per-episode metrics, not just reward.** Extend `run_episode` (`train.py:88-161`) to collect the §1 metrics into a per-episode record (read the Gate-A game variables at episode end; track tics, death, completion), and have `evaluate()` (`train.py:238-259`) aggregate the record list instead of a scalar mean.
- **A `--metrics` / dashboard mode.** Emit a reproducible artifact in the repo's existing tradition (markdown tables + simple figures, like §3.1 / the dodge ablation): per-episode and binned-over-training curves for each metric, plus a **par/reference comparison** for time. One command produces it.
- **Map-agnostic from the start.** It already keys on `--scenario`; add `--wad`/`--map`/`--cfg` (Gate B) so the *same* dashboard runs on any built-in scenario **and** on Freedoom `MAP01`. map01 is a default target, not a special case.

**Success (Done when):** running the dashboard on at least one scenario produces a reproducible figure showing metrics **improving across training** (e.g. damage↓, kills↑), and the identical harness runs end-to-end on Freedoom `MAP01` (Gate B) — even if early map01 numbers are poor. This is the demo: *"watch it get better."*

---

## 3. Track 2 — Map-aware encoder  *(the real bottleneck)*

**Perceive where you are.** Add **position, heading, and a coarse visited-grid** to the structured encoder so recall can distinguish "this room" from "that room." `POSITION_X/Y` + `ANGLE` are already read in the navigation encoder (`brain/encoder/navigation.py:36-41`); the new dims slot into `make_encoder` (`brain/encoder/__init__.py:18-67`, `structured_dim` at `structured.py:40-48`). Keep dims tight — every dim is bytes × N-experiences (Track 2's capacity lesson).

**Validate on `my_way_home` first** (a navigation scenario) before full maps, using Track 1's progress/time metrics. **Success:** the progress and time-to-goal metrics on `my_way_home` improve clearly over a no-position baseline.

---

## 4. Track 3 — Waypoint memory + subgoal chaining

**A map of places, not just moments.** Layer a coarse **topological waypoint graph** (`graph_rag::KnowledgeGraph`, Gate D — or a Python place-graph) *over* the fine experience store: place-nodes joined by "leads-to" edges. Turn one-step `WorldModel.plan` into **subgoal chaining** — plan to the next waypoint, then let reactive recall (Tracks 1–3 of the skills plan) drive to it. Extend `plan` toward multi-step rollouts to the current subgoal.

**Success:** on Freedoom `MAP01`, the agent reaches the exit at a rate **measurably above** a reactive-only baseline (Track 1 completion metric).

---

## 5. Track 4 — Progress shaping + capacity for long horizons  *(throughout)*

**Credit over thousands of steps.** Reactive episodic control is weak over long horizons, so add **progress shaping**: small dense rewards for new-area-discovered, distance-to-exit-reduced, and next-waypoint-reached (mirrors Track 1/3 shaping in `brain/policy/reward.py`; eval always reports the *unshaped* completion/time so shaping can't be gamed). Pair with a **capacity budget** for `MAP01` (Gate C): dim discipline + `max_elements` + eviction quality, plus periodic compaction for the tombstone-creep found in skills §3 (the lifetime-insert ceiling).

**Success:** across training on `MAP01`, completion-time and deaths improve (Track 1 curves) while RSS stays within the Pi budget.

---

## 6. Sequencing & dependencies

```
Track 1 (Dashboard)  ── harness + metrics + map-agnostic loader ──┐  ships FIRST, standalone
                                                                   │
Track 2 (Map encoder) ── position/heading/visited-grid ───────────┤  (validate on my_way_home)
                                                                   ├─► Full-level play on MAP01,
Track 3 (Waypoints)   ── graph + subgoal chaining ────────────────┤     measured by Track 1
                                                                   │
Track 4 (Shaping/cap) ── long-horizon credit + capacity ──────────┘  (throughout)
```

- **Track 1 first** — independent, low-risk, and it's the scoreboard for everything else.
- **Track 2 before Track 3** — waypoints are useless if the encoder can't tell places apart.
- **Track 4 throughout** — shaping and capacity get tuned as 2 and 3 land.

Each track is its own shippable PR, in the §6.5 order, exactly like the skills plan.

---

## 6.5 Tasks (shipyard checklist)

Run with `shipyard N --tasks docs/plans/doom-vector-full-level-northstar.md`. One checkbox per track, in dependency order.

- [ ] **Gate — Feasibility (§0.5).** depends: none. Run Gates A–D; record results (amend tracks; honest negatives are fine, per repo tradition). Done when: A–D answered with evidence, the Freedoom `MAP01` cfg loads, and the completion signal is confirmed.
- [ ] **Track 1 — Dashboard & full-level harness (§2).** depends: Gate (A, B). Per-episode metrics + `--metrics` dashboard + `--wad/--map/--cfg` loader. Done when: a reproducible improvement-over-training figure exists for ≥1 scenario AND the harness runs end-to-end on Freedoom `MAP01`.
- [ ] **Track 2 — Map-aware encoder (§3).** depends: Track 1. position/heading/visited-grid dims. Done when: `my_way_home` progress/time metrics beat a no-position baseline.
- [ ] **Track 3 — Waypoints + subgoal chaining (§4).** depends: Track 2, Gate D. graph place-memory + multi-step plan. Done when: `MAP01` exit-reach rate beats a reactive-only baseline.
- [ ] **Track 4 — Progress shaping + capacity (§5).** depends: Track 1 (metrics), composes with 2–3. Done when: `MAP01` completion-time/deaths improve across training within the Pi RAM budget.

---

## 7. Risks & unknowns (tracked, in the doc's tradition)

- **Encoder is the bottleneck, not the graph** (skills §8). If position/heading/visited-grid don't separate places well, waypoints won't help — Track 2 must prove out on `my_way_home` first.
- **Long-horizon credit assignment.** Episodic control degrades over thousands of steps; Track 4's progress shaping is the mitigation, but it's unproven at full-map scale.
- **Capacity with no quantization.** A full map may need 100k+ experiences; the only levers are dim/cap/eviction (skills §0.5). If `MAP01` doesn't fit the Pi, fall back to a smaller-WAD or coarser visited-grid.
- **Freedoom specifics.** Exit-triggers episode-end, the secrets denominator, and whether a meaningful par/reference time exists are Gate-B/A unknowns; the reference time may be a captured baseline rather than an engine value.
- **`graph_rag` may be stubbed** like quantization (Gate D) → Python place-graph fallback.
- **A53 throughput** at 100k+ experiences with multi-step rollouts is unmeasured (skills §3.1 covered single scenarios). Track 1's recall-latency metric watches this.

---

## 8. Generalization

`MAP01` is the **first** target, not a hardcode. The harness keys on `--wad/--map/--cfg` and the metric map is engine-level (`GameVariable`s + position), so any WAD/level — other Freedoom maps, `doom2.wad`, custom maps — drops in by pointing the loader at a new cfg. The plan's success criteria are phrased per-map so the same scoreboard travels to the next level.
