# Doom-Vector: Teaching Skills & Showcasing RuVector

**Status:** Plan / queued for implementation
**Date:** 2026-06-12
**Goal:** Push the agent from "learns toy scenarios" toward real Doom competence — **aim, dodge, and (north star) finish a level** — while turning the repo into a **plain-language explainer of what RuVector can do**. Each work track teaches the agent one skill *and* lights up one RuVector feature we currently leave on the table.

This plan covers four tracks the maintainer selected, sequenced by dependency:

1. **Aim** — learn to shoot on `defend_the_center` (showcases *metadata-filtered recall* + *MMR-diverse re-ranking* + reward shaping).
2. **Capacity** — stretch the Pi's experience budget (quantization gate failed — see §0.5 — so this is now a *dim / cap / eviction* lever, not a quantization kwarg).
3. **Dodge** — learn to avoid damage on `take_cover` (showcases *recall-uncertainty as a safety signal*, with native *conformal prediction* as a verified stretch).
4. **Explainer** — a repo doc that uses the agent to teach each RuVector feature in everyday language.

The north star past these four is **whole-level completion**, which needs everything here *plus* hierarchy/waypoints and progress shaping — scoped at the end as the follow-on, not built in this plan.

---

## 0. Where we are (the substrate these tracks build on)

The winning recipe from §9 of the [self-learning plan](doom-vector-self-learning-agent.md) is **reactive value-weighted per-step recall**: encode the frame → k-NN recall from RuVector → vote for the action whose recalled neighbours had the best return-to-go → store the outcome. No gradients. It learns `basic`, `health_gathering`, and `deadly_corridor`, on the real 32-bit Pi Zero 2 W, within ~51 MiB.

What the bridge (`bridge/ruvector_py/src/lib.rs`) exposes today: `insert`, `search`, `delete`, plus `WorldModel.observe/plan`. Two things to notice for the tracks below:

- `search` already constructs `SearchQuery { vector, k, filter: None, ef_search: None }` (`lib.rs:72`). The `filter` field is real and usable; the `ef_search` field is **dead** in 2.2.0 (see §0.5) — Track 1 turns `filter` on.
- Metadata is stored as `serde_json::Value` (`lib.rs:61`) but only floats are read back. Filtering on float metadata is therefore feasible without changing the write path.

---

## 0.5 Feasibility gate results (verified against ruvector-core 2.2.0 source, 2026-06-13)

Both pre-wiring gates from §7 were run by extracting the installed crate (`~/.cargo/registry/.../ruvector-core-2.2.0.crate`) and reading `types.rs`, `vector_db.rs`, `index/hnsw.rs`, and `quantization.rs`. Results below; they amend the track plans and risk table.

**Gate 1 — `SearchQuery.filter` (Track 1): PASS, with two corrections.**
- **Type & equality:** `filter: Option<HashMap<String, serde_json::Value>>` (`types.rs:41`). Match is exact `serde_json::Value` equality (`vector_db.rs:190`: `metadata.get(key).is_some_and(|v| v == value)`). The bridge stores floats as `json!(f64)`, so the filter value must be built the same way — a `HashMap<String, f64>` mapped through `json!` matches cleanly. (The §7 "HashMap-equality" guess was right in spirit; the value type is `Value`, not `f64`.)
- **It is a POST-filter, not index-side.** `search` runs k-NN for the full `k` first, then `results.retain(...)` drops non-matching entries (`vector_db.rs:185-195`). Consequence: a filtered query returns **≤ k** results — possibly 0. **Track 1 must over-fetch** (search with a larger `k`, e.g. `k_raw = k * over_fetch`) and let the filter prune down. This corrects §2's "Pi note" — filtering is cheap CPU but is *not* free recall; under-fetching silently starves the vote.
- **`ef_search` per-query is a DEAD field.** `SearchQuery.ef_search` is never read anywhere in 2.2.0 (only ever written `None`); `VectorDB::search` calls `index.search(vector, k)` which uses the *static* `HnswConfig.ef_search` (`hnsw.rs:331-333`). `set_ef_search()` is a no-op stub (`hnsw.rs:130`). **The only real ef_search dial is `HnswConfig.ef_search` at construction** — which the bridge already passes next to `max_elements`. So "expose ef_search on search" is dropped; if we want to tune it, expose it on `RuVectorMemory::new` instead.

**Gate 2 — quantization via `DbOptions` (Track 2): NEGATIVE (as §7 feared).**
- `DbOptions.quantization: Option<QuantizationConfig>` exists (`types.rs:71`, enum: `None / Scalar(int8,4×) / Product{subspaces,k} / Binary(32×)` — **no int4 variant**), but it is **inert**: `grep QuantizationConfig` across the crate (minus its definition) returns zero hits. `VectorDB::new` persists the field then builds the index from only `dimensions / distance_metric / hnsw_config` (`vector_db.rs:81-98`); neither `HnswIndex` nor storage ever sees it. Setting `quantization=` would change nothing.
- Silver lining: because the field is ignored, the bridge is **not** secretly quantizing despite `DbOptions::default().quantization == Some(Scalar)` — current behavior is genuinely full-precision f32.
- A standalone, exported `quantization` module *does* exist (`ScalarQuantized`, `ProductQuantized`, `Int4Quantized`, `BinaryQuantized` — int4 lives here, oddly, not in the enum) operating on `&[f32] → Vec<u8>` with its own distance fns. But it is **not wired into VectorDB's HNSW/storage**; using it for real RAM savings means building a side-index outside VectorDB — a major architectural change, not a kwarg.
- **Decision:** Track 2 cannot ship as "add a `quantization=` kwarg." Fall back to the §7 mitigation — smaller embedding dim and/or tighter `max_elements` cap — for capacity, and record this as a documented negative result (it becomes an honest entry in the Track-4 explainer: "quantization is in the type system but not yet load-bearing in 2.2.0's `VectorDB`").

**Track 3 stretch note:** `advanced_features/conformal_prediction.rs` exists and is exported, so the calibrated-uncertainty stretch is reachable later — but Track 3's first cut needs no bridge change regardless.

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
| Quantization (int8/product/binary) | "Compress memories, keep 4–32× more" | Coverage for big maps | **inert in 2.2.0** (see §0.5) |
| Recall distance/spread → uncertainty | "Know when I'm somewhere unfamiliar" | Dodge: when unsure, play safe | **Track 3** |
| `MMRSearch.rerank` (diverse recall) | "Get a spread of advice, not 16 echoes" | Robust votes, better eviction | **Track 1** (composes over `search`, §1.5) |
| `ConformalPredictor` (calibrated uncertainty) | "Know *how sure* I am, with a guarantee" | Dodge: principled evade threshold | **Track 3 stretch** (composes, §1.5) |
| `graph_rag::KnowledgeGraph` | "A map of places and how they connect" | Waypoints / subgoal chaining | **north-star track** (§8, §1.5) |
| `ef_search` knob | "Trade recall quality vs. speed" | Free tuning dial | construction-time only (§0.5) |
| Hybrid dense+sparse / multi-vector | "Match on looks AND on tags" | Symbolic + visual recall | follow-on (§1.5) |
| `agenticdb` policy store | "A ready-made RL memory" | — | **rejected** — buggy/redundant (§1.5) |

---

## 1.5 Additional reachable features (source-audited 2026-06-13)

Beyond the two §0.5 gates, we audited the rest of 2.2.0's advanced surface against the same standard — *is it load-bearing, or just present?* The decisive question for each was **reachability tier**: a feature that **composes over our existing `search`** (takes a `search_fn` closure or post-processes results we already fetch) is cheap to adopt; one tied to a separate DB type we don't use is inert for us.

**Adopt (compose over `search` — near-free):**
- **`MMRSearch.rerank(query, candidates, k)`** (`advanced_features/mmr.rs`) — pure post-processing over already-fetched candidates. We over-fetch (we're doing that anyway for the Track-1 filter), then MMR-rerank so the value vote sees a *diverse* neighbour set instead of 16 near-duplicates. Could even be reimplemented in Python. → **wired into Track 1** (§2) and doubles as the "diverse eviction" lever Track 2 leans on (§3).
- **`ConformalPredictor`** (`advanced_features/conformal_prediction.rs`) — wraps a `search_fn` closure, so it sits on top of our `search`. This is the *real* calibrated-uncertainty path (distinct from `agenticdb`'s toy `predict_with_confidence`). → **Track 3's stretch is confirmed reachable** (§4).

**Adopt later (standalone structures we'd maintain alongside the store):**
- **`graph_rag::KnowledgeGraph`** (`advanced_features/graph_rag.rs`) — entities + relations + `get_neighbors` + `local/global_search`; a standalone topological graph, *not* tied to `VectorDB`. This is the single most navigation-relevant primitive in the crate and anchors the §8 north-star hierarchy.
- **`hybrid_search`/BM25, `multi_vector`, `matryoshka`** — richer recall (symbolic+visual, multi-channel states, coarse-to-fine). Real builds, self-contained; second-wave (§8).

**Rejected after a stub smoke test:**
- **`agenticdb` (`PolicyMemoryStore`, `predict_with_confidence`)** — *looks* tailor-made (it's value-weighted episodic control with `{action, reward, q_value, state_embedding}`), but reading the bodies showed it (a) merely re-wraps what `experience_store.py` + `episodic.py` already do, and (b) its one additive method, `update_q_value`, is a **destructive stub** — it deletes the entry, ignores the new value, and returns `Ok(())` (silent data loss). Its `predict_with_confidence` is a linear-scan heuristic on a separate session table, continuous-action-shaped, not our discrete vote. **Filed upstream:** [ruvnet/RuVector#562](https://github.com/ruvnet/RuVector/issues/562) (the stub) and [#563](https://github.com/ruvnet/RuVector/issues/563) (the inert `DbOptions.quantization` from §0.5). Do not adopt; the hand-rolled store + composable MMR/Conformal are strictly better.

**Meta-finding (worth a line in the Track-4 doc):** ruvector 2.2.0's *core primitives* (insert/search/delete/filtered-search) are solid and behaved exactly as their source claims; its *higher-level advanced/agentic layer* is partly aspirational (inert quantization, dead per-query `ef_search`, stubbed policy store). The verified-reachable wins for us are the **composable** ones.

---

## 2. Track 1 — Aim (`defend_the_center`)

**Why this scenario:** the agent stands still and enemies close in from all sides. It isolates aiming from navigation — the cleanest place to prove "learns to shoot."

**RuVector feature showcased:** *metadata-filtered recall.* The aim policy should be advised only by moments where an enemy was actually on screen; corridor-empty memories shouldn't dilute the trigger decision.

**What the agent must perceive (encoder).** Extend the structured encoder (`brain/encoder/structured.py`) with explicit aim signals — recall can't learn "pull the trigger when lined up" if alignment isn't a dimension:
- horizontal offset of the nearest enemy from screen centre (≈0 ⇒ on target),
- nearest-enemy distance,
- enemy-visible flag (also used as the filter key, below).

**What counts as good (reward).** Stop relying on kill-only reward; it's too sparse to shape aiming. ViZDoom exposes `HITCOUNT` / `DAMAGECOUNT` game variables — add a small per-step bonus for damage just dealt, in `experiments/train.py` (the reward post-processing already lives there). Dense "you hit something" feedback arrives many tics before a kill.

**The RuVector change (bridge).** Expose `filter` on `search` (verified in §0.5; `ef_search` per-query is dead, so it is *not* exposed here):
- `bridge/ruvector_py/src/lib.rs:71-72` — accept an optional `filter: Option<HashMap<String, f64>>`, convert each value via `serde_json::json!(v)` to match the stored `Value` (the write path already stores `json!(f64)`), and pass it into `SearchQuery.filter` instead of `None`. Leave `ef_search: None`.
- **Over-fetch to survive the post-filter.** The 2.2.0 filter is applied *after* k-NN (`vector_db.rs:185`), so a filtered query returns ≤ k. The bridge (or `experience_store`) must search with an inflated `k_raw` (e.g. `k * over_fetch`, default over_fetch≈4) when a filter is set, then return the top-k that survive. Without this the value vote silently starves.
- *(Optional, if we ever want the ef_search dial)* expose `ef_search` on `RuVectorMemory::new` → `HnswConfig.ef_search`, since that is the only place it is honored. Not needed for Track 1.
- `brain/memory/experience_store.py` — thread an optional `filter=` (and the over-fetch logic) through `search`.
- `brain/policy/episodic.py` — when an enemy is visible, recall with `filter={"enemy_visible": 1.0}`; otherwise recall unfiltered.

**Diverse re-ranking (MMR — §1.5, near-free).** The over-fetched candidate pool we already build for the filter is exactly what MMR wants. Before the value vote, re-rank the `k_raw` survivors down to `k` by relevance *minus* redundancy, so a cluster of 16 near-identical "enemy dead-ahead" memories can't drown out the one neighbour that tried a different action. Because `MMRSearch.rerank(query, candidates, k)` is pure post-processing over fetched results, the first cut can live **in Python** in `episodic.py` (cosine-diversity over the returned vectors) — no bridge change — and only move into the bridge if it proves hot. This is the same lever Track 2 reuses for diverse *eviction* (§3).

**Success:** greedy-eval kill count on `defend_the_center` climbs clearly above random, and a recorded GIF shows the agent tracking and dropping enemies. Two mini-ablations become Track-4 figures: **filtered vs. unfiltered recall**, and **MMR-diverse vs. raw top-k** vote.

**Pi note:** the filter is an app-side `retain` over the fetched `k` (§0.5) — negligible CPU, zero RAM cost, but remember it *reduces* effective recall, so over-fetch rather than shrinking `k`. Keep the *returned* `k` small; let `k_raw` absorb the filtering.

---

## 3. Track 2 — Capacity (quantization gate failed; fall back to dim/cap/eviction)

**Why now:** this is the "how do I maximize it" lever, and it's the prerequisite for the full-level north star. A single room fits in 20k experiences; a whole map needs far more state coverage. The original plan was to buy that coverage with quantization (int8 ≈ 4×, binary ≈ 32×) *in the same 512 MB* — but §0.5 found quantization inert in 2.2.0's `VectorDB`, so capacity must come from embedding dim, the `max_elements` cap, and eviction quality instead.

**RuVector feature showcased:** *vector quantization* (scalar/product/binary). The §0.5 gate found it **inert in 2.2.0**: the `DbOptions.quantization` field is stored but never consumed by `VectorDB`/`HnswIndex`/storage, and the standalone `quantization` module is a parallel API not wired into the index. So the original "add a `quantization=` kwarg" plan is **not viable** without bypassing `VectorDB` entirely.

**Gate result: NEGATIVE — recorded (see §0.5).** No `cargo doc` smoke test remains to run; the source was read directly. Two honest paths forward, in priority order:

1. **Capacity without quantization (do this).** Buy coverage with the levers that *are* load-bearing today:
   - **Smaller embedding dim** — the nav encoder is already dim-8; audit whether any track's additions can be kept tight (every dim is bytes × N-experiences).
   - **Tighter `max_elements` + smarter eviction** — we already evict worst-return entries; capacity is really "useful experiences per MB," so better eviction (e.g. MMR-diverse, reusing Track 1's reranker — §1.5) may beat raw count.
   - Benchmark experiences-held vs. dim and cap on the Seed, mirroring §9's RSS tables — that becomes the capacity figure instead of a quantization table.
2. **Native quantization (only if capacity proves blocking).** Would require either upstreaming a fix so `VectorDB` honors `DbOptions.quantization`, or building a side-index on the standalone `Int4Quantized`/`BinaryQuantized` types outside `VectorDB`. Both are real projects, not in scope for this plan — flag for a future RuVector-core bump.

**Success (revised):** a capacity table showing experiences-held and recall-latency vs. *dim and cap* on the A53 (not per quantization scheme), plus the documented negative result. The honest finding — "quantization is in 2.2.0's type system but not yet load-bearing in `VectorDB`" — is itself a Track-4 explainer entry.

### 3.1 Benchmark results — real Seed (Pi Zero 2 W, armv7l, 2026-06-13)

Measured with `deploy/bench_seed.py matrix` on the actual 32-bit Seed — **store
only** (no ViZDoom), so these isolate the *memory's* RAM/throughput, not the full
agent (§9's *full-agent* SCALE runs include ViZDoom's ~27 MiB base; §9's
*hardware-only* table — which the coverage curve below cross-checks — is store-only
like this one, on a ~10–13 MiB python+`ruvector_py` base). Reproduce on-device:
`python3 deploy/bench_seed.py matrix` (rootfs store, ~25 min). The bench drives
the stable `insert/search/delete` API directly and **replicates
`ExperienceStore._evict_native` line-for-line**, so the eviction numbers reflect
the real mechanism. Caveats baked into the output: `*`insert/s is **fsync-bound**
(redb on the SD card), not a CPU figure; `**`search/s is the **raw native recall
rate — an upper bound**; the live per-decision rate is lower by the encode +
marshal in `ExperienceStore.search`; query vectors are uniform-random, so recall
*timing* is indicative, not encoder-faithful.

**Cap lever — `max_elements` is a *lazy* ceiling (empty-store RAM, dim 8, fresh process per cap).**

| `max_elements` | empty-store RAM (over the ~10.9 MiB python+`ruvector_py` base) |
|---|---|
| 20,000 | 2.34 MiB |
| 50,000 | 2.35 MiB |
| 100,000 | 2.35 MiB |

**Flat** — the graph grows as needed *within* the cap, so raising `max_elements`
does **not** eagerly pre-allocate (a 100k store costs the same ~2.3 MiB as a 20k
one; contrast the 10 M default's ~661 MB, §9 — that OOM is the default *config*,
not a per-cap scaling law). So the prod default (100k) is essentially free when
the store isn't full; capacity cost is dominated by **held vectors and
tombstones**, not the cap value. (The bench measures one cap per process: a
within-process multi-cap sweep is allocator-noisy — a deleted store's retained
pages distort the next construct's reading — so `matrix` isolates each.)

**Coverage curve — RSS + recall vs held-count (dim 8, cap 100k = prod default).**

| held | RSS | insert/s `*` | recall (search/s) `**` | ms/search |
|---|---|---|---|---|
| 2,000 | 16.1 MiB | 30 | 345 | 2.9 |
| 5,000 | 20.0 MiB | 23 | 334 | 3.0 |
| 10,000 | 27.5 MiB | 23 | 307 | 3.3 |
| 20,000 | 43.3 MiB | 20 | 283 | 3.5 |

≈ **1.5 KiB RSS per held vector** at dim 8 ((43.3 − 13.1) MiB ÷ 20,000, over a
13.1 MiB empty store). Cross-checks §9's hardware table (20k ≈ 40.4 MiB, ~305/s) —
this run's 20k row (43.3 MiB / 283/s) sits ~5–7% off, run-to-run drift not a
regression — and recall holds **~283–345 decisions/s — ~30–38× the ~9/s real-time
bar** (small-store recall runs higher and is variance-prone, ±~15% run-to-run).

**Dim lever — RSS & recall vs embedding dim (held 5,000, cap 100k).**

| dim | RSS | recall (search/s) `**` | ms/search |
|---|---|---|---|
| 8 | 20.0 MiB | 334 | 3.0 |
| 16 | 21.7 MiB | 213 | 4.7 |
| 32 | 24.4 MiB | 136 | 7.3 |

RSS grows ~**linearly with dim** (≈ 0.19 MiB per added dim per 5,000 vectors ≈
40 B/dim/vector — ~10× the raw 4 B/dim f32, the rest being HNSW links + redb +
metadata), and recall-ms ~2.4× from dim 8→32 (no-SIMD scalar distance on the
A53, §9). **Dim discipline is the cheapest capacity lever** — every dim is
bytes × held — and, with no quantization safety net (§0.5), the *only* compression
lever we have. Keep encoder additions (Tracks 1/3) to a handful of dims.

**Eviction lever — value-based, hard-bounds live count; tombstones creep (dim 8, capacity 5,000).**

| inserts | phase | live count | RSS |
|---|---|---|---|
| 2,000 | fill | 2,000 | 16.6 MiB |
| 4,000 | fill | 4,000 | 19.9 MiB |
| 6,000 | churn | **5,000** | 21.7 MiB |
| 8,000 | churn | **5,000** | 23.7 MiB |
| 10,000 | churn | **5,000** | 25.9 MiB |
| 12,000 | churn | **5,000** | 28.1 MiB |

The live count **pins exactly at the capacity (5,000)** under sustained churn —
the memory is hard-bounded, as designed (drop the lowest-`return` entry on every
overflow). **But** RSS still creeps ≈ **1.09 KiB per evicting-insert** (7.5 MiB
over 6,999 churn inserts) because HNSW `delete` *tombstones* the slot; RAM is
reclaimed only on a compaction/rebuild we never trigger (the `experience_store`
docstring and §9 stress warn this). So capacity has **two distinct ceilings**:

- **Live-count / RSS ceiling** — the eviction `capacity` bounds live experiences
  and steady-state RSS (modulo tombstone creep). This is the working-set knob.
- **Lifetime-insert ceiling** — `max_elements` (prod default 100k) bounds *total
  inserts over the store's life*: every insert consumes a slot whether or not it
  is later evicted. A run inserting > `max_elements` experiences exhausts the
  slots regardless of how small the live cap is. At the agent's per-step insert
  rate that is a many-thousands-of-episodes concern; the fix is a periodic store
  rebuild/compaction (out of scope here — flagged for a future RuVector bump).

**Capacity levers, ranked (the §3 deliverable):**

1. **embedding dim** — linear in RAM, the cheapest and (post-§0.5) only
   compression lever; hold the line on encoder bloat.
2. **eviction `capacity`** — hard-bounds live RAM; size it to the working set the
   scenario needs (a single room fits in a few thousand; a full map needs more).
3. **`max_elements`** — free to set generously (lazy), but it is the *lifetime*-
   insert ceiling, so pair a high cap with periodic compaction on very long runs.

Quantization would have multiplied this budget 4–32× *in the same 512 MB*, but
it is inert in 2.2.0 (§0.5). So on the Pi the honest recipe is: **keep dim tight,
size the eviction capacity to the scenario, and rebuild periodically on long
runs** — no compression shortcut exists today.

---

## 4. Track 3 — Dodge (`take_cover`)

**Why this scenario:** `take_cover` gives the agent no weapon and incoming fireballs — it isolates "avoid getting shot" from everything else.

**RuVector feature showcased:** *recall uncertainty as a safety signal.* We already get neighbour **distances** back from `search`; we don't have to add native conformal prediction to benefit. When the nearest neighbours are far away or their action-votes disagree, the agent is in unfamiliar territory — and in a dodge task, the right default under uncertainty is *evade*, not freeze.

**What the agent must perceive (encoder).** A threat-aware encoder (new `brain/encoder/threat.py` or an extension of navigation):
- nearest projectile/enemy relative position and distance (ViZDoom labels fireballs in this scenario),
- **Δhealth** — damage taken this step. Prior phases proved HEALTH is decisive for survival; the *change* in it is the dodge signal.

**What counts as good (reward).** Penalize health loss per step in `experiments/train.py`, not just death. Dodging is continuous and needs continuous feedback.

**The mechanism (policy, mostly Python — cheap and honest).** In `brain/policy/episodic.py`, derive an uncertainty score from the recall already returned (e.g. mean neighbour distance, or vote entropy across actions). When uncertainty is high: bias toward an evasive default (strafe away from the nearest threat) and raise exploration locally. No bridge change required for the first cut.
- *Stretch (verified reachable — §1.5):* use `ruvector-core`'s real `ConformalPredictor` (`advanced_features/conformal_prediction.rs`) for a *calibrated* uncertainty with a coverage guarantee, and compare against the cheap distance heuristic. It wraps a `search_fn` closure, so it composes over our existing `search` — expose it through the bridge as a thin wrapper (don't confuse it with `agenticdb::predict_with_confidence`, which is a rejected linear-scan toy, §1.5). Only pursue if the distance heuristic proves too noisy.

**Success:** greedy-eval survival time on `take_cover` rises above random; a GIF shows the agent sidestepping fireballs; the uncertainty-gated safe-fallback measurably reduces deaths vs. the same policy without it (ablation → Track-4 figure).

---

## 5. Track 4 — The explainer doc

**Goal:** make the repo teach RuVector. One short doc (`docs/ruvector-by-example.md`, linked from the README) that walks the §1 table top to bottom, and for each feature gives: the everyday-language meaning, the one-line Doom payoff, the exact call site in this repo, and — where Tracks 1–3 produced one — a GIF or ablation figure as proof.

**Structure:**
- *The one idea:* the agent has no neural net; its "brain" is a memory of past moments, and RuVector is that memory. (Reuse the framing from §1 of the self-learning plan.)
- *Per feature:* recall → metadata → eviction → filtered recall → MMR-diverse recall → uncertainty (→ conformal) → graph-RAG waypoints (teaser). Each anchored to a real call site (`experience_store.py`, `episodic.py`, `lib.rs`) so a reader can jump from concept to code.
- *Honest negative results are content too:* a short "what's in the type system but not yet load-bearing" box — inert `DbOptions.quantization`, dead per-query `ef_search`, the rejected `agenticdb` policy store — with links to the upstream issues we filed ([#562](https://github.com/ruvnet/RuVector/issues/562), [#563](https://github.com/ruvnet/RuVector/issues/563)). This is the §1.5 meta-finding: core primitives solid, advanced layer partly aspirational.
- *Per scenario as a showcase:* `basic` = recall+metadata; `defend_the_center` = filtered recall + MMR vote; `take_cover` = uncertainty; `deadly_corridor`/`health_gathering` = eviction at scale; capacity bench = dim/cap.

**Why last:** it harvests the GIFs, ablations, and benchmark tables that Tracks 1–3 generate, so it ends up evidence-backed rather than aspirational. Draft the skeleton early, fill figures as each track lands.

---

## 6. Sequencing & dependencies

```
Track 1 (Aim)        ── bridge: filter (+over-fetch) + MMR rerank ─┐
                                                                    ├─► Track 4 (Explainer: harvests figures)
Track 2 (Capacity)   ── dim/cap/eviction (quant gate failed) ───────┤
                                                                    │
Track 3 (Dodge)      ── policy: uncertainty (+conformal stretch) ───┘
```

- **Track 1 first** — smallest change, most visible payoff, and it establishes the filtered-recall + reward-shaping pattern the others reuse.
- **Track 2 second** — independent infra; the quantization gate **failed (§0.5)**, so this is now a dim/cap/eviction benchmark rather than a bridge change; still informs the full-level north star.
- **Track 3 third** — mostly Python; reuses Track 1's encoder/reward scaffolding.
- **Track 4 throughout** — skeleton early, figures as they arrive.

Each track is a curriculum rung: `basic` ✓ → **`defend_the_center`** → **`take_cover`** → `deadly_corridor` ✓ → `my_way_home` ✓ → **full level**.

---

## 6.5 Tasks (shipyard checklist)

One checkbox per track, in the §6 dependency order. Each ships as its own
quality-gated PR; honor the `depends:` notes — don't start a task whose
dependency is still unchecked. Sub-bullets are scope, not separate tasks.

- [x] **Track 1 — Aim (`defend_the_center`).** depends: none. (see §2 + §0.5)
  - Bridge: turn on `SearchQuery.filter` with over-fetch (`k_raw = k * over_fetch`) so the post-filter doesn't starve the value vote; add MMR-diverse re-ranking.
  - Policy: aim/threat encoder dims + small hit-bonus reward shaping (eval on the *unshaped* scenario score).
  - Done when: the agent measurably learns to shoot on `defend_the_center` and the filtered-recall + reward-shaping pattern is reusable by later tracks.
- [x] **Track 2 — Capacity (dim/cap/eviction).** depends: none (independent infra; sequence after Track 1). (see §3 + §3.1 + §0.5)
  - Quantization gate failed (§0.5) — so benchmark the Pi's experience budget via smaller embedding dim, tighter `max_elements` cap, and better eviction, not a `quantization=` kwarg.
  - Done when: there's a documented capacity/coverage benchmark on the real Pi with the dim/cap/eviction levers characterized. → **done:** `deploy/bench_seed.py matrix` measured all three levers on the real Seed; results + findings (lazy `max_elements` ceiling, ~1.5 KiB/held-vector, linear dim cost, hard-bounded live count with tombstone creep, the two capacity ceilings) in §3.1.
- [ ] **Track 3 — Dodge (`take_cover`).** depends: Track 1. (see §4)
  - Reuse Track 1's encoder/reward scaffolding; use recall-uncertainty as a safety signal; conformal-prediction calibration is an optional stretch (no bridge change needed for the first cut).
  - Done when: the agent measurably reduces damage taken on `take_cover`.
- [ ] **Track 4 — Explainer doc.** depends: Tracks 1–3 (skeleton early, figures harvested as they land). (see §5 + §1 table)
  - A plain-language repo doc teaching each RuVector feature via the agent, including the honest negative result that quantization is in the type system but not load-bearing in 2.2.0 (§0.5).
  - Done when: the doc covers all four tracks with figures/results drawn from their PRs.

---

## 7. Risks & unknowns (tracked, in the doc's tradition)

| Risk | Why it matters | Mitigation |
|---|---|---|
| ~~`SearchQuery.filter` semantics unconfirmed~~ **RESOLVED (§0.5)** | Track 1 bridge change depends on it | Verified: native filter is an app-side post-filter over `k`; works with `json!(f64)` equality. Mitigation realized → **over-fetch `k_raw`** so the filter doesn't starve the vote |
| ~~Quantization reachable via `DbOptions`?~~ **RESOLVED: NO (§0.5)** | Track 2 hinged on it | Field is inert in 2.2.0. Fallback adopted: capacity via smaller dim / tighter cap / better eviction; native quant deferred to a future core bump |
| ~~Approximate distance breaks the value vote~~ **MOOT** | Was a quantization risk | No quantization ⇒ distances stay exact f32. Risk retired |
| Reward shaping can be gamed | Hit-bonus might encourage spray, damage-penalty might encourage cowering | Keep shaping small relative to scenario reward; eval on the *unshaped* scenario score |
| Encoder bloat raises dim/RAM | More features → bigger vectors on a 512 MB device | Keep aim/threat additions to a handful of dims; **no quantization safety net** (§0.5) — dim discipline matters more, since Track 2 can't compress us out of bloat |
| Full-level credit assignment | Episodic control is weak over long horizons | Out of scope here; the north-star section flags hierarchy + progress shaping as the next plan |

---

## 8. North star (follow-on, not in this plan): finish a level

These four tracks make the agent a competent reactive fighter that perceives threats and scales its memory. A *whole level* (find key → cross map → reach exit) additionally needs:
- **Capacity** (Track 2 delivers what it can via dim/cap/eviction; the bigger multiplier — native quantization — awaits a RuVector-core bump, per §0.5),
- **Hierarchy via `graph_rag::KnowledgeGraph`** (§1.5) — the concrete anchor for the navigation track. Build a coarse **topological waypoint memory** *over* the fine reactive store: place-nodes (entities, encoded by coarse position/landmark) joined by "leads-to" relations as the agent traverses. `get_neighbors` then turns `WorldModel.plan` from 1-step imagination into **subgoal chaining** (plan to the next waypoint, hand off to reactive recall to get there), and `local/global_search` answers "which known place is nearest my goal?" This is a standalone structure we maintain alongside the experience store — a real build, hence follow-on, not bolted onto Tracks 1–3.
- **What the agent must perceive (encoder), again first** — the hierarchy is useless without position/heading and a coarse visited-grid as dimensions (the §0.5 "recall is only as good as the encoder" rule). This, not the graph API, is the likely bottleneck for `my_way_home` → full level.
- **Progress shaping** — reward new-area-discovered / distance-to-exit / next-waypoint-reached so credit propagates over thousands of steps,
- **Multi-step `WorldModel.plan` rollout** — short n-step rollouts toward the current graph subgoal, turning reactive twitching into planning,
- **Curriculum** to a full `freedoom`/`doom2.wad` map via a custom `.cfg`.

That's the next plan (anchored on `graph_rag` + the encoder/reward/rollout trio). This one earns the right to attempt it.
