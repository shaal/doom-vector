# RuVector by example — what the Doom agent teaches about a vector DB

**A plain-language tour of [RuVector](https://github.com/ruvnet/RuVector) through one
working program: a Doom agent whose entire brain is a vector database.**

Every feature below is anchored to a real call site in this repo, paired with the
everyday idea it stands for, the Doom payoff it buys, and — where we have one — a
figure or measured result as proof. Read it top to bottom and you'll have seen the
whole of how `ruvector-core` 2.2.0 is (and isn't) load-bearing in a real agent.

> Companion to the design docs: [the self-learning plan](plans/doom-vector-self-learning-agent.md)
> (why a vector DB is a policy at all) and [the skills & features plan](plans/doom-vector-skills-and-features.md)
> (the track-by-track build this doc harvests).

---

## The one idea

**The agent has no neural network.** There are no weights, no gradients, no training
loop in the usual sense. Its "brain" is a *memory of past moments*, and RuVector is
that memory.

Each game tic the loop is the same four steps:

1. **Encode** the current frame into a short vector — "what does it look like right
   now?" (`brain/encoder/`).
2. **Recall** the nearest past moments from RuVector — "what did I do last time it
   looked like this, and how did it turn out?" (`search`).
3. **Vote** for the action whose recalled neighbours had the best return-to-go
   (`brain/policy/episodic.py`).
4. **Store** what actually happened, so the next encounter is a little wiser
   (`insert`).

That's the whole policy. Everything that follows is a refinement of those four
steps — *recall better*, *remember the right things*, *know when you're guessing* —
and each refinement maps to one RuVector primitive.

---

## The feature map

This is the spine of the tour. The leftmost columns are the RuVector feature and
its plain meaning; the rightmost is how load-bearing it actually is in 2.2.0 — a
distinction this repo took seriously and verified against the crate source
(2026-06-13).

| RuVector feature | Plain meaning | Doom payoff | Status in this repo |
|---|---|---|---|
| HNSW k-NN `search` | "What did I do last time it looked like this?" | The whole policy | **used** — §1 |
| Float metadata (`action_idx`, `return`) | "...and how did that turn out?" | Credit assignment | **used** — §2 |
| `delete` eviction | "Forget my worst outcomes when memory's full" | Bounded RAM on the Pi | **used** — §3 |
| `WorldModel.plan` | "Imagine one move ahead" | Stepping stone to planning | **used** — §4 |
| `SearchQuery.filter` | "Only recall *relevant* memories" | Aim: recall only enemy-visible moments | **Track 1** — §5 |
| `MMRSearch.rerank` | "Get a spread of advice, not 16 echoes" | Robust votes, diverse eviction | **Track 1** — §6 |
| Recall distance/spread → uncertainty | "Know when I'm somewhere unfamiliar" | Dodge: when unsure, play safe | **Track 3** — §7 |
| `ConformalPredictor` | "Know *how sure* I am, with a guarantee" | Principled evade threshold | **Track 3 stretch** (reachable, deferred) — §7 |
| `graph_rag::KnowledgeGraph` | "A map of places and how they connect" | Waypoints / subgoal chaining | **north-star teaser** — §8 |
| Quantization (int8/product/binary) | "Compress memories, keep 4–32× more" | Coverage for big maps | **inert in 2.2.0** — §9 |
| `ef_search` per-query knob | "Trade recall quality vs. speed" | A tuning dial | **dead per-query; construction-only** — §9 |
| `agenticdb` policy store | "A ready-made RL memory" | — | **rejected** — §9 |

---

## 1. Recall — `search` is the policy

**Everyday meaning:** "What did I do last time things looked like this?"

**Doom payoff:** this single call *is* the agent. Encode the frame, ask RuVector for
the *k* nearest past frames, and let their outcomes decide the move.

**Call site:** the bridge exposes k-NN recall at
[`bridge/ruvector_py/src/lib.rs:85`](../bridge/ruvector_py/src/lib.rs) — it builds a
`SearchQuery` and calls into the HNSW index:

```rust
let query = SearchQuery { vector, k, filter, ef_search: None };   // lib.rs:100
```

On the Python side, `ExperienceStore.search` (`brain/memory/experience_store.py:110`)
wraps that into the encode→marshal→recall path, and `choose_action`
(`brain/policy/episodic.py:15`) turns the recalled neighbours into a move.

**Proof:** the agent learns `basic`, `health_gathering`, and `deadly_corridor` from
recall alone — no gradients — within ~51 MiB on a real 32-bit Pi Zero 2 W. The two
replay GIFs in the repo root, `demo_deadly_corridor.gif` (desktop) and
`demo_seed_deadly_corridor.gif` (rendered *on the Pi*), are greedy episodes driven
entirely by this recall loop.

---

## 2. Metadata — turning a neighbour into advice

**Everyday meaning:** "...and how did that turn out, all things considered?"

**Doom payoff:** a nearest neighbour is useless unless you know *what you did* there
and *what it earned you*. Every stored vector carries two floats — the action taken
(`action_idx`) and the return-to-go (`return`) — and the vote weighs each
neighbour's return to pick an action. That's credit assignment with no backprop.

**Call site:** metadata rides along on `insert`
([`bridge/ruvector_py/src/lib.rs:52`](../bridge/ruvector_py/src/lib.rs), the
`VectorEntry { id, vector, metadata }` at line 63). `choose_action`
(`brain/policy/episodic.py:15`) reads it back, weighting each neighbour by its
recall similarity, `(score - min_score) + 1e-6`, and averaging the return per
candidate action.

**A subtle bug this surfaced (honest entry):** the value vote weights by
`score - min_score`, so the *sign* of the recall score matters. ruvector-core's
native `search` returns raw **L2 distance** (smaller = closer); the numpy fallback
returns **negative** L2 (bigger = closer). On the Pi the weighting was silently
*inverted* — the farthest neighbour dominated. It still learned (return-to-go
dominates the k-nearest set), which is why it went unnoticed for two tracks. The fix
negates the native score at the backend boundary so both backends agree
(`brain/memory/experience_store.py:154`):

```python
cands = [(r[0], -r[1], r[2], r[3]) for r in raw]   # native L2 → match numpy's −L2
```

A regression test, `tests/test_backend_parity.py`, locks the two backends to
identical scores. This corrects the value vote for *every* track, not just dodge.

---

## 3. Eviction — `delete` keeps the Pi alive

**Everyday meaning:** "When memory's full, forget my worst outcomes first."

**Doom payoff:** a 512 MB board can't hold unbounded experience. On every overflow
the store drops the lowest-`return` entry, so the memory stays pinned to a fixed
working set of *useful* moments.

**Call site:** `delete` is the bridge primitive
([`bridge/ruvector_py/src/lib.rs:126`](../bridge/ruvector_py/src/lib.rs)); the policy
lives in `ExperienceStore._evict_native` (`brain/memory/experience_store.py:93`),
which selects the minimum-return slot and deletes it.

**Proof (measured on the real Seed):** under sustained churn the live count pins
*exactly* at the eviction capacity — the memory is hard-bounded as designed. But
HNSW `delete` only *tombstones* a slot; RAM is reclaimed on a compaction we never
trigger, so RSS still creeps ≈1.09 KiB per evicting-insert (`deploy/bench_seed.py
evict`):

| inserts | phase | live count | RSS |
|---|---|---|---|
| 2,000 | fill | 2,000 | 16.6 MiB |
| 4,000 | fill | 4,000 | 19.9 MiB |
| 6,000 | churn | **5,000** | 21.7 MiB |
| 8,000 | churn | **5,000** | 23.7 MiB |
| 10,000 | churn | **5,000** | 25.9 MiB |
| 12,000 | churn | **5,000** | 28.1 MiB |

So capacity has *two* ceilings: the **live-count** cap (the working-set knob) and a
**lifetime-insert** cap (`max_elements`, since every insert burns a slot even if
later evicted). The fix for the latter is a periodic rebuild — flagged for a future
core bump. The `deadly_corridor` GIFs above are this eviction running at scale.

---

## 4. One-step imagination — `WorldModel.plan`

**Everyday meaning:** "Before I move, imagine one move ahead."

**Doom payoff:** a stepping stone from pure reaction toward planning. Given a state,
the world model picks the action that maximises `r(s,a) + γ·V(s')` — a one-step
Bellman backup over recalled transitions.

**Call site:** `WorldModel.observe` records each transition
([`bridge/ruvector_py/src/world_model.rs:96`](../bridge/ruvector_py/src/world_model.rs)),
and `WorldModel.plan` (world_model.rs:121) does the one-step lookahead, calling
`predict()` (a `k=1` `SearchQuery`, world_model.rs:34) and `state_value()`
internally.

**Status:** used as a primitive and a proving ground; the *multi-step* rollout it
hints at is north-star work (§8). One move ahead today, subgoal chaining later.

---

## 5. Filtered recall — "only ask about relevant moments" *(Track 1 — Aim)*

**Everyday meaning:** "Don't let empty-corridor memories dilute my aim — only recall
moments where an enemy was actually on screen."

**Doom payoff:** on `defend_the_center` the agent stands still and shoots; the
trigger decision should be advised only by frames where a target was visible. This is
`SearchQuery.filter`, turned on for Track 1.

**Call site:** the bridge now accepts an optional metadata filter
([`bridge/ruvector_py/src/lib.rs:90`](../bridge/ruvector_py/src/lib.rs),
`filter: Option<HashMap<String, f64>>`) and passes it straight into the query at
lib.rs:100 (above) instead of the old hardcoded `None`.

**The catch — and the fix.** In 2.2.0 the filter is a **post-filter**: `search` runs
k-NN for the full *k* first, then drops non-matching results. A filtered query
therefore returns **≤ k** — possibly zero — which would silently starve the value
vote. So `ExperienceStore.search` **over-fetches**: it searches an inflated `k_raw`
and lets the filter prune down to *k* (`brain/memory/experience_store.py:110`,
`over_fetch` parameter). This corrects the naive "filtering is free" assumption — it
costs no RAM but it *reduces effective recall*, so you widen the net, not shrink it.

The aim signals the filter keys on are added to the encoder
(`brain/encoder/structured.py:33`, `AIM_DIMS = 3`): the nearest enemy's horizontal
offset from centre, its on-screen size, and an `enemy_visible` flag — which doubles
as the filter key. Reward shaping (next paragraph) makes "you hit something"
feedback dense.

**Reward shaping:** kill-only reward is too sparse to teach aiming. `hit_shaped_reward`
(`brain/policy/reward.py:22`) adds a small per-step bonus for damage just dealt:

```python
return reward + bonus * dmg_delta   # reward.py:28 — dense "you hit something"
```

Kept small relative to the scenario reward, and **eval always reports the unshaped
score**, so the shaping can't be gamed into the headline number.

**Proof:** enable it with `python experiments/train.py --scenario defend_the_center
--encoder structured --aim` (`experiments/train.py:182`). The regression test
`tests/test_aim_learning.py:102` asserts the learned policy clears the baseline by a
real margin:

```python
assert learned > baseline + 0.8, f"no learning: baseline={baseline:.2f} learned={learned:.2f}"
```

---

## 6. MMR-diverse recall — "advice, not 16 echoes" *(Track 1)*

**Everyday meaning:** "Give me a *spread* of past advice, not sixteen copies of the
same memory."

**Doom payoff:** the over-fetched candidate pool from §5 is often a cluster of
near-identical "enemy dead-ahead" frames. If they all voted the same way, the one
neighbour that tried a *different* action would be drowned out. Maximal Marginal
Relevance re-ranks the pool by relevance *minus* redundancy, so the vote sees a
diverse set.

**Call site:** because MMR is pure post-processing over already-fetched results, the
first cut lives in Python — `mmr_rerank`
([`brain/policy/mmr.py:33`](../brain/policy/mmr.py), signature
`def mmr_rerank(query, candidates, k, *, lam: float = 0.5)`). It greedily builds the
output: at each step it picks the candidate maximising `lam · relevance − (1 − lam) ·
(max similarity to anything already chosen)`, so a near-duplicate of a memory you've
already taken scores low and a genuine outlier gets a turn. `lam=0.5` balances the
two; `lam=1` collapses to plain top-k.

`ExperienceStore.search` invokes it when `diversify=True`
(`brain/memory/experience_store.py:158`), re-ranking the `k_raw` survivors down to
*k*. This is the same lever Track 2 reuses for *diverse eviction* — keeping a varied
working set, not just a high-return one.

**Proof:** `tests/test_mmr.py:32` checks the rerank actually preserves the outlier a
plain top-k would drop:

```python
assert "other" in ids, f"diverse rerank dropped the outlier: {ids}"
```

---

## 7. Recall uncertainty — "know when you're guessing" *(Track 3 — Dodge)*

**Everyday meaning:** "When the nearest memories are far away or disagree with each
other, I'm somewhere unfamiliar — and in a dodge task the safe default is *evade*,
not freeze."

**Doom payoff:** on `take_cover` the agent has no weapon, only incoming fireballs.
The recall *already returns neighbour distances*; we don't need new machinery to know
when we're uncertain. When the value vote is weak, fall back to a hard-coded evade.

**Call site:** `recall_uncertainty` (`brain/policy/episodic.py:50`) computes the
normalized Shannon entropy of the similarity-weighted action vote — `1.0` on empty
recall, near `0.0` when the neighbours agree. `choose_action_safe`
(`brain/policy/episodic.py:96`) gates on it: when uncertainty ≥ `--evade-threshold`
it strafes *away from the nearest projectile* instead of trusting the weak vote.

The threat encoder (`brain/encoder/structured.py:37`, `THREAT_DIMS = 4`) perceives
what dodging needs: the nearest *projectile's* dx/size/visible (keyed on the fireball
`_projectiles`, not the wall monster that fired it) plus per-step **Δhealth** — which
makes the encoder *stateful*, tracking previous HEALTH across tics. The reward
mirrors Track 1's bonus as a penalty (`dodge_shaped_reward`, `brain/policy/reward.py:41`):
`reward - penalty * health_loss`.

**Proof (real ViZDoom `take_cover`, `living_reward=1` so survival ticks = inverse of
damage taken; 3 training seeds × 30 eval episodes):**

| policy | mean survival |
|---|---|
| random | 309.9 |
| evade-always (heuristic only) | 314.7 |
| value-vote only (recall, no fallback) | 322.2 |
| **vote + uncertainty-gated evade @0.6** | **336.8** |
| vote + evade @0.7 | 356.2 |
| vote + evade @0.8 | 321.7 |

The thesis lands: *neither* the evade heuristic alone (~315) *nor* learned recall
alone (~322) is strong — the **uncertainty-gated combination** (337–356) beats both.
Trust learned recall when confident; fall back to a safe evade when not. Reproduce
with `python experiments/train.py --scenario take_cover --encoder structured --dodge`,
which prints a paired `[dodge ablation]` line (`experiments/train.py:289`).

**Stretch (reachable, deferred):** ruvector-core ships a real `ConformalPredictor`
(`advanced_features/conformal_prediction.rs`) that wraps a `search_fn` closure — so
it *composes over our existing `search`* — to give calibrated uncertainty with a
coverage guarantee. It's verified reachable but unnecessary while the cheap distance
heuristic works; pursue it only if that proves too noisy. (Do not confuse it with
`agenticdb`'s rejected toy — §9.)

---

## 8. Graph-RAG waypoints — the north-star teaser

**Everyday meaning:** "A map of *places* and how they connect."

**Doom payoff:** finishing a whole level (find key → cross map → reach exit) needs
more than reactive recall — it needs a coarse **topological waypoint memory** layered
*over* the fine experience store. `graph_rag::KnowledgeGraph`
(`advanced_features/graph_rag.rs`) is a standalone entities-and-relations structure
with `get_neighbors` and `local/global_search`; place-nodes joined by "leads-to"
relations would turn `WorldModel.plan` from one-step imagination into **subgoal
chaining**.

**Status:** not built here — it's a standalone structure we'd maintain alongside the
store, hence the *next* plan, not this one. Flagged so the tour is honest about where
the road leads. The likely bottleneck isn't the graph API; it's the encoder (the
"recall is only as good as the encoder" rule), which would need position, heading,
and a coarse visited-grid before any of this helps.

---

## 9. The honest negatives — "in the type system, not yet load-bearing"

A vector DB's *core primitives* (insert / search / delete / filtered search) behaved
**exactly** as their source claims, on x86 and on the real Pi. Its higher-level
*advanced/agentic* layer is partly aspirational. Naming that plainly is part of the
tour — these are real findings audited against the 2.2.0 crate source, two of them
filed upstream.

- **Quantization is inert.** `DbOptions.quantization` exists in the type system
  (`None / Scalar / Product / Binary` — no int4 variant) but `VectorDB::new` persists
  the field then builds the index from only `dimensions / distance_metric /
  hnsw_config`; the index and storage never see it. Setting `quantization=` changes
  nothing. A standalone `quantization` module (`Int4Quantized`, `BinaryQuantized`, …)
  *does* exist but isn't wired into `VectorDB`'s HNSW — using it for real RAM savings
  means a side-index, a major change, not a kwarg. So Track 2's "keep 4–32× more
  memories in the same RAM" promise is **not available** in 2.2.0; capacity instead
  comes from the levers that *are* load-bearing — embedding dim, the `max_elements`
  cap, and eviction quality (§3). Filed upstream:
  [ruvnet/RuVector#563](https://github.com/ruvnet/RuVector/issues/563).
- **Per-query `ef_search` is dead.** `SearchQuery.ef_search` is never read anywhere
  in 2.2.0; `VectorDB::search` uses the *static* `HnswConfig.ef_search`, and the
  per-query setter is a no-op stub. The only real dial is `HnswConfig.ef_search` at
  construction — and this bridge doesn't even expose that on
  `RuVectorMemory::new` ([`lib.rs:42`](../bridge/ruvector_py/src/lib.rs) passes only
  `max_elements`). The recall path hardcodes `ef_search: None`.
- **The `agenticdb` policy store is rejected.** `PolicyMemoryStore` *looks*
  tailor-made — value-weighted episodic control with `{action, reward, q_value,
  state_embedding}` — but reading the bodies showed it merely re-wraps what
  `experience_store.py` + `episodic.py` already do, and its one additive method,
  `update_q_value`, is a **destructive stub**: it deletes the entry, ignores the new
  value, and returns `Ok(())` (silent data loss). Its `predict_with_confidence` is a
  continuous-action linear scan, not our discrete vote. The hand-rolled store plus
  composable MMR/Conformal are strictly better. Filed upstream:
  [ruvnet/RuVector#562](https://github.com/ruvnet/RuVector/issues/562).

**The meta-finding:** adopt the **composable** wins — the ones that sit on top of the
`search` you already call (filter, MMR, conformal). Be skeptical of the agentic layer
until its source, not its type signature, proves load-bearing.

---

## Capacity, measured — the Track 2 levers

With no quantization safety net, capacity on the Pi is three knobs, characterized on
the real Seed with `python3 deploy/bench_seed.py matrix`:

- **Embedding dim** — linear in RAM (≈40 B/dim/vector; recall-ms ~2.4× from dim 8→32
  with no SIMD on the A53). The cheapest and *only* compression lever — keep encoder
  additions (the §5/§7 dims) to a handful.
- **Eviction `capacity`** — hard-bounds live RAM (§3); size it to the working set
  (a single room fits in a few thousand; a full map needs far more).
- **`max_elements`** — a *lazy* ceiling (a 100k store costs the same ~2.3 MiB empty
  as a 20k one), free to set generously, but it's the *lifetime*-insert cap, so pair
  a high value with periodic compaction on long runs.

The coverage curve, measured on the real Seed (dim 8, cap 100k = prod default):

| held | RSS | recall (search/s) | ms/search |
|---|---|---|---|
| 2,000 | 16.1 MiB | 345 | 2.9 |
| 5,000 | 20.0 MiB | 334 | 3.0 |
| 10,000 | 27.5 MiB | 307 | 3.3 |
| 20,000 | 43.3 MiB | 283 | 3.5 |

That's ≈1.5 KiB RSS per held vector at dim 8, recalling ~283–345 decisions/s —
**~30–38× the ~9/s real-time bar.** The dim and lazy-`max_elements` tables, and the
cross-check against the earlier hardware run, are in
[the skills & features plan §3.1](plans/doom-vector-skills-and-features.md).

---

## Per-scenario cheat sheet

| Scenario | RuVector feature it showcases |
|---|---|
| `basic` | recall + metadata (§1, §2) — the bare loop |
| `defend_the_center` | filtered recall + MMR-diverse vote (§5, §6) |
| `take_cover` | recall-uncertainty as a safety signal (§7) |
| `deadly_corridor` / `health_gathering` | eviction at scale (§3) — the demo GIFs |
| capacity bench | dim / cap / eviction levers (Track 2) |

---

## Where to look in the code

| Concept | File |
|---|---|
| The four-step loop / value vote | `brain/policy/episodic.py` |
| Encode the frame (aim + threat dims) | `brain/encoder/structured.py` |
| Store, recall, over-fetch, evict | `brain/memory/experience_store.py` |
| Reward shaping (hit bonus, dodge penalty) | `brain/policy/reward.py` |
| Diverse re-ranking | `brain/policy/mmr.py` |
| The RuVector bridge (insert/search/delete) | `bridge/ruvector_py/src/lib.rs` |
| One-step world model | `bridge/ruvector_py/src/world_model.rs` |
| Train & ablate | `experiments/train.py` |
| Capacity benchmark | `deploy/bench_seed.py` |
| Regression tests | `tests/` (parity, aim, dodge, mmr, reward) |

The agent has no neural net. It remembers, recalls, and votes — and RuVector is the
memory that makes that a policy.
