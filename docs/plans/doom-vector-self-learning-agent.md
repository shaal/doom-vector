# Doom-Vector: A Self-Learning Doom Agent on RuVector

**Status:** Draft / ideas for discussion
**Date:** 2026-06-10
**Goal:** Build a ViZDoom-style RL environment where an agent learns to play Doom, using `ruvnet/RuVector` as the "self-learning" substrate — packaged as a **watchable working demo** that ideally **runs on a Raspberry Pi Zero 2 W** (quad-core Cortex-A53 @ 1 GHz, **512 MB RAM**, no GPU).

---

## 0. Decisions locked in (2026-06-10)

These were chosen with the user and constrain everything below:

- **End goal: a working demo.** A satisfying, watchable agent that visibly learns to play Doom — optimize for a real result fast, not benchmark rigor.
- **Target device: Raspberry Pi Zero 2 W if at all possible.** 512 MB RAM, no GPU, ARM aarch64. This is the dominant constraint.
- **Architecture: the device picks it.** On 512 MB with no GPU you **cannot gradient-train a deep RL policy on-device**. But **episodic control (Architecture B) "learns" simply by storing experiences** — no training step — so it is the natural and recommended path. Deep-RL baselines (Arch A) become *desktop-only reference points*, not the deliverable.
- **Bridge: in-process PyO3, no Node runtime.** ViZDoom ships aarch64 Python wheels; RuVector is Rust with NEON SIMD. Wrap `ruvector-core` as a native Python module via PyO3/maturin so there's **no second runtime** eating RAM (a Node.js process alone is ~24–30 MB idle). Complexity is acceptable; efficiency wins.
- **Workflow: develop on desktop/WSL, then port to the Pi.** Iterate fast on a big machine, keep the footprint honest, validate on real hardware at the end of each phase.

**Feasibility: confirmed-with-caveats.** aarch64 wheels exist for ViZDoom (`pip install vizdoom`, headless via `set_window_visible(False)`, lowest res `RES_160X120` + `GRAY8`). RuVector's README lists aarch64 + NEON. PyO3 is well-supported on aarch64. The **two genuine unknowns that require testing on the actual device** are (1) ViZDoom's real RSS on a Pi Zero 2 W and (2) RuVector's actual aarch64 build + footprint — both are Phase-0 gates.

---

## 1. Vision in one paragraph

An agent that plays Doom and *gets better by remembering*. It perceives the screen, encodes what it sees into a vector, recalls similar situations it has lived through before (stored in RuVector with the rewards they led to), uses that recall to choose an action, and writes the outcome back into memory. Over time the memory — and RuVector's adaptive re-ranking of it — becomes the thing that learned. The headline question we're exploring: *can a fast, reward-aware vector memory substitute for (or augment) gradient-trained deep RL in a hard visual control task?*

---

## 2. What each piece actually gives us

### 2.1 ViZDoom (the environment) — solid, mature, exactly what we need

- RL research platform built on the **ZDoom** engine. Latest release **1.3.0** (Feb 2026), Farama Foundation.
- **Architecture:** ZDoom C++ engine → C++ core lib (`DoomGame`) → thin Python bindings. A *scenario* = a `.cfg` settings file + a `.wad` map. WADs author goals/rewards via **ACS scripting**.
- **Observations** via `get_state()` → `GameState`: `screen_buffer` (RGB24 H×W×3 / GRAY8 / CRCGCB, uint8), `depth_buffer`, `labels_buffer` (semantic segmentation), `automap_buffer`, `audio_buffer`, plus structured `labels`/`objects`/`sectors` and `game_variables` (health, ammo, position…).
- **Actions:** binary `Button`s (ATTACK, MOVE_FORWARD, TURN_LEFT…) + delta buttons. `make_action(action, tics)` applies an action with built-in frame-skip and returns cumulative reward.
- **Framework glue:** ships **Gymnasium** wrappers (`VizdoomBasic-v1`, etc.) and **PettingZoo** (up to 8 networked players for deathmatch/MARL). Works with stable-baselines3, Sample Factory (~100k FPS async), CleanRL.
- **Built-in scenarios** (and the skill each trains): Basic (aim), Defend the Center/Line (turret combat), Health Gathering [Supreme] (survival), Deadly Corridor (navigation under fire), My Way Home (maze exploration), Predict Position (projectile lead), Take Cover (dodging), Deathmatch (everything).
- **Reward model:** per-scenario `.cfg` (`living_reward`, `death_penalty`, `kill_reward`) + custom ACS. Shaped rewards available (e.g. Deadly Corridor on Δdistance-to-goal).
- **Performance:** up to ~7000 FPS single thread (sync mode), headless rendering, deterministic PLAYER/SPECTATOR sync modes ideal for training.

**Takeaway:** ViZDoom is a turnkey, well-supported gym. We don't need to clone it — we *use* it and put our novelty in the brain.

### 2.2 RuVector (the "self-learning" substrate) — powerful, but it is a *memory*, not a *policy*

This is the single most important finding and it reframes the whole project.

- RuVector is a **Rust-native vector database + graph engine with a learned re-ranking layer** ("Real-Time, Self-Learning AI, Vector GNN, Memory DB"). Core abstraction: HNSW vector similarity search, augmented by a GNN attention layer that re-ranks and *adapts to usage*.
- **It is not an RL framework and not a game-AI engine.** Closest analog: an agent-memory / RAG substrate (semantic recall, knowledge graph, reward-weighted experience store).
- **Runtime:** Rust (~78%) with TypeScript/JS (~17%). Ships **WASM** + **Node.js** bindings (`@ruvector/*`), ONNX/GGUF support, SIMD-optimized. **No first-class Python binding** (this is a real integration cost — see §4).
- **API surface:** crates `ruvector-core` (`VectorDB`, `VectorEntry`, `SearchQuery`, `DbOptions`), `ruvector-graph`, `ruvector-gnn`, `ruvector-sona`, `rvf-runtime`, `ruvllm`. Node: `npm install ruvector` → `new VectorDB({dimensions, storagePath})`, `db.insert({vector, metadata})`, `db.search({vector, k})`.
- **What "self-learning" means here:** *usage-based adaptive recall*, in three tiers — MicroLoRA per-request weight nudges (<1ms), session-level GNN attention reinforcement (~10ms), long-term EWC++ consolidation (~100ms). The **SONA** module adds a **trajectory API**: start a trajectory with a query embedding, record steps, end with a **quality/reward score**; **ReasoningBank** stores task→output patterns with reward values.
- **Crucially:** there *is* a reward signal, but it tunes *retrieval/routing*, not an action policy. **We must supply the action-selection brain ourselves.**
- **Maturity:** ~4.2k stars, active through May 2026, but many crates are 0.1.x and the docs are marketing-heavy. Treat the 0.x surface as experimental and verify each crate before depending on it.
- **Ecosystem:** same author as **ruv-FANN** (Rust neural-net library — the *neural* layer), **ruv-swarm** (multi-agent orchestration), claude-flow, Flow Nexus. Loosely coupled, individually packaged.

**Takeaway:** RuVector is a legitimate, fast, reward-aware memory backend. The project's intellectual core is therefore: *what policy do we wrap around that memory, and how much of the "learning" can the memory itself carry?*

---

## 3. The design space — five candidate architectures

Each shares the same skeleton: **perceive → encode to vector → recall from RuVector → decide → act in ViZDoom → store (state, action, reward) back**. They differ in *where the policy lives* and *how much RuVector does*.

### A. Memory-augmented deep RL  *(safe, proven)*
A standard PPO/DQN policy (stable-baselines3 or Sample Factory) plays Doom. RuVector stores episodic experiences keyed by state embedding; retrieved neighbors are concatenated into the policy's observation as extra context. Self-learning = both the net (gradients) and the memory (adaptive recall) improve.
- ➕ Lowest risk; strong baselines exist; RuVector is additive.
- ➖ RuVector is a bolt-on, not the star. Less novel.

### B. Episodic-control / retrieval-as-policy  *(novel, RuVector is the star)* ⭐
Inspired by **Model-Free Episodic Control** and **Neural Episodic Control**. *There is no policy network.* To act, the agent embeds the current frame, queries RuVector for the k most-similar past states, and picks the action whose stored neighbors had the **highest reward-weighted return**. RuVector's reward-aware re-ranking (SONA) does the credit assignment. Learning = writing better experiences and letting the memory consolidate (EWC++).
- ➕ Plays directly to RuVector's strengths; genuinely distinctive; CPU-friendly; fast to "learn" early (no warm-up gradient training).
- ➖ Quality hinges on the embedding (a bad encoder = bad recall); long-horizon credit assignment is hard; unproven at this scale.

### C. Hybrid: retrieval-warmed RL  *(best-of-both)*
Start as B (pure episodic control) for fast early competence, then distill the memory into a small policy net (ruv-FANN or torch) that takes over for fine control, with RuVector remaining as long-term episodic recall + a tie-breaker. This mirrors how NEC bootstraps then stabilizes.
- ➕ Fast early learning + stable late performance; a clean research narrative ("memory teaches the net").
- ➖ Two systems to build and keep in sync.

### D. VLM/LLM-as-policy with gameplay RAG  *(reasoning-first, slow)*
A vision-language model looks at the frame (or just `game_variables` + `labels`), RuVector supplies the most relevant past experiences as in-context examples ("RAG over your own gameplay"), the model reasons out an action. Self-learning = the retrieved memory gets better and reward-weighted.
- ➕ Interpretable ("why did you turn left?"); leverages strong priors; great demos.
- ➖ Far too slow for 35 tics/s real-time control; better for a slow, turn-based or "deliberation" variant. Token cost.

### E. Swarm / collective memory  *(ambitious, later)*
Multiple agents (ViZDoom PettingZoo deathmatch) share *one* RuVector memory via **ruv-swarm**, so experience from any agent improves all. Co-evolution / population-based.
- ➕ Showcases the whole ruvnet stack; rich emergent behavior.
- ➖ Multiplies every integration risk. Strictly a phase-3+ idea.

---

## 4. The hard part: the Python ↔ Rust/Node bridge

ViZDoom's good path is **Python**; RuVector's is **Rust/Node/WASM** with **no first-class Python binding**. Every architecture above must cross this boundary. Options, roughly in order of pragmatism:

1. **RuVector as a local service.** Run RuVector behind a thin HTTP/gRPC/IPC server (Node or Rust); the Python agent calls `insert`/`search`/`trajectory`. Simplest to stand up; adds per-step latency (mitigate with batching / Unix socket / shared memory).
2. **PyO3 wrapper** around `ruvector-core` → a native Python module. Best latency, most work; ties us to specific crate versions (0.x churn risk).
3. **WASM via `wasmtime`/`wasmer-py`.** Load RuVector's WASM build in-process from Python. Medium effort; sidesteps Node.
4. **Co-process over stdio** to a small Node script. Quick hack for a spike; not for production.
5. **Sidestep entirely:** do the whole thing in Rust against a Rust Doom binding, or in Node — losing ViZDoom's mature Python/Gym ecosystem. Not recommended unless we want a pure-Rust artifact.

**Recommendation (revised for the Pi target): go straight to (2) PyO3.** On a 512 MB device, the local-service approach (1) would mean running a whole second runtime (a Node.js process is ~24–30 MB idle, plus IPC overhead) — wasteful when RAM is the bottleneck. PyO3 compiles `ruvector-core` into a native Python `.so` that loads in-process with near-zero overhead and uses NEON SIMD on ARM. WASM-in-Python (`wasmtime-py`) is technically possible but adds a sandbox runtime for no benefit here.

Keep the interface narrow regardless of binding: `encode(frame|state) → vector`, `recall(vector, k) → experiences`, `store(state, action, reward, next)`, `end_episode(score)`. During early desktop spikes a quick stdio/service shim is fine to move fast, but the **Pi deliverable uses PyO3**.

---

## 5. Recommended path

**The Pi target picks the architecture: Architecture B (episodic-control) is the deliverable.** It "learns" by *storing experiences* rather than running a training step, so it fits a GPU-less 512 MB device — and it makes RuVector the protagonist, not a bolt-on. The narrative is genuinely cool: *a self-learning Doom agent that runs entirely on a $15 computer and keeps getting better the more it plays.* Architecture A (deep RL) is kept only as a **desktop reference score**; C (distill memory → a tiny ruv-FANN net) is a **stretch** since ruv-FANN nets are small enough to *infer* (not train) on-device.

### Memory budget on 512 MB (the thing to respect)

Rough simultaneous footprint, headless 64-bit Pi OS:

| Component | Approx RAM |
|---|---|
| OS (headless, 64-bit) | ~80–120 MB |
| ViZDoom engine + buffers (160×120 GRAY8) | ~tens of MB |
| Python interpreter + NumPy | ~30–50 MB |
| RuVector index (≤100k vecs @ dim-128 f32, incl. HNSW graph) | ~80–100 MB |
| Agent code | small |

Lands around **300–400 MB — workable with little headroom.** Levers: cap the index size and evict low-value experiences (or lean on RuVector's EWC++ consolidation), use a **small embedding dim**, prefer **structured-state vectors over pixels**, and enable **zram** (compressed RAM swap) rather than SD-card swap. 64-bit OS is required for NEON ML perf even though it costs more RAM than 32-bit.

### Phased roadmap (develop on desktop → validate on Pi)

**Phase 0 — De-risk on desktop, then smoke-test the Pi (the two real unknowns)**
- Desktop: ViZDoom `Basic` headless in Python; random agent; read `screen_buffer` + reward.
- Desktop: build the PyO3 binding to `ruvector-core`; round-trip `insert`/`search` from Python.
- **Pi gate:** `pip install vizdoom` on 64-bit Pi OS and measure real RSS; confirm `ruvector-core` cross-compiles for aarch64 and measure index RSS. *These two numbers decide whether the Pi dream survives — everything else is desktop-confirmed.*
- Pick the encoder, cheapest first: **structured state** (`game_variables` + object `labels`: types/positions/distances) as a compact vector — this skips pixel processing entirely and is ideal for the Pi. Downscaled grayscale is the fallback.

### Phase 0 validation strategy: *emulate to build, Pi to prove*

We have a real Pi Zero 2 W, so the device is the final judge — but we don't iterate on it. Three tiers:

**Tier 1 — Cross-build on the fast machine (Docker / `cross` / maturin).** Produce the aarch64 artifacts off-device: ViZDoom is a prebuilt `aarch64` wheel; the RuVector PyO3 binding is cross-compiled. Easiest cross-compile path for PyO3 is **`maturin build --release --target aarch64-unknown-linux-gnu --zig`** (uses `ziglang` as the cross-linker, no toolchain wrangling); `cargo install cross` is the fallback. *Why:* compiling Rust on a 512 MB Pi is slow and can OOM at link time — build wheels on the PC, copy them over.

**Tier 2 — aarch64 correctness sandbox (Docker `--platform linux/arm64`).** Run the full loop on the ARM ISA before touching hardware. Catches NEON/endianness/build breakage. Cap it with `--memory=512m --memory-swap=512m` for an early OOM signal.
- On an **Apple Silicon Mac** this runs *natively* (fast, real NEON) — the better sandbox.
- On **Windows/WSL x86** it runs under QEMU (`docker run --privileged --rm tonistiigi/binfmt --install arm64`): correct but slow, and **timing numbers are meaningless** — use it only to confirm *it works*, never for perf.

**Tier 3 — The real Pi (authoritative — the only thing that "proves it works on device").** Flash **64-bit Raspberry Pi OS Lite (Bookworm)**, headless, enable **zram**. Deploy the Tier-1 artifacts and measure what emulation can't:
- **RAM fit:** `free -m`, `VmRSS` from `/proc/<pid>/status`, or `smem` — does the whole stack stay under ~512 MB?
- **Throughput:** steps/sec and recall latency — does it keep up with the game (pin the ViZDoom engine to one core, the brain to another)?

**Bottom line:** Tiers 1–2 get us to a deployable artifact fast and kill ARM-correctness bugs cheaply; only Tier 3 settles the two real gates (RAM ceiling + A53 throughput). Emulation accelerates the path; it does not replace the device.

**Phase 1 — Episodic control on `Basic` (the demo skeleton)**
- Implement perceive → encode → recall → act → store.
- Action = reward-weighted k-NN over retrieved neighbors (ε-greedy exploration early).
- Wire SONA trajectories: each life = a trajectory ended with its score, so recall becomes reward-aware.
- Bound the memory; add eviction. Success = visibly beats random and trends upward over episodes.
- Record a replay video for the "watchable demo."

**Phase 2 — Harder scenarios + on-device run**
- Defend the Center, Health Gathering, Deadly Corridor.
- Get the loop running on the actual Pi Zero 2 W within the memory budget; capture a replay of it learning on-device.
- Optional stretch (C): distill the memory into a tiny ruv-FANN net for inference on-device; compare against the desktop PPO reference.

**Phase 3 — Reach goals (optional, off the Pi)**
- Deathmatch + ruv-swarm collective memory (E), or the VLM deliberation variant (D) as a separate slow-mode demo on a bigger machine.

### What we'd build (repo shape, tentative)
```
doom-vector/
  docs/plans/            # this doc + design notes
  envs/                  # ViZDoom config, scenario wrappers
  brain/
    encoder/             # frame → vector
    memory/              # RuVector client (service bridge)
    policy/              # episodic-control + (later) ruv-FANN distill
  bridge/                # PyO3 binding to ruvector-core (maturin) + Python client
  experiments/           # training scripts, baselines, metrics
  eval/                  # scoring vs PPO baselines
  deploy/                # Pi Zero 2 W setup: cross-compile, zram, footprint checks
```

---

## 6. Key risks & open questions (tracked)

| Risk | Why it matters | Mitigation |
|---|---|---|
| RuVector ≠ policy | The "self-learning" brand is about recall, not control | Own the policy (Arch B); use RuVector for what it's good at |
| **512 MB RAM ceiling** | Whole stack must coexist in ~300–400 MB | Bound + evict the index, small embedding dim, structured-state vectors, zram |
| ViZDoom + RuVector RSS on Pi unverified | The two facts not confirmable without the device | Phase-0 on-device smoke test gates the whole Pi goal |
| RuVector 0.x churn / aarch64 build | Crates are early; ARM not benchmarked by upstream | Pin versions; thin adapter interface; confirm cross-compile early |
| Embedding quality | Retrieval is only as good as the vector | Try structured (`game_variables`/`labels`) vectors before pixels |
| Credit assignment over long horizons | Episodic control is weak here | Reward shaping (ViZDoom supports it); start with short-horizon scenarios |
| A53 @ 1 GHz throughput | Per-step recall must keep up with the game | Headless low-res, frame-skip (`make_action` tics), small k, one core for the engine |

---

## 7. Prior art worth reading before building
- *Model-Free Episodic Control* (Blundell et al., 2016) — the conceptual ancestor of Architecture B.
- *Neural Episodic Control* (Pritzel et al., 2017) — differentiable memory + RL; basis for the hybrid (C).
- ViZDoom paper (Kempka et al., 2016, arXiv:1605.02097) and *Sample Factory* (arXiv:2006.11751) for baselines.
- RuVector README + `ruvector-sona` crate docs for the trajectory/reward API.

---

## 9. Progress log

**Phase 0 — DONE, de-risked across all off-device tiers (2026-06-10).**
- ViZDoom runs headless (Python 3.11; 3.13 lacks a wheel — use 3.11/3.12).
- PyO3 binding compiles **unmodified** against `ruvector-core` 2.2.0 on x86 *and* aarch64 (maturin + zig) → ~1.8 MB self-contained abi3 wheel.
- Full loop runs in a **512 MB arm64 Docker sandbox** on the native backend at ~85 MiB RSS, no OOM.

**Phase 1 — the agent learns (2026-06-10).**
- Episodic control over RuVector learns `basic`: mean greedy-eval reward **−152 (random) → +27 by ep 25**, holding +30…+50. Native backend, ~33-dim structured encoder, RSS ~50 MiB.
- **Structured encoder (game vars + labeled-object geometry) beats the pixel thumbnail decisively** — thumbnail never reached positive. Default to `structured` for combat scenarios.
- Bounded memory with value-based eviction verified (caps exactly at capacity). `.lmp` recording + replay path work.

**SONA — evaluated and SKIPPED.** `ruvector-sona` 0.2.0 is real and PyO3-friendly (sync, standalone), but its recall (`find_similar`) ranks by cosine similarity only and **never re-ranks by reward** (never reads `avg_quality`). Its "self-learning" is a separate rank-1 LoRA adapter that doesn't fit episodic control. Our reward-weighted k-NN already does what SONA's recall path doesn't. Revisit only if upstream wires reward into recall.

**Phase 2 — path/plan prediction (Options A + B): honest, mostly-negative results (2026-06-10).**
Both were built, verified to run, and *underperformed Phase 1's reactive value-recall* — a useful negative result.
- **Option A — open-loop trajectory retrieve-and-follow** (`experiments/train_path.py`, `brain/policy/trajectory_follow.py`, `brain/memory/trajectory_store.py`): recall the best-return nearby past trajectory and replay its action sequence with periodic replanning. Result: **flat on `health_gathering`** (dynamic — kits are consumed/respawn, so a replayed path stops transferring once the world diverges) and a **small bump then plateau on `my_way_home`** (static maze, where a fixed action sequence partly transfers). *Open-loop replay only helps in static environments.*
- **Option B — model-based rollout planning with a RuVector forward model** (`bridge/ruvector_py/src/world_model.rs` — all-Rust rollout loop — + `experiments/train_world_model.py`): one RuVector index per action (`state → next_state, reward`); rollouts run entirely in Rust. Compiled first-try; `plan()` is correct when rewards differ (smoke-tested). Result: **collapses to no-op (action 0) on `health_gathering`** — confirmed by a histogram of `{0: 96/96}` greedy steps, constant 284.0 reward. Cause (not a bug): **every observed transition carries the same +4.0 reward** (living reward × frameskip), constant across all actions/states, and we skip the terminal transition so the planner never sees the death cost. With nothing to discriminate, all rollouts tie and `argmax` picks index 0. Greedy *1-step reward* gives no gradient toward distant kits — it needs a *value/return-to-go* signal.
- **Synthesis:** what learns is **reward/value-weighted *per-step* recall** (Phase 1's `choose_action`). Both A (open-loop replay) and B (1-step-reward model) lack a forward-looking *value* signal that distinguishes actions in dynamic dense-reward settings. Principled fixes: (1) a **closed-loop reactive value-vote** on the navigation encoder — pick the action whose recalled neighbors had the best return-to-go; and/or (2) make Option B optimize predicted **return-to-go (value)** rather than 1-step reward.
- **Speed (Pi-relevant, important):** Option B's Rust rollout = **14 ms/decision (71/s) on x86** (8 actions, horizon 4 → 256 searches/decision). An A53 at 1 GHz is ~15–30× slower → **~3–5 decisions/s on the Pi**, below the ~9/s needed for real-time at frameskip 4. So **B is borderline-to-not-viable for real-time on the Pi**; `my_way_home` (32 actions, ~4 s/decision) is out. Cost is quadratic in action count. A reactive **value-vote is 1 search/decision** — trivially Pi-real-time. WorldModel memory is currently unbounded (no eviction); RSS grew ~5 MiB/25 ep.
- **Known minor issues (Option B):** WorldModel has no eviction yet; `unknown_dist` filtering uses an inverted comparison for the cosine metric (harmless at the default `f32::MAX`, wrong if set).

**Phase 2 follow-up — value signal added, *still* flat on `health_gathering` (2026-06-10).**
Both fixes were built and tested: (1) the **reactive value-vote** on the navigation encoder (`train.py --encoder navigation`), and (2) **Option B v2**, a **value-bootstrapped 1-step Bellman backup** (`score(a) = r + γ·V(s')`, `world_model.rs` rewritten; V = best recalled return-to-go). On `health_gathering`:
- Reactive vote: 376 (random) → drifts to ~320, *below* the random baseline. Does not learn.
- B v2: 284 → ~320; the value bootstrap **fixed the no-op collapse** (it no longer sits at exactly 284) but there is no learning trend.

**Diagnosis — state representation, not the planner:** (a) the navigation encoder **omits HEALTH**, the decisive survival variable, so two states at the same pose but different health are indistinguishable to recall; (b) in a survival task, **return-to-go is confounded with time-/health-remaining** rather than action quality, so value-by-recall is noisy. `basic` worked because the structured encoder captured the decisive variable (monster position) and reward was cleanly tied to the action.

**Phase 2 RESOLVED — navigation learns once the state is right (2026-06-10).** Both targeted fixes worked:
- (i) **Adding HEALTH to the navigation encoder** turned `health_gathering` around: the reactive value-vote climbs from a ~320 random baseline to ~400 (peak 492), versus drifting *below* baseline without HEALTH. The diagnosis was right — it was the missing decisive state variable.
- (ii) **`deadly_corridor`** (distance-shaped reward) learns strongly: −30 (random) → **+330** by ep 150, clean upward trend — the best navigation result, even with 128 actions, in ~40 s at ~57 MiB.

**The winning recipe is reactive value-weighted per-step recall over RuVector with an adequate state encoding.** Three learning scenarios now: `basic` (combat, structured encoder), `health_gathering` (survival, nav encoder + HEALTH), `deadly_corridor` (navigation, nav encoder). All native backend, 1 search/decision, Pi-viable (~45–80 MiB). Open-loop replay (A) and reward-sum rollout (B) were the wrong mechanisms.

**Tier 3 — REAL hardware results on the Cognitum Seed (Pi Zero 2 W, 2026-06-11).**
The Seed runs **32-bit Raspbian (armv7l / armhf)**, not the assumed 64-bit. Running the RuVector brain on it took three fixes (all now in the binding):
1. **Cross-build for `armv7-unknown-linux-gnueabihf`** (maturin + zig) → a 1.9 MB armv7 wheel (no prebuilt 32-bit ARM wheels exist for anything in the stack).
2. **Bound `HnswConfig.max_elements`** — the default is **10,000,000**, which pre-allocates **~661 MB** and instantly OOMs a 512 MB device. Exposed `max_elements` (default 100k) on `RuVectorMemory`/`WorldModel`.
3. **Disable the `simd` feature** — `ruvector-core`'s `simsimd` C kernels **SIGSEGV on armv7** during distance/search. The pure-Rust scalar fallback fixes it (negligible at dim 8); HNSW kept (its `anndists` has no SIMD).

Benchmark on the real A53 (dim-8 nav-encoder states, k=16 — the reactive value-vote = **1 search/decision**):

| store size | RSS | recall throughput |
|---|---|---|
| 5k | 17.8 MiB | 358/s (2.8 ms) |
| 20k | 40.4 MiB | 305/s (3.3 ms) |
| 50k | 86.5 MiB | 271/s (3.7 ms) |

**Both Tier-3 gates pass on real hardware:** RSS is modest (~40 MiB at our 20k experiment scale), and recall is **~270–510 decisions/s — ~30–55× the ~9/s real-time bar**. The self-learning brain is comfortably real-time on a $15 32-bit Pi. (The 100k checkpoint hit `No space left on device` — the redb store filled `/tmp`; bound the store or use a roomier path on-device.)

**Still open:** ViZDoom has **no 32-bit ARM wheel**, so running the *environment* on the Seed needs a 64-bit Pi OS (our aarch64 wheel + ViZDoom's aarch64 wheel are both verified) or an armhf source build. The brain is proven on-device regardless; the watchable Doom video is produced on the desktop.

**Next:** to close the full on-device loop, run on 64-bit Pi OS (so ViZDoom installs); or keep env-on-desktop + brain-on-Seed as a split demo.

**Tier 3 — FULL agent (ViZDoom + RuVector) runs AND learns on the 32-bit Seed (2026-06-11).**
We then got the *environment* onto the Seed too (there is no 32-bit ViZDoom wheel anywhere):
- **Cross-built a ViZDoom armhf wheel** by compiling from source in a QEMU `linux/arm/v7` Debian-**trixie** container on the desktop (matched the Seed's OS/Python 3.13 for ABI), with `-j<cores>` — far faster than a native A53 build → `vizdoom-1.3.0-cp313-cp313-linux_armv7l.whl`. (Needed `libboost-all-dev`; a native on-Seed build also works but is slow.)
- **Fixed an empty `vizdoom.pk3`**: the container's build produced a 22-byte stub (missing zip step) → engine error "No IWAD definitions found". Copied the working 630 KB pk3 from the desktop install — it's arch-independent game data.
- **numpy** from **piwheels** (`--extra-index-url https://www.piwheels.org/simple --only-binary numpy`) + `apt install libopenblas0-pthread` (its runtime).
- **Switched the distance metric to Euclidean.** The default Cosine + scalar fallback (no `simsimd` on armv7) returns tiny *negative* distances for near-identical vectors, which panics `hnsw_rs` (`assertion failed: dist_to_ref <= 0`). L2 is always ≥ 0 and also matches the numpy fallback's metric.

**Result on the Seed** (`basic`, structured encoder, reactive value-vote): **−118 (random) → +92 by episode 30**, holding ~+90; **~27 MiB RSS; 60 episodes + evals in 49 s** on the A53. The complete self-learning agent — environment *and* brain — runs and learns directly on a 32-bit \$15 Raspberry Pi.

**Next:** scale on-device episodes / try a navigation scenario on the Seed; optionally a 64-bit Pi OS (avoids the armhf workarounds and re-enables SIMD). Filed [cognitum-one/support#60](https://github.com/cognitum-one/support/issues/60) asking why the Seed ships 32-bit and for a supported 64-bit path.

## Next steps (planned)

Queued for a future session. The Seed is reached via the SSH helper in the parent
`cognitum/` dir (credentials live in the maintainer's local notes, not this public
repo); the agent code + venv are already deployed on the Seed under `~/doom-vector`
and `~/dv-bench`.

### A. Scale on-device
Push past the 60-episode `basic` smoke — longer runs and a navigation scenario,
measuring learning + RAM + throughput at scale on the A53.
- Run `train.py` on the Seed headless (`SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy`),
  store on **rootfs** (`--store ~/dvstore.rvf`), NOT the 32 MB `/tmp` tmpfs; keep
  `--capacity` modest (≤ 20k — the Seed shares ~300 MB free with its agent stack).
- Try `deadly_corridor` (nav encoder; distance-shaped reward — learned +330 on
  desktop) and `health_gathering` (nav encoder, HEALTH already included) on-device.
- Capture: eval curve, RSS-at-scale, episodes/sec on the A53. Watch for OOM; the
  store/HNSW is bounded (`max_elements`) but verify under longer runs.

### B. Record an on-device gameplay GIF (rendered on the Seed)
A clip rendered on the Seed itself (vs the current desktop GIF).
- `experiments/record_demo.py` already grabs RGB frames headlessly. **Open question:**
  whether ViZDoom fills `screen_buffer` with no display. We ran training with
  `SDL_VIDEODRIVER=dummy`, which likely yields blank frames — so the first step is
  rendering headless: most reliable is **`xvfb-run`** (a virtual framebuffer:
  `sudo apt install xvfb`, then `xvfb-run -a python experiments/record_demo.py ...`).
- Install `imageio` + `pillow` in the Seed venv (`--extra-index-url https://www.piwheels.org/simple`).
- Likely cheapest path: capture frames on the Seed → copy raw frames back to the
  desktop → encode the GIF on the desktop (avoids heavy encoding on the A53).
- Deliverable: `demo_seed_<scenario>.gif`, committed + embedded in the README as the
  "rendered on the actual Seed" artifact.

### C. (optional) Go 64-bit
If [cognitum-one/support#60](https://github.com/cognitum-one/support/issues/60)
recommends it: a 64-bit Pi OS drops the armhf workarounds (prebuilt ViZDoom wheel,
re-enable SIMD, no Euclidean-only constraint).

## 8. Sources
- ViZDoom: https://github.com/Farama-Foundation/ViZDoom · https://vizdoom.farama.org/
- RuVector: https://github.com/ruvnet/RuVector · `ruvector-sona` on crates.io
- ruv-FANN: https://github.com/ruvnet/ruv-FANN
