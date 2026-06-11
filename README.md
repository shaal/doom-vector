# doom-vector

A self-learning Doom agent built on **ViZDoom** (environment) + **RuVector** (`ruvector-core`, the reward-aware vector memory that does the "learning"). The agent acts by *recall* — episodic control — rather than gradient-trained deep RL, so it can "learn" with no training step and run on tiny hardware. **Stretch target: a Raspberry Pi Zero 2 W (512 MB, no GPU).**

**It works:** on ViZDoom's `basic`, the agent learns from a random baseline of **−152** to a **positive score in under 25 episodes** (holding +30 to +50) — using a ~33-dim structured state encoder, the native RuVector backend, and ~50 MiB RAM. No neural network is trained; "learning" is reward-weighted experiences accumulating in the vector memory.

See [`docs/plans/doom-vector-self-learning-agent.md`](docs/plans/doom-vector-self-learning-agent.md) for the full design, rationale, and roadmap.

## Repo layout

```
envs/         ViZDoom environment factories (headless, low-res)
brain/
  encoder/    game state -> fixed-length vector
  memory/     RuVector-backed experience store (numpy fallback included)
  policy/     reward-weighted k-NN action selection (episodic control)
bridge/
  ruvector_py/  PyO3 binding -> ruvector-core (native Python module, no Node runtime)
experiments/  runnable spikes (random agent, memory loop)
eval/         scoring helpers
deploy/       cross-compile + arm64 Docker sandbox + Pi Zero 2 W setup
docs/plans/   design docs
```

## Quickstart — train & watch it learn

```bash
python3 -m venv .venv && source .venv/bin/activate   # Python 3.11/3.12 (vizdoom wheel)
pip install -r requirements.txt
pip install maturin && maturin develop --release -m bridge/ruvector_py/Cargo.toml  # native memory (optional; numpy fallback otherwise)

# train on `basic` with the structured encoder; prints a greedy-eval learning curve
python experiments/train.py --scenario basic --encoder structured --episodes 200 --eval-every 25

# record a greedy episode, then replay it in a window (on a desktop)
python experiments/train.py --episodes 200 --record demo.lmp
python experiments/replay.py demo.lmp --scenario basic
```

The `structured` encoder (game variables + labeled-object geometry) beats the
`thumbnail` pixel encoder decisively on `basic` — use it.

## Phase 0 — validation tiers (emulate to build, Pi to prove)

### Tier 0 — desktop spike (x86, fastest)
```bash
python3 -m venv .venv && source .venv/bin/activate   # use Python 3.11/3.12 if vizdoom has no wheel for yours
pip install -r requirements.txt
python experiments/spike_random_basic.py             # ViZDoom Basic, random agent: prints reward + RSS
python experiments/spike_memory_loop.py              # ViZDoom + RuVector memory loop (numpy fallback if binding unbuilt)
```

### Tier 1 — build the native RuVector binding
```bash
pip install maturin
maturin develop -m bridge/ruvector_py/Cargo.toml --release   # installs `ruvector_py` into the venv
python -c "import ruvector_py; m=ruvector_py.RuVectorMemory(3); m.insert([0.1,0.2,0.3]); print(m.search([0.1,0.2,0.3],1))"
```
Now `spike_memory_loop.py` uses the native backend instead of the numpy fallback.

### Tier 2 — aarch64 correctness sandbox (Docker)
```bash
bash deploy/setup_binfmt.sh                          # x86 hosts only; skip on Apple Silicon (native arm64)
bash deploy/build_pyo3_arm64.sh                      # cross-compile the arm64 wheel (maturin + zig)
docker build -f deploy/Dockerfile.arm64 -t doom-vector:arm64 .
docker run --rm --memory=512m --memory-swap=512m doom-vector:arm64   # early OOM signal; timing is NOT meaningful
```
On an **Apple Silicon Mac** this runs natively (real NEON, fast). On **WSL/x86** it runs under QEMU — correctness only.

### Tier 3 — the real Pi Zero 2 W (authoritative)
See [`deploy/pi_setup.md`](deploy/pi_setup.md). Flash 64-bit Pi OS Lite, enable zram, copy the arm64 wheel + code, then measure **RAM fit** (`free -m`, `deploy/measure_rss.py`) and **throughput** (steps/sec). Only this tier settles the two real gates.

## Status & verified results

- **Phase 0 (de-risked, all tiers):** ViZDoom runs headless; the PyO3 binding compiles **unmodified** against `ruvector-core` **2.2.0** on x86 **and** aarch64 (maturin + zig, ~1.8 MB self-contained abi3 wheel); the full loop runs in a **512 MB arm64 sandbox** on the native backend at ~85 MiB.
- **Phase 1 (the agent learns):** episodic control over RuVector learns `basic` from −152 → +30…+50 in <25 episodes with the structured encoder. Bounded memory (value-based eviction) verified; replay recording works. Pixel/thumbnail encoder underperforms — skip it for combat scenarios.
- **Phase 2 (path/plan prediction — honest negative results):** two extensions were built and tested. **Option A** (open-loop trajectory retrieve-and-follow) is flat on dynamic `health_gathering` and only bumps slightly on the static `my_way_home` maze — replay transfers only in static worlds. **Option B** (model-based rollout planning with an all-Rust RuVector forward model) compiles and runs but **collapses to no-op** on `health_gathering` (the dense living reward is constant across actions → no gradient) and is **~3–5 decisions/s on a Pi (too slow for real-time)**. Two follow-up fixes (a closed-loop reactive value-vote on the nav encoder, and a value-bootstrapped world model `r+γ·V(s')`) were then built and tested — the bootstrap fixed B's no-op collapse, but **neither learns `health_gathering`**. Diagnosis: the navigation encoder **omits HEALTH** (the decisive survival variable) and **return-to-go is confounded with time-remaining** in survival tasks. What demonstrably learns today is **`basic`** (combat, structured encoder, reactive value recall). Next options: add HEALTH to the encoder, try a distance-shaped scenario (`deadly_corridor`), or take the working `basic` agent to the Pi.
- **SONA evaluated, skipped:** `ruvector-sona`'s recall ranks by cosine only (never re-ranks by reward), so it adds nothing over our reward-weighted k-NN. Revisit only if upstream wires reward into `find_similar`.
- **Remaining gate:** Tier 3 on a real Pi Zero 2 W — RSS as the store fills, and A53 throughput (steps/sec). See [`deploy/pi_setup.md`](deploy/pi_setup.md).
