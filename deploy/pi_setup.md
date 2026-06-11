# Tier 3 — Raspberry Pi Zero 2 W setup (the authoritative test)

Only the real device settles the two gates: **does the stack fit in 512 MB**, and
**does recall keep up with the game**.

> ## Results from the real device (Cognitum Seed, 2026-06-11)
> The Seed runs **32-bit Raspbian (armv7l/armhf)**, not 64-bit. The RuVector brain
> runs on it after three fixes (all now in the binding); benchmarked on the A53:
> **~40 MiB at 20k experiences, ~270–510 recall/s** (≈ decisions/s — far above the
> ~9/s real-time bar). The brain is comfortably real-time. Gotchas discovered:
> 1. **No 32-bit ARM wheels** for the stack → cross-build the binding for
>    `armv7-unknown-linux-gnueabihf` (maturin + zig), not aarch64.
> 2. **`HnswConfig.max_elements` defaults to 10M** → ~661 MB pre-alloc → OOM. Pass a
>    small `max_elements` (the binding now defaults to 100k).
> 3. **`simsimd` SIGSEGVs on armv7** → build `ruvector-core` with
>    `default-features=false` (no `simd`); scalar distance is free at low dim.
> 4. **ViZDoom has no 32-bit wheel** → to run the *environment* on the Pi you need
>    **64-bit Pi OS** (below) or an armhf source build. The brain works on 32-bit.
> Cross-build for 32-bit: `maturin build --release --target armv7-unknown-linux-gnueabihf --zig -m bridge/ruvector_py/Cargo.toml --out dist`

## Running the FULL agent on a 32-bit (armhf) Seed — the recipe

The complete agent (ViZDoom env + RuVector brain) was made to run + learn on the
32-bit Seed. Reproducible steps:

1. **RuVector brain (Rust):** cross-build for armv7 (above). The binding already
   sets `max_elements` (avoid the 10M default OOM) and **Euclidean** distance
   (the scalar cosine fallback, no simsimd on armv7, returns negative distances
   that panic hnsw_rs).
2. **ViZDoom (C++): no armhf wheel exists — build from source.** Fastest is a
   QEMU `linux/arm/v7` Debian-trixie container on a dev box (match the Seed's OS
   + Python 3.13 for ABI):
   ```bash
   docker run --privileged --rm tonistiigi/binfmt --install arm
   docker run --rm --platform linux/arm/v7 -v /tmp/out:/out arm32v7/debian:trixie-slim bash -c '
     apt-get update && apt-get install -y python3 python3-dev python3-venv cmake g++ make \
       libsdl2-dev libopenal-dev libboost-all-dev zlib1g-dev libbz2-dev
     python3 -m venv /v && /v/bin/pip install -U pip wheel
     CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) /v/bin/pip wheel vizdoom --no-deps -w /out'
   ```
3. **Fix the empty `vizdoom.pk3`** (the container build leaves a ~22-byte stub →
   "No IWAD definitions found"): copy the working `vizdoom.pk3` (~630 KB) from a
   desktop ViZDoom install over the stub — it's arch-independent game data.
4. **On the Seed**, in the venv:
   ```bash
   pip install vizdoom-1.3.0-cp313-cp313-linux_armv7l.whl \
       --extra-index-url https://www.piwheels.org/simple --only-binary numpy
   sudo apt install -y libopenblas0-pthread libgfortran5   # numpy runtime
   ```
5. **Run headless** (no display): `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy`,
   and put the RuVector store on rootfs (`--store ~/dvstore.rvf`), not the 32 MB
   `/tmp` tmpfs.

Verified result: `basic` learns −118 → +92 in 30 episodes at ~27 MiB on the A53.

## 1. Flash the OS
- **64-bit Raspberry Pi OS Lite (Bookworm)** — 64-bit is required for the ViZDoom
  aarch64 wheel and NEON SIMD; Lite (no desktop) saves ~100+ MB RAM.
- Use Raspberry Pi Imager; preset Wi-Fi + SSH so you can run headless.

## 2. Enable zram (compressed RAM swap — never swap to the SD card)
```bash
sudo apt-get update && sudo apt-get install -y zram-tools
echo -e "ALGO=zstd\nPERCENT=50" | sudo tee /etc/default/zramswap
sudo systemctl restart zramswap
free -m   # confirm a zram swap device appears
```

## 3. System deps + Python
```bash
sudo apt-get install -y python3-venv python3-pip libsdl2-2.0-0 libopenal1 libgomp1
python3 -m venv ~/dv && source ~/dv/bin/activate
pip install --upgrade pip
```

## 4. Copy the project + the cross-built wheel
From your build machine (after `deploy/build_pyo3_arm64.sh`):
```bash
rsync -av --exclude .venv --exclude target --exclude dist ./ pi@<pi-host>:~/doom-vector/
scp dist/*aarch64*.whl pi@<pi-host>:~/doom-vector/dist/
```
On the Pi:
```bash
cd ~/doom-vector
pip install -r requirements.txt          # pulls the aarch64 vizdoom wheel
pip install dist/*aarch64*.whl           # native ruvector_py (else numpy fallback)
```

## 5. Run + measure
```bash
# RAM fit — watch while the loop runs (run in a second SSH session):
watch -n1 'python deploy/measure_rss.py $(pgrep -f spike_memory_loop)'

# the loop itself (headless): prints reward + RSS per episode
python experiments/spike_memory_loop.py --episodes 20

# throughput: time a fixed number of steps
time python experiments/spike_random_basic.py --episodes 5
```

## What to record (the Phase 0 findings)
- Peak RSS of the whole loop vs. the 512 MB ceiling (and how much zram was used).
- Steps/sec on the A53 — is real-time (≥ a few decisions/sec) achievable headless at 160x120?
- Did the native `ruvector_py` wheel import, or did it fall back to numpy?
- Memory growth as the experience store fills (this sets the eviction/cap policy).
