# Tier 3 — Raspberry Pi Zero 2 W setup (the authoritative test)

Only the real device settles the two gates: **does the stack fit in 512 MB**, and
**does recall keep up with the game**.

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
