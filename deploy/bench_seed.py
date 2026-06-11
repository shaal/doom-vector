"""Benchmark the RuVector brain on real hardware (Cognitum Seed / Pi Zero 2 W).

Measures the two Tier-3 gates WITHOUT needing ViZDoom on the device: RSS as the
experience store fills, and recall/insert throughput for the reactive value-vote
workload (dim-8 navigation-encoder states, k=16 recall). For that policy it's
1 search/decision, so search/s ~= decisions/s achievable on the A53.

Output is flushed per line so a SIGKILL (OOM) still shows how far it got.

    python3 deploy/bench_seed.py
"""
import os
import random
import time

import ruvector_py as r

DIM = 8
K = 16
CHECKPOINTS = [500, 2000, 5000, 10000, 20000, 50000, 100000]


def rss_mb() -> float:
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024.0
    return float("nan")


def log(s: str) -> None:
    print(s, flush=True)


def main() -> None:
    rng = random.Random(0)
    log(f"machine={os.uname().machine} dim={DIM} k={K} start_rss={rss_mb():.1f}MiB")
    # max_elements bounds HNSW pre-allocation (default 10M ~= 661 MB OOMs a Pi)
    m = r.RuVectorMemory(DIM, f"/tmp/bench_{os.getpid()}.rvf", 120_000)
    log(f"store_created rss={rss_mb():.1f}MiB")

    def rv():
        return [rng.random() for _ in range(DIM)]

    n = 0
    prev = 0
    for cap in CHECKPOINTS:
        t0 = time.time()
        while n < cap:
            m.insert(rv(), None, {"action_idx": float(rng.randrange(8)), "return": rng.random()})
            n += 1
        ins = (cap - prev) / max(1e-6, time.time() - t0)
        prev = cap
        QN = 200
        t1 = time.time()
        for _ in range(QN):
            m.search(rv(), K)
        sps = QN / max(1e-6, time.time() - t1)
        log(f"n={n:7d}  rss={rss_mb():6.1f}MiB  insert={ins:8.0f}/s  search={sps:7.1f}/s ({1000.0/sps:.2f} ms)")


if __name__ == "__main__":
    main()
