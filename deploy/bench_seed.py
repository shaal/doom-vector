"""Capacity benchmark for the RuVector brain on real hardware (Cognitum Seed / Pi Zero 2 W).

Track 2 (capacity). Quantization is inert in ruvector-core 2.2.0 (plan §0.5), so
the Pi's experience budget is bought with three load-bearing levers instead:

  * embedding dim   -- bytes per stored vector (RSS at a fixed held-count)
  * max_elements    -- HNSW pre-allocation ceiling (a fixed RAM tax, paid empty);
                       the deployed ExperienceStore leaves this at the 100_000
                       default, so that is the cap the agent actually ships with
  * eviction        -- value-based: drop the lowest-return experience once the
                       logical *capacity* is full, so a bounded store keeps its
                       *useful* experiences. This is the real capacity ceiling
                       (distinct from max_elements, which is only a RAM floor).

It measures the Tier-3 gates WITHOUT needing ViZDoom on the device: RSS as the
experience store fills, and recall/insert throughput for the reactive value-vote
workload. For that policy it's 1 search/decision.

HONEST CAVEATS (printed by `matrix`, repeated here so a reader of the table
knows what the numbers do and don't mean):
  * insert/s is FSYNC-BOUND -- the redb store lives on the SD card and commits
    per insert, so insert/s reflects storage durability, not A53 compute.
  * search/s is the RAW native recall rate -- an UPPER BOUND on decisions/s. The
    real policy goes through ExperienceStore.search, which adds a per-decision
    _nonzero() scan + .tolist() marshal + slice that this self-contained bench
    deliberately omits (it imports only ruvector_py, never the `brain` package).
  * query/stored vectors are uniform-random, not encoder embeddings, so recall
    *timing* is indicative of the workload shape, not encoder-faithful.
  * eviction RAM creep is real: HNSW `delete` tombstones a slot rather than
    reusing it (RAM reclaimed only on a compaction/rebuild we never trigger), so
    total slots consumed == total inserts, not live count.

The `evict` mode replicates, line for line, the policy in
`brain.memory.experience_store.ExperienceStore._evict_native` (keep an
(id, return) list; on overflow delete the lowest-return entry), so the numbers
reflect the real eviction mechanism.

Store goes on rootfs by default, NOT /tmp -- the Pi's /tmp is a 32 MB tmpfs the
redb store overruns at scale ("No space left").

Output is flushed per line so a SIGKILL (OOM) still shows how far it got (the
`finally` cleanup cannot run on SIGKILL -- the pid-stamped default store path is
the real guard against stale-store collisions). Result rows are prefixed `ROW `
for easy harvesting into the capacity table; lines starting `# ` are context.

    python3 deploy/bench_seed.py matrix                  # full dim/cap/eviction sweep -> the committed table
    python3 deploy/bench_seed.py fill  --dim 8 --cap 100000 --checkpoints 2000,10000,20000
    python3 deploy/bench_seed.py fill  --dim 16 --cap 100000 --count 5000   # one dim-sweep cell
    python3 deploy/bench_seed.py evict --dim 8 --capacity 5000 --count 20000
"""
import argparse
import os
import random
import shutil
import subprocess
import sys
import time

import ruvector_py as r


def rss_mb() -> float:
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024.0
    return float("nan")


def log(s: str) -> None:
    print(s, flush=True)


def _fresh(path: str) -> None:
    """Remove any store left at `path` so the run starts from an empty index.
    redb persists the store; a stale file would silently load old vectors and
    corrupt the count, so each run wipes its path first (and cleans up after)."""
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def _rv(rng: random.Random, dim: int) -> list:
    return [rng.random() for _ in range(dim)]


def _search_ms(m, rng: random.Random, dim: int, k: int, n: int = 200, warmup: int = 20):
    """(searches_per_sec, ms_per_search) over n random queries -- the recall hot
    path the reactive policy runs once per decision. A short warm-up first pays
    the cold-cache / lazy-alloc cost so the timed window measures steady-state
    recall, not first-touch. This is the RAW native rate (an upper bound on real
    decisions/s -- see the module caveats)."""
    for _ in range(warmup):
        m.search(_rv(rng, dim), k)
    t = time.time()
    for _ in range(n):
        m.search(_rv(rng, dim), k)
    sps = n / max(1e-6, time.time() - t)
    return sps, 1000.0 / sps


_STORE_SEQ = [0]


def _store_path(arg: str) -> str:
    # rootfs, never /tmp (32 MB tmpfs on the Pi). Default paths are pid- AND
    # sequence-stamped so repeated stores in one process never collide. (Cross-
    # *dim* isolation is handled at a coarser level -- the `matrix` runs each dim
    # in its own subprocess, because ruvector-core 2.2.0 fixes the embedding dim
    # process-globally; see `_cell`.)
    if arg:
        return arg
    _STORE_SEQ[0] += 1
    return os.path.expanduser(f"~/dvstore_bench_{os.getpid()}_{_STORE_SEQ[0]}.rvf")


def run_fill(dim, cap, k, queries, store="", count=20_000, checkpoints=None) -> None:
    """Fill the store to each checkpoint; report held-count, RSS, and insert /
    recall throughput. One (dim, cap) per process keeps the RSS reading clean,
    so a dim- or cap-sweep is just this run repeated. `cap` sets max_elements
    (the pre-alloc tax, read off the `store_created` line) but `fill` never
    evicts -- held-count is driven purely by `count`/`checkpoints`, so this
    isolates the dim and the held->RSS curve; the eviction ceiling is `evict`."""
    rng = random.Random(0)
    path = _store_path(store)
    _fresh(path)
    cps = checkpoints if checkpoints else [count]
    log(
        f"# fill machine={os.uname().machine} dim={dim} max_elements={cap} "
        f"k={k} start_rss={rss_mb():.1f}MiB"
    )
    m = r.RuVectorMemory(dim, path, cap)
    log(f"# store_created rss={rss_mb():.1f}MiB  (pre-alloc tax for cap={cap})")
    log("# ROW fill <dim> <max_elements> <held> <rss_mb> <insert_s> <search_s> <ms_search>")
    n = 0
    prev = 0
    try:
        for cp in cps:
            t0 = time.time()
            while n < cp:
                m.insert(
                    _rv(rng, dim),
                    None,
                    {"action_idx": float(rng.randrange(8)), "return": rng.random()},
                )
                n += 1
            ins = (cp - prev) / max(1e-6, time.time() - t0)
            prev = cp
            sps, ms = _search_ms(m, rng, dim, k, queries)
            log(f"ROW fill {dim} {cap} {n} {rss_mb():.1f} {ins:.0f} {sps:.1f} {ms:.2f}")
    finally:
        del m
        _fresh(path)


def run_evict(dim, capacity, count, k, report_every, store="", max_elements=0) -> None:
    """Value-based eviction churn: insert well past `capacity` so the store
    spends most of the run at steady state, evicting the lowest-return entry on
    every insert. Surfaces (1) that the live count pins exactly at `capacity`,
    (2) the RSS creep from HNSW tombstones (deletes reclaim RAM only on a
    compaction/rebuild), and (3) the churn-phase insert cost (the O(n) min-scan
    over the value index plus the native delete). The per-window ROWs trace the
    RSS trajectory; `rss_at_cap` is sampled at the FIRST eviction, so `creep` is
    the growth across the whole churn phase (read the ROWs for the curve)."""
    rng = random.Random(0)
    path = _store_path(store)
    _fresh(path)
    # max_elements must cover EVERY insert: HNSW tombstones deleted slots rather
    # than reusing them, so total slots consumed == total inserts, not live count.
    me = max_elements if max_elements else count + capacity
    log(
        f"# evict machine={os.uname().machine} dim={dim} capacity={capacity} "
        f"count={count} max_elements={me} k={k} start_rss={rss_mb():.1f}MiB"
    )
    m = r.RuVectorMemory(dim, path, me)
    index: list = []  # (id, return) -- mirrors ExperienceStore._index
    log("# ROW evict-<phase> <n_inserted> <held> <rss_mb> <insert_s> <ms_search>")
    rss_at_cap = None
    n_at_cap = None
    win_inserts = 0
    win_secs = 0.0
    try:
        for n in range(1, count + 1):
            ti = time.time()
            val = rng.random()
            vid = m.insert(
                _rv(rng, dim),
                None,
                {"action_idx": float(rng.randrange(8)), "return": val},
            )
            index.append((vid, val))
            # value-based eviction: drop the lowest-return entry once over capacity
            # (identical to ExperienceStore._evict_native).
            if len(index) > capacity:
                i = min(range(len(index)), key=lambda j: index[j][1])
                ev_id, _ = index.pop(i)
                try:
                    m.delete(ev_id)
                except Exception:
                    pass
                if rss_at_cap is None:
                    rss_at_cap = rss_mb()
                    n_at_cap = n
            dt = time.time() - ti
            win_inserts += 1
            win_secs += dt
            if n % report_every == 0 or n == count:
                ins = win_inserts / max(1e-6, win_secs)
                _, ms = _search_ms(m, rng, dim, k, 100)
                phase = "churn" if len(index) >= capacity else "fill"
                log(
                    f"ROW evict-{phase} {n} {len(index)} {rss_mb():.1f} {ins:.0f} {ms:.2f}"
                )
                win_inserts = 0
                win_secs = 0.0
        if rss_at_cap is not None:
            churn = count - n_at_cap
            creep = rss_mb() - rss_at_cap
            per = creep * 1024.0 / max(1, churn)  # KiB per evicting-insert
            log(
                f"# steady_len={len(index)} (target capacity={capacity})  "
                f"rss_at_cap={rss_at_cap:.1f}MiB final_rss={rss_mb():.1f}MiB  "
                f"creep={creep:.1f}MiB over {churn} churn-inserts ({per:.2f} KiB/insert)"
            )
        else:
            log(f"# WARNING: count={count} <= capacity={capacity}; never evicted")
    finally:
        del m
        _fresh(path)


def run_captax(dim, caps, store="") -> None:
    """max_elements pre-alloc RAM tax on an EMPTY store. Each cap is constructed
    in isolation (prev store deleted first) and the RSS jump *across that one
    construction* is the marginal pre-alloc cost -- which answers whether
    max_elements pre-allocates eagerly (tax scales with cap) or lazily (flat)."""
    log("# ROW captax <dim> <max_elements> <prealloc_mb> <rss_mb>")
    for cap in caps:
        path = _store_path(store)
        _fresh(path)
        before = rss_mb()
        m = r.RuVectorMemory(dim, path, cap)
        after = rss_mb()
        log(f"ROW captax {dim} {cap} {after - before:.2f} {after:.1f}")
        del m
        _fresh(path)


def cmd_fill(a) -> None:
    cps = [int(x) for x in a.checkpoints.split(",")] if a.checkpoints else None
    run_fill(a.dim, a.cap, a.k, a.queries, a.store, a.count, cps)


def cmd_evict(a) -> None:
    run_evict(a.dim, a.capacity, a.count, a.k, a.report_every, a.store, a.cap)


def cmd_captax(a) -> None:
    run_captax(a.dim, [int(x) for x in a.caps.split(",")], a.store)


def _cell(args, title) -> None:
    """Run one matrix cell as its OWN subprocess and stream its output. Separate
    processes are REQUIRED, not just tidy: ruvector-core 2.2.0 pins the embedding
    dimension process-globally -- a second RuVectorMemory of a different `dim` in
    the same process trips `Dimension mismatch` on insert -- so the dim-sweep
    cells cannot share a process. check=False keeps one failed cell from aborting
    the rest of the sweep."""
    log(f"# --- {title} ---")
    try:
        subprocess.run([sys.executable, os.path.abspath(__file__)] + args, check=False)
    except Exception as exc:  # spawn failure shouldn't lose the rest of the run
        log(f"# CELL FAILED: {title}: {exc!r}")


def cmd_matrix(a) -> None:
    """Run the full dim/cap/eviction lever sweep and print every ROW, so
    `python3 deploy/bench_seed.py matrix` reproduces the committed capacity table
    end to end. Each cell runs as a fresh SUBPROCESS (see `_cell`) -- mandatory
    because ruvector-core 2.2.0 fixes the embedding dim per process."""
    k, q = str(a.k), str(a.queries)
    log("# === RuVector capacity matrix (dim / cap / eviction) ===")
    log(f"# machine={os.uname().machine} k={a.k} start_rss={rss_mb():.1f}MiB")
    log("# CAVEATS: insert/s is fsync-bound (redb on SD), NOT a CPU number.")
    log("#   search/s is the RAW native recall rate -- an upper bound; the real")
    log("#   per-decision rate is lower by ExperienceStore.search's encode+marshal.")
    log("#   query/stored vectors are uniform-random, so recall timing is")
    log("#   indicative of the workload shape, not encoder-faithful.")

    # One cap PER subprocess: a fresh process gives a clean empty-store pre-alloc
    # reading. (Measuring several caps in one process is allocator-noisy -- a
    # deleted store's retained pages distort the next construct's marginal delta.)
    for cap in ("20000", "50000", "100000"):
        _cell(["captax", "--dim", "8", "--caps", cap],
              f"cap lever: max_elements pre-alloc tax (empty store, dim=8, cap={cap})")
    _cell(["fill", "--dim", "8", "--cap", "100000",
           "--checkpoints", "2000,5000,10000,20000", "--queries", q, "--k", k],
          "coverage curve: RSS + recall throughput vs held (dim=8, cap=100000=prod default)")
    for dim in ("16", "32"):
        _cell(["fill", "--dim", dim, "--cap", "100000", "--count", "5000",
               "--queries", q, "--k", k],
              f"dim lever: RSS vs embedding dim at held=5000, dim={dim} (cap=100000)")
    _cell(["evict", "--dim", "8", "--capacity", "5000", "--count", "12000",
           "--report-every", "2000", "--k", k],
          "eviction lever: live count pins at capacity; tombstone creep (dim=8, capacity=5000)")
    log("# === MATRIX DONE ===")


def main() -> None:
    p = argparse.ArgumentParser(description="RuVector capacity benchmark (dim / cap / eviction).")
    sub = p.add_subparsers(dest="mode", required=True)

    f = sub.add_parser("fill", help="fill the store; report RSS + throughput at checkpoints")
    f.add_argument("--dim", type=int, default=8)
    f.add_argument("--cap", type=int, default=100_000, help="HnswConfig.max_elements (prod default)")
    f.add_argument("--k", type=int, default=16)
    f.add_argument("--count", type=int, default=20_000, help="target size when --checkpoints unset")
    f.add_argument("--checkpoints", type=str, default="", help="comma list, e.g. 2000,10000,20000")
    f.add_argument("--queries", type=int, default=200, help="recall samples per checkpoint")
    f.add_argument("--store", type=str, default="", help="store path (default: rootfs, pid-stamped)")
    f.set_defaults(func=cmd_fill)

    e = sub.add_parser("evict", help="value-based eviction churn: RSS creep + steady-state cost")
    e.add_argument("--dim", type=int, default=8)
    e.add_argument("--cap", type=int, default=0, help="max_elements (default: count + capacity)")
    e.add_argument("--capacity", type=int, default=5_000, help="logical eviction cap (live count)")
    e.add_argument("--count", type=int, default=20_000, help="total inserts (> capacity to churn)")
    e.add_argument("--k", type=int, default=16)
    e.add_argument("--report-every", type=int, default=2_500, dest="report_every")
    e.add_argument("--store", type=str, default="", help="store path (default: rootfs, pid-stamped)")
    e.set_defaults(func=cmd_evict)

    c = sub.add_parser("captax", help="max_elements pre-alloc RAM tax on an empty store")
    c.add_argument("--dim", type=int, default=8)
    c.add_argument("--caps", type=str, default="20000,50000,100000", help="comma list of max_elements")
    c.add_argument("--store", type=str, default="", help="store path (default: rootfs, pid-stamped)")
    c.set_defaults(func=cmd_captax)

    mx = sub.add_parser("matrix", help="full dim/cap/eviction sweep -> the committed capacity table")
    mx.add_argument("--k", type=int, default=16)
    mx.add_argument("--queries", type=int, default=200, help="recall samples per checkpoint")
    mx.set_defaults(func=cmd_matrix)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
