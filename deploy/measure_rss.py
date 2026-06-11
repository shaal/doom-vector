"""Print system memory and optionally a process's RSS. Stdlib only, Pi-friendly.

    python deploy/measure_rss.py [PID]
"""
import sys
from pathlib import Path


def meminfo_kb() -> dict[str, int]:
    out = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, val = line.partition(":")
        out[key] = int(val.split()[0])
    return out


def proc_rss_kb(pid: str) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except FileNotFoundError:
        return None
    return None


def main() -> None:
    m = meminfo_kb()
    total = m["MemTotal"] / 1024
    avail = m.get("MemAvailable", 0) / 1024
    print(f"system: total={total:.0f}MiB available={avail:.0f}MiB used={total - avail:.0f}MiB")
    if len(sys.argv) > 1:
        pid = sys.argv[1]
        rss = proc_rss_kb(pid)
        print(f"pid {pid}: rss={rss / 1024:.1f}MiB" if rss is not None else f"pid {pid}: not found")


if __name__ == "__main__":
    main()
