#!/usr/bin/env bash
# Register QEMU so an x86_64 host can run linux/arm64 Docker images.
# NOT needed on Apple Silicon (arm64 runs natively). Requires Docker.
set -euo pipefail
docker run --privileged --rm tonistiigi/binfmt --install arm64
echo "arm64 binfmt registered. Verify:"
ls /proc/sys/fs/binfmt_misc/ | grep -i aarch64 || echo "  (not found — re-run with a working Docker daemon)"
