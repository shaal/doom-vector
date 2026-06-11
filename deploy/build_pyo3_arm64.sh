#!/usr/bin/env bash
# Cross-compile the ruvector_py wheel for aarch64 (Pi Zero 2 W) using maturin + zig.
# Build on your fast machine: compiling ruvector-core on a 512 MB Pi is impractical.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

pip install --upgrade maturin ziglang
rustup target add aarch64-unknown-linux-gnu

# --zig uses ziglang as the cross-linker, avoiding a full cross toolchain.
maturin build --release \
  --target aarch64-unknown-linux-gnu \
  --zig \
  -m "$ROOT/bridge/ruvector_py/Cargo.toml" \
  --out "$ROOT/dist"

echo
echo "Built aarch64 wheel(s) in $ROOT/dist :"
ls -1 "$ROOT/dist" | grep -i aarch64 || true
echo "Copy the aarch64 wheel to the Pi and: pip install <wheel>"
