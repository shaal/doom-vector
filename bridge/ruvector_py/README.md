# ruvector_py — native RuVector binding

A thin PyO3 wrapper over `ruvector-core` (v2.2.0). Compiles to a single `.so`
that loads in-process from Python — no Node.js runtime, which matters on the
Pi's 512 MB.

## Build (desktop / x86)
```bash
pip install maturin
maturin develop -m bridge/ruvector_py/Cargo.toml --release
python -c "import ruvector_py; m=ruvector_py.RuVectorMemory(3); \
  m.insert([0.1,0.2,0.3], None, {'return': 1.0, 'action_idx': 2.0}); \
  print(m.search([0.1,0.2,0.3], 1))"
```

> **Dev-env note:** maturin aborts if both `VIRTUAL_ENV` and `CONDA_PREFIX` are
> set — `unset CONDA_PREFIX` first. A `uv`-created venv has no `pip`; install
> maturin with `uv pip install maturin` and build with
> `maturin develop --release --uv -m bridge/ruvector_py/Cargo.toml` (the `--uv`
> flag is required so maturin uses uv instead of pip to install the module).

## Cross-build for the Pi (aarch64)
See `deploy/build_pyo3_arm64.sh` (maturin + zig). Build on a fast machine; copy
the resulting `dist/*aarch64*.whl` to the Pi and `pip install` it.

## API
```python
RuVectorMemory(dimensions: int, storage_path: str | None = None)
  .insert(vector: list[float], id: str | None = None, metadata: dict[str,float] | None = None) -> str
  .search(vector: list[float], k: int = 8) -> list[tuple[id, score, metadata]]
```
Metadata is float-valued only (sufficient for episodic control: `action_idx`,
`return`). Extend `lib.rs` if you need string/JSON metadata.

**Persistence gotcha:** storage is on by default. With no `storage_path`, the
store writes a redb file at `./ruvector_store` **relative to CWD** that persists
across runs and rebuilds the index on reopen — so omitting it can silently reuse
or collide on an old store (especially if the vector dim changes). Always pass an
explicit per-instance `storage_path` (a fresh one for clean runs; a stable one to
deliberately keep learning across runs).

## Verified (x86_64, 2026-06-10)
Compiled **unmodified** against `ruvector-core` 2.2.0 + `pyo3` 0.23.5 (Rust 1.95).
maturin produced a **2.15 MB abi3 (py3.9+) wheel** linking only libc/libm/libgcc_s
— self-contained, ideal to cross-build and drop onto the Pi. Smoke test (insert →
search, float metadata round-trip) passed.
