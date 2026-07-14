# Decisions log (solo)

Append-only. Newest at bottom.

## 2026-07-14 — Dual-machine ops

- **Decided:** Solo ownership (no A/B split). All workstreams owned by Arush.
- **Decided:** Cursor agents develop on CCC/login checkout and push to `Arush777/Priority_KV`.
- **Decided:** H200 is human-operated only: `git pull` + `uv` + GPU runs. No agents on H200.
- **Decided:** Env manager is `uv` with `pyproject.toml` + lockfile; GPU extras via `uv sync --extra gpu`.
- **Decided:** Primary model pin remains `Qwen/Qwen3-8B` @ `b968826d9c46dd6066d109eabc6255188de91218`.
