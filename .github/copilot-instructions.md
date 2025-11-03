## Squiggy — Copilot / AI Agent Instructions

Short, actionable guidance to help an AI coding agent be productive in this repository.

1) Big picture (where to look first)
  - Frontend (Positron extension): `src/` — entry: `src/extension.ts`. UI panels live in `src/views/` and `src/webview/`.
  - Backend (Python package): `squiggy/` — core I/O, plotting and normalization. Key files: `squiggy/io.py`, `squiggy/plot_factory.py`, `squiggy/plot_strategies/`, `squiggy/rendering/`, `squiggy/normalization.py`.
  - Kernel comms: `src/backend/squiggy-positron-runtime.ts` (Positron runtime) and `src/backend/squiggy-python-backend.ts` (subprocess fallback).

2) Critical integration patterns (do exactly this)
  - NEVER use `print()` in Python to pass data to TypeScript. Use the Positron variable-access pattern implemented in `src/backend/squiggy-positron-runtime.ts`.
    Example: execute code silently to set a Python variable, then call `positron.runtime.getSessionVariables(sessionId, [[varname]])` to retrieve JSON.
  - When generating plot HTML, Python returns a Bokeh HTML string (see `squiggy.plot_read()` in `squiggy/__init__.py`). TypeScript displays it directly in a webview.

3) Build / test / dev commands (what actually works here)
  - Preferred full environment: pixi (recommended). Common commands:
    - `pixi install && pixi run setup` — install Python + Node deps
    - `pixi run dev` — watch / dev
    - `pixi run test` — run Python + TypeScript tests
    - `pixi run build` — produce `.vsix`
  - If using standard tools:
    - `npm run watch` / `npm run compile` / `npm run package`
    - `pytest tests/` (Python tests)
  - Useful npm scripts (see `package.json`): `compile`, `watch`, `test`, `package`, `sync` (runs `scripts/sync-version.js`).

4) Repo-specific conventions & gotchas
  - Positron runtime is required for full integration; extension behavior differs when running in plain VS Code (subprocess fallback used).
  - Python state is kept in kernel globals — functions like `load_pod5()` set `_current_pod5_reader` in `squiggy/io.py`.
  - BAM files must be indexed (`.bai`) and use MM/ML tags for modification support.
  - POD5 VBZ compressed files may require `vbz_h5py_plugin` (installed via pod5 package).
  - Version sync: `npm run sync` updates `squiggy/__init__.py` from `package.json` — keep both in sync when releasing.

5) Where to add tests & what to run
  - TypeScript tests use Jest; look under `src/**/__tests__/` and run `npm test`.
  - Python tests live in `tests/` and use pytest. Use `pytest tests/ -v`.
  - Unit tests generally avoid real POD5 files; integration tests use `tests/data/` (Git LFS) — run `git lfs pull` before running tests that need data.

6) Small, high-value examples for automation
  - Re-run a plot with changed options (TypeScript): `vscode.commands.executeCommand('squiggy.plotRead', readId)` — used throughout `src/extension.ts` to refresh when options change.
  - Retrieve a slice of reads (Positron pattern): execute silent code to assign `temp = json.dumps(my_python_object)` then `getSessionVariables(sessionId, [["temp"]])` and `del temp`.

7) Files to inspect for further context (quick checklist)
  - `src/extension.ts` — activation and command registration
  - `src/backend/squiggy-positron-runtime.ts` — how the runtime fetches Python variables
  - `squiggy/io.py` — global kernel state and load functions
  - `squiggy/plot_factory.py` — Strategy Pattern factory for plot generation
  - `squiggy/plot_strategies/` — Individual plot strategy implementations (5 strategies: SINGLE, OVERLAY, STACKED, EVENTALIGN, AGGREGATE)
  - `squiggy/rendering/` — Reusable rendering components (ThemeManager, BaseAnnotationRenderer, ModificationTrackBuilder)
  - `package.json` / `pyproject.toml` — scripts and dependencies
  - `scripts/sync-version.js` — version sync between TS and Python

8) When to ask the human
  - If a change requires a new CI credential or secret (do not attempt to access secrets).
  - If you need clarification on which release channel (alpha vs stable) to publish to.

If this looks correct, I will commit this file. Tell me if you want more examples (small snippets) added, or a stricter lint/test checklist.
