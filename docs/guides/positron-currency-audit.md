# Positron currency audit (2026-10-11)

This audit checks how well Squiggy keeps up with upstream Positron as of the 2026.10.0 release (2026-10-08). It also covers standard VS Code extension practices. The previous audit was #182 (2026-06-26), which synced the type stub to upstream commit `1212541`.

Branch: `chore/positron-currency`

## 1. What changed upstream

Each API was dated by finding the first release tag whose `src/positron-dts/positron.d.ts` declares it.

Sources:
- GitHub releases: <https://github.com/posit-dev/positron/releases>
- d.ts history: <https://github.com/posit-dev/positron/commits/main/src/positron-dts/positron.d.ts>

| Release (date) | Change relevant to Squiggy |
|---|---|
| 2025.07 (07-02) | Kernels no longer start in a login shell by default (`kernelSupervisor.runInShell` restores it). |
| 2025.10 (10-10) | `runtime.deleteSession()` is added. |
| 2025.12 (12-02) | Pyright is no longer bundled. The upstream *Python Environments* extension is documented as not working in Positron. |
| 2026.04 (04-07) | `runtime.evaluateCode(lang, code, token?, sessionId?)` is added.<br>`restartSession()` now returns `Thenable<boolean>` (it was `void`).<br>`executeCode()` gains `documentUri`. |
| 2026.05 (05-04) | The default extension gallery moves from Open VSX to Posit Public Package Manager (P3M), which mirrors Open VSX. |
| 2026.09 (09-03) | New `RuntimeBusyBehavior` and `evaluateCode(..., whenBusy)`.<br>New `RuntimeCodeExecutionMode.Unprocessed`, `executeCode(..., attributionMetadata)` and `getConsoleHistory()`.<br>`BaseLanguageRuntimeSession` gains optional `getRuntimeState()`, `onDidChangeRuntimeState`, `onDidDisconnect` and `onDidReconnect`. |
| 2026.10 (10-08) | `startLanguageRuntime(..., options?: { owner?, activate? })` (posit-dev/positron#16341). `activate: false` starts a session without taking the foreground.<br>`registerRuntimeFromPath` and the command `workbench.action.language.runtime.registerRuntimeFromPath(languageId, path)` (#16378).<br>New `interpreters.definitions` and `interpreters.discovery` settings (#16289).<br>`positron.context.ephemeralState` (process-lifetime workspace memento).<br>Code OSS base 1.134.0. |

Things that have **not** changed:
- **Background sessions:** `LanguageRuntimeSessionMode.Background` exists, but the public API still cannot create one. `extHost.positron.api.impl.ts` hard-codes Console or Notebook mode.
- **Bokeh routing:** `bokeh.io.show()` is still intercepted and routed to the Plots pane. `PositronViewerBrowser` checks the call stack for `bokeh.io.showing.show` (see `positron-python/python_files/posit/positron/ui.py` and `patch/bokeh.py`). No routing changes appear in 2025–2026 release notes.
- **Deprecations:** the upstream d.ts has no `@deprecated` tags.

Platform and tooling (as of 2026-10-11):
- **VS Code base:** stable Positron 2026.10.0 is on Code OSS **1.134.0**, and main is on 1.138.0. `@types/vscode` latest is 1.140.0.
- **`@posit-dev/positron`:** latest is **0.2.11** (2026-10-08). Its README recommends installing it as a devDependency. The official extension template declares `engines.positron: ^2025.6.x` and `activationEvents: []`.
- **Publishing:** Positron installs from P3M, which mirrors Open VSX, so publishing to Open VSX (`ovsx publish`) is still the route.
  - Tool versions: ovsx 1.2.0, @vscode/vsce 4.0.0.

## 2. What the repo does today

- **Kernel** (`src/backend/squiggy-kernel-manager.ts`): a dedicated *console* session started with `startLanguageRuntime(runtimeId, 'Squiggy Dedicated Kernel')`.
  - Every call passes an explicit `sessionId`.
  - Variable reads use `evaluateCode` when it is available, with the `getSessionVariables` round-trip as a fallback.
  - The session ID is persisted in `workspaceState` so the extension can reconnect after a window reload.
- **Interpreter** (`src/backend/venv-manager.ts`): creates `~/.venvs/squiggy` with `uv` and relies on Positron discovering `~/.venvs`.
  - The kernel manager polls `getRegisteredRuntimes()` for a `runtimePath` containing `.venvs/squiggy`. If it finds none, it falls back to `getPreferredRuntime('python')`.
  - After creating the venv, the extension asks the user to restart Positron.
- **Type stub** (`src/types/squiggy-positron.d.ts`): a hand-maintained subset of the upstream d.ts, last synced 2026-06-26.
- **Webviews:**
  - Five React panels share `getReactWebviewHtml` (nonce CSP).
  - The Motif Explorer is a legacy plain-HTML panel.
  - No panel uses `retainContextWhenHidden`.
- **Manifest:** `engines: { vscode: ^1.96.0, positron: ^2025.6.0 }` and `activationEvents: [onStartupFinished]`.
  - `@posit-dev/positron@^0.1.0` is listed as a runtime dependency but is never imported.
- **Plots:** `squiggy/utils/plotting.py::_route_to_plots_pane()` calls `bokeh.io.show()`. This is still the supported path.

## 3. Gaps, prioritized

### A. Breaking, incorrect or security issues (fixed on this branch)

| # | Issue | Files | Fix |
|---|---|---|---|
| A1 | The Motif Explorer webview had **no Content-Security-Policy**. It used inline `on*` handlers and inserted FASTA-derived strings (chrom names, sequences) through `innerHTML` without escaping. | `src/views/squiggy-motif-panel.ts` | Added a nonce CSP (`default-src 'none'`). Handlers now use `addEventListener` and event delegation. Match fields are HTML-escaped. `localResourceRoots: []`. Tests added. |
| A2 | The CSP nonce was generated with `Math.random()`. | `src/utils/webview-utils.ts` | Now uses `crypto.randomBytes(16)`. Also dropped the unused `connect-src 'self'` and narrowed `localResourceRoots` to `build/`. |
| A3 | The Squiggy kernel **takes over the user's foreground console session** when it starts. | `squiggy-kernel-manager.ts` | Now passes `{ activate: false }` to `startLanguageRuntime` (2026.10+). Older versions ignore the extra argument. |
| A4 | `restartSession()` returns `false` when the user declines to interrupt a busy kernel. The code ignored this, waited for readiness, timed out, and then the recovery path **started a second session**. | `squiggy-kernel-manager.ts` | Detects `false`, keeps the existing session and restores the `Ready` state. Test added. |
| A5 | `waitForReady()` resolved through its `1+1` probe without setting `Ready`. If no state event followed (the kernel was already idle), later `execute()` calls threw "not ready". | `squiggy-kernel-manager.ts` | The probe now records `Ready`. `getRuntimeState()` (2026.09+) gives an early exit. Tests added. |
| A6 | The type stub drifted from upstream:<br>• it widened `getForegroundSession()` to `LanguageRuntimeSession`<br>• it was missing `startLanguageRuntime` options, `evaluateCode` `whenBusy`, `executeCode` `attributionMetadata`, `RuntimeBusyBehavior`, `Unprocessed`, `LanguageRuntimeSessionMode`, `RuntimeExitReason`/`LanguageRuntimeExit` and the optional state accessors on the base session<br>• `LanguageRuntimeMetadata` was incomplete | `src/types/squiggy-positron.d.ts`, `src/backend/positron-runtime-client.ts` | Synced to main as of 2026-10-11, with each newer member annotated with the release that introduced it. `PositronRuntimeClient` now guards the optional event. |
| A7 | Dead `view/item/context` menus targeted `squiggyReadList`. That view is a webview, so `viewItem` tree menus never apply. | `package.json` | Removed. |
| A8 | The manifest was missing `license`, `homepage`, `bugs` and `keywords` (vsce and Open VSX show these). It also did not declare that the extension cannot run in untrusted or virtual workspaces. | `package.json` | Added the metadata, `extensionKind: ["workspace"]`, and `capabilities.untrustedWorkspaces` / `virtualWorkspaces` set to `supported: false` with descriptions. |

Note on A8: the next `npm install` will add `"license": "MIT"` to the root entry of `package-lock.json`. That lock change was deliberately left out of this branch to avoid conflicting with the concurrent dependency-bump branch.

### B. Recommended (deferred)

| # | Issue | Files | Suggested fix |
|---|---|---|---|
| B1 | `engines.positron: ^2025.6.0` understates the real minimum. `deleteSession()` (2025.10+) is called without a guard in `shutdown()`/`dispose()`, and on older Positron it throws a `TypeError` synchronously inside `dispose()`. | `package.json`, `squiggy-kernel-manager.ts` | Raise to `^2025.10.0`, or guard with `typeof positron.runtime.deleteSession === 'function'`. If the minimum is raised to `^2026.04.0`, the `evaluateCode` feature detection and the `getSessionVariables` fallback can be deleted. |
| B2 | **Do not raise `engines.vscode` / `@types/vscode` above `^1.134.0`.** Stable Positron is on Code OSS 1.134.0, so a higher value blocks installation. The current `^1.96.0` is fine. | `package.json` | Leave as is, or raise to at most `^1.134.0` together with an `engines.positron: ^2026.10.0` bump. |
| B3 | Venv discovery is brittle:<br>• `runtimePath.includes('.venvs/squiggy')` fails on Windows (backslashes) and ignores `$SQUIGGY_VENV`.<br>• Positron 2026.10's `interpreters.discovery: "definitionsOnly"` hides `~/.venvs` entirely.<br>• The "restart Positron after venv creation" prompt is a workaround for discovery. | `squiggy-kernel-manager.ts`, `venv-manager.ts`, `extension.ts` | Compare against `VenvManager.getVenvPython()` using `path.resolve`. On 2026.10+, call `workbench.action.language.runtime.registerRuntimeFromPath('python', venvPython)` after creating the venv, use the returned runtime directly, and drop the restart prompt. Note that the command also persists the path in Python settings. |
| B4 | The fallback to `getPreferredRuntime('python')` can silently start a kernel in the user's environment that has no `squiggy`. That only fails later, in `verifySquiggyAvailable`. | `squiggy-kernel-manager.ts` | Fail fast with an actionable error ("Reset Python Environment"), or verify the package before adopting the runtime. |
| B5 | The session ID is persisted in `workspaceState`. That state survives an application restart, when the session cannot be reconnected. | `squiggy-kernel-manager.ts` | Use `positron.context.ephemeralState` (2026.10+, `await whenReady`), which upstream documents for exactly this "runtime session mapping" case. Fall back to `workspaceState`. |
| B6 | `@posit-dev/positron@^0.1.0` is a runtime `dependency` but is never imported. Types come from the local stub, and the module is resolved through webpack `externals`. | `package.json`, `package-lock.json` | Either move it to `devDependencies` and bump to `^0.2.11`, then replace the hand-written stub with its types (`export type * from 'positron'`); or remove it. Coordinate with the dependency-bump branch. |
| B7 | `PositronRuntimeClient` (foreground-session client) is dead code. Only tests reference it, and the single-kernel design (#168) removed its callers. | `src/backend/positron-runtime-client.ts` and its test | Delete it and its test, and update the comments in `runtime-client-interface.ts` and `squiggy-runtime-api.ts`. |
| B8 | CLAUDE.md still documents the temp-variable + `getSessionVariables` + `display_value` parsing pattern as the primary approach. The code now prefers `evaluateCode`. | `CLAUDE.md`, `docs/developer.md` | Document `evaluateCode(..., sessionId)` as primary and the old pattern as a fallback for Positron < 2026.04. |
| B9 | The release workflow treats a failed Open VSX publish as a warning, so a release can ship without reaching Positron users. | `.github/workflows/release.yml` | Make the publish failure fatal for tagged stable releases, except for "version already exists". Bump `ovsx` to 1.x and `@vscode/vsce` to 4.x on the dependency branch. |

### C. Nice to have

| # | Item | Suggested change |
|---|---|---|
| C1 | `getRuntimeState()` / `onDidDisconnect` / `onDidReconnect` (2026.09+) | Use them in the reconnect path to resync state after a supervisor disconnect instead of re-probing. |
| C2 | `evaluateCode(..., whenBusy: RuntimeBusyBehavior.Reject)` | For UI reads, such as read lists and modification summaries, where queueing behind a long plot is worse than failing fast. |
| C3 | `activationEvents: ["onStartupFinished"]` | This is reasonable, because the extension sets context keys and auto-starts the kernel when enabled. If startup cost matters, rely on implicit `onView`/`onCommand` activation (VS Code ≥ 1.74) plus `workspaceContains:**/squiggy-session.json` for pipeline sessions. |
| C4 | Webview state | Use `vscode.getState()`/`setState()` in the React panels so filters and scroll position survive the panel being hidden, without `retainContextWhenHidden`. |
| C5 | Motif Explorer | Port it to the shared React webview (per CLAUDE.md "React-first"). After that, every panel goes through `getReactWebviewHtml`. |
| C6 | Testing | Run tests against a real Positron with `posit-dev/setup-positron` and `@posit-dev/positron-test-electron` (used by the official template). |

## 4. Not verified

- Whether `bokeh.io.show()` called from `Silent`-mode `executeCode` still reaches the Plots pane on 2026.10. It should, because the routing goes through `webbrowser`, but it needs a manual check in Positron.
- Whether `{ activate: false }` behaves as intended on 2026.10.0. The PR was merged 2026-10-05 and is in the 2026.10.0 tag's d.ts. It needs a manual check that the user's console stays in the foreground.
- How quickly P3M picks up new Open VSX releases.
