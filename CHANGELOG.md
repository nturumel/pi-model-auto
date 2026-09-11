# Changelog

## Unreleased

### Breaking changes

- `modeModels` entries now require `{ "model": "provider/id", "thinking": "level" }`, pinning both the endpoint and reasoning effort for each configured capability mode.

### Changed

- Refreshed the bundled Ramp SWE-Bench data against the August 2026 78-task result set, updated all measured resolve-rate, cost, and tool-call values, and added the high-effort DeepSeek V4 Flash result. The full measured table remains available while Ramp's highlighted score-versus-spend wall is tracked separately for routing priority.
- Replaced the default length/context heuristic with a checksum-pinned, local INT8 ModernBERT complexity classifier. The heuristic remains available with `classifier: "off"`, while `classifier: "llm"` or `classifierModel` retains the bounded provider-backed strategy.
- Replaced Transformers.js's unused native image dependency with a text-only stub so the local classifier loads in OMP's embedded runtime; image prompts still conservatively route High before classifier inference.
- Reworked the router footer as a policy-to-model breadcrumb: `AUTO`, `PIN`, and `ONCE` identify route lifetime; active turns show the selected concrete model; settled and reset states retain the last model without implying it is pinned.
- Added unified one-turn `@route:<mode|provider/model>` overrides on any interactive turn. Exact-model overrides are restricted to endpoints admitted to the router pool, and the next turn returns to automatic routing.
- Added the CLI-safe `--route ultra|provider/model` flag for print mode, where Pi consumes leading `@...` tokens as file references before extensions receive the prompt.
- Isolated router context, selections, classifier state, quota state, and footer decisions with a composite OMP session-ID/session-file key. Provider requests resolve through the session ID, while ambiguous nested shutdown events retain state because OMP does not identify their target.
- Declared the Medium capability model as Pi Router's compaction target, preserving normal provider-native compaction while isolating maintenance from task routing.
- Added session-scoped sticky routing through `/route <mode|provider/model>`, with argument completion for `auto`, modes, and admitted exact models. `/route auto` and `@route:auto <prompt>` restore automatic selection, while the status line and `/auto` output identify forced routing explicitly.
- Shared router session state across per-session extension factory bindings so OMP subagent provider rebinding cannot orphan or misroute an active parent request.

## 0.2.1 - 2026-07-30

### Changed

- Added the Ramp SWE-Bench result for Kimi K3 at high effort: 86.1% resolve rate, $1.63 average cost per task, and 91 average tool-call turns. It is routed in the Ultra capability mode.

## 0.2.0 - 2026-07-26

### Breaking changes

- Replaced user-facing `cheap` / `strong` routing hints with capability-mode hints: `low`, `medium`, `high`, and `ultra`.
- `@low`, `@medium`, `@high`, `@ultra`, and `@model:provider/model` are now only honored on the first user turn of a conversation. Mid-conversation prefixes are ignored and the user is warned, avoiding accidental high-cost routing over long existing histories.
- Core API `resolveRouteModel({ hint })` now accepts `low | medium | high | ultra | auto | provider/model`; `cheap` and `strong` no longer resolve.
- Router config now uses `modeModels` for optional `low` / `medium` / `high` / `ultra` endpoint pins instead of cheap/strong tier pins.
- Classifier output now uses the same mode vocabulary directly (`mode=<low|medium|high|ultra>`) instead of the old `trivial` / `normal` / `hard` / `max` labels.
- The LLM classifier is now opt-in: it only runs when `classifierModel` pins an exact model; default routing uses the local heuristic with no extra classifier model call.

### Changed

- Artificial Analysis (`aa`) routing now maps models onto the same Low / Medium / High / Ultra capability modes using conservative Intelligence Index bands (`Ultra >= 56`, `High >= 52`, `Medium > 41`, `Low <= 41`), so both `ramp` and `aa` share the same user-facing mode vocabulary.
- README now documents representative models for each capability mode and explains fallback behavior when those models are not available locally.
