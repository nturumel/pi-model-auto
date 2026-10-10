# pi-model-auto

`pi-model-auto` adds one Pi model: **Pi Router (Auto)**.

Choose it once with `/model`. After that, Pi keeps using the router, and the router chooses one of your authenticated models for each turn.

## Start

Install from npm:

```bash
pi install npm:pi-model-auto
```

Or install from git (pin a release with `@vX.Y.Z`, or omit it to track the default branch):

```bash
pi install git:github.com/maynewong/pi-model-auto@v0.1.0
```

Update later with `pi update --extensions`.

To try it once without installing, point `-e` at a local checkout:

```bash
pi -e /path/to/pi-model-auto
```

Then:

1. Run `/model`.
2. Choose **Pi Router (Auto)**.
3. Use Pi normally.

No config is required at first. If the router says no authenticated models are available, run `/login` for the providers you want to use, then reload Pi.

## Use

Check what the router sees:

```text
/auto
```

The footer is a routing breadcrumb. `AUTO → gpt-5.6-sol · Ultra · xhigh` identifies the persistent automatic policy and the concrete model selected for the active turn; after completion it becomes `AUTO · last: gpt-5.6-sol · Ultra · xhigh`. Sticky routes use `PIN Ultra`, and inline overrides use `ONCE High` while active. Immediately after `/route auto`, `AUTO · next: dynamic · last: <model>` makes the reset explicit without pretending the next model has already been selected. `/auto` reports the detailed routing decision.

Keep a route across turns with `/route`, then return to automatic routing with `/route auto`:

```text
/route ultra
/route openai-codex/gpt-5.6-sol
/route auto
```

Sticky routing is session-scoped. `/route` argument completion lists `auto`, all capability modes, and exact models currently admitted to the router pool. `@route:auto <prompt>` clears sticky routing and routes that prompt automatically.

Most turns should use automatic routing. To override any single turn—including in the middle of a conversation—start that prompt with `@route:`:

```text
@route:low summarize this file
@route:medium implement this small change
@route:high debug this failing test
@route:ultra investigate this architecture issue
@route:openai-codex/gpt-5.6-sol use Sol here
```

- `@route:low|medium|high|ultra` selects one configured capability option.
- `@route:provider/model-id` selects one exact endpoint already admitted to the router pool.
- While `pi-router/auto` is selected, typing `@route:` offers the configured modes and exact models from the router pool.
- The override applies only to that turn; the following prompt returns to automatic routing.
- The prefix must be the first character and is removed before the model sees the prompt.
- Without a prefix, task difficulty chooses the route automatically.

For non-interactive CLI use, `@...` is reserved for file expansion before extensions see the prompt. Pass the same route expression through `--route`:

```bash
pi --route ultra -p "investigate this architecture issue"
omp --route openai-codex/gpt-5.6-sol -p "use Sol here"
```

Leading whitespace does not activate an override. `cheap` and `strong` are not supported route names; use `low` and `ultra`.

## Configure When Needed

Most people only need config for two reasons:

1. Limit the pool to providers they trust.
2. Tell the router what each model really costs them.

Config lives in either file:

- `~/.pi/agent/model-router.json`
- `.pi/model-router.json` in trusted projects

Project config overrides user config.

```jsonc
{
  "router": {
    "modelFilter": { "include": ["openai-codex"], "exclude": [] },
    "modeModels": {
      "low": { "model": "openai-codex/gpt-5.6-luna", "thinking": "high" },
      "medium": { "model": "openai-codex/gpt-5.6-terra", "thinking": "high" },
      "high": { "model": "openai-codex/gpt-5.6-sol", "thinking": "high" },
      "ultra": { "model": "openai-codex/gpt-5.6-sol", "thinking": "xhigh" }
    },
    "modelOverrides": {
      "anthropic/claude-3-5-sonnet-20241022": { "costCoef": 0.05 },
      "z-ai/glm-5.2": {
        "costCoef": 0.4,
        "costCoefHours": [{ "hours": [14, 18], "factor": 3 }]
      }
    }
  }
}
```

Use provider/model ids exactly as they appear in your Pi registry. Each `modeModels` entry pairs its endpoint with the reasoning effort used for that mode, so two modes may use the same model at different effort levels. `modeModels` is optional; if you omit it, the router builds each mode and effort from benchmark metadata.

### `costCoef`

`costCoef` multiplies the benchmark cost:

- `< 1`: cheaper for you than the benchmark, such as a subscription or discount.
- `= 1`: roughly benchmark cost.
- `> 1`: more expensive for you.

Avoid setting a limited subscription near zero unless you want it to win almost every turn.

### `costCoefHours`

Use this when a model is more expensive during local hours:

```jsonc
"z-ai/glm-5.2": {
  "costCoef": 0.4,
  "costCoefHours": [{ "hours": [14, 18], "factor": 3 }]
}
```

This means `0.4` normally, `1.2` from 14:00 through 17:59. Windows are half-open `[start, end)`. `[22, 2]` wraps across midnight.

## How It Chooses

The router compares quality against your effective cost.

Quality comes from one benchmark table. Cost starts from the same table, then applies your `costCoef` and any active time window. When the table reports reasoning effort, the router treats `(model, effort)` as the measured operating point. The router keeps to the efficient frontier: a model-effort variant is only worth considering if no other available variant is both better and cheaper.

`capabilitySource` chooses the benchmark:

- `"ramp"` (default): this package's [Ramp SWE-Bench](https://labs.ramp.com/swebench#score-vs-spend) table, using coding-agent resolve rate and measured cost per task. It is a narrow coding-agent slice, not a general model score. The task family follows [SWE-bench](https://arxiv.org/abs/2310.06770).
- `"aa"`: [Artificial Analysis](https://artificialanalysis.ai/models) model data, using its Intelligence Index and blended price metrics.

The numeric tables live in [`src/canonical-models.ts`](src/canonical-models.ts). The two sources are not mixed.

The bundled Ramp table keeps every measured row because users rarely have identical model access. Routing first filters that table to the locally available, authenticated, quota-eligible models and recomputes the Pareto frontier with effective `costCoef` pricing. Models on Ramp's highlighted score-versus-spend wall receive priority only when their marginal capability gain remains within the selected mode's willingness budget; otherwise the router keeps the locally better fallback.

The user-facing status label is a capability mode. Cost is a separate routing axis and never determines this label.

For the Ramp source, the mode is derived only from SWE-bench solve rate:

| mode | Ramp solve rate | representative measured models |
| --- | --- | --- |
| `Ultra` | `>= 85%` | `claude-opus-5`, `claude-fable-5`, `kimi-k3` |
| `High` | `80–85%` | `gpt-5.5`, `gpt-5.6-sol`, `grok-4.5`, `glm-5.2` |
| `Medium` | `75–80%` | `deepseek-v4-flash`, `claude-opus-4-8`, `gpt-5.6-terra`, `claude-sonnet-5` |
| `Low` | `< 75%` | `gpt-5.4-nano`, `qwen3.7-plus`, `gpt-5.6-luna`, `gemini-3.1-pro` |

For the AA source, the same four modes are mapped from Artificial Analysis Intelligence Index bands:

| mode | AA Intelligence Index | representative AA models |
| --- | --- | --- |
| `Ultra` | `>= 56` | `claude-opus-5`, `claude-fable-5`, `gpt-5.6-sol`, `kimi-k3` |
| `High` | `52–56` | `gpt-5.6-terra`, `grok-4.5`, `claude-opus-4-7`, `claude-sonnet-5`, `gpt-5.5` |
| `Medium` | `> 41 and < 52` | `gpt-5.4`, `gpt-5.6-luna`, `glm-5.2`, `gemini-3.1-pro`, `kimi-k2.7-code` |
| `Low` | `<= 41` | `deepseek-v4-flash`, `glm-5.1`, `gpt-5.4-mini`, `gpt-5.4-nano`, `gemini-3.1-flash-lite` |

Representative models are examples from the bundled benchmark tables, not requirements. The router only uses models that are present in your local Pi registry, authenticated, and allowed by `modelFilter`. If none of your available models are in the target mode, the router borrows the nearest stronger mode; if no stronger mode exists, it uses the strongest available lower mode. If no authenticated benchmark-backed models are available at all, it asks you to `/login` or configure model metadata.

Task difficulty chooses the mode, while the continuous difficulty score sets a capability floor inside that mode. A configured `modeModels` entry pins both the model and its reasoning effort for that mode. Without a configured entry, the router picks the cheapest effective-cost model meeting the floor, then permits only affordable `willingness` upgrades inside the same mode. It never enters the next mode early. Models without capability-mode metadata retain Pareto routing.

In automatic routing, configured mode effort takes precedence over benchmark-backed effort, which takes precedence over Pi's session default before the model's `thinkingLevelMap` is applied. Exact-model overrides still honor Pi's selected effort. When a turn is harder than it looks, prefix that prompt with `@route:high` or `@route:ultra`.

The classifier runs locally and makes no provider call.

### Classifier and Coarse Auto Detection

The default classifier is
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
(22.7M parameters, Apache-2.0) fine-tuned with [`training/train.py`](training/train.py) on the
Apache-2.0 [`llm-query-complexity-benchmark`](https://huggingface.co/datasets/anasnassar/llm-query-complexity-benchmark)
train split, then exported to ONNX with per-channel int8 weights (23 MB). Installation downloads the
model and tokenizer from this fork's `classifier-v2` release, verifies fixed SHA-256 checksums, and
fails the install on any mismatch. Inference is fully local and disables remote model loading. A
classifier directory that ships `onnx/model.onnx` is loaded in fp32; otherwise `model_quantized.onnx`.

Only the latest user message is classified, after removing host-injected `<system-reminder>` blocks.
The model is warmed when the Pi session starts, input is capped at 128 tokens, and the session is
released on shutdown. Its `LOW`, `MEDIUM`, and `HIGH` labels map directly to the router's `low`,
`medium`, and `high` modes; it never guesses `ultra`. Image turns route to `high`, and predictions
below the 0.45 confidence floor fail upward to `high`. One-turn overrides and tool-call continuations
do not run another classification.

Measured with [`training/bench.py`](training/bench.py) on the 1,200-query benchmark test split and on
the 500 SWE-bench Verified issues (human difficulty: `<15 min` = low, `15 min–1 h` = medium, longer = high):

| classifier | test macro-F1 | High recall | severe under-route | SWE issues sent to `low` | p50 / p95 (Bun) | disk | process RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MiniLM-L6 fine-tune, int8 per-channel (default) | 0.663 (3 seeds: 0.668 ± 0.007) | 69.8% | 0.7% | 0.0% | 1.1 / 1.7 ms | 23 MB | ~180 MB |
| ModernBERT `classifier-v1`, int8 (previous) | 0.575 | 62.5% | 4.6% | 4.4% | 6.7 / 10 ms | 151 MB | ~390 MB |

Larger encoders (Ettin 68M/150M, ModernBERT-base, granite-small-r2), frozen embeddings with logistic
regression, and zero-shot local LLMs (Qwen3 0.6B/1.7B, Qwen2.5 1.5B) all scored lower or cost more;
zero-shot LLMs took 240–370 ms per prompt and collapsed to a single label. No prompt-only classifier
separated short from long SWE-bench fixes (ROC-AUC ≈ 0.55), so treat coding-task difficulty as
unresolved: the router's job there is to avoid sending real repository work to `low`, which it does.

#### Heuristic fallback

Set `"classifier": "off"` to disable semantic classification. The fallback score remains:

```text
score = normalize(estimatedContextTokens, 8_000, 120_000) * weights.contextTokens
      + normalize(lastUserMessage.length, 120, 1_200) * weights.lastUserLen
      + min(1, recentToolResults / 8) * weights.toolDensity
```

It maps `< 0.30` to `low`, `0.30–0.52` to `medium`, `0.52–0.74` to `high`, and `>= 0.74` to
`ultra`. This fallback is deterministic and free, but it cannot recognize semantically hard short
requests.

```jsonc
{
  "router": {
    "classifier": "off"
  }
}
```

#### Optional LLM classifier

To replace local inference with the existing bounded LLM classifier, pin an exact endpoint:

```jsonc
{
  "router": {
    "classifier": "llm",
    "classifierModel": "gateway/gpt-5.4-nano"
  }
}
```

`classifierModel` also selects the LLM strategy when `classifier` is omitted. The LLM receives only
the estimated context size, image flag, and latest user message; output is capped at 80 tokens with
temperature 0, no internal retry, and the configured timeout. Repeated failures trigger its existing
cooldown and heuristic fallback.

One user turn keeps one model, including tool-call continuations. Automatic routing also avoids
quota-cooled plans and avoids switching away from a useful warm cache when the switch is not worth it.

### Context Window and Compaction

Pi displays context usage and decides when to compact from the selected `pi-router/auto` model. The router keeps that virtual model's `contextWindow` synchronized with the concrete model selected for the current turn. It preselects the route before Pi's preflight compaction check, refreshes the value again when the provider request starts, and restores the most recent concrete model's window when resuming a session.

As a result, a turn routed to a 272K model displays and compacts against 272K; a later turn routed to a 1M model switches the same limits to 1M. The initial 1M registration is only a startup placeholder before a session target can be identified.

## Settings

| setting | use |
| --- | --- |
| `capabilitySource` | Choose `"ramp"` or `"aa"`. |
| `modelFilter` | Include or exclude providers/models by substring. |
| `modeModels` | Pin an exact endpoint and thinking level for `low`, `medium`, `high`, or `ultra`. |
| `modelOverrides` | Adjust cost or metadata for known/private/local models. |
| `willingness` | Control affordable same-mode upgrades after a capability mode is selected. |
| `cacheAware` | Keep warm prompt caches when switching is not worth it. Enabled by default. |
| `quota` | Skip cooled-down plans after rate-limit headers or `429`. Enabled by default. |
| `classifier` | Select `"local"` (default), `"llm"`, or `"off"`. Object tuning remains available for LLM timeout/cooldown settings. |
| `classifierModel` | Pin the optional LLM classifier to an exact provider/model or variant; setting it selects the LLM strategy. |
| `weights` | Language-neutral difficulty-scoring weights. Advanced. |
| `log` | Append routing decisions to `.pi/router.log`. |

Useful override fields:

| field | meaning |
| --- | --- |
| `costCoef` | Cost multiplier. |
| `costCoefHours` | Local-hour multipliers. |
| `canonical` | Name shown in `/auto`. |
| `costTier` | Cost-only classification: `cheap`, `standard`, `premium`, or `unknown`. |
| `capabilityMode` | Override the capability mode: `low`, `medium`, `high`, or `ultra`. |
| `profiles` | `deep`, `fast`, `coder`, `balanced`, `vision`, `frontier`. |
| `frontier` | Whether the model can appear in the frontier display/pool. |
| `benchmarkEffort` | Pi-normalized effort (`minimal`, `low`, `medium`, `high`, `xhigh`) backing a manual metric row. |
| `priceBlended`, `intelligence`, `scores`, `tps` | Raw metrics for models without benchmark data. |

Quota state is stored at `~/.pi/agent/quota-state.json`. Providers without remaining-quota headers only cool down after a real `429`.

## Core API

Other Pi extensions can resolve a model without an `ExtensionContext`:

```ts
import { resolveRouteModel } from "pi-model-auto/core";

const selection = resolveRouteModel({
  models: availableModels,
  hint: "high", // routing hint: low | medium | high | ultra | auto | provider/model
  context,
});
```

The core API loads user-level `model-router.json` and quota state by default. It never reads project config because that requires a trust decision from the host. Pass `cfg` to supply an explicit configuration or `filterQuota: false` to disable persisted cooldown filtering.

## Develop

```bash
npm run build
npm run typecheck
npm test
```

Maintainers: see [RELEASING.md](RELEASING.md) for the publish flow.
