/**
 * Pi manages context against the selected `pi-router/auto` model, while this provider chooses a
 * concrete model per turn. The virtual model is therefore refreshed after routing so Pi's footer
 * and compaction threshold follow the model that actually serves the request.
 */
import { appendFileSync, existsSync, mkdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import {
  buildSessionContext,
  CONFIG_DIR_NAME,
  convertToLlm,
  getAgentDir,
  type ExtensionAPI,
  type ExtensionContext,
} from "@earendil-works/pi-coding-agent";
import {
  clampThinkingLevel,
  completeSimple,
  createAssistantMessageEventStream,
  streamSimple as aiStreamSimple,
  type Api,
  type AssistantMessage,
  type AssistantMessageEvent,
  type Context,
  type ImageContent,
  type Model,
  type SimpleStreamOptions,
} from "@earendil-works/pi-ai";
import {
  AA_WILLINGNESS,
  RAMP_WILLINGNESS,
  DEFAULT_CONFIG,
  axisValue,
  buildAutoPool,
  cacheAwareSelect,
  createClassifierState,
  createRoutingState,
  decide,
  enableClassifierForPinnedModel,
  estimateContextTokens,
  frontierChain,
  lastUserText,
  loadUserRouterConfig,
  matchesModelFilter,
  mergeClassifierConfig,
  modelKey,
  parseCapabilityMode,
  parseClassificationOutput,
  recordClassifierFailure,
  recordClassifierSuccess,
  recordRoutingUsage,
  repriceForTimeOfDay,
  resolveModel,
  routingReasoning,
  routingTurnKey,
  selectFromPool,
  selectClassifierModel,
  shouldReuseTurnSelection,
  userTurnIndex,
  variantKey,
  type CacheReason,
  type ClassificationResult,
  type ClassifierState,
  type Decision,
  type Pool,
  type ResolvedModel,
  type RouterConfig,
  type RoutingState,
  type Selection,
  type TaskClassifier,
  type CapabilityMode,
} from "./router-core.ts";
import {
  classifyLocally,
  warmLocalClassifier,
  disposeLocalClassifier,
  classifierInput,
  type LocalComplexityLabel,
} from "./local-classifier.ts";
import { QuotaState, buildPlanKey, filterPoolByQuota, type PlanState } from "./quota.ts";
import { createEscalationState, isAbove, modeIndex, updateEscalation, type EscalationState } from "./escalation.ts";

type ForcedRoute = { mode: CapabilityMode } | { model: string };
type ResolvedAuth = { ok: true; apiKey?: string; headers?: Record<string, string>; env?: Record<string, string> };
type QuotaPlanLookup = Map<string, { planKey: string; auth: Awaited<ReturnType<ExtensionContext["modelRegistry"]["getApiKeyAndHeaders"]>> }>;

const LOCAL_MODE: Record<LocalComplexityLabel, CapabilityMode> = {
  LOW: "low",
  MEDIUM: "medium",
  HIGH: "high",
};

const ZERO_USAGE = {
  input: 0,
  output: 0,
  cacheRead: 0,
  cacheWrite: 0,
  totalTokens: 0,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

interface LastDecision extends Decision {
  chosen: string;
  planKey: string;
  canonical: string | null;
  costTier: string;
  capabilityMode?: string;
  profile?: string;
  benchmarkEffort?: string;
  reasoning?: string;
  confidence: string;
  alternatives: string[];
  cacheReason?: CacheReason;
}
type RouterSessionKey = `${string}\0${string}`;


interface RouterSessionState {
  ctx: ExtensionContext;
  cfg: RouterConfig;
  quota: QuotaState;
  pool: Pool;
  forcedRoute?: ForcedRoute;
  cliForcedRoute?: ForcedRoute;
  stickyRoute?: ForcedRoute;
  lastDecision?: LastDecision;
  turnSelection?: { key: string; decision: Decision; selection: Selection; cacheReason?: CacheReason; pending?: boolean };
  routingState: RoutingState;
  classifierState: ClassifierState;
  localClassifierErrorShown: boolean;
  escalation: EscalationState;
  /** A throwaway copy used for omp side calls (recaps, probes); never touches the UI or the parent's turn. */
  side?: boolean;
}

interface RouterProcessState {
  sessions: Map<RouterSessionKey, RouterSessionState>;
  sessionKeysById: Map<string, RouterSessionKey>;
  memorySessionSequence: number;
  /** Most recently started session; routes calls whose session id the router cannot match. */
  lastStartedKey?: RouterSessionKey;
}

const ROUTER_PROCESS_STATE_KEY = "__pi_model_auto_router_process_state_v1__" as const;

type RouterGlobal = typeof globalThis & {
  [ROUTER_PROCESS_STATE_KEY]?: RouterProcessState;
};

function getRouterProcessState(): RouterProcessState {
  const host = globalThis as RouterGlobal;
  const existing = host[ROUTER_PROCESS_STATE_KEY];
  if (existing) return existing;

  const created: RouterProcessState = {
    sessions: new Map(),
    sessionKeysById: new Map(),
    memorySessionSequence: 0,
  };
  host[ROUTER_PROCESS_STATE_KEY] = created;
  return created;
}


export default function modelRouter(pi: ExtensionAPI) {
  const processState = getRouterProcessState();
  const { sessions, sessionKeysById } = processState;
  const ownedSessions = new Map<RouterSessionKey, string>();
  let routerContextWindow: number | undefined;
  let routerCompactionModel: string | undefined;
  const autocompleteContexts = new WeakSet<object>();
  const initialRouterConfig = loadUserRouterConfig();
  syncRouterContextWindow(1_000_000, initialRouterConfig.modeModels.medium?.model);

  pi.registerFlag("route", {
    type: "string",
    description: "Override the first turn with low|medium|high|ultra or provider/model",
  });

  pi.registerCommand("auto", {
    description: "Show Pi Model Router pool and last decision",
    handler: async (_args, ctx) => {
      const state = sessionState(ctx);
      ctx.ui.notify(
        describeRouter(
          repriceForTimeOfDay(state.pool, new Date().getHours()),
          state.cfg,
          state.lastDecision,
          state.quota,
          state.stickyRoute,
        ),
        "info",
      );
    },
  });

  pi.registerCommand("route", {
    description: "Keep routing through a mode or model until /route auto",
    getArgumentCompletions(argumentPrefix) {
      if (argumentPrefix.includes(" ")) return null;
      const normalized = argumentPrefix.trim().toLowerCase();
      const items: Array<{ value: string; label: string; description: string }> = [
        { value: "auto", label: "auto", description: "Clear sticky routing" },
        ...(["low", "medium", "high", "ultra"] as const).map((mode) => ({
          value: mode,
          label: mode,
          description: `Keep routing through ${mode[0].toUpperCase() + mode.slice(1)}`,
        })),
      ];
      const seen = new Set<string>();
      for (const state of sessions.values()) {
        for (const candidate of state.pool.all) {
          const key = modelKey(candidate.model);
          const normalizedKey = key.toLowerCase();
          if (seen.has(normalizedKey)) continue;
          seen.add(normalizedKey);
          items.push({
            value: key,
            label: key,
            description: "Keep routing through this exact model",
          });
        }
      }
      const filtered = items.filter((item) => item.label.toLowerCase().startsWith(normalized));
      return filtered.length > 0 ? filtered : null;
    },
    handler: async (args, ctx) => {
      const state = sessionState(ctx);
      const target = args.trim();
      if (!target) {
        ctx.ui.notify(`Pi Router: ${state.stickyRoute ? stickyRouteLabel(state.stickyRoute) : "Auto"}. Use /route <mode|provider/model|auto>.`, "info");
        return;
      }
      if (target.toLowerCase() === "auto") {
        state.stickyRoute = undefined;
        state.forcedRoute = undefined;
        state.turnSelection = undefined;
        state.escalation = createEscalationState();
        updateRouterStatus(ctx, readyStatus(undefined, state.lastDecision, state.quota));
        ctx.ui.notify("Pi Router: sticky route cleared; automatic routing restored.", "info");
        return;
      }

      const route = parseRouteHint(target);
      if (!route || ("model" in route && !state.pool.all.some((item) => modelKey(item.model) === route.model))) {
        ctx.ui.notify(`Pi Router: invalid or unavailable route: ${target}`, "warning");
        return;
      }
      state.stickyRoute = route;
      state.forcedRoute = undefined;
      state.turnSelection = undefined;
      updateRouterStatus(ctx, readyStatus(state.stickyRoute, state.lastDecision, state.quota));
      ctx.ui.notify(`${stickyRouteLabel(route)} until /route auto.`, "info");
    },
  });

  pi.on("session_start", async (_event, ctx) => {
    startSession(ctx);
  });

  /** Build and register router state for ctx's session (also used when session_start was missed). */
  function startSession(ctx: ExtensionContext): RouterSessionState {
    const cfg = loadConfig(ctx);
    const cliRoute = pi.getFlag("route");
    const cliForcedRoute = typeof cliRoute === "string" ? parseRouteHint(cliRoute) : undefined;
    const quota = new QuotaState(cfg.quota);
    quota.load(quotaStateFile());
    const sessionId = ctx.sessionManager.getSessionId();
    const key = routerSessionKey(ctx, ++processState.memorySessionSequence);
    const state: RouterSessionState = {
      ctx,
      cfg,
      quota,
      pool: applyConfiguredTiers(buildAutoPool(ctx.modelRegistry.getAvailable(), cfg), cfg, ctx),
      forcedRoute: cliForcedRoute,
      cliForcedRoute,
      routingState: createRoutingState(),
      classifierState: createClassifierState(),
      localClassifierErrorShown: false,
      escalation: createEscalationState(),
    };
    const previousKey = sessionKeysById.get(sessionId);
    if (previousKey) {
      sessions.delete(previousKey);
      ownedSessions.delete(previousKey);
    }
    sessions.set(key, state);
    sessionKeysById.set(sessionId, key);
    ownedSessions.set(key, sessionId);
    processState.lastStartedKey = key;
    if (!autocompleteContexts.has(ctx)) {
      autocompleteContexts.add(ctx);
      ctx.ui.addAutocompleteProvider((current) => ({
        triggerCharacters: ["@"],
        async getSuggestions(lines, line, col, options) {
          const beforeCursor = (lines[line] ?? "").slice(0, col);
          const match = beforeCursor.match(/^@route:([^\s]*)$/i);
          if (!isRouterModel(ctx.model) || !match) {
            return current.getSuggestions(lines, line, col, options);
          }

          const typedTarget = (match[1] ?? "").toLowerCase();
          const modes = (["low", "medium", "high", "ultra"] as const).map((mode) => {
            const configured = sessionState(ctx).cfg.modeModels[mode];
            const details = configured
              ? `${configured.model}${configured.thinking ? ` · ${configured.thinking}` : ""} · one turn`
              : "Automatic model selection · one turn";
            return { value: `@route:${mode} `, label: `@route:${mode}`, description: details };
          });
          const seen = new Set<string>();
          const models = sessionState(ctx).pool.all.flatMap((candidate) => {
            const key = modelKey(candidate.model);
            const normalized = key.toLowerCase();
            if (seen.has(normalized)) return [];
            seen.add(normalized);
            return [{ value: `@route:${key} `, label: `@route:${key}`, description: "Exact model · one turn" }];
          });
          const items = [...modes, ...models].filter((item) =>
            item.label.slice("@route:".length).toLowerCase().startsWith(typedTarget)
          );
          return { prefix: match[0], items };
        },
        applyCompletion(lines, line, col, item, prefix) {
          return current.applyCompletion(lines, line, col, item, prefix);
        },
        shouldTriggerFileCompletion(lines, line, col) {
          const beforeCursor = (lines[line] ?? "").slice(0, col);
          if (isRouterModel(ctx.model) && /^@route:[^\s]*$/i.test(beforeCursor)) return false;
          return current.shouldTriggerFileCompletion?.(lines, line, col) ?? true;
        },
      }));
    }
    if (typeof cliRoute === "string" && !cliForcedRoute) {
      ctx.ui.notify(`Pi Router: invalid --route value: ${cliRoute}`, "warning");
    }
    if (cfg.classifier.enabled && cfg.classifier.strategy === "local") {
      void warmLocalClassifier().catch((error) => {
        if (state.localClassifierErrorShown) return;
        state.localClassifierErrorShown = true;
        ctx.ui.notify(`Pi Router: local classifier unavailable; using heuristic (${String(error)})`, "warning");
      });
    }

    const recentModel = mostRecentConcreteModel(ctx);
    syncRouterContextWindow(
      recentModel?.contextWindow ?? routerContextWindow ?? 1_000_000,
      state.cfg.modeModels.medium?.model,
    );

    updateRouterStatus(ctx, state.pool.all.length === 0 ? "No routed models" : readyStatus(state.stickyRoute, state.lastDecision, state.quota));
    return state;
  }

  pi.on("model_select", async (event, ctx) => {
    const state = sessionStateById(ctx.sessionManager.getSessionId());
    if (isRouterModel(event.model)) {
      updateRouterStatus(
        ctx,
        state?.lastDecision
          ? settledStatus(state.lastDecision, state.quota, state.stickyRoute)
          : readyStatus(state?.stickyRoute, undefined, state?.quota),
      );
    } else {
      ctx.ui.setStatus("router", undefined);
    }
  });

  pi.on("session_shutdown", async () => {
    // A factory normally owns one session. If a shared runner reports several starts, the
    // shutdown event does not identify its target; retaining them is safer than deleting a parent.
    if (ownedSessions.size !== 1) return;

    const entry = ownedSessions.entries().next().value;
    if (!entry) return;
    const [key, sessionId] = entry;
    ownedSessions.clear();
    sessions.delete(key);
    if (sessionKeysById.get(sessionId) === key) sessionKeysById.delete(sessionId);
    if (sessions.size === 0) await disposeLocalClassifier();
  });

  pi.on("input", async (event, ctx) => {
    if (event.source === "extension") return { action: "continue" };
    const state = sessionState(ctx);

    const parsed = parseForcedRoute(event.text);
    const initialTurn = isInitialUserTurn(ctx);
    if (parsed?.auto) {
      state.stickyRoute = undefined;
      state.turnSelection = undefined;
      updateRouterStatus(ctx, readyStatus(undefined, state.lastDecision, state.quota));
    }
    state.forcedRoute = parsed ? parsed.route : (initialTurn ? state.cliForcedRoute : undefined);
    const routedText = parsed?.text ?? event.text;
    // Pi checks preflight compaction after input hooks but before `streamSimple`, so the target
    // window must be installed here or a large→small model switch can miss its compact threshold.
    if (isRouterModel(ctx.model) && !event.streamingBehavior) {
      try {
        await preselectTurn(
          state,
          inputRoutingContext(ctx, routedText, event.images),
          state.forcedRoute ?? state.stickyRoute,
        );
      } catch {
        state.turnSelection = undefined;
      }
    }

    return parsed
      ? { action: "transform", text: routedText, images: event.images }
      : { action: "continue" };
  });

  function streamSimple(routerModel: Model<Api>, context: Context, options?: SimpleStreamOptions) {
    const stream = createAssistantMessageEventStream();

    let activeState: RouterSessionState | undefined;

    void (async () => {
      try {
        const state = streamSessionState(context, options);
        activeState = state;
        const { ctx, cfg, pool, quota } = state;
        if (pool.all.length === 0) {
          throw new Error("Pi Router: no authenticated models. Run /login or configure an API key, then /reload.");
        }

        // Compaction is maintenance summarization, not task-complexity work. Preserve Pi's normal
        // compaction request unchanged, while pinning only its model selection to Medium.
        const requestRoute: ForcedRoute | undefined = options?.codexCompaction
          ? { mode: "medium" }
          : (state.forcedRoute ?? state.stickyRoute);
        const quotaPlans = cfg.quota.enabled && !requestRoute ? await resolveQuotaPlans(ctx, pool) : new Map();
        const turnKey = routingTurnKey(context);
        const cachedSelection = state.turnSelection;
        const preselectedTurn =
          !options?.codexCompaction && cachedSelection?.pending === true && cachedSelection.key === turnKey;
        const reuseTurnSelection =
          preselectedTurn ||
          (!options?.codexCompaction &&
            shouldReuseTurnSelection(context) &&
            cachedSelection?.key === turnKey &&
            cachedSelection.decision.cls !== "model");
        const currentClassifier = !reuseTurnSelection && !options?.codexCompaction && !shouldReuseTurnSelection(context)
          ? await classifyCurrentTurnWhenUseful(state, context)
          : undefined;
        const classified = reuseTurnSelection
          ? cachedSelection!.decision
          : decide(context, options, requestRoute, cfg, currentClassifier ? { classify: () => currentClassifier } : undefined);
        const decision = applyEscalation(state, context, classified, requestRoute, Boolean(options?.codexCompaction), turnKey);
        const reuse = reuseTurnSelection && decision === classified;
        const statusRoute = options?.codexCompaction ? undefined : state.stickyRoute;
        const oneTurnRoute = options?.codexCompaction ? undefined : state.forcedRoute;
        // Re-evaluate time-of-day shadow-price windows once per turn (clock read here, pick reused
        // within the turn), so a window boundary like GLM 14:00–18:00 takes effect without a /reload.
        const selectionPool = decision.cls === "model" || reuse
          ? pool
          : repriceForTimeOfDay(
            requestRoute ? pool : usablePoolForQuota(ctx, cfg, pool, quota, Date.now(), quotaPlans),
            new Date().getHours(),
          );

        let selection: Selection;
        let cacheReason: CacheReason | undefined = preselectedTurn ? cachedSelection?.cacheReason : undefined;
        if (reuse) {
          const reuseReason = preselectedTurn ? "preselected before compaction" : "reused within user turn";
          selection = { ...cachedSelection!.selection, reason: `${cachedSelection!.selection.reason}; ${reuseReason}` };
        } else {
          const fresh = selectModel(decision, selectionPool, context, options, cfg);
          // Explicit model/mode pins are contracts; cache economics may only replace unpinned auto selections.
          const configuredPin = decision.cls === "model" ? undefined : cfg.modeModels[decision.cls];
          if (!requestRoute && decision.cls !== "model" && !configuredPin) {
            const result = cacheAwareSelect(fresh, state.routingState, selectionPool, context, cfg);
            selection = result.selection;
            cacheReason = result.cacheReason;
          } else {
            selection = fresh;
          }
        }
        const target = selection.selected.model;

        state.turnSelection = { key: turnKey, decision, selection, cacheReason };

        const selectedAuth = quotaPlans.get(modelKey(target))?.auth;
        const auth = selectedAuth ?? await ctx.modelRegistry.getApiKeyAndHeaders(target);
        if (!auth.ok) throw new Error(auth.error);
        const planKey = quotaPlans.get(modelKey(target))?.planKey ?? modelPlanKey(target, auth);

        if (!state.side) syncRouterContextWindow(target.contextWindow, cfg.modeModels.medium?.model);

        const configuredMode = decision.cls === "model" ? undefined : cfg.modeModels[decision.cls];
        const requestedReasoning = routingReasoning(
          configuredMode?.thinking,
          selection.benchmarkEffort,
          options?.reasoning,
          decision.cls === "model",
        );
        const clampedReasoning = target.reasoning ? clampThinkingLevel(target, requestedReasoning) : "off";
        const reasoning = clampedReasoning === "off" ? undefined : clampedReasoning;
        const maxTokens = Math.min(options?.maxTokens ?? target.maxTokens, target.maxTokens);

        const streamDecision: LastDecision = {
          ...decision,
          chosen: modelKey(target),
          planKey,
          canonical: selection.selected.canonicalKey,
          costTier: selection.selected.costTier,
          capabilityMode: selection.selected.capabilityMode,
          profile: selection.profile,
          benchmarkEffort: selection.benchmarkEffort,
          reasoning: clampedReasoning,
          confidence: selection.selected.confidence,
          reason: selection.reason,
          alternatives: selection.alternatives,
          cacheReason,
        };
        state.lastDecision = streamDecision;

        if (!state.side) updateRouterStatus(ctx, activeStatus(streamDecision, quota, statusRoute, oneTurnRoute));
        logDecision(ctx, cfg, streamDecision);

        const inner = aiStreamSimple(target, context, {
          ...options,
          apiKey: auth.apiKey,
          headers: auth.headers,
          env: auth.env,
          reasoning,
          maxTokens,
          onResponse: async (response, model) => {
            await options?.onResponse?.(response, model);
            if (!cfg.quota.enabled) return;

            recordQuotaChange(ctx, cfg, quota, planKey, () =>
              quota.recordResponse(planKey, response.status, response.headers, target.provider, Date.now()),
            );
            if (!state.side) updateRouterStatus(ctx, activeStatus(streamDecision, quota, statusRoute, oneTurnRoute));
          },
        });

        for await (const event of inner) {
          if (event.type === "error" && looksRateLimited(event.error) && cfg.quota.enabled) {
            recordQuotaChange(ctx, cfg, quota, planKey, () =>
              quota.recordRateLimited(planKey, undefined, undefined, Date.now()),
            );
            if (!state.side) updateRouterStatus(ctx, activeStatus(streamDecision, quota, statusRoute, oneTurnRoute));
          }

          stream.push(event);
          if (event.type === "done" && cfg.cacheAware.enabled) {
            // Re-establish the warm lease from realized usage so the next turn can weigh staying vs switching.
            recordRoutingUsage(state.routingState, selection.selected, event.message.usage, context);
          }
          if (event.type === "done" || event.type === "error") {
            logTerminalEvent(ctx, cfg, streamDecision, event);
            state.forcedRoute = undefined;
            if (!state.side) updateRouterStatus(ctx, settledStatus(streamDecision, quota, state.stickyRoute));
          }
        }
        stream.end();
        activeState = undefined;
      } catch (error) {
        if (activeState) {
          activeState.forcedRoute = undefined;
          if (!activeState.side) updateRouterStatus(
            activeState.ctx,
            readyStatus(activeState.stickyRoute, activeState.lastDecision, activeState.quota),
          );
        }
        stream.push(makeRouterError(routerModel, error));
        stream.end();
      }
    })();

    return stream;
  }

  async function preselectTurn(
    state: RouterSessionState,
    context: Context,
    requestRoute: ForcedRoute | undefined,
  ): Promise<void> {
    const { ctx, cfg, pool, quota } = state;
    if (pool.all.length === 0) return;

    const quotaPlans = cfg.quota.enabled && !requestRoute ? await resolveQuotaPlans(ctx, pool) : new Map();
    const currentClassifier = requestRoute
      ? undefined
      : await classifyCurrentTurnWhenUseful(state, context);
    const classified = decide(context, undefined, requestRoute, cfg, currentClassifier ? { classify: () => currentClassifier } : undefined);
    const decision = applyEscalation(state, context, classified, requestRoute, false, routingTurnKey(context));
    const selectionPool = decision.cls === "model"
      ? pool
      : repriceForTimeOfDay(
        requestRoute ? pool : usablePoolForQuota(ctx, cfg, pool, quota, Date.now(), quotaPlans),
        new Date().getHours(),
      );
    const fresh = selectModel(decision, selectionPool, context, undefined, cfg);
    let selection = fresh;
    let cacheReason: CacheReason | undefined;
    const configuredPin = decision.cls === "model" ? undefined : cfg.modeModels[decision.cls];
    if (!requestRoute && decision.cls !== "model" && !configuredPin) {
      const result = cacheAwareSelect(fresh, state.routingState, selectionPool, context, cfg);
      selection = result.selection;
      cacheReason = result.cacheReason;
    }

    state.turnSelection = { key: routingTurnKey(context), decision, selection, cacheReason, pending: true };
    syncRouterContextWindow(selection.selected.model.contextWindow, cfg.modeModels.medium?.model);
  }

  function inputRoutingContext(
    ctx: ExtensionContext,
    text: string,
    images: ImageContent[] | undefined,
  ): Context {
    const session = buildSessionContext(ctx.sessionManager.getBranch());
    return {
      systemPrompt: ctx.getSystemPrompt(),
      messages: [
        ...convertToLlm(session.messages),
        {
          role: "user",
          content: [{ type: "text", text }, ...(images ?? [])],
          timestamp: Date.now(),
        },
      ],
    };
  }

  function mostRecentConcreteModel(ctx: ExtensionContext): Model<Api> | undefined {
    const messages = buildSessionContext(ctx.sessionManager.getBranch()).messages;
    for (let index = messages.length - 1; index >= 0; index--) {
      const message = messages[index];
      if (message.role !== "assistant" || message.provider === "pi-router") continue;
      const model = ctx.modelRegistry.find(message.provider, message.model);
      if (model) return model;
    }
    return undefined;
  }

  function syncRouterContextWindow(contextWindow: number, compactionModel?: string): void {
    if (routerContextWindow === contextWindow && routerCompactionModel === compactionModel) return;
    routerContextWindow = contextWindow;
    routerCompactionModel = compactionModel;
    pi.registerProvider("pi-router", {
      name: "Pi Router",
      api: "pi-router-api",
      baseUrl: "https://router.local",
      apiKey: "pi-router-dummy-key",
      models: [
        {
          id: "auto",
          name: "Pi Router (Auto)",
          reasoning: true,
          input: ["text", "image"],
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
          contextWindow,
          maxTokens: 64_000,
          ...(routerCompactionModel ? { compactionModel: routerCompactionModel } : {}),
        } as Model<Api> & { compactionModel?: string },
      ],
      streamSimple,
    });
  }

  /** Raise the decision to the session's escalation floor; explicit routes and compaction are left alone. */
  function applyEscalation(
    state: RouterSessionState,
    context: Context,
    decision: Decision,
    requestRoute: ForcedRoute | undefined,
    compaction: boolean,
    turnKey: string,
  ): Decision {
    // decision.forced also covers tool-call continuations that reuse a forced decision after the
    // one-turn route itself has been consumed by the turn's first request.
    if (requestRoute || compaction || decision.forced || decision.cls === "model") return decision;
    const { ctx, cfg } = state;
    const result = updateEscalation(state.escalation, context, cfg.escalation, decision.cls, cfg.modeModels, turnKey);
    if (result.reset && cfg.log) appendJsonLine(ctx, { ts: new Date().toISOString(), escalation: "reset" });
    if (result.raised) {
      const { from, to, reason } = result.raised;
      if (!state.side) ctx.ui.notify(`Pi Router: escalating ${from} → ${to} (${reason}).`, "info");
      if (cfg.log) appendJsonLine(ctx, { ts: new Date().toISOString(), escalation: { from, to, reason } });
    }
    if (!isAbove(result.floor, decision.cls)) return decision;
    return {
      ...decision,
      cls: result.floor,
      modeBucket: modeIndex(result.floor),
      reason: `escalated to ${result.floor}${result.raised ? `: ${result.raised.reason}` : " (held)"}`,
    };
  }

  async function classifyCurrentTurnWhenUseful(
    state: RouterSessionState,
    context: Context,
  ): Promise<ClassificationResult | undefined> {
    const { ctx, cfg } = state;
    if (state.forcedRoute || !cfg.classifier.enabled) return undefined;
    if (cfg.classifier.strategy === "local") {
      if (contextHasImageForClassifier(context)) {
        return { mode: "high", reason: "local classifier: image safety fallback" };
      }
      try {
        const text = lastUserText(context);
        const startedAt = performance.now();
        const result = await classifyLocally(text);
        if (cfg.log) {
          appendJsonLine(ctx, {
            ts: new Date().toISOString(),
            localClassifier: result.label,
            scores: result.scores,
            latencyMs: Math.round(performance.now() - startedAt),
            classifiedChars: classifierInput(text).length,
          });
        }
        return {
          mode: LOCAL_MODE[result.label],
          reason: `local classifier: ${result.label.toLowerCase()} (${result.scores[result.label].toFixed(3)})`,
        };
      } catch (error) {
        if (!state.localClassifierErrorShown) {
          state.localClassifierErrorShown = true;
          ctx.ui.notify(`Pi Router: local classifier unavailable; using heuristic (${String(error)})`, "warning");
        }
        return undefined;
      }
    }
    return classifyCurrentTurn(state, context);
  }

  async function classifyCurrentTurn(state: RouterSessionState, context: Context): Promise<ClassificationResult | undefined> {
    const { ctx, cfg, pool } = state;
    const turn = userTurnIndex(context);
    const candidate = selectClassifierModel(repriceForTimeOfDay(pool, new Date().getHours()), cfg, state.classifierState, turn);
    if (!candidate) {
      if (cfg.log) appendJsonLine(ctx, { ts: new Date().toISOString(), classifierError: "configured model unavailable" });
      return undefined;
    }

    const key = modelKey(candidate.model);
    const classifierReasoning = candidate.model.reasoning
      ? clampThinkingLevel(candidate.model, "minimal")
      : "off";
    try {
      const auth = await ctx.modelRegistry.getApiKeyAndHeaders(candidate.model);
      if (!auth.ok) {
        recordClassifierFailure(state.classifierState, key, turn, cfg);
        if (cfg.log) appendJsonLine(ctx, { ts: new Date().toISOString(), classifierError: "classifier authentication unavailable", chosen: key });
        return undefined;
      }

      const message = await completeSimple(candidate.model, classifierContext(context), {
        apiKey: auth.apiKey,
        headers: auth.headers,
        env: auth.env,
        reasoning: classifierReasoning === "off" ? undefined : classifierReasoning,
        maxTokens: 80,
        temperature: 0,
        timeoutMs: cfg.classifier.timeoutMs,
        maxRetries: 0,
      });
      const parsed = parseClassificationOutput(assistantText(message));
      if (!parsed) {
        recordClassifierFailure(state.classifierState, key, turn, cfg);
        if (cfg.log) appendJsonLine(ctx, { ts: new Date().toISOString(), classifierError: "invalid classifier response", chosen: key, response: assistantText(message) });
        return undefined;
      }

      recordClassifierSuccess(state.classifierState, key);
      return parsed;
    } catch (error) {
      recordClassifierFailure(state.classifierState, key, turn, cfg);
      if (cfg.log) appendJsonLine(ctx, { ts: new Date().toISOString(), classifierError: String(error), chosen: key });
      return undefined;
    }
  }

  function sessionState(ctx: ExtensionContext): RouterSessionState {
    const sessionId = ctx.sessionManager.getSessionId();
    // Pi can deliver events for a session whose session_start this extension never saw (for example
    // after a reload or for a subagent); start it on demand rather than failing the event.
    return sessionStateById(sessionId) ?? startSession(ctx);
  }

  function sessionStateById(sessionId: string): RouterSessionState | undefined {
    const key = sessionKeysById.get(sessionId);
    return key ? sessions.get(key) : undefined;
  }

  /** Same config, pool and quota as the parent; fresh per-turn state so side traffic cannot alter the parent's turn. */
  function sideState(parent: RouterSessionState): RouterSessionState {
    return {
      ...parent,
      forcedRoute: undefined,
      turnSelection: undefined,
      lastDecision: undefined,
      routingState: createRoutingState(),
      classifierState: createClassifierState(),
      escalation: createEscalationState(),
      side: true,
    };
  }

  function routerSessionKey(ctx: ExtensionContext, memorySequence: number): RouterSessionKey {
    const sessionId = ctx.sessionManager.getSessionId();
    const sessionFile = ctx.sessionManager.getSessionFile?.();
    return `${sessionId}\0${sessionFile ?? `memory:${memorySequence}`}`;
  }

  function streamSessionState(context: Context, options: SimpleStreamOptions | undefined): RouterSessionState {
    if (options?.sessionId) {
      const state = sessionStateById(options.sessionId);
      if (state) return state;
      // omp side calls (subagent completion probes, idle recaps) use "<session>:side:<hash>".
      const side = options.sessionId.indexOf(":side:");
      const parent = side > 0 ? sessionStateById(options.sessionId.slice(0, side)) : undefined;
      if (parent) return sideState(parent);
    }

    const turnKey = routingTurnKey(context);
    const matching = [...sessions.values()].filter((state) => {
      const messages = convertToLlm(buildSessionContext(state.ctx.sessionManager.getBranch()).messages);
      return routingTurnKey({ messages }) === turnKey;
    });
    if (matching.length === 1) return matching[0];
    if (sessions.size === 1) return sessions.values().next().value!;
    const latest = processState.lastStartedKey ? sessions.get(processState.lastStartedKey) : undefined;
    const fallback = latest ?? [...sessions.values()].at(-1);
    if (fallback) return sideState(fallback);

    const suffix = options?.sessionId ? `: ${options.sessionId}` : "";
    throw new Error(`Pi Router: no matching session state${suffix}`);
  }
}

function classifierContext(context: Context): Context {
  return {
    systemPrompt: [
      "You are Pi Router's classifier. Classify the current user turn for model routing.",
      "Return exactly one line and no prose: mode=<low|medium|high|ultra> profile=<balanced|coder|deep|fast|vision> score=<0..1>.",
      "Mode is required capability, not cost. Cost is handled later by the router.",
      "Mode rubric:",
      "- low: trivial, explain, summarize, docs/copy edits, tiny localized change, low ambiguity.",
      "- medium: routine coding/debugging/refactor, small scoped implementation, ordinary tests, a few files.",
      "- high: multi-file work, failing tests, unknown root cause, API/design decisions, non-trivial tool use, sizeable context.",
      "- ultra: architecture/security/migrations, production-risk changes, large refactor, long-horizon investigation, high ambiguity or high cost of failure.",
      "Profile rubric:",
      "- vision if hasImage=true.",
      "- fast only when the user explicitly prioritizes speed/latency or the task is a simple transform.",
      "- coder for implementation, debugging, refactoring, tests, or code review.",
      "- deep for architecture, root-cause analysis, security, planning, or long-horizon reasoning.",
      "- balanced otherwise.",
      "Score is position inside the chosen mode: low 0.00-0.29, medium 0.30-0.51, high 0.52-0.73, ultra 0.74-1.00.",
      "If uncertain between adjacent modes, choose the lower mode unless the task is risky, irreversible, security-sensitive, or explicitly asks for deep investigation.",
    ].join("\n"),
    messages: [{
      role: "user",
      content: [
        `estimatedContextTokens=${estimateContextTokens(context)}`,
        `hasImage=${contextHasImageForClassifier(context)}`,
        "lastUserMessage:",
        lastUserText(context),
      ].join("\n"),
      timestamp: Date.now(),
    }],
  };
}

function assistantText(message: AssistantMessage): string {
  return message.content
    .filter((part) => part.type === "text")
    .map((part) => part.text)
    .join("\n");
}

function contextHasImageForClassifier(context: Context): boolean {
  return context.messages.some(
    (message) => Array.isArray(message.content) && message.content.some((part) => part.type === "image"),
  );
}

function selectModel(
  decision: Decision,
  pool: Pool,
  context: Context,
  options: SimpleStreamOptions | undefined,
  cfg: RouterConfig,
) {
  if (decision.cls === "model") {
    const selected = pool.all.find((candidate) =>
      modelKey(candidate.model).toLowerCase() === decision.chosen.toLowerCase()
    );
    if (!selected) throw new Error(`Pi Router: forced model is not available in the router pool: ${decision.chosen}`);
    return { selected, profile: selected.profiles[0] ?? "balanced", benchmarkEffort: selected.benchmarkEffort, reason: "forced model", alternatives: [] };
  }
  const configured = decision.cls === "model" ? undefined : cfg.modeModels[decision.cls];
  if (configured) {
    const selected = pool.all.find((candidate) =>
      modelKey(candidate.model).toLowerCase() === configured.model.toLowerCase()
    );
    if (selected) {
      const pinned = selectFromPool(decision, { ...pool, all: [selected] }, context, options, cfg);
      if (pinned) return { ...pinned, reason: `configured ${decision.cls}: ${configured.model}` };
    }
  }

  const selection = selectFromPool(decision, pool, context, options, cfg);
  if (!selection) throw new Error("Pi Router: model pool is empty");
  return selection;
}

function applyConfiguredTiers(pool: Pool, cfg: RouterConfig, ctx: ExtensionContext): Pool {
  const next: Pool = {
    cheapPool: [...pool.cheapPool],
    strongPool: [...pool.strongPool],
    standardPool: [...pool.standardPool],
    unknownPool: [...pool.unknownPool],
    all: [...pool.all],
  };

  for (const mode of ["low", "medium", "high", "ultra"] as const) {
    const configured = cfg.modeModels[mode];
    if (!configured) continue;

    const model = findModelByRef(ctx, configured.model);
    if (!model) {
      ctx.ui.notify(`Pi Router: configured ${mode} model not found or unauthenticated: ${configured.model}`, "warning");
      continue;
    }

    const resolved = { ...resolveModel(model, cfg), capabilityMode: mode };
    if (!matchesModelFilter(resolved, cfg.modelFilter)) {
      ctx.ui.notify(`Pi Router: configured ${mode} model rejected by modelFilter: ${configured.model}`, "warning");
      continue;
    }

    prependUnique(next.all, resolved);
  }

  return next;
}

function prependUnique(items: ResolvedModel[], item: ResolvedModel) {
  const key = modelKey(item.model);
  const existing = items.findIndex((candidate) => modelKey(candidate.model) === key);
  if (existing >= 0) items.splice(existing, 1);
  items.unshift(item);
}

function findModelByRef(ctx: ExtensionContext, ref: string): Model<Api> | undefined {
  const [provider, ...idParts] = ref.split("/");
  const id = idParts.join("/");
  if (!provider || !id) return undefined;

  const model = ctx.modelRegistry.find(provider, id);
  if (!model || !ctx.modelRegistry.hasConfiguredAuth(model)) return undefined;
  return model;
}

function updateRouterStatus(ctx: ExtensionContext, text: string) {
  if (ctx.model && !isRouterModel(ctx.model)) {
    ctx.ui.setStatus("router", undefined);
    return;
  }
  ctx.ui.setStatus("router", text);
}

function isRouterModel(model: Model<Api> | undefined): boolean {
  return model?.provider === "pi-router" && model.id === "auto";
}

function parseRouteHint(text: string): ForcedRoute | undefined {
  const normalized = text.trim();
  const mode = parseCapabilityMode(normalized);
  if (mode) return { mode };
  return /^[^\s/]+\/[^\s]+$/.test(normalized) ? { model: normalized } : undefined;
}

function parseForcedRoute(text: string): { route?: ForcedRoute; auto: boolean; text: string } | undefined {
  const match = text.match(/^@route:([^\s]+)\s+([\s\S]*)$/i);
  if (!match) return undefined;
  if (match[1].toLowerCase() === "auto") return { auto: true, text: match[2] };
  const route = parseRouteHint(match[1]);
  return route ? { route, auto: false, text: match[2] } : undefined;
}

function isInitialUserTurn(ctx: ExtensionContext): boolean {
  return buildSessionContext(ctx.sessionManager.getBranch()).messages.every((message) => message.role !== "user");
}

function loadConfig(ctx: ExtensionContext): RouterConfig {
  let cfg = DEFAULT_CONFIG;
  let userWillingness: Partial<RouterConfig["willingness"]> = {};
  for (const file of configPaths(ctx)) {
    if (!existsSync(file)) continue;

    try {
      const parsed = JSON.parse(readFileSync(file, "utf8"));
      const router = parsed.router ?? parsed;
      cfg = {
        ...cfg,
        ...router,
        weights: { ...cfg.weights, ...(router.weights ?? {}) },
        modeModels: { ...cfg.modeModels, ...(router.modeModels ?? router.models ?? {}) },
        modelFilter: { ...cfg.modelFilter, ...(router.modelFilter ?? {}) },
        modelOverrides: { ...cfg.modelOverrides, ...(router.modelOverrides ?? router.overrides ?? {}) },
        cacheAware: { ...cfg.cacheAware, ...(router.cacheAware ?? {}) },
        quota: { ...cfg.quota, ...(router.quota ?? {}) },
        escalation: { ...cfg.escalation, ...(router.escalation ?? {}) },
        classifierModel: typeof router.classifierModel === "string" ? router.classifierModel : cfg.classifierModel,
        classifier: enableClassifierForPinnedModel(
          mergeClassifierConfig(router.classifier, cfg.classifier),
          typeof router.classifierModel === "string" ? router.classifierModel : cfg.classifierModel,
          router.classifier,
        ),
      };
      if (router.willingness) userWillingness = { ...userWillingness, ...router.willingness };
    } catch (error) {
      ctx.ui.notify(`Pi Router: failed to read ${file}: ${error instanceof Error ? error.message : String(error)}`, "warning");
    }
  }

  // The $/quality-point budgets live on different scales per source (list price vs measured cost-per-task),
  // so the base willingness follows capabilitySource; explicit user values overlay it.
  const baseWillingness = cfg.capabilitySource === "aa" ? AA_WILLINGNESS : RAMP_WILLINGNESS;
  cfg = { ...cfg, willingness: { ...baseWillingness, ...userWillingness } };
  return cfg;
}

function configPaths(ctx: ExtensionContext): string[] {
  const paths = [join(getAgentDir(), "model-router.json")];
  if (ctx.isProjectTrusted()) paths.push(join(ctx.cwd, CONFIG_DIR_NAME, "model-router.json"));
  return paths;
}

function quotaStateFile(): string {
  return join(getAgentDir(), "quota-state.json");
}

function describeRouter(
  pool: Pool,
  cfg: RouterConfig,
  lastDecision: LastDecision | undefined,
  quota: QuotaState,
  stickyRoute: ForcedRoute | undefined,
): string {
  const lines = [
    "Pi Router",
    `capabilitySource: ${cfg.capabilitySource}`,
    `cacheAware: ${cfg.cacheAware.enabled}`,
    `modelFilter: include=[${cfg.modelFilter.include.join(", ") || "*"}] exclude=[${cfg.modelFilter.exclude.join(", ") || "none"}]`,
    `quota: ${cfg.quota.enabled ? "enabled" : "disabled"}`,
    `routing: ${stickyRoute ? stickyRouteLabel(stickyRoute) : "Auto"}`,
    `costCheapPool: ${pool.cheapPool.map((item) => modelKey(item.model)).join(", ") || "none"}`,
    `frontierPool: ${pool.strongPool.map((item) => `${modelKey(item.model)}(${item.canonicalKey ?? "unknown"}/${item.costTier}/${item.profiles.join("+")})`).join(", ") || "none"}`,
    `costStandardPool: ${pool.standardPool.map((item) => modelKey(item.model)).join(", ") || "none"}`,
    `unknownPool: ${pool.unknownPool.map((item) => modelKey(item.model)).join(", ") || "none"}`,
    "frontier (auto climbs these Low→Ultra by mode):",
    ...(["coder", "deep", "balanced"] as const).map((profile) => {
      const chain = frontierChain(pool.all, profile);
      const points = chain
        .map((item) => `${displayVariantName(item)}(${axisValue(item, profile).toFixed(0)}@$${item.priceBlended})`)
        .join(" → ");
      return `  ${profile}: ${points || "none"}`;
    }),
    "overrides:",
    "  sticky: /route low|medium|high|ultra|provider/model",
    "  clear sticky: /route auto",
    "  one turn: @route:low|medium|high|ultra <prompt>",
    "  one turn, exact endpoint: @route:provider/model <prompt>",
    "  clear sticky + auto turn: @route:auto <prompt>",
    "  CLI/print mode: --route ultra|provider/model",
  ];

  if (lastDecision) {
    lines.push(
      "last:",
      `  chosen: ${lastDecision.chosen}`,
      `  planKey: ${lastDecision.planKey}`,
      `  canonical: ${lastDecision.canonical ?? "unknown"}`,
      `  costTier: ${lastDecision.costTier}`,
      `  capabilityMode: ${lastDecision.capabilityMode ?? "unknown"}`,
      `  profile: ${lastDecision.profile ?? "unknown"}`,
      `  benchmarkEffort: ${lastDecision.benchmarkEffort ?? "unknown"}`,
      `  reasoning: ${lastDecision.reasoning ?? "off"}`,
      `  confidence: ${lastDecision.confidence}`,
      `  cacheReason: ${lastDecision.cacheReason ?? "n/a"}`,
      `  reason: ${lastDecision.reason ?? "none"}`,
      `  alternatives: ${lastDecision.alternatives.join(", ") || "none"}`,
    );
  } else {
    lines.push("last: none");
  }

  const plans = quota.snapshots();
  if (plans.length > 0) {
    lines.push("plans:");
    for (const plan of plans) {
      lines.push(`  ${formatPlanState(plan)}`);
    }
  } else {
    lines.push("plans: none");
  }

  return lines.join("\n");
}

function displayVariantName(item: ResolvedModel): string {
  if (!item.canonicalKey) return variantKey(item);
  return item.benchmarkEffort ? `${item.canonicalKey}@${item.benchmarkEffort}` : item.canonicalKey;
}

function logDecision(ctx: ExtensionContext, cfg: RouterConfig, decision: LastDecision | undefined) {
  if (!cfg.log || !decision) return;
  appendJsonLine(ctx, { ts: new Date().toISOString(), ...decision });
}

function logTerminalEvent(
  ctx: ExtensionContext,
  cfg: RouterConfig,
  decision: LastDecision | undefined,
  event: AssistantMessageEvent,
) {
  if (!cfg.log || !decision) return;
  if (event.type !== "done" && event.type !== "error") return;

  const message = event.type === "done" ? event.message : event.error;
  appendJsonLine(ctx, {
    ts: new Date().toISOString(),
    chosen: decision.chosen,
    stopReason: message.stopReason,
    usage: message.usage,
  });
}

function appendJsonLine(ctx: ExtensionContext, value: unknown) {
  const file = join(ctx.cwd, CONFIG_DIR_NAME, "router.log");
  mkdirSync(dirname(file), { recursive: true });
  appendFileSync(file, `${JSON.stringify(value)}\n`, "utf8");
}

function usablePoolForQuota(
  ctx: ExtensionContext,
  cfg: RouterConfig,
  pool: Pool,
  quota: QuotaState,
  now: number,
  quotaPlans: QuotaPlanLookup,
): Pool {
  if (!cfg.quota.enabled) return pool;

  let changed = false;
  for (const planKey of new Set(pool.all.map((item) => quotaPlanKeyFor(item, quotaPlans)))) {
    const before = quota.snapshot(planKey);
    quota.isAvailable(planKey, now);
    const after = quota.snapshot(planKey);
    if (quotaStateChanged(before, after)) {
      changed = true;
      logQuotaChange(ctx, cfg, after);
    }
  }
  if (changed) quota.persist(quotaStateFile());

  return filterPoolByQuota(pool, quota, now, new Set(), (item) => quotaPlanKeyFor(item, quotaPlans));
}

function recordQuotaChange(
  ctx: ExtensionContext,
  cfg: RouterConfig,
  quota: QuotaState,
  planKey: string,
  update: () => PlanState,
): PlanState {
  const before = quota.snapshot(planKey);
  const after = update();
  quota.persist(quotaStateFile());
  if (quotaStateChanged(before, after)) logQuotaChange(ctx, cfg, after);
  return after;
}

function quotaStateChanged(before: PlanState | undefined, after: PlanState | undefined): boolean {
  return (
    before?.status !== after?.status ||
    before?.reason !== after?.reason ||
    before?.cooldownUntil !== after?.cooldownUntil
  );
}

function logQuotaChange(ctx: ExtensionContext, cfg: RouterConfig, state: PlanState | undefined) {
  if (!cfg.log || !state) return;
  appendJsonLine(ctx, {
    ts: new Date().toISOString(),
    planKey: state.planKey,
    status: state.status,
    reason: state.reason,
    cooldownUntil: state.cooldownUntil,
  });
}

function looksRateLimited(message: AssistantMessage): boolean {
  const text = `${message.stopReason ?? ""} ${message.errorMessage ?? ""}`.toLowerCase();
  return text.includes("429") || text.includes("rate limit") || text.includes("too many requests") || text.includes("quota");
}

function readyStatus(
  stickyRoute?: ForcedRoute,
  lastDecision?: LastDecision,
  quota?: QuotaState,
): string {
  const policy = stickyRoute ? `${stickyRouteLabel(stickyRoute)} · next: pinned` : "AUTO · next: dynamic";
  if (!lastDecision) return policy;
  const quotaTag = quota ? quotaStatusTag(quota.snapshot(lastDecision.planKey), Date.now()) : "";
  return `${policy} · last: ${selectedModelName(lastDecision)}${quotaTag}`;
}

function stickyRouteLabel(route: ForcedRoute): string {
  return routeLabel("PIN", route);
}

function routeLabel(prefix: "PIN" | "ONCE", route: ForcedRoute): string {
  const target = "mode" in route
    ? route.mode[0].toUpperCase() + route.mode.slice(1)
    : route.model;
  return `${prefix} ${target}`;
}

function activeStatus(
  decision: LastDecision,
  quota: QuotaState,
  stickyRoute?: ForcedRoute,
  oneTurnRoute?: ForcedRoute,
): string {
  const model = selectedModelName(decision);
  const effort = decisionEffort(decision);
  const mode = decisionMode(decision);
  const quotaTag = quotaStatusTag(quota.snapshot(decision.planKey), Date.now());
  if (oneTurnRoute) {
    const label = routeLabel("ONCE", oneTurnRoute);
    return "mode" in oneTurnRoute
      ? `${label} → ${model} · ${effort}${quotaTag}`
      : `${label} · ${mode} · ${effort}${quotaTag}`;
  }
  if (stickyRoute) {
    const label = stickyRouteLabel(stickyRoute);
    return "mode" in stickyRoute
      ? `${label} → ${model} · ${effort}${quotaTag}`
      : `${label} · ${mode} · ${effort}${quotaTag}`;
  }
  return `AUTO → ${model} · ${mode} · ${effort}${quotaTag}`;
}

function settledStatus(decision: LastDecision, quota: QuotaState, stickyRoute?: ForcedRoute): string {
  const quotaTag = quotaStatusTag(quota.snapshot(decision.planKey), Date.now());
  if (stickyRoute && "model" in stickyRoute) {
    return `${stickyRouteLabel(stickyRoute)} · ${decisionMode(decision)} · ${decisionEffort(decision)}${quotaTag}`;
  }
  const policy = stickyRoute ? stickyRouteLabel(stickyRoute) : "AUTO";
  return `${policy} · last: ${selectedModelName(decision)} · ${decisionMode(decision)} · ${decisionEffort(decision)}${quotaTag}`;
}

function selectedModelName(decision: LastDecision): string {
  return decision.chosen.split("/").at(-1) ?? decision.chosen;
}

function decisionEffort(decision: LastDecision): string {
  return decision.reasoning && decision.reasoning !== "off" ? decision.reasoning : "off";
}

function decisionMode(decision: LastDecision): string {
  return decision.capabilityMode
    ? decision.capabilityMode[0].toUpperCase() + decision.capabilityMode.slice(1)
    : `cost:${decision.costTier}`;
}

function quotaStatusTag(state: PlanState | undefined, now: number): string {
  if (state?.status === "cooldown" && state.cooldownUntil != null) {
    return ` ⏳${Math.max(0, Math.ceil((state.cooldownUntil - now) / 60_000))}m`;
  }

  const snapshot = state?.lastSnapshot;
  if (snapshot?.remaining != null && snapshot.limit != null && snapshot.limit > 0) {
    return ` ${Math.round((100 * snapshot.remaining) / snapshot.limit)}%`;
  }

  return "";
}

function formatPlanState(state: PlanState): string {
  const remaining = quotaStatusTag(state, Date.now()).trim();
  const cooldownUntil = state.cooldownUntil ? ` cooldownUntil=${new Date(state.cooldownUntil).toISOString()}` : "";
  return `${state.planKey}: ${state.status}${state.reason ? ` reason=${state.reason}` : ""}${cooldownUntil}${remaining ? ` ${remaining}` : ""}`;
}

async function resolveQuotaPlans(ctx: ExtensionContext, pool: Pool): Promise<QuotaPlanLookup> {
  const entries = await Promise.all(
    pool.all.map(async (item) => {
      const auth = await ctx.modelRegistry.getApiKeyAndHeaders(item.model);
      const planKey = auth.ok ? modelPlanKey(item.model, auth) : modelPlanKey(item.model);
      return [modelKey(item.model), { planKey, auth }] as const;
    }),
  );
  return new Map(entries);
}

function quotaPlanKeyFor(item: ResolvedModel, quotaPlans: QuotaPlanLookup): string {
  return quotaPlans.get(modelKey(item.model))?.planKey ?? modelPlanKey(item.model);
}

function modelPlanKey(model: Model<Api>, auth?: ResolvedAuth): string {
  return buildPlanKey({
    provider: model.provider,
    baseUrl: model.baseUrl,
    apiKey: auth?.apiKey,
    headers: auth?.headers,
    env: auth?.env,
  });
}

function makeRouterError(model: Model<Api>, error: unknown): AssistantMessageEvent {
  const errorMessage = error instanceof Error ? error.message : String(error);
  const message: AssistantMessage = {
    role: "assistant",
    content: [{ type: "text", text: `Pi Router error: ${errorMessage}` }],
    api: model.api,
    provider: model.provider,
    model: model.id,
    usage: structuredClone(ZERO_USAGE),
    stopReason: "error",
    errorMessage,
    timestamp: Date.now(),
  };

  return { type: "error", reason: "error", error: message };
}
