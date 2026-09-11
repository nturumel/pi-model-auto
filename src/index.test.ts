/**
 * Exercise the extension boundary where the router's virtual model mirrors the concrete model that
 * will serve the turn. This catches regressions that pure routing tests cannot observe in Pi's UI.
 */
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Api, Context, Model } from "@earendil-works/pi-ai";
import type { ExtensionAPI, ExtensionContext, ProviderConfig } from "@earendil-works/pi-coding-agent";
import { afterEach, describe, expect, it, vi } from "vitest";
import modelRouter from "./index.ts";

type Handler = (...args: any[]) => unknown;

const temporaryDirs: string[] = [];

afterEach(() => {
  delete process.env.PI_CODING_AGENT_DIR;
  for (const directory of temporaryDirs.splice(0)) rmSync(directory, { recursive: true, force: true });
});

function model(provider: string, id: string, contextWindow: number): Model<Api> {
  return {
    provider,
    id,
    name: id,
    api: "openai-completions",
    baseUrl: "https://example.invalid",
    reasoning: true,
    input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow,
    maxTokens: 16_384,
  };
}

function createHarness(target: Model<Api> | Model<Api>[], branch: unknown[] = [], routeFlag?: string, sessionId = "parent") {
  const targets = Array.isArray(target) ? target : [target];
  const agentDir = mkdtempSync(join(tmpdir(), "pi-model-auto-agent-"));
  temporaryDirs.push(agentDir);
  process.env.PI_CODING_AGENT_DIR = agentDir;
  writeFileSync(join(agentDir, "model-router.json"), JSON.stringify({ router: { classifier: "off" } }));

  const providers: ProviderConfig[] = [];
  const commands = new Map<string, {
    getArgumentCompletions?: (prefix: string) => Array<{ value: string; label: string; description?: string }> | null;
    handler: (args: string, ctx: ExtensionContext) => Promise<void>;
  }>();
  const handlers = new Map<string, Handler>();
  const pi = {
    registerProvider: (_name: string, config: ProviderConfig) => providers.push(config),
    registerCommand: (name: string, command: {
      getArgumentCompletions?: (prefix: string) => Array<{ value: string; label: string; description?: string }> | null;
      handler: (args: string, ctx: ExtensionContext) => Promise<void>;
    }) => commands.set(name, command),
    registerFlag: vi.fn(),
    getFlag: (name: string) => name === "route" ? routeFlag : undefined,
    on: (event: string, handler: Handler) => handlers.set(event, handler),
  } as unknown as ExtensionAPI;

  modelRouter(pi);
  const initialProvider = providers[0];
  const routerDefinition = initialProvider.models![0];
  const routerModel: Model<Api> = {
    ...routerDefinition,
    provider: "pi-router",
    api: initialProvider.api as Api,
    baseUrl: initialProvider.baseUrl!,
  };
  const registry = {
    getAvailable: () => targets,
    find: (provider: string, id: string) =>
      targets.find((candidate) => provider === candidate.provider && id === candidate.id),
    hasConfiguredAuth: () => true,
    getApiKeyAndHeaders: async () => ({ ok: true as const, apiKey: "test-key" }),
  };
  const ctx = {
    cwd: agentDir,
    model: routerModel,
    modelRegistry: registry,
    sessionManager: { getSessionId: () => sessionId, getBranch: () => branch },
    getSystemPrompt: () => "",
    isProjectTrusted: () => false,
    setModel: async () => true,
    ui: { notify: vi.fn(), setStatus: vi.fn(), addAutocompleteProvider: vi.fn() },
  } as unknown as ExtensionContext;

  return { providers, handlers, commands, initialProvider, routerModel, ctx, agentDir };
}

describe("router context window", () => {
  it("restores the window of the most recent concrete model when a session starts", async () => {
    const target = model("magi-codex", "gpt-5.6-sol", 272_000);
    const branch = [{
      type: "message",
      id: "assistant-1",
      parentId: null,
      timestamp: new Date().toISOString(),
      message: {
        role: "assistant",
        api: target.api,
        provider: target.provider,
        model: target.id,
        content: [{ type: "text", text: "done" }],
        usage: { input: 1, output: 1, cacheRead: 0, cacheWrite: 0, totalTokens: 2, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
        stopReason: "stop",
        timestamp: Date.now(),
      },
    }];
    const { providers, handlers, ctx } = createHarness(target, branch);

    await handlers.get("session_start")!({}, ctx);

    expect(providers.at(-1)?.models?.[0].contextWindow).toBe(272_000);
  });

  it("selects the automatic route window before Pi performs its preflight compaction check", async () => {
    const target = model("magi-codex", "gpt-5.6-sol", 272_000);
    const { providers, handlers, ctx } = createHarness(target);

    await handlers.get("session_start")!({}, ctx);
    expect(ctx.ui.setStatus).toHaveBeenCalledWith(
      "router",
      "AUTO · next: dynamic",
    );
    const inputResult = await handlers.get("input")!({ source: "interactive", text: "test" }, ctx);

    expect(inputResult).toEqual({ action: "continue" });
    expect(providers.at(-1)?.models?.[0].contextWindow).toBe(272_000);
  });

  it("updates the auto model to the concrete model window selected for the turn", async () => {
    const target = model("magi-codex", "gpt-5.6-sol", 272_000);
    const { providers, handlers, initialProvider, routerModel, ctx } = createHarness(target);

    await handlers.get("session_start")!({}, ctx);
    const inputResult = await handlers.get("input")!(
      { source: "interactive", text: "@route:magi-codex/gpt-5.6-sol test" },
      ctx,
    );
    expect(inputResult).toEqual({ action: "transform", text: "test", images: undefined });
    expect(providers.at(-1)?.models?.[0].contextWindow).toBe(272_000);

    const context: Context = {
      messages: [{ role: "user", content: "test", timestamp: Date.now() }],
    };
    for await (const _event of initialProvider.streamSimple!(routerModel, context)) {
      // Consume the stream so the router's asynchronous selection finishes.
    }

    expect(providers.at(-1)?.models?.[0].contextWindow).toBe(272_000);
    expect(ctx.ui.setStatus).toHaveBeenCalledWith(
      "router",
      expect.stringMatching(/^ONCE magi-codex\/gpt-5\.6-sol · High · high/),
    );
    expect(ctx.ui.setStatus).toHaveBeenLastCalledWith(
      "router",
      expect.stringMatching(/^AUTO · last: gpt-5\.6-sol · High · high/),
    );
  });

  it("accepts a CLI-safe route flag without an @-prefixed prompt", async () => {
    const target = model("openai-codex", "gpt-5.6-sol", 372_000);
    const { handlers, initialProvider, routerModel, ctx, agentDir } = createHarness(target, [], "ultra");
    writeFileSync(join(agentDir, "model-router.json"), JSON.stringify({
      router: {
        modeModels: {
          ultra: { model: "openai-codex/gpt-5.6-sol", thinking: "xhigh" },
        },
      },
    }));

    await handlers.get("session_start")!({}, ctx);
    const inputResult = await handlers.get("input")!({ source: "interactive", text: "test" }, ctx);
    expect(inputResult).toEqual({ action: "continue" });

    const context: Context = {
      messages: [{ role: "user", content: "test", timestamp: Date.now() }],
    };
    for await (const _event of initialProvider.streamSimple!(routerModel, context)) {
      // Consume the stream so the router's asynchronous selection finishes.
    }

    expect(ctx.ui.setStatus).toHaveBeenLastCalledWith(
      "router",
      expect.stringMatching(/^AUTO · last: gpt-5\.6-sol · Ultra · high/),
    );
  });

  it("routes manual and automatic compaction through Medium", async () => {
    const medium = model("openai-codex", "gpt-5.6-terra", 372_000);
    const ultra = model("openai-codex", "gpt-5.6-sol", 372_000);
    const { providers, handlers, initialProvider, routerModel, ctx, agentDir } = createHarness([medium, ultra], [], "ultra");
    writeFileSync(join(agentDir, "model-router.json"), JSON.stringify({
      router: {
        modeModels: {
          medium: { model: "openai-codex/gpt-5.6-terra", thinking: "high" },
          ultra: { model: "openai-codex/gpt-5.6-sol", thinking: "xhigh" },
        },
      },
    }));
    await handlers.get("session_start")!({}, ctx);
    expect(providers.at(-1)?.models?.[0]).toMatchObject({
      compactionModel: "openai-codex/gpt-5.6-terra",
    });
    const context: Context = {
      messages: [{ role: "user", content: "a very large compaction transcript", timestamp: Date.now() }],
    };

    for (const trigger of ["manual", "auto"] as const) {
      for await (const _event of initialProvider.streamSimple!(routerModel, context, {
        codexCompaction: { trigger, reason: trigger === "manual" ? "user_requested" : "context_limit", phase: "standalone_turn" },
      })) {
        // Consume the stream so routing finishes.
      }
      expect(ctx.ui.setStatus).toHaveBeenLastCalledWith(
        "router",
        expect.stringMatching(/^AUTO · last: gpt-5\.6-terra · Medium · high/),
      );
    }
  });

  it("keeps a sticky route across turns and exposes its forced UI state", async () => {
    const medium = model("openai-codex", "gpt-5.6-terra", 372_000);
    const ultra = model("openai-codex", "gpt-5.6-sol", 372_000);
    const { handlers, commands, initialProvider, routerModel, ctx, agentDir } =
      createHarness([medium, ultra]);
    writeFileSync(join(agentDir, "model-router.json"), JSON.stringify({
      router: {
        modeModels: {
          medium: { model: "openai-codex/gpt-5.6-terra", thinking: "high" },
          ultra: { model: "openai-codex/gpt-5.6-sol", thinking: "xhigh" },
        },
      },
    }));
    await handlers.get("session_start")!({}, ctx);
    await commands.get("route")!.handler("ultra", ctx);
    expect(ctx.ui.setStatus).toHaveBeenLastCalledWith(
      "router",
      "PIN Ultra · next: pinned",
    );

    for (const text of ["first sticky turn", "second sticky turn"]) {
      await handlers.get("input")!({ source: "interactive", text }, ctx);
      const context: Context = {
        messages: [{ role: "user", content: text, timestamp: Date.now() }],
      };
      for await (const _event of initialProvider.streamSimple!(routerModel, context)) {
        // Consume the stream so routing finishes.
      }
      expect(ctx.ui.setStatus).toHaveBeenCalledWith(
        "router",
        expect.stringMatching(/^PIN Ultra → gpt-5\.6-sol · high/),
      );
      expect(ctx.ui.setStatus).toHaveBeenLastCalledWith(
        "router",
        expect.stringMatching(/^PIN Ultra · last: gpt-5\.6-sol · Ultra · high/),
      );
    }

    await commands.get("route")!.handler("auto", ctx);
    expect(ctx.ui.setStatus).toHaveBeenLastCalledWith(
      "router",
      "AUTO · next: dynamic · last: gpt-5.6-sol",
    );
  });

  it("clears sticky routing with an @route:auto prompt", async () => {
    const target = model("openai-codex", "gpt-5.6-sol", 372_000);
    const { handlers, commands, ctx } = createHarness(target);
    await handlers.get("session_start")!({}, ctx);
    await commands.get("route")!.handler("ultra", ctx);

    await expect(handlers.get("input")!(
      { source: "interactive", text: "@route:auto resume automatic routing" },
      ctx,
    )).resolves.toEqual({
      action: "transform",
      text: "resume automatic routing",
      images: undefined,
    });
    expect(ctx.ui.setStatus).toHaveBeenLastCalledWith(
      "router",
      "AUTO · next: dynamic",
    );
  });

  it("accepts a one-turn mode override in the middle of a session", async () => {
    const target = model("openai-codex", "gpt-5.6-sol", 372_000);
    const branch = [{
      type: "message",
      id: "user-1",
      parentId: null,
      timestamp: new Date().toISOString(),
      message: {
        role: "user",
        content: [{ type: "text", text: "earlier turn" }],
        timestamp: Date.now(),
      },
    }];
    const { handlers, ctx } = createHarness(target, branch);

    await handlers.get("session_start")!({}, ctx);
    await expect(handlers.get("input")!(
      { source: "interactive", text: "@route:ultra inspect this" },
      ctx,
    )).resolves.toEqual({ action: "transform", text: "inspect this", images: undefined });
    await expect(handlers.get("input")!(
      { source: "interactive", text: "continue normally" },
      ctx,
    )).resolves.toEqual({ action: "continue" });
  });

  it("completes one-turn routes only while the router model is selected", async () => {
    const target = model("openai-codex", "gpt-5.6-sol", 372_000);
    const { handlers, commands, ctx, agentDir } = createHarness(target);
    writeFileSync(join(agentDir, "model-router.json"), JSON.stringify({
      router: {
        classifier: "off",
        modeModels: {
          ultra: { model: "openai-codex/gpt-5.6-sol", thinking: "xhigh" },
        },
      },
    }));

    await handlers.get("session_start")!({}, ctx);
    const factory = vi.mocked(ctx.ui.addAutocompleteProvider).mock.calls[0][0];
    const current = {
      getSuggestions: vi.fn().mockResolvedValue({ prefix: "", items: [] }),
      applyCompletion: vi.fn(),
      shouldTriggerFileCompletion: vi.fn(() => true),
    };
    const provider = factory(current);

    await expect(provider.getSuggestions(["@route:u"], 0, 8, {})).resolves.toEqual({
      prefix: "@route:u",
      items: [{
        value: "@route:ultra ",
        label: "@route:ultra",
        description: "openai-codex/gpt-5.6-sol · xhigh · one turn",
      }],
    });
    await expect(provider.getSuggestions(["@route:openai"], 0, 14, {})).resolves.toEqual({
      prefix: "@route:openai",
      items: [{
        value: "@route:openai-codex/gpt-5.6-sol ",
        label: "@route:openai-codex/gpt-5.6-sol",
        description: "Exact model · one turn",
      }],
    });
    expect(commands.get("route")!.getArgumentCompletions!("u")).toEqual([{
      value: "ultra",
      label: "ultra",
      description: "Keep routing through Ultra",
    }]);
    expect(commands.get("route")!.getArgumentCompletions!("openai")).toEqual([{
      value: "openai-codex/gpt-5.6-sol",
      label: "openai-codex/gpt-5.6-sol",
      description: "Keep routing through this exact model",
    }]);
    expect(commands.get("route")!.getArgumentCompletions!("a")).toEqual([{
      value: "auto",
      label: "auto",
      description: "Clear sticky routing",
    }]);
    expect(provider.shouldTriggerFileCompletion(["@route:u"], 0, 8)).toBe(false);

    ctx.model = target;
    await provider.getSuggestions(["@route:u"], 0, 8, {});
    expect(current.getSuggestions).toHaveBeenCalledWith(["@route:u"], 0, 8, {});
    expect(provider.shouldTriggerFileCompletion(["@route:u"], 0, 8)).toBe(true);
  });

  it("keeps the parent router usable when nested shutdown reports the parent context", async () => {
    const target = model("openai-codex", "gpt-5.6-sol", 372_000);
    const { handlers, initialProvider, routerModel, ctx } = createHarness(target);
    const childCtx = {
      ...ctx,
      sessionManager: { getSessionId: () => "child", getBranch: () => [] },
      ui: { notify: vi.fn(), setStatus: vi.fn(), addAutocompleteProvider: vi.fn() },
    } as unknown as ExtensionContext;

    await handlers.get("session_start")!({}, ctx);
    await handlers.get("session_start")!({}, childCtx);
    await handlers.get("session_shutdown")!({ reason: "quit" }, ctx);

    const context: Context = {
      messages: [{ role: "user", content: "continue parent work", timestamp: Date.now() }],
    };
    const events = [];
    for await (const event of initialProvider.streamSimple!(routerModel, context, { sessionId: "parent" })) {
      events.push(event);
    }

    expect(events.some((event) =>
      event.type === "error" && event.error.errorMessage?.includes("session state")
    )).toBe(false);
    expect(ctx.ui.setStatus).toHaveBeenLastCalledWith(
      "router",
      expect.stringMatching(/^AUTO · last: gpt-5\.6-sol/),
    );
  });

  it("keeps parent state visible through a provider rebound by a starting child", async () => {
    const parent = createHarness([], [], undefined, "parent-rebound");
    await parent.handlers.get("session_start")!({}, parent.ctx);

    // OMP rebinds the process-global custom API to a fresh extension factory before
    // that child's session_start. The replacement handler must still see its parent.
    const child = createHarness([], [], undefined, "child-rebound");
    const context: Context = {
      messages: [{ role: "user", content: "continue parent work", timestamp: Date.now() }],
    };
    const events = [];
    for await (const event of child.initialProvider.streamSimple!(
      child.routerModel,
      context,
      { sessionId: "parent-rebound" },
    )) {
      events.push(event);
    }

    expect(events.some((event) =>
      event.type === "error" && event.error.errorMessage?.includes("no authenticated models")
    )).toBe(true);

    await child.handlers.get("session_shutdown")!({}, child.ctx);
    await parent.handlers.get("session_shutdown")!({}, parent.ctx);
  });
});
