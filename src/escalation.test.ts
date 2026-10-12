import type { Context } from "@earendil-works/pi-ai";
import { describe, expect, it } from "vitest";
import { createEscalationState, nextMode, turnSignals, updateEscalation } from "./escalation.ts";
import { DEFAULT_ESCALATION, type CapabilityMode, type ModeModelConfig } from "./router-core.ts";

const MODES: Partial<Record<CapabilityMode, ModeModelConfig>> = {
  low: { model: "anthropic/haiku", thinking: "low" },
  medium: { model: "anthropic/sonnet", thinking: "medium" },
  high: { model: "anthropic/sonnet", thinking: "high" },
  ultra: { model: "anthropic/opus", thinking: "xhigh" },
};

const user = (text: string) => ({ role: "user" as const, content: text, timestamp: 0 });
const assistant = () => ({ role: "assistant" as const, content: [], api: "x", provider: "x", model: "x",
  usage: {} as never, stopReason: "toolUse" as const, timestamp: 0 }) as Context["messages"][number];
const tool = (toolName: string, text: string, isError = false) =>
  ({ role: "toolResult" as const, toolCallId: "t", toolName, content: [{ type: "text" as const, text }], isError, timestamp: 0 });
const ctx = (...messages: Context["messages"]): Context => ({ messages });

describe("turnSignals", () => {
  it("counts tool errors and failing shell runs, but not passing runs or file contents", () => {
    const s = turnSignals(ctx(
      user("fix the build"),
      assistant(), tool("bash", "===== 2 failed, 10 passed in 1.2s ====="),
      assistant(), tool("bash", "src/a.ts(3,1): error TS2345: Argument of type"),
      assistant(), tool("edit", "old_string not found", true),
      assistant(), tool("bash", "===== 0 failed, 12 passed ====="),
      assistant(), tool("read", "expect(result).toBe('FAILED')"),
    ));
    expect(s).toMatchObject({ failures: 3, steps: 5, repeatedError: false, feedback: false });
  });

  it("only looks at the current user turn", () => {
    const s = turnSignals(ctx(user("first"), assistant(), tool("bash", "1 failed", true), user("next"), assistant()));
    expect(s.failures).toBe(0);
  });

  it("detects the same failure re-run with different numbers", () => {
    const s = turnSignals(ctx(user("go"), assistant(), tool("bash", "FAILED test_x.py::t - assert 1 == 2 (0.31s)"),
      assistant(), tool("bash", "FAILED test_x.py::t - assert 1 == 2 (0.29s)")));
    expect(s.repeatedError).toBe(true);
  });

  it("reads follow-up feedback but never the first message, and ignores host reminders", () => {
    expect(turnSignals(ctx(user("this doesn't work offline, why?"))).feedback).toBe(false);
    expect(turnSignals(ctx(user("hi"), assistant(), user("<system-reminder>cwd</system-reminder>\nstill broken, same error"))).feedback).toBe(true);
    expect(turnSignals(ctx(user("hi"), assistant(), user("thanks, looks good"))).feedback).toBe(false);
  });
});

describe("nextMode", () => {
  it("steps to the next configured model at its highest mode", () => {
    expect(nextMode("low", MODES, "ultra")).toBe("high");
    expect(nextMode("medium", MODES, "ultra")).toBe("ultra");
    expect(nextMode("high", MODES, "ultra")).toBe("ultra");
    expect(nextMode("ultra", MODES, "ultra")).toBeUndefined();
  });

  it("respects the ceiling and falls back to the next mode without pins", () => {
    expect(nextMode("low", MODES, "high")).toBe("high");
    expect(nextMode("high", MODES, "high")).toBeUndefined();
    expect(nextMode("low", {}, "ultra")).toBe("medium");
  });
});

describe("updateEscalation", () => {
  const cfg = DEFAULT_ESCALATION;

  it("escalates once on the third failure and does not raise again in the same turn", () => {
    const state = createEscalationState();
    const msgs: Context["messages"] = [user("fix it")];
    const results = [];
    for (let i = 0; i < 5; i++) {
      msgs.push(assistant(), tool("bash", `Error in ${["parser", "lexer", "emitter", "loader", "cli"][i]}\nexit code 1`));
      results.push(updateEscalation(state, ctx(...msgs), cfg, "low", MODES, "1:a"));
    }
    expect(results.map((r) => r.floor)).toEqual([undefined, undefined, "high", "high", "high"]);
    expect(results.filter((r) => r.raised)).toHaveLength(1);
    expect(results[2].raised).toMatchObject({ from: "low", to: "high" });
  });

  it("escalates on a 'still broken' follow-up and goes up a tier per turn, not per signal", () => {
    const state = createEscalationState();
    updateEscalation(state, ctx(user("fix it")), cfg, "low", MODES, "1:a");
    const r2 = updateEscalation(state, ctx(user("fix it"), assistant(), user("still broken")), cfg, "low", MODES, "2:b");
    expect(r2.raised).toMatchObject({ from: "low", to: "high" });
    const r3 = updateEscalation(state, ctx(user("fix it"), assistant(), user("still broken"), assistant(), user("same error again")),
      cfg, "low", MODES, "3:c");
    expect(r3.raised).toMatchObject({ from: "high", to: "ultra" });
  });

  it("holds the floor for quiet turns, then resets", () => {
    const state = createEscalationState();
    const failing = ctx(user("a"), assistant(), tool("x", "boom", true), assistant(), tool("x", "boom 2", true), assistant(), tool("x", "boom 3", true));
    expect(updateEscalation(state, failing, cfg, "low", MODES, "1:a").floor).toBe("high");
    const floors = ["2:b", "3:c", "4:d", "5:e"].map((k) => updateEscalation(state, ctx(user("a"), assistant(), user("ok next")), cfg, "low", MODES, k));
    expect(floors.map((r) => r.floor)).toEqual(["high", "high", "high", undefined]);
    expect(floors[3].reset).toBe(true);
  });

  it("does nothing when disabled", () => {
    const state = createEscalationState();
    const failing = ctx(user("a"), assistant(), tool("x", "1", true), assistant(), tool("x", "2", true), assistant(), tool("x", "3", true));
    expect(updateEscalation(state, failing, { ...cfg, enabled: false }, "low", MODES, "1:a")).toEqual({});
  });
});
