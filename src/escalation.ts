/**
 * In-session escalation: move a task up a model tier when the session shows it is struggling.
 *
 * Prompt-only routing cannot see whether the cheap model is coping. These signals can, and they are
 * read straight from the conversation (no model call, no extra latency):
 *   - failing tool results in the current user turn (tool errors, failing tests/builds in shell output)
 *   - the same error text repeated within the turn
 *   - a follow-up like "still broken" / "that's wrong" / "same error"
 *   - an unusually long agent loop within one user turn
 * Escalation only moves up, to the next mode that maps to a different configured model, and holds
 * for a few quiet user turns before resetting. Explicit routes (@route, --route, /route) are never
 * overridden; the caller skips escalation for them.
 */
import type { Context } from "@earendil-works/pi-ai";
import { stripHostReminders, type CapabilityMode, type EscalationConfig, type ModeModelConfig } from "./router-core.ts";

export interface EscalationState {
  floor?: CapabilityMode;
  turnKey?: string;
  quietTurns: number;
  signalThisTurn: boolean;
  fired: Set<string>;
}

export interface EscalationResult {
  floor?: CapabilityMode;
  raised?: { from: CapabilityMode; to: CapabilityMode; reason: string };
  reset?: boolean;
}

export interface TurnSignals {
  failures: number;
  repeatedError: boolean;
  steps: number;
  feedback: boolean;
}

const MODES: CapabilityMode[] = ["low", "medium", "high", "ultra"];
const rank = (mode: CapabilityMode) => MODES.indexOf(mode);

const SHELL_TOOL = /^(bash|shell|sh|zsh|exec|run|terminal|command|cmd|python)/i;
// Conservative: test-runner summaries, compiler errors and non-zero exits, not any line containing "error".
const FAILURE_PATTERNS = [
  /\b[1-9]\d* (?:failed|failing)\b/i,
  /^FAIL\b/m,
  /\bFAILED\b/,
  /--- FAIL:/,
  /\bTraceback \(most recent call last\)/,
  /\berror TS\d{3,5}:/,
  /\berror\[E\d{4}\]/,
  /npm ERR!/,
  /\bBUILD FAILED\b/i,
  /\bcompilation (?:failed|terminated)\b/i,
  /\bexit (?:code|status):? ?[1-9]\d*/i,
  /\bpanicked at\b/,
  /\bSegmentation fault\b/,
  /\bAssertionError\b/,
];
const FEEDBACK_PATTERNS = [
  /\b(?:still|again)\b[^.!?\n]{0,40}\b(?:broken|failing|fails|failed|not working|doesn'?t work|wrong|errors?|crash(?:es|ing)?)\b/i,
  /\b(?:doesn'?t|does not|didn'?t|did not|isn'?t|is not|not) (?:work|working|fixed|compile|compiling|pass|passing)\b/i,
  /\b(?:that'?s|this is|it'?s) (?:wrong|incorrect|not right|broken)\b/i,
  /\bsame (?:error|issue|problem|bug|failure)\b/i,
  /\byou broke\b/i,
];

export function createEscalationState(): EscalationState {
  return { quietTurns: 0, signalThisTurn: false, fired: new Set() };
}

function textOf(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content.filter((part) => part?.type === "text").map((part) => part.text as string).join("\n");
}

/** Signals from the messages of the current user turn (everything after the last user message). */
export function turnSignals(context: Context): TurnSignals {
  const messages = context.messages;
  let lastUser = -1;
  let userCount = 0;
  for (let i = 0; i < messages.length; i++) {
    if (messages[i].role === "user") { lastUser = i; userCount += 1; }
  }
  let failures = 0;
  let steps = 0;
  let repeatedError = false;
  const seen = new Set<string>();
  for (const message of messages.slice(lastUser + 1)) {
    if (message.role === "assistant") steps += 1;
    if (message.role !== "toolResult") continue;
    const text = textOf(message.content);
    const failed = message.isError || (SHELL_TOOL.test(message.toolName) && FAILURE_PATTERNS.some((re) => re.test(text)));
    if (!failed) continue;
    failures += 1;
    // Normalise numbers (line numbers, timings, ids) so a re-run of the same failure matches.
    const signature = text.replace(/\d+/g, "#").replace(/\s+/g, " ").trim().slice(0, 240);
    if (signature && seen.has(signature)) repeatedError = true;
    seen.add(signature);
  }
  const userText = lastUser >= 0 ? stripHostReminders(textOf(messages[lastUser].content)) : "";
  const feedback = userCount >= 2 && FEEDBACK_PATTERNS.some((re) => re.test(userText));
  return { failures, repeatedError, steps, feedback };
}

/**
 * Next mode up that changes the serving model: with modeModels {low: Haiku, medium/high: Sonnet,
 * ultra: Opus}, low -> high (Sonnet at its highest configured effort) and medium/high -> ultra.
 */
export function nextMode(current: CapabilityMode, modeModels: Partial<Record<CapabilityMode, ModeModelConfig>>,
                         ceiling: CapabilityMode): CapabilityMode | undefined {
  const candidates = MODES.slice(rank(current) + 1, rank(ceiling) + 1);
  if (candidates.length === 0) return undefined;
  const currentModel = modeModels[current]?.model;
  if (!currentModel) return candidates[0];
  const firstDifferent = candidates.find((mode) => modeModels[mode]?.model && modeModels[mode]?.model !== currentModel);
  if (!firstDifferent) return undefined;
  const target = modeModels[firstDifferent]!.model;
  return [...candidates].reverse().find((mode) => modeModels[mode]?.model === target);
}

/** Update escalation state for one routing request and return the current floor. */
export function updateEscalation(state: EscalationState, context: Context, cfg: EscalationConfig, current: CapabilityMode,
                                 modeModels: Partial<Record<CapabilityMode, ModeModelConfig>>, turnKey: string): EscalationResult {
  if (!cfg.enabled) return {};
  let reset = false;
  if (turnKey !== state.turnKey) {
    if (state.turnKey !== undefined && state.floor && !state.signalThisTurn) state.quietTurns += 1;
    state.turnKey = turnKey;
    state.fired.clear();
    state.signalThisTurn = false;
    if (state.floor && cfg.decayTurns > 0 && state.quietTurns >= cfg.decayTurns) {
      state.floor = undefined;
      state.quietTurns = 0;
      reset = true;
    }
  }

  const s = turnSignals(context);
  const triggers: [string, boolean, string][] = [
    ["feedback", cfg.userFeedback && s.feedback, "user reported it is still not working"],
    ["failures", cfg.toolFailures > 0 && s.failures >= cfg.toolFailures, `${s.failures} failing tool results this turn`],
    ["repeated", cfg.repeatedError && s.repeatedError, "the same error repeated"],
    ["steps", cfg.maxSteps > 0 && s.steps > cfg.maxSteps, `${s.steps} agent steps this turn`],
  ];
  const newlyFired = triggers.filter(([id, on]) => on && !state.fired.has(id));
  if (newlyFired.length === 0) return { floor: state.floor, reset };
  for (const [id] of newlyFired) state.fired.add(id);
  // At most one step up per user turn, so a burst of signals cannot jump straight to the top tier.
  if (state.signalThisTurn) return { floor: state.floor, reset };

  const base = state.floor && rank(state.floor) > rank(current) ? state.floor : current;
  const to = nextMode(base, modeModels, cfg.ceiling);
  state.signalThisTurn = true;
  state.quietTurns = 0;
  if (!to) return { floor: state.floor, reset };
  state.floor = to;
  return { floor: to, raised: { from: base, to, reason: newlyFired.map(([, , why]) => why).join("; ") }, reset };
}

export function isAbove(a: CapabilityMode | undefined, b: CapabilityMode): a is CapabilityMode {
  return a !== undefined && rank(a) > rank(b);
}

export function modeIndex(mode: CapabilityMode): number {
  return rank(mode);
}
