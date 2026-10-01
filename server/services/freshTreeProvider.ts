import { callLLM } from "../llm";
import { streamFreshTreeCompletion, streamFreshTreeResearch, type FreshTreeStreamObserver } from "./freshTreeStream";

const PROVIDERS = [
  { name: "OpenAI", id: "openai", key: "OPENAI_API_KEY" },
  { name: "Perplexity", id: "perplexity", key: "PERPLEXITY_API_KEY" },
  { name: "Anthropic", id: "anthropic", key: "ANTHROPIC_API_KEY" },
  { name: "Grok", id: "grok", key: "GROK_API_KEY" },
  { name: "Gemini", id: "gemini", key: "GEMINI_API_KEY" },
  { name: "DeepSeek", id: "deepseek", key: "DEEPSEEK_API_KEY" },
] as const;

type Provider = typeof PROVIDERS[number]["id"];
type RunState = { unavailable: Set<Provider>; active?: Provider; notify?: (message: string) => void; stream?: FreshTreeStreamObserver };
const runs = new WeakMap<AbortSignal, RunState>();

function stateFor(signal: AbortSignal): RunState {
  let state = runs.get(signal);
  if (!state) {
    state = { unavailable: new Set() };
    runs.set(signal, state);
  }
  return state;
}

function failureReason(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  if (/credit_balance_exhausted|insufficient_quota|no credits remaining/i.test(message)) return "quota exhausted";
  if (/\b429\b|rate.limit/i.test(message)) return "rate limited or quota exhausted";
  if (/\b401\b|\b403\b|unauthori[sz]ed|invalid.api.key/i.test(message)) return "authentication rejected";
  if (/not configured/i.test(message)) return "not configured";
  return "request failed";
}

export function reportFreshTreeProviders(signal: AbortSignal, notify: (message: string) => void, stream?: FreshTreeStreamObserver): void {
  stateFor(signal).notify = notify;
  stateFor(signal).stream = stream;
}

export function freshTreeProviderUnavailable(signal: AbortSignal, provider: Provider): boolean {
  return stateFor(signal).unavailable.has(provider);
}

export function markFreshTreeProviderUnavailable(signal: AbortSignal, provider: Provider, error: unknown): void {
  if (signal.aborted) return;
  const state = stateFor(signal);
  if (state.unavailable.has(provider)) return;
  state.unavailable.add(provider);
  const name = PROVIDERS.find((item) => item.id === provider)!.name;
  state.notify?.(`${name} failed (${failureReason(error)}); trying the next configured API.`);
}

export async function callFreshTreeLLM(prompt: string, signal: AbortSignal, stage = "Generating or reviewing tree nodes"): Promise<string> {
  const state = stateFor(signal);
  for (const provider of PROVIDERS) {
    if (!process.env[provider.key] || state.unavailable.has(provider.id)) continue;
    if (signal.aborted) throw new Error("Fresh Tree generation cancelled.");
    try {
      state.stream?.start(provider.name, stage);
      const text = state.stream
        ? await streamFreshTreeCompletion(provider.id, prompt, signal, (chunk) => state.stream?.chunk(chunk))
        : await callLLM(provider.id, prompt, signal);
      if (typeof text !== "string" || !text.trim()) throw new Error("No text returned");
      state.stream?.complete();
      if (state.unavailable.size && state.active !== provider.id) {
        state.notify?.(`Fresh Tree is now using ${provider.name}.`);
      }
      state.active = provider.id;
      return text;
    } catch (error) {
      if (signal.aborted) throw error;
      markFreshTreeProviderUnavailable(signal, provider.id, error);
    }
  }
  throw new Error("No working Fresh Tree API is available. Check the provider messages above.");
}

export async function callFreshTreeResearch(prompt: string, signal: AbortSignal): Promise<string> {
  const observer = stateFor(signal).stream;
  observer?.start("OpenAI", "Researching supporting facts");
  const text = await streamFreshTreeResearch(prompt, signal, (chunk) => observer?.chunk(chunk));
  observer?.complete();
  return text;
}