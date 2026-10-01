import { callFreshTreeLLM, callFreshTreeResearch, freshTreeProviderUnavailable, markFreshTreeProviderUnavailable } from "./freshTreeProvider";
import type { TractatusStatement } from "./tractatusTree";

export type FreshTreeStatement = TractatusStatement;
export type FreshTreeSource = { node: string; marker?: string; url: string };
type TierCandidate = FreshTreeStatement & {
  parent: string;
  url?: string;
  searched?: boolean;
  legitimacyLabel?: LegitimacyResult["label"];
  reviewFailure?: { label: LegitimacyResult["label"]; reason: string };
};

export const DEFAULT_FRESH_INSTRUCTIONS = 
  "Under each node, add 1 or 2 child nodes that ESTABLISH that exact parent. Prefer a logical child when the parent is a distinction, definition, or inference: restate the author's own contrast or example in one sentence, or give a full mini-derivation. Prefer an empirical child only when the parent is an empirical claim. Fresh means: do not paraphrase the parent and do not repeat a child already in the tree. Source examples are ALLOWED when they are the author's proof of that parent. One sentence per node. No commentary.";
export const NODE_LEGITIMACY_RULE = `Every node must ESTABLISH its parent, by evidence or by logic:
(a) EMPIRICAL: a specific fact, measurement, experiment, observed case, or real-world instance, stated concretely (what, where, when, how much).
(b) LOGICAL: a derivation, a demonstration, or a counterexample, stated in full.
A node that reports what someone SAYS, ARGUES, or BELIEVES establishes nothing and is forbidden. A person may appear only as part of a fact (e.g. 'In 1847 Semmelweis cut maternal deaths in his ward from about 18% to about 2% by requiring handwashing'), never as an authority (e.g. 'Semmelweis argued that handwashing matters').
Vague appeals are forbidden: 'studies show', 'research indicates', 'a study found', 'experts agree', 'it is widely accepted', 'a classic illustration', 'a well-known example'.
The node must support its parent's EXACT claim, not a neighboring claim that shares a word with it.
Citations are allowed only as a footnote marker (¹, ², ...) at the end of a factual node. The citation never replaces the fact. A sentence of the form "Author (year) demonstrated/showed/found that P" is AUTHORITY unless P is an observable result (a measurement, contrast, intervention, or raw performance) that is identical with the parent's claim, not with the parent's topic.
If the parent is a distinction (A not B, framework not contents, trigger not instruct, cognitive not merely physiological), the child must use both poles of that distinction. A paper about only one pole is OFF-TARGET.`;

const AUTHORITY_PATTERNS = [
  /\bargued\b/i, /\bargues\b/i, /\bposits\b/i, /\bposited\b/i, /\bcontends\b/i, /\bmaintains that\b/i,
  /\bclaims that\b/i, /\baccording to\b/i, /\bas noted in\b/i, /\bas discussed in\b/i, /\bis discussed in\b/i,
  /\bhighlights\b/i, /\bemphasizes\b/i, /\bsuggests that\b/i, /\bstates that\b/i, /\bnotes that\b/i,
  /\bthe article\b/i, /\bthe paper\b/i, /\bthe book\b/i, /\bthe encyclopedia\b/i, /\bstanford encyclopedia\b/i,
  /\binternet encyclopedia\b/i, /\bSEP\b/i, /\bIEP\b/i,/\bdemonstrated that\b/i,
  /\bshowed that\b/i,
  /\bfound that\b/i,
  /\bobserved that\b/i,
  /\brevealed that\b/i,
  /\bet al\.\b/i,
];
const VAGUE_PATTERNS = [
  /\bstudies show\b/i, /\bresearch indicates\b/i,
  /\bresearch shows\b/i, /\ba study found\b/i, /\bexperts\b/i, /\bwidely accepted\b/i,
  /\bclassic illustration\b/i, /\bwell-known example\b/i,
];

type AddTierOptions = {
  shouldStop?: () => boolean;
  onProgress?: (tier: number, thesis: number, totalTheses: number) => void;
  onWarning?: (message: string) => void;
  tierNumber?: number;
  completedTheses?: string[];
  onThesisComplete?: (tierNumber: number, thesisNumber: string, statements: FreshTreeStatement[], sources: FreshTreeSource[]) => void;
  research?: (prompt: string, signal: AbortSignal, shouldStop?: () => boolean) => Promise<{ text: string; searched: boolean }>;
};

function assertNotAborted(signal: AbortSignal) {
  if (signal.aborted) throw new Error("Fresh Tree generation cancelled.");
}

/** Parse numbered prose defensively and discard every level deeper than requested. */
export function parseTreeLines(raw: string, maximumParts: number): FreshTreeStatement[] {
  const result: FreshTreeStatement[] = [];
  const seen = new Set<string>();
  for (const line of raw.split("\n")) {
    const match = /^\s*(\d+(?:\.\d+)+)\s+(.+?)\s*$/.exec(line.replace(/^[-*]\s*/, ""));
    if (!match) continue;
    const parts = match[1].split(".");
    if (parts.length > maximumParts || parts.length < 2 || seen.has(match[1])) continue;
    if (parts.length === 2 && parts[1] !== "0" && !result.some((item) => item.number === `${parts[0]}.0`)) continue;
    seen.add(match[1]);
    const depth = parts.length === 2 ? (parts[1] === "0" ? 0 : 1) : parts.length - 1;
    result.push({ number: match[1], text: match[2].trim(), depth });
  }
  return result;
}

export function freshTreeParentsAtTier(existing: FreshTreeStatement[], tierNumber: number): FreshTreeStatement[] {
  return existing.filter((item) => item.depth === tierNumber - 2);
}

export async function generateFreshTree(source: string, signal: AbortSignal): Promise<FreshTreeStatement[]> {
  assertNotAborted(signal);
  const required = "Build a compressed two-tier architecture of the author's theory, not an outline. Each n.0 is a load-bearing thesis, not a heading. Each n.1, n.2, ... must ESTABLISH that exact parent using only the source (reason, distinction, definition, author's example, or author's counterexample). Do not emit one root per paragraph. Each thesis needs at least two children. No lines deeper than tier 2. No outside facts.";
  let previous = "";
  for (let attempt = 0; attempt < 2; attempt++) {
    const correction = attempt ? `\nThe previous response was not a valid two-tier tree. Correct the format: each thesis must be numbered 1.0, 2.0, etc., followed by at least one child numbered 1.1, 2.1, etc. No deeper lines. Recheck every line against the source.\n<previous-response>\n${previous}\n</previous-response>` : "";
    const raw = await callFreshTreeLLM(`Build a plain numbered tree. ${required}\nEvery thesis must have at least one direct sub-claim. Output numbered lines only, with one sentence per line. Do not use Markdown.${correction}\n\n<source>\n${source}\n</source>`, signal, attempt ? "Correcting the first two tiers" : "Building the first two tiers");
    assertNotAborted(signal);
    const statements = parseTreeLines(raw, 2);
    const roots = statements.filter((item) => item.number.endsWith(".0"));
    if (roots.length && roots.every((root) => statements.some((item) => item.number.startsWith(`${root.number.slice(0, -1)}`) && item.number !== root.number))) {
      return statements;
    }
    previous = raw;
  }
  throw new Error("The model did not return a valid two-tier tree after a correction attempt.");
}

async function research(prompt: string, signal: AbortSignal, shouldStop?: () => boolean): Promise<{ text: string; searched: boolean }> {
  const key = process.env.OPENAI_API_KEY;
  if (!key || freshTreeProviderUnavailable(signal, "openai")) {
    return { text: await callFreshTreeLLM(prompt, signal, "Generating support without verified web search"), searched: false };
  }
  try {
    const text = await callFreshTreeResearch(prompt, signal);
    if (!text.trim()) throw new Error("web search returned no text");
    return { text, searched: true };
  } catch (error) {
    if (signal.aborted) throw error;
    if (shouldStop?.()) return { text: "", searched: false };
    markFreshTreeProviderUnavailable(signal, "openai", error);
    return { text: await callFreshTreeLLM(prompt, signal, "Generating support without verified web search"), searched: false };
  }
}

const REPUTABLE_HOSTS = [
  "bbc.com", "cell.com", "doi.org", "jamanetwork.com", "nationalgeographic.com", "nature.com",
  "ncbi.nlm.nih.gov", "newscientist.com", "noaa.gov", "nasa.gov", "pnas.org", "sciencemag.org",
  "scientificamerican.com", "science.org", "springer.com", "thelancet.com", "who.int", "wiley.com",
];

function isReputableSource(value: unknown): value is string {
  if (typeof value !== "string") return false;
  try {
    const hostname = new URL(value).hostname.toLowerCase().replace(/^www\./, "");
    return hostname.endsWith(".gov") || hostname.endsWith(".edu") || hostname.endsWith(".ac.uk") ||
      REPUTABLE_HOSTS.some((host) => hostname === host || hostname.endsWith(`.${host}`));
  } catch {
    return false;
  }
}

function words(value: string): Set<string> {
  return new Set(value.toLowerCase().match(/[\p{L}\p{N}]+/gu) || []);
}

/** Percentage of the proposed node's words that already occur in another line. */
export function wordOverlap(proposed: string, comparison: string): number {
  const proposedWords = words(proposed);
  if (!proposedWords.size) return 1;
  const comparisonWords = words(comparison);
  let shared = 0;
  proposedWords.forEach((word) => { if (comparisonWords.has(word)) shared++; });
  return shared / proposedWords.size;
}
function studyKey(text: string): string | undefined {
  const year = text.match(/\b(19|20)\d{2}\b/);
  const who = text.match(/\b([A-Z][a-z]+(?:\s+and\s+[A-Z][a-z]+)?(?:\s+et al\.?)?)\b/);
  if (!year || !who) return undefined;
  return `${who[1].toLowerCase()}|${year[0]}`;
}

function sameStudy(a: string, b: string): boolean {
  const ka = studyKey(a);
  const kb = studyKey(b);
  return Boolean(ka && kb && ka === kb);
}
function duplicatesAny(candidate: TierCandidate, lines: Array<{ text: string }>): boolean {
  return lines.some((line) =>
    wordOverlap(candidate.text, line.text) > 0.6 ||
    sameStudy(candidate.text, line.text)
  );
}

function childCount(instructions: string, tier: number): { min: number; max: number } {
  const userOverride = Boolean(instructions.trim()) && instructions.trim() !== DEFAULT_FRESH_INSTRUCTIONS;
  const explicit = userOverride
    ? /\b(one|1|two|2|three|3|four|4|five|5)\s+(?:new\s+)?(?:child(?:ren)?|nodes?)\b/i.exec(instructions)
    : null;
  const exampleCount = explicit || (userOverride
    ? /\b(one|1|two|2|three|3|four|4|five|5)\s+(?:counterexample|example)\b/i.exec(instructions)
    : null);
  if (exampleCount) {
    const count = ({ one: 1, two: 2, three: 3, four: 4, five: 5 } as Record<string, number>)[exampleCount[1].toLowerCase()] || Number(exampleCount[1]);
    return { min: count, max: count };
  }
  if (userOverride && /\b1\s+or\s+2\b|\bone\s+or\s+two\b/i.test(instructions)) return { min: 1, max: 2 };
  return tier === 3 ? { min: 1, max: 2 } : { min: 1, max: 1 };
}

function parseTier(raw: string, parents: FreshTreeStatement[], tier: number, count: { min: number; max: number },
  allowMissing = false): TierCandidate[] {
  const start = raw.indexOf("{");
  const end = raw.lastIndexOf("}");
  if (start < 0 || end < start) throw new Error(`Tier ${tier} returned invalid structured output.`);
  const parsed = JSON.parse(raw.slice(start, end + 1));
  const statements: TierCandidate[] = [];
  for (const parent of parents) {
    const children = parsed[parent.number];
    if (!Array.isArray(children) || children.length < count.min) {
      if (allowMissing) continue;
      throw new Error(`Tier ${tier} is missing the required ${count.min === count.max ? count.min : `${count.min}-${count.max}`} children for ${parent.number}.`);
    }
    const selected = children.slice(0, count.max);
    if (selected.some((child: any) => typeof child?.text !== "string" || !child.text.trim())) {
      if (allowMissing) continue;
      throw new Error(`Tier ${tier} contains an invalid child for ${parent.number}.`);
    }
    selected.forEach((child: any, index: number) => {
      const number = `${parent.number}.${index + 1}`;
      statements.push({ number, parent: parent.number, text: child.text.trim(), depth: tier - 1,
        url: isReputableSource(child.url) ? child.url : undefined });
    });
  }
  return statements;
}

async function reviewAndDeduplicate(candidates: TierCandidate[], existing: FreshTreeStatement[], researchResult: string,
  source: string, signal: AbortSignal): Promise<TierCandidate[]> {
  const duplicateNumbers = candidates.filter((candidate, index) =>
    duplicatesAny(candidate, [...existing, ...candidates.slice(0, index)])).map((candidate) => candidate.number);
  const prompt = `Review proposed child nodes. Return ONLY a JSON object keyed by every exact node number, shaped {"1.1.1":{"text":"one sentence","directSupport":true}}.
- directSupport is true only when the node directly supports its IMMEDIATE parent's specific claim; otherwise false.
- For these code-detected duplicates, regenerate the text ONCE: ${duplicateNumbers.join(", ") || "none"}.
- A regenerated node must use a concrete fact already present in the WEB RESEARCH RESULT and must retain the candidate's meaning and source; do not introduce an unsupported fact.
- Leave every nonduplicate node's text exactly unchanged.
- A child may be LOGICAL (author's contrast, definition, derivation, or source example) or EMPIRICAL (a concrete result that is the parent's claim). Named studies are required only for EMPIRICAL children.
- A restatement or paraphrase of its parent or any other tree line is forbidden.

PARENTS AND PROPOSED CHILDREN:
${candidates.map((item) => `${item.parent} ${existing.find((line) => line.number === item.parent)?.text}\n${item.number} ${item.text}`).join("\n")}

FULL EXISTING TREE:
${existing.map((item) => `${item.number} ${item.text}`).join("\n")}

WEB RESEARCH RESULT (the only factual material allowed for rewrites):
${researchResult}

SOURCE TEXT (freshness exclusion list):
${source}`;
  const reviewedRaw = await callFreshTreeLLM(prompt, signal, "Reviewing support and removing duplicates");
  const start = reviewedRaw.indexOf("{");
  const end = reviewedRaw.lastIndexOf("}");
  if (start < 0 || end < start) throw new Error("Fresh-tier review returned invalid structured output.");
  const reviewed = JSON.parse(reviewedRaw.slice(start, end + 1));
  const accepted: TierCandidate[] = [];
  const rejected: TierCandidate[] = [];
  for (const candidate of candidates) {
    const result = reviewed[candidate.number];
    if (!result || result.directSupport !== true || typeof result.text !== "string" || !result.text.trim()) {
      rejected.push({ ...candidate, reviewFailure: { label: "OFF-TARGET", reason: "The node did not pass the existing direct-support review." } });
      continue;
    }
    const wasDuplicate = duplicateNumbers.includes(candidate.number);
    const updated = { ...candidate, text: wasDuplicate ? result.text.trim() : candidate.text };
    // This is the required post-regeneration code gate. A second duplicate is dropped, not retried.
    if (duplicatesAny(updated, [...existing, ...accepted])) {
      rejected.push({ ...updated, reviewFailure: { label: "VAGUE", reason: "The node remained too similar to an existing tree node after one regeneration." } });
      continue;
    }
    accepted.push(updated);
  }
  const byNumber = new Map([...accepted, ...rejected].map((candidate) => [candidate.number, candidate]));
  return candidates.map((candidate) => byNumber.get(candidate.number)!).filter(Boolean);
}

type LegitimacyResult = { label: "EMPIRICAL" | "LOGICAL" | "AUTHORITY" | "VAGUE" | "OFF-TARGET"; reason: string };

function parseClassifications(raw: string, candidates: TierCandidate[]): Record<string, LegitimacyResult> {
  const start = raw.indexOf("{");
  const end = raw.lastIndexOf("}");
  if (start >= 0 && end >= start) {
    try {
      const parsed = JSON.parse(raw.slice(start, end + 1));
      const result: Record<string, LegitimacyResult> = {};
      for (const candidate of candidates) {
        const item = parsed[candidate.number];
        if (item && ["EMPIRICAL", "LOGICAL", "AUTHORITY", "VAGUE", "OFF-TARGET"].includes(item.label) &&
          typeof item.reason === "string" && item.reason.trim()) {
          result[candidate.number] = { label: item.label, reason: item.reason.trim() };
        }
      }
      if (candidates.every((candidate) => result[candidate.number])) return result;
    } catch {
      // Try the requested line-oriented response format below.
    }
  }
  const result: Record<string, LegitimacyResult> = {};
  for (const line of raw.split("\n")) {
    const match = /^\s*(\d+(?:\.\d+)+)\s*[,|:\-]\s*(EMPIRICAL|LOGICAL|AUTHORITY|VAGUE|OFF-TARGET)\s*[,|:\-]\s*(.+)$/i.exec(line);
    if (match) result[match[1]] = { label: match[2].toUpperCase() as LegitimacyResult["label"], reason: match[3].trim() };
  }
  const missing = candidates.filter((candidate) => !result[candidate.number]).map((candidate) => candidate.number);
  if (missing.length) throw new Error(`Fresh-tier legitimacy classifier returned no valid label and reason for: ${missing.join(", ")}.`);
  return result;
}

async function classifyLegitimacy(candidates: TierCandidate[], existing: FreshTreeStatement[], signal: AbortSignal,
  allowCitationNodes: boolean): Promise<Record<string, LegitimacyResult>> {
  if (!candidates.length) return {};
  const prompt = `Label each node. EMPIRICAL = a specific, concrete fact or case. LOGICAL = a derivation, demonstration, or counterexample. AUTHORITY = its content is that someone says/argues/believes something. VAGUE = gestures at evidence without stating it. OFF-TARGET = does not establish the parent's exact claim. Return: node number, label, reason. Return ONLY a JSON object keyed by node number, each value {"label":"EMPIRICAL|LOGICAL|AUTHORITY|VAGUE|OFF-TARGET","reason":"one-line reason"}.
 A restatement or paraphrase of the parent is VAGUE, not LOGICAL: an actual derivation needs distinct premises and a complete inference. "A study found people conform to groups" is VAGUE, not EMPIRICAL, because it gives no specific study, measurement, place, or observed case. "The Stanford Encyclopedia of Philosophy notes that induction lacks a non-circular justification" is AUTHORITY, not LOGICAL. Do not mistake a matching topic for support of the exact parent.
${allowCitationNodes ? 'The user explicitly requested "allow citation nodes": AUTHORITY citation nodes may pass, but OFF-TARGET and VAGUE nodes must still fail.' : ""}

${NODE_LEGITIMACY_RULE}

PARENTS AND NODES:
${candidates.map((item) => `${item.parent} ${existing.find((line) => line.number === item.parent)?.text}\n${item.number} ${item.text}`).join("\n")}`;
  const raw = await callFreshTreeLLM(prompt, signal);
  return parseClassifications(raw, candidates);
}

function patternLabel(text: string, allowCitationNodes: boolean): "VAGUE" | "AUTHORITY" | undefined {
  if (VAGUE_PATTERNS.some((pattern) => pattern.test(text))) return "VAGUE";
  if (!allowCitationNodes && (AUTHORITY_PATTERNS.some((pattern) => pattern.test(text)) ||
    /^(?:\s*(?:https?:\/\/\S+|[¹²³⁴⁵⁶⁷⁸⁹⁰]+|\[\d+\])\s*[.,]?\s*)$/.test(text))) return "AUTHORITY";
  return undefined;
}

function patternFailure(text: string, allowCitationNodes: boolean): string | undefined {
  const label = patternLabel(text, allowCitationNodes);
  return label === "VAGUE" ? "The node contains a forbidden vague-evidence phrase." :
    label === "AUTHORITY" ? "The node contains a forbidden authority phrase or only a citation." : undefined;
}

function legitimacyPasses(label: string | undefined, allowCitationNodes: boolean): boolean {
  return label === "EMPIRICAL" || label === "LOGICAL" || (allowCitationNodes && label === "AUTHORITY");
}

async function recheckDirectSupport(candidates: TierCandidate[], existing: FreshTreeStatement[], signal: AbortSignal): Promise<Set<string>> {
  if (!candidates.length) return new Set();
  const prompt = `Review regenerated child nodes. For every exact node number return directSupport true only if the node directly establishes its immediate parent's specific claim; otherwise false. Return ONLY a JSON object keyed by node number, shaped {"1.1.1":{"directSupport":true}}.

PARENTS AND REGENERATED NODES:
${candidates.map((item) => `${item.parent} ${existing.find((line) => line.number === item.parent)?.text}\n${item.number} ${item.text}`).join("\n")}

FULL EXISTING TREE:
${existing.map((item) => `${item.number} ${item.text}`).join("\n")}`;
  const raw = await callFreshTreeLLM(prompt, signal);
  const start = raw.indexOf("{");
  const end = raw.lastIndexOf("}");
  if (start < 0 || end < start) throw new Error("Fresh-tier regenerated-node review returned invalid structured output.");
  const reviewed = JSON.parse(raw.slice(start, end + 1));
  return new Set(candidates.filter((candidate) => reviewed[candidate.number]?.directSupport === true).map((candidate) => candidate.number));
}

async function enforceLegitimacy(candidates: TierCandidate[], existing: FreshTreeStatement[], source: string, researchResult: string,
  instructions: string, signal: AbortSignal, onWarning: (message: string) => void, shouldStop?: () => boolean): Promise<TierCandidate[]> {
  const allowCitationNodes = /\ballow citation nodes\b/i.test(instructions);
  const firstReview = await classifyLegitimacy(candidates, existing, signal, allowCitationNodes);
  if (shouldStop?.()) return [];
  const failed = candidates.filter((candidate) => candidate.reviewFailure || patternFailure(candidate.text, allowCitationNodes) ||
    !legitimacyPasses(firstReview[candidate.number]?.label, allowCitationNodes) ||
    firstReview[candidate.number]?.label === "OFF-TARGET");
  const regenerated: TierCandidate[] = [];
  for (const candidate of failed) {
    assertNotAborted(signal);
    const classification = firstReview[candidate.number] || { label: "VAGUE" as const, reason: "The classifier did not return a label." };
    const rejection = candidate.reviewFailure;
    const reason = rejection?.reason || patternFailure(candidate.text, allowCitationNodes) || classification.reason;
    const label = rejection?.label || patternLabel(candidate.text, allowCitationNodes) || classification.label;
    const prompt = `Regenerate this rejected node exactly once. Your node was rejected as ${label}: ${reason}. Replace it with a node that establishes the parent by a concrete fact or by logic.

${NODE_LEGITIMACY_RULE}
${allowCitationNodes ? 'The user explicitly requested "allow citation nodes", but the replacement must still establish its parent and must not be OFF-TARGET.' : ""}

Return ONLY one sentence for node ${candidate.number}, with no node number or links. If factual, put any citation marker (¹, ², ...) only at the end. Use only the provided web research result for factual claims.

USER INSTRUCTIONS:
${instructions.trim() || DEFAULT_FRESH_INSTRUCTIONS}
PARENT ${candidate.parent}: ${existing.find((line) => line.number === candidate.parent)?.text}
REJECTED NODE ${candidate.number}: ${candidate.text}
WEB RESEARCH RESULT: ${researchResult}
SOURCE TEXT (freshness exclusion list): ${source}`;
    const replacement = await callFreshTreeLLM(prompt, signal, "Regenerating a rejected support node");
    assertNotAborted(signal);
    if (shouldStop?.()) return [];
    const text = replacement.trim().replace(/^\s*\d+(?:\.\d+)+\s+/, "").split("\n")[0].trim();
    if (text) regenerated.push({ ...candidate, text });
  }
  if (shouldStop?.()) return [];
  const recheck = regenerated.length ? await classifyLegitimacy(regenerated, existing, signal, allowCitationNodes) : {};
  if (shouldStop?.()) return [];
  const supportRecheck = regenerated.length ? await recheckDirectSupport(regenerated, existing, signal) : new Set<string>();
  if (shouldStop?.()) return [];
  const accepted: TierCandidate[] = [];
  const failedNumbers = new Set(failed.map((candidate) => candidate.number));
  const byNumber = new Map(regenerated.map((item) => [item.number, item]));
  for (const candidate of candidates) {
    const replacement = byNumber.get(candidate.number);
    const result = replacement ? recheck[candidate.number] : firstReview[candidate.number];
    const text = replacement?.text || candidate.text;
    const pattern = patternFailure(text, allowCitationNodes);
    const supportFailed = Boolean(replacement && !supportRecheck.has(candidate.number));
    const regenerationMissing = failedNumbers.has(candidate.number) && !replacement;
    if (pattern || result?.label === "OFF-TARGET" || !legitimacyPasses(result?.label, allowCitationNodes) || supportFailed || regenerationMissing) {
      const label = supportFailed ? "OFF-TARGET" : patternLabel(text, allowCitationNodes) || result?.label || candidate.reviewFailure?.label || "VAGUE";
      const why = supportFailed ? "The regenerated node did not pass the existing direct-support review." :
        regenerationMissing ? "The one allowed regeneration returned no replacement." :
          pattern || result?.reason || candidate.reviewFailure?.reason || "The classifier did not return a label.";
      onWarning(`Node ${candidate.number} dropped: ${label} — ${why}`);
      continue;
    }
    accepted.push({ ...candidate, text, legitimacyLabel: result!.label });
  }
  return accepted;
}

function superscript(value: number): string {
  const digits = ["⁰", "¹", "²", "³", "⁴", "⁵", "⁶", "⁷", "⁸", "⁹"];
  return String(value).split("").map((digit) => digits[Number(digit)]).join("");
}

export async function addNextTier(source: string, existing: FreshTreeStatement[], instructions: string, signal: AbortSignal,
  onTier: (statements: FreshTreeStatement[], sources: FreshTreeSource[]) => void, options: AddTierOptions = {}): Promise<boolean> {
  if (!existing.length) throw new Error("The existing tree has no statements.");
  const deepest = Math.max(...existing.map((item) => item.depth));
  const tierNumber = options.tierNumber ?? deepest + 2;
  if (!Number.isInteger(tierNumber) || tierNumber < 3) throw new Error("The requested tier number must be 3 or greater.");
  const parents = freshTreeParentsAtTier(existing, tierNumber);
  if (!parents.length && options.tierNumber === undefined) throw new Error("The existing tree has no deepest-tier nodes.");
  const completed = new Set(options.completedTheses || []);
  const roots = existing.filter((item) => item.number.endsWith(".0")).filter((root) => !completed.has(root.number));
  if (!roots.length) return false;
  const maxTheses = roots.length;
  const count = childCount(instructions, tierNumber);
  const effectiveInstructions = !instructions.trim() || instructions.trim() === DEFAULT_FRESH_INSTRUCTIONS
    ? (tierNumber > 3
      ? DEFAULT_FRESH_INSTRUCTIONS.replace("add 1 or 2 child nodes", "add exactly 1 child node")
      : DEFAULT_FRESH_INSTRUCTIONS)
    : instructions.trim();
  const allowCitationNodes = /\ballow citation nodes\b/i.test(instructions);
  let markerNumber = existing.reduce((max, statement) => {
    const trailingMarker = /([⁰¹²³⁴⁵⁶⁷⁸⁹]+)$/.exec(statement.text.trim())?.[1];
    if (!trailingMarker) return max;
    const numeric = Number(trailingMarker.replace(/[⁰¹²³⁴⁵⁶⁷⁸⁹]/g, (digit) =>
      String("⁰¹²³⁴⁵⁶⁷⁸⁹".indexOf(digit))));
    return Number.isFinite(numeric) ? Math.max(max, numeric) : max;
  }, 0);
  for (let rootIndex = 0; rootIndex < roots.length; rootIndex++) {
    const root = roots[rootIndex];
    assertNotAborted(signal);
    if (options.shouldStop?.()) return true;
    const prefix = root.number.split(".")[0] + ".";
    const group = parents.filter((item) => item.number.startsWith(prefix));
    if (!group.length) {
      options.onThesisComplete?.(tierNumber, root.number, [], []);
      continue;
    }
    options.onProgress?.(tierNumber, rootIndex + 1, maxTheses);
    const prompt = `For EACH parent listed under PARENTS FOR THIS CALL, search separately using that parent's full sentence as the query. Do not search the top-level thesis topic. Do not reuse one paper for two parents. If a parent is a distinction, definition, or inference, you may return a LOGICAL child with url "". A LOGICAL child may use the source example that proves that parent. Follow the user's instructions. Return ONLY a JSON object keyed by every exact parent number. Each value is an array of ${count.min === count.max ? count.min : `${count.min} or ${count.max}`} objects shaped {"text":"one sentence","url":"direct supporting source URL"}. Include every parent exactly once. Do not put links in text.

A child may be LOGICAL (author's contrast, definition, derivation, or source example) or EMPIRICAL (a concrete result that is the parent's claim). Named studies are required only for EMPIRICAL children. It must directly support its IMMEDIATE parent's specific claim. A restatement or paraphrase of its parent or any other tree line is forbidden.

Use only primary or reputable sources: journals, universities, government agencies, or major science outlets. Never use career sites, content farms, or pop-psychology explainer sites. Confirm that each source's date and content match the node's claim.

${NODE_LEGITIMACY_RULE}
${allowCitationNodes ? 'The user explicitly requested "allow citation nodes"; citation/authority nodes are allowed, but vague and OFF-TARGET nodes remain forbidden.' : ""}

USER INSTRUCTIONS:
${effectiveInstructions}

FULL EXISTING TREE (do not alter it):
${existing.map((item) => `${item.number} ${item.text}`).join("\n")}

PARENTS FOR THIS CALL:
${group.map((item) => `${item.number} ${item.text}`).join("\n")}

SOURCE TEXT (reference and freshness exclusion list):
<source>
${source}
</source>`;
    try {
      const answer = await (options.research || research)(prompt, signal, options.shouldStop);
      assertNotAborted(signal);
      if (options.shouldStop?.()) return true;
      const parseAvailable = (raw: string, requested: FreshTreeStatement[]) => {
        try { return parseTier(raw, requested, tierNumber, count, true); }
        catch { return [] as TierCandidate[]; }
      };
      const parsed = parseAvailable(answer.text, group);
      let missing = group.filter((parent) => !parsed.some((child) => child.parent === parent.number));
      if (missing.length) {
        const repaired = await callFreshTreeLLM(`The prior tier response omitted or misnumbered a required parent. Return ONLY a JSON object with exactly these parent keys: ${missing.map((item) => item.number).join(", ")}. Under each key put ${count.min === count.max ? count.min : `${count.min} or ${count.max}`} child objects shaped {"text":"one concrete sentence establishing this exact parent","url":"direct reputable supporting source URL"}. Use facts from the research response below; do not invent claims or cite someone merely saying something.\n\n${NODE_LEGITIMACY_RULE}\n\nPARENTS:\n${missing.map((item) => `${item.number} ${item.text}`).join("\n")}\n\nRESEARCH RESPONSE:\n${answer.text}\n\nSOURCE TEXT (freshness exclusion list):\n${source}`, signal);
        assertNotAborted(signal);
        if (options.shouldStop?.()) return true;
        parsed.push(...parseAvailable(repaired, missing));
        missing = missing.filter((parent) => !parsed.some((child) => child.parent === parent.number));
        for (const parent of missing) {
          try {
            const individual = await callFreshTreeLLM(`Return ONLY a JSON array of ${count.min} ${count.min === 1 ? "child" : "children"} for parent ${parent.number}: ${parent.text}. Each child is {"text":"one sentence establishing the exact parent by a concrete fact or full logical counterexample","url":"direct reputable source URL if factual"}. Use only facts supported by the research below, no authority assertions.\n\n${NODE_LEGITIMACY_RULE}\n\nRESEARCH:\n${answer.text}\n\nSOURCE TEXT (freshness exclusion list):\n${source}`, signal);
            assertNotAborted(signal);
            if (options.shouldStop?.()) return true;
            const opening = individual.indexOf("[");
            const closing = individual.lastIndexOf("]");
            const children = opening >= 0 && closing > opening ? JSON.parse(individual.slice(opening, closing + 1)) : [];
            parsed.push(...parseTier(JSON.stringify({ [parent.number]: children }), [parent], tierNumber, count));
          } catch (error: any) {
            if (signal.aborted) throw error;
            options.onWarning?.(`Node ${parent.number} has no legitimate support: the model did not supply a valid child.`);
          }
        }
      }
      if (!parsed.length) throw new Error(`No valid child nodes were returned for thesis ${root.number}.`);
      const accepted = parsed.length ? await reviewAndDeduplicate(parsed, existing, answer.text, source, signal) : [];
      assertNotAborted(signal);
      if (options.shouldStop?.()) return true;
      const supported = await enforceLegitimacy(
        accepted.map((candidate) => ({ ...candidate, searched: answer.searched })),
        existing, source, answer.text, instructions, signal,
        (message) => options.onWarning?.(message), options.shouldStop,
      );
      assertNotAborted(signal);
      if (options.shouldStop?.()) return true;
      const parentNumbers = new Set(supported.map((item) => item.parent));
      for (const parent of group) if (!parentNumbers.has(parent.number)) {
        options.onWarning?.(`Node ${parent.number} has no legitimate support.`);
      }
      const statements: FreshTreeStatement[] = [];
      const sources: FreshTreeSource[] = [];
      for (const { parent: _parent, url, searched, legitimacyLabel, reviewFailure: _reviewFailure, ...statement } of supported) {
        const factualText = statement.text.replace(/[¹²³⁴⁵⁶⁷⁸⁹⁰]/g, "").trim();
        if (url && searched && legitimacyLabel === "EMPIRICAL") {
          markerNumber++;
          const marker = superscript(markerNumber);
          statements.push({ ...statement, text: `${factualText}${marker}` });
          sources.push({ node: statement.number, marker, url });
        } else {
          statements.push({ ...statement, text: factualText });
        }
      }
      if (statements.length) onTier(statements, sources);
      options.onThesisComplete?.(tierNumber, root.number, statements, sources);
    } catch (error: any) {
      throw new Error(`Tier ${tierNumber}, thesis ${root.number}: ${error?.message || "generation failed"}`);
    }
  }
  return false;
}
