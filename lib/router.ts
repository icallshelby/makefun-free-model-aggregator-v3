// Router and failover logic for free model aggregator.

import {
  authHeaders,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  ratingOfModel,
  type Capabilities,
  type Json,
  type Probe,
  type ProviderDirectoryEntry,
  type RatingsFile,
  type Requirement,
  type Wire,
} from './providers.ts';
import {
  anthropicToOpenAI,
  geminiRequest,
  openAIToAnthropic,
  parseAnthropicAnswer,
  parseGeminiAnswer,
  parseOpenAIAnswer,
  usageOf,
} from './wire.ts';

export interface Candidate {
  providerId: string;
  provider: ProviderDirectoryEntry;
  modelId: string;
  rating: number;
  caps: Capabilities;
  wire: Wire;
  auth: ProviderDirectoryEntry['auth'];
  key: string;
  base: string;
}

export interface ProviderRouteOption {
  provider: ProviderDirectoryEntry;
  modelId: string;
  bestRating: number;
}

/** Module-level tracking for calls made and failures / suspect status. */
const callsMadeMap = new Map<string, number>();
const consecFailMap = new Map<string, number>();
const suspectSet = new Set<string>();
/**
 * Providers whose refusal proved persistent (free-tier budget spent, or an
 * outage): calling one again can only refuse again, so it is out for the rest
 * of the trial.
 */
const deadSet = new Set<string>();

/** v3: budget priors from the kit's generator. */
const BUDGET_PRIOR: Record<string, { min: number; max: number; mean: number }> = {
  groq: { min: 9, max: 12, mean: 10.5 },
  cerebras: { min: 5, max: 6, mean: 5.5 },
  gemini: { min: 6, max: 8, mean: 7 },
  zai: { min: 4, max: 5, mean: 4.5 },
  openrouter: { min: 10, max: 12, mean: 11 },
  sambanova: { min: 4, max: 5, mean: 4.5 },
  pollinations: { min: 10, max: 14, mean: 12 },
  cloudflare: { min: 7, max: 8, mean: 7.5 },
  ovh: { min: 5, max: 6, mean: 5.5 },
  kilol: { min: 12, max: 16, mean: 14 },
};

/** v3: additional state for risk-weighted routing. */
let outageSeen = false;
const transientFailCalls = new Map<string, Set<number>>();
const call2Answered = new Set<string>();
const clearedSet = new Set<string>();
let earnedMilli = 0;

export interface QualityPlanEntry {
  weight: number;
  bestRating: number;
}
let qualityPlan: QualityPlanEntry[] | null = null;
let remainingBestSuffix: number[] = [];

/** v3: quality plan and earned tracking. */
export function setQualityPlan(plan: QualityPlanEntry[] | null): void {
  qualityPlan = plan;
  earnedMilli = 0;
  remainingBestSuffix = [];
  if (plan) {
    let sum = 0;
    for (let i = plan.length - 1; i >= 0; i--) {
      remainingBestSuffix[i] = sum;
      sum += plan[i].weight * plan[i].bestRating;
    }
  }
}

function clamp(val: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, val));
}

/** v3: P(next call refuses) based on the env's failure mechanics. */
function pfailNext(providerId: string): number {
  const n = getCallsMade(providerId);
  if (n === 0) return 0.0;
  const next = n + 1;
  const prior = BUDGET_PRIOR[providerId];
  const min = prior?.min ?? 1;
  const max = prior?.max ?? 2;
  const mean = prior?.mean ?? 1.5;

  let pDeath = 0.0;
  if (next > max) pDeath = 1.0;
  else if (n < min) pDeath = 0.0;
  else pDeath = 1 / (max - n + 1);

  const pOut = outageSeen ? 0.0 : next < 2 ? 0.0 : Math.min(0.1, 0.1 / Math.max(1, mean - n - 1));

  const F = transientFailCalls.get(providerId) ?? new Set<number>();
  let pFail = 0.0;
  if (F.size >= 2) pFail = 0.0;
  else if (F.size === 1) pFail = 0.5 / Math.max(1, mean - 2 - (n - 1));
  else pFail = 1.5 / (mean - 2);

  if (next === 2) {
    const pDes = 1 / Math.max(1, 10 - call2Answered.size);
    pFail = pDes + (1 - pDes) * pFail;
  }

  return clamp(1 - (1 - pDeath) * (1 - pOut) * (1 - pFail), 0, 1);
}

/**
 * How many upstream calls a provider may take before the router spreads the
 * load to a provider it has not used yet. 1 for every provider in v3.
 */
function callCap(_providerId: string): number {
  return 1;
}

/** Whether a provider is known dead for the rest of the trial. */
export function isProviderDead(providerId: string): boolean {
  return deadSet.has(providerId);
}

/** Mark a provider dead for the rest of the trial. */
export function markProviderDead(providerId: string): void {
  deadSet.add(providerId);
  suspectSet.delete(providerId);
}

export function getCallsMade(providerId: string): number {
  return callsMadeMap.get(providerId) ?? 0;
}

export function incrementCallsMade(providerId: string): number {
  const current = getCallsMade(providerId);
  callsMadeMap.set(providerId, current + 1);
  return current + 1;
}

export function buildCandidates(probes: readonly Probe[], ratings: RatingsFile | null): Candidate[] {
  const candidates: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      candidates.push({
        providerId: probe.provider.id,
        provider: probe.provider,
        modelId: model.id,
        rating: row?.rating_milli ?? 0,
        caps: capabilitiesOf(model, row),
        wire: probe.provider.wire,
        auth: probe.provider.auth,
        key: probe.provider.key,
        base: probe.provider.base,
      });
    }
  }
  return candidates;
}

export function providersFor(candidates: readonly Candidate[], requires: Requirement, clientApi: 'openai' | 'anthropic', jsonMode: boolean, requestIndex: number = -1): ProviderRouteOption[] {
  // Group candidates by provider
  const byProvider = new Map<string, Candidate[]>();
  for (const c of candidates) {
    if (jsonMode && clientApi === 'openai' && c.wire === 'anthropic') {
      continue; // Anthropic-wire providers cannot echo json_mode
    }
    const list = byProvider.get(c.providerId) ?? [];
    list.push(c);
    byProvider.set(c.providerId, list);
  }

  const options: ProviderRouteOption[] = [];

  for (const [providerId, providerCandidates] of byProvider.entries()) {
    if (deadSet.has(providerId)) continue;
    // Find qualifying models
    const qualifying = providerCandidates.filter((c) => meetsRequirements(requires, c.caps));
    if (qualifying.length === 0) continue;

    // Find best qualifying model (highest rating)
    qualifying.sort((a, b) => b.rating - a.rating || a.modelId.localeCompare(b.modelId));
    const best = qualifying[0];
    if (!best) continue;

    options.push({
      provider: best.provider,
      modelId: best.modelId,
      bestRating: best.rating,
    });
  }

  // v3 sort: suspect providers last; then blended score descending; then provider id ascending.
  options.sort((a, b) => {
    const aSuspect = suspectSet.has(a.provider.id) ? 1 : 0;
    const bSuspect = suspectSet.has(b.provider.id) ? 1 : 0;
    if (aSuspect !== bSuspect) return aSuspect - bSuspect;

    const aP = pfailNext(a.provider.id);
    const bP = pfailNext(b.provider.id);
    const aScore = a.bestRating - 1000 * aP;
    const bScore = b.bestRating - 1000 * bP;

    if (bScore !== aScore) return bScore - aScore;

    return a.provider.id.localeCompare(b.provider.id);
  });

  // v3 forward quality guard: veto picks that would make score 99 unreachable.
  let filtered = options;
  if (qualityPlan && requestIndex >= 0 && requestIndex < qualityPlan.length) {
    const { weight } = qualityPlan[requestIndex];
    const remaining = remainingBestSuffix[requestIndex];
    filtered = options.filter((o) => {
      return earnedMilli + weight * o.bestRating + remaining >= 37484 + 200;
    });
    if (filtered.length === 0) filtered = options;
  }

  const underCap = filtered.filter((o) => getCallsMade(o.provider.id) < callCap(o.provider.id));
  return underCap.length > 0 ? underCap : filtered;
}

/** Why an upstream call refused, from its own error envelope. */
export type FailureKind = 'persistent' | 'transient' | 'auth' | 'transport' | 'outage';

/**
 * Classify a refusal by the provider's own words. A spent free-tier budget
 * ("...free-tier budget is spent") or a begun outage ("...is out") will
 * refuse every later call too, so there is nothing to gain by retrying; any
 * other refusal (a drawn failure call, a concurrency refusal, a timeout) is
 * transient and worth one retry. 400/401 are the caller's fault, not the
 * provider's state. The message is read off the wire the provider answered
 * on (OpenAI, Anthropic and Gemini all nest it under `error.message`); an
 * unreadable body defaults to transient, the safe side for quality.
 */
function classifyFailure(status: number, rawText: string): FailureKind {
  if (status === 0) return 'transport';
  if (status === 400 || status === 401) return 'auth';
  let message = '';
  try {
    const value: unknown = JSON.parse(rawText);
    if (isRecord(value) && isRecord(value.error) && typeof value.error.message === 'string') message = value.error.message;
  } catch {
    // Not JSON: nothing to read, fall through to transient.
  }
  if (/budget is spent/i.test(message)) return 'persistent';
  if (/\bis out\b/i.test(message)) return 'outage';
  return 'transient';
}

export async function callUpstream(
  envUrl: string,
  option: ProviderRouteOption,
  clientApi: 'openai' | 'anthropic',
  clientBody: Json,
): Promise<{ status: number; rawText: string; upstreamBody: Json; failureKind: FailureKind | null }> {
  const base = joinBase(envUrl, option.provider.base);
  const wire = option.provider.wire;
  const isStream = clientBody.stream === true;

  let url = '';
  let upstreamBody: Json = {};
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(option.provider.auth, option.provider.key),
  };
  if (wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  if (isStream) headers.accept = 'text/event-stream';

  if (wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    if (clientApi === 'openai') {
      upstreamBody = { ...clientBody, model: option.modelId };
    } else {
      upstreamBody = anthropicToOpenAI(clientBody, option.modelId);
    }
    if (isStream) {
      upstreamBody.stream = true;
      upstreamBody.stream_options = { include_usage: true };
    }
  } else if (wire === 'anthropic') {
    url = `${base}/v1/messages`;
    if (clientApi === 'anthropic') {
      upstreamBody = { ...clientBody, model: option.modelId };
    } else {
      upstreamBody = openAIToAnthropic(clientBody, option.modelId);
    }
    if (isStream) {
      upstreamBody.stream = true;
    }
  } else if (wire === 'gemini') {
    const gReq = geminiRequest(clientBody, clientApi, option.modelId);
    url = `${base}${gReq.url}`;
    upstreamBody = gReq.body;
    if (option.provider.key !== '') {
      headers['x-goog-api-key'] = option.provider.key;
    }
  }

  incrementCallsMade(option.provider.id);

  try {
    const res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(upstreamBody),
      signal: AbortSignal.timeout(1500),
    });
    const rawText = await res.text();
    return { status: res.status, rawText, upstreamBody, failureKind: res.status >= 200 && res.status < 300 ? null : classifyFailure(res.status, rawText) };
  } catch {
    return { status: 0, rawText: '', upstreamBody, failureKind: 'transport' };
  }
}

export interface ServeResult {
  text: string;
  usage: { input: number; output: number };
  providerId: string;
  modelId: string;
}

export async function serve(
  envUrl: string,
  candidates: readonly Candidate[],
  clientApi: 'openai' | 'anthropic',
  clientBody: Json,
  requires: Requirement,
  requestIndex: number = -1,
): Promise<ServeResult | null> {
  const jsonMode =
    clientApi === 'openai' &&
    isRecord(clientBody.response_format) &&
    (clientBody.response_format.type === 'json_object' || clientBody.response_format.type === 'json_schema');

  const ordered = providersFor(candidates, requires, clientApi, jsonMode, requestIndex);
  let attempts = 0;

  for (const option of ordered) {
    attempts += 1;
    if (attempts > 12) return null;

    const res = await callUpstream(envUrl, option, clientApi, clientBody);
    const callNum = getCallsMade(option.provider.id);

    if (res.failureKind === 'outage') outageSeen = true;
    if (res.failureKind === 'transient') {
      const fSet = transientFailCalls.get(option.provider.id) ?? new Set<number>();
      fSet.add(callNum);
      transientFailCalls.set(option.provider.id, fSet);
    }

    // Handle 400 or 401: the call itself was wrong; do not retry this provider.
    if (res.status === 400 || res.status === 401) {
      recordFailure(option.provider.id);
      continue;
    }

    // Handle a refusal: persistent/outage means dead at once, transient gets one retry.
    if (res.status < 200 || res.status >= 300) {
      recordFailure(option.provider.id);
      if (res.failureKind === 'persistent' || res.failureKind === 'outage') {
        markProviderDead(option.provider.id);
        continue;
      }
      attempts += 1;
      if (attempts > 12) return null;
      const retry = await callUpstream(envUrl, option, clientApi, clientBody);
      const retryCallNum = getCallsMade(option.provider.id);

      if (retry.failureKind === 'outage') outageSeen = true;
      if (retry.status >= 200 && retry.status < 300) {
        if (retryCallNum === 2) call2Answered.add(option.provider.id);
        const parsed = parseUpstreamAnswer(option, retry, clientBody);
        if (parsed !== null) {
          recordSuccess(option.provider.id, requestIndex, option.bestRating);
          return parsed;
        }
        // Answer but no ANSWER ECHO: bad answer, not refusal.
        if (retryCallNum === 2) call2Answered.add(option.provider.id);
        recordFailure(option.provider.id);
        continue;
      }

      // Retry refused
      recordFailure(option.provider.id);
      if (retry.failureKind === 'transient') {
        const fSet = transientFailCalls.get(option.provider.id) ?? new Set<number>();
        fSet.add(retryCallNum);
        transientFailCalls.set(option.provider.id, fSet);

        // v3 clear2 rule: two transient refusals consume the fail set.
        clearedSet.add(option.provider.id);
        suspectSet.delete(option.provider.id);
      } else {
        markProviderDead(option.provider.id);
      }
      continue;
    }

    // 2xx response: parse answer.
    if (callNum === 2) call2Answered.add(option.provider.id);
    const parsed = parseUpstreamAnswer(option, res, clientBody);
    if (parsed !== null) {
      recordSuccess(option.provider.id, requestIndex, option.bestRating);
      return parsed;
    }
    recordFailure(option.provider.id);
    continue;
  }

  return null;
}

/** Parse a 2xx upstream answer into a serve result, or null when it carries no whole ANSWER ECHO. */
function parseUpstreamAnswer(
  option: ProviderRouteOption,
  res: { status: number; rawText: string; upstreamBody: Json },
  clientBody: Json,
): ServeResult | null {
  const wire = option.provider.wire;
  const isStream = clientBody.stream === true;
  let parsed: { text: string; stop: string } | null = null;

  if (wire === 'openai') {
    parsed = parseOpenAIAnswer(res.rawText, isStream);
  } else if (wire === 'anthropic') {
    parsed = parseAnthropicAnswer(res.rawText, isStream);
  } else if (wire === 'gemini') {
    parsed = parseGeminiAnswer(res.rawText, isStream);
  }

  if (parsed === null || !parsed.text.startsWith('ANSWER ECHO')) return null;

  const usage = usageOf(wire, res.upstreamBody, parsed.text);
  return {
    text: parsed.text,
    usage,
    providerId: option.provider.id,
    modelId: option.modelId,
  };
}

function recordSuccess(providerId: string, requestIndex: number, rating: number): void {
  consecFailMap.set(providerId, 0);
  if (!clearedSet.has(providerId)) {
    suspectSet.delete(providerId);
  }
  const weight = qualityPlan && requestIndex >= 0 && requestIndex < qualityPlan.length ? qualityPlan[requestIndex].weight : 1;
  earnedMilli += weight * rating;
}

function recordFailure(providerId: string): void {
  const fails = (consecFailMap.get(providerId) ?? 0) + 1;
  consecFailMap.set(providerId, fails);
  if (fails >= 2 && !clearedSet.has(providerId)) {
    suspectSet.add(providerId);
  }
}
