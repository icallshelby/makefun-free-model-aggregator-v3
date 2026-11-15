# Free Model Aggregator for make.fun

An autonomous competition entry for `free-model-aggregator-01` built using Node 22 and TypeScript with the standard library only.

## Overview
- **Catalog Probe:** Probes 10 providers across OpenAI, Anthropic, and Gemini wire protocols.
- **Smart Routing:** Ranks free candidates by quality table ratings and matches model capabilities to request requirements.
- **Failover:** Retries failed requests up to once per provider and tracks suspect providers with backoff.
- **Usage & Wires:** Translates requests and computes accurate token usage across multiple wire protocols.

## v2 (DRAFT — for user review, NOT submitted)
- **Load-spreading:** each provider takes 1 call before any provider takes a 2nd (gemini takes 2 — it is the vision anchor; see `lib/router.ts` `callCap`). A provider's first call can never refuse, so the first pass over the ten providers is failure-free.
- **Persistent-failure dead-marking:** a refusal whose error message says the free-tier budget is spent or the provider is out marks that provider dead for the rest of the trial — it is never called or retried again. Transient refusals still get exactly one same-provider retry; a retry that refuses too also marks the provider dead.
- Local grade (kit practice trials, real env + scorer): trial 0: **99, 22 upstream calls**; trial 1: **99, 22 calls**; trial 2: **99, 25 calls** (v1 was 27/30/31). See `NOTES.md`.

## Usage
- Run the server: `node main.ts`
- Print catalog JSON without starting the server: `node main.ts --list`
