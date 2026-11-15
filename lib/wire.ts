// The client wires and upstream wire helpers for OpenAI, Anthropic, and Gemini.

import { isRecord, type Json, type Wire } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
export function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** textOf(content): string -> return blocksText(content). (anthropicToOpenAI and the parsers call textOf; it must exist.) */
export function textOf(content: unknown): string {
  return blocksText(content);
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
export function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
export function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
export function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** OpenAI client REQUEST -> Anthropic provider REQUEST. */
export function openAIToAnthropic(clientBody: Json, model: string): Json {
  const messages: Json[] = [];
  const clientMessages = Array.isArray(clientBody.messages) ? clientBody.messages : [];
  const systemBlocks: string[] = [];

  for (const m of clientMessages) {
    if (!isRecord(m)) continue;
    if (m.role === 'system') {
      systemBlocks.push(textOf(m.content));
    }
  }
  const system = systemBlocks.join('\n');

  for (const m of clientMessages) {
    if (!isRecord(m)) continue;
    if (m.role === 'system') continue;
    if (m.role === 'tool') continue; // skip tool messages entirely

    const assistant = m.role === 'assistant';
    const content = m.content;

    if (typeof content === 'string') {
      messages.push({
        role: assistant ? 'assistant' : 'user',
        content,
      });
      continue;
    }

    if (Array.isArray(content)) {
      if (assistant) {
        messages.push({
          role: 'assistant',
          content: textOf(content),
        });
      } else {
        const parts: Json[] = [];
        for (const block of content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') {
            parts.push({ type: 'text', text: block.text });
          } else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
            const url = block.image_url.url;
            if (url.startsWith('data:')) {
              const comma = url.indexOf(',');
              if (comma !== -1) {
                const header = url.slice(5, comma);
                const data = url.slice(comma + 1);
                const mediaType = header.split(';')[0] || 'image/png';
                parts.push({
                  type: 'image',
                  source: {
                    type: 'base64',
                    media_type: mediaType,
                    data,
                  },
                });
              }
            } else {
              parts.push({
                type: 'image',
                source: {
                  type: 'url',
                  url,
                },
              });
            }
          }
        }
        messages.push({
          role: 'user',
          content: parts.length > 0 ? parts : textOf(content),
        });
      }
    }
  }

  const out: Json = { model, messages };
  if (system !== '') out.system = system;
  if (typeof clientBody.max_tokens === 'number') out.max_tokens = clientBody.max_tokens;
  if (Array.isArray(clientBody.stop)) {
    out.stop_sequences = clientBody.stop;
  } else if (typeof clientBody.stop === 'string') {
    out.stop_sequences = [clientBody.stop];
  }
  if (typeof clientBody.temperature === 'number') out.temperature = clientBody.temperature;
  if (typeof clientBody.top_p === 'number') out.top_p = clientBody.top_p;

  if (Array.isArray(clientBody.tools)) {
    const tools = clientBody.tools.flatMap((raw) => {
      if (!isRecord(raw)) return [];
      const fn = isRecord(raw.function) ? raw.function : raw;
      const name = typeof fn.name === 'string' ? fn.name : '';
      if (!name) return [];
      return [{
        name,
        description: typeof fn.description === 'string' ? fn.description : '',
        input_schema: isRecord(fn.parameters) ? fn.parameters : { type: 'object', properties: {} },
      }];
    });
    if (tools.length > 0) out.tools = tools;
  }

  const choice = clientBody.tool_choice;
  if (typeof choice === 'string') {
    if (choice === 'auto') out.tool_choice = { type: 'auto' };
    else if (choice === 'required' || choice === 'any') out.tool_choice = { type: 'any' };
  } else if (isRecord(choice)) {
    if (choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
      out.tool_choice = { type: 'tool', name: choice.function.name };
    }
  }

  if (clientBody.stream === true) {
    out.stream = true;
  }

  return out;
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Gemini parts text helper. */
export function geminiPartsText(parts: unknown): string {
  if (!Array.isArray(parts)) return '';
  return parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('\n');
}

/** Parse upstream body for usage calculation. */
export function parseUpstreamForUsage(wire: Wire, body: Json): { systemText: string; userText: string; tools: string[] } {
  let systemText = '';
  let userText = '';
  const tools: string[] = [];

  if (wire === 'openai') {
    const messages = Array.isArray(body.messages) ? body.messages : [];
    const sysBlocks: string[] = [];
    let lastUserContent: unknown = '';
    for (const m of messages) {
      if (!isRecord(m)) continue;
      if (m.role === 'system') {
        sysBlocks.push(blocksText(m.content));
      } else if (m.role === 'user') {
        lastUserContent = m.content;
      }
    }
    systemText = sysBlocks.join('\n');
    userText = blocksText(lastUserContent);
    const bodyTools = Array.isArray(body.tools) ? body.tools : [];
    for (const t of bodyTools) {
      if (isRecord(t) && isRecord(t.function) && typeof t.function.name === 'string') {
        tools.push(t.function.name);
      }
    }
  } else if (wire === 'anthropic') {
    systemText = blocksText(body.system);
    const messages = Array.isArray(body.messages) ? body.messages : [];
    let lastUserContent: unknown = '';
    for (const m of messages) {
      if (!isRecord(m)) continue;
      if (m.role === 'user') {
        lastUserContent = m.content;
      }
    }
    userText = blocksText(lastUserContent);
    const bodyTools = Array.isArray(body.tools) ? body.tools : [];
    for (const t of bodyTools) {
      if (isRecord(t) && typeof t.name === 'string') {
        tools.push(t.name);
      }
    }
  } else if (wire === 'gemini') {
    const sysInst = isRecord(body.systemInstruction) ? body.systemInstruction : null;
    if (sysInst && Array.isArray(sysInst.parts)) {
      systemText = geminiPartsText(sysInst.parts);
    }
    const contents = Array.isArray(body.contents) ? body.contents : [];
    let lastUserParts: unknown = [];
    for (const c of contents) {
      if (!isRecord(c)) continue;
      if (c.role === 'user') {
        lastUserParts = c.parts;
      }
    }
    userText = geminiPartsText(lastUserParts);
    const gTools = Array.isArray(body.tools) ? body.tools : [];
    for (const gt of gTools) {
      if (!isRecord(gt)) continue;
      const decls = Array.isArray(gt.functionDeclarations) ? gt.functionDeclarations : [];
      for (const d of decls) {
        if (isRecord(d) && typeof d.name === 'string') {
          tools.push(d.name);
        }
      }
    }
  }
  return { systemText, userText, tools };
}

/** Compute usage from upstream body and answer text. */
export function usageOf(upstreamWire: Wire, upstreamBody: Json, answerText: string): { input: number; output: number } {
  const { systemText, userText, tools } = parseUpstreamForUsage(upstreamWire, upstreamBody);
  const toolStr = tools.join(',');
  const inputLen = systemText.length + userText.length + toolStr.length;
  const input = 1 + Math.floor(inputLen / 4);
  const output = 1 + Math.floor(answerText.length / 4);
  return { input, output };
}

/** Split text into 13-character pieces. */
export function splitText(text: string): string[] {
  if (text === '') return [''];
  const pieces: string[] = [];
  for (let i = 0; i < text.length; i += 13) {
    pieces.push(text.slice(i, i + 13));
  }
  return pieces;
}

export interface SseFrame {
  event: string | null;
  data: string;
}

/** Parse SSE frames. */
export function parseSseFrames(text: string): SseFrame[] {
  const norm = text.replace(/\r\n/g, '\n');
  const blocks = norm.split('\n\n');
  const out: SseFrame[] = [];
  for (const block of blocks) {
    if (block.trim() === '') continue;
    let event: string | null = null;
    const dataLines: string[] = [];
    for (const line of block.split('\n')) {
      if (line.startsWith('event:')) {
        event = line.slice(6).trim();
      } else if (line.startsWith('data:')) {
        dataLines.push(line.slice(5).trim());
      }
    }
    out.push({ event, data: dataLines.join('\n') });
  }
  return out;
}

/** Gemini request builder. */
export function geminiRequest(clientBody: Json, clientApi: 'openai' | 'anthropic', model: string): { url: string; body: Json } {
  let systemText = '';
  const contents: Json[] = [];

  if (clientApi === 'openai') {
    const messages = Array.isArray(clientBody.messages) ? clientBody.messages : [];
    const sysBlocks: string[] = [];
    for (const m of messages) {
      if (!isRecord(m)) continue;
      if (m.role === 'system') {
        sysBlocks.push(textOf(m.content));
      }
    }
    systemText = sysBlocks.join('\n');

    for (const m of messages) {
      if (!isRecord(m)) continue;
      if (m.role === 'system') continue;
      if (m.role === 'tool') continue;

      const assistant = m.role === 'assistant';
      const role = assistant ? 'model' : 'user';
      const content = m.content;
      const parts: Json[] = [];

      if (typeof content === 'string') {
        if (content !== '') parts.push({ text: content });
      } else if (Array.isArray(content)) {
        for (const block of content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') {
            parts.push({ text: block.text });
          } else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
            const url = block.image_url.url;
            if (url.startsWith('data:')) {
              const comma = url.indexOf(',');
              if (comma !== -1) {
                const header = url.slice(5, comma);
                const data = url.slice(comma + 1);
                const mimeType = header.split(';')[0] || 'image/png';
                parts.push({ inlineData: { mimeType, data } });
              }
            } else {
              parts.push({ fileData: { fileUri: url } });
            }
          }
        }
      }
      if (parts.length > 0) {
        contents.push({ role, parts });
      }
    }
  } else {
    systemText = blocksText(clientBody.system);
    const messages = Array.isArray(clientBody.messages) ? clientBody.messages : [];
    for (const m of messages) {
      if (!isRecord(m)) continue;
      const assistant = m.role === 'assistant';
      const role = assistant ? 'model' : 'user';
      const content = m.content;
      const parts: Json[] = [];

      if (typeof content === 'string') {
        if (content !== '') parts.push({ text: content });
      } else if (Array.isArray(content)) {
        for (const block of content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') {
            parts.push({ text: block.text });
          } else if (block.type === 'image' && isRecord(block.source)) {
            const source = block.source;
            if (source.type === 'base64' && typeof source.data === 'string') {
              const mimeType = typeof source.media_type === 'string' ? source.media_type : 'image/png';
              parts.push({ inlineData: { mimeType, data: source.data } });
            } else if (source.type === 'url' && typeof source.url === 'string') {
              parts.push({ fileData: { fileUri: source.url } });
            }
          }
        }
      }
      if (parts.length > 0) {
        contents.push({ role, parts });
      }
    }
  }

  const body: Json = { contents };
  if (systemText !== '') {
    body.systemInstruction = { parts: [{ text: systemText }] };
  }

  const rawTools = clientBody.tools;
  if (Array.isArray(rawTools)) {
    const decls: Json[] = [];
    for (const t of rawTools) {
      if (!isRecord(t)) continue;
      const fn = isRecord(t.function) ? t.function : t;
      const name = typeof fn.name === 'string' ? fn.name : '';
      if (!name) continue;
      decls.push({
        name,
        description: typeof fn.description === 'string' ? fn.description : '',
        parameters: isRecord(fn.parameters) ? fn.parameters : (isRecord(fn.input_schema) ? fn.input_schema : { type: 'object', properties: {} }),
      });
    }
    if (decls.length > 0) {
      body.tools = [{ functionDeclarations: decls }];
    }
  }

  const genConfig: Json = {};
  const maxTokens = clientBody.max_tokens;
  if (typeof maxTokens === 'number' && Number.isFinite(maxTokens)) {
    genConfig.maxOutputTokens = maxTokens;
  }
  const stops = clientApi === 'openai' ? (Array.isArray(clientBody.stop) ? clientBody.stop : (typeof clientBody.stop === 'string' ? [clientBody.stop] : [])) : (Array.isArray(clientBody.stop_sequences) ? clientBody.stop_sequences : []);
  if (stops.length > 0) {
    genConfig.stopSequences = stops;
  }
  const format = isRecord(clientBody.response_format) ? clientBody.response_format.type : undefined;
  if (format === 'json_object' || format === 'json_schema') {
    genConfig.responseMimeType = 'application/json';
  }
  if (Object.keys(genConfig).length > 0) {
    body.generationConfig = genConfig;
  }

  const isStream = clientBody.stream === true;
  const encodedModel = encodeURIComponent(model);
  const action = isStream ? `streamGenerateContent?alt=sse` : `generateContent`;
  const url = `/v1beta/models/${encodedModel}:${action}`;

  return { url, body };
}

/** Parse OpenAI upstream answer. */
export function parseOpenAIAnswer(rawText: string, stream: boolean): { text: string; stop: string } | null {
  if (!stream) {
    try {
      const obj = JSON.parse(rawText);
      if (!isRecord(obj)) return null;
      const choices = Array.isArray(obj.choices) ? obj.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (!choice) return null;
      const message = isRecord(choice.message) ? choice.message : {};
      const content = message.content;
      const text = typeof content === 'string' ? content : blocksText(content);
      const stop = typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop';
      return { text, stop };
    } catch {
      return null;
    }
  } else {
    try {
      const frames = parseSseFrames(rawText);
      let text = '';
      let stop = 'stop';
      for (const frame of frames) {
        if (frame.data === '[DONE]') continue;
        const chunk = JSON.parse(frame.data);
        if (!isRecord(chunk)) continue;
        const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
        const choice = isRecord(choices[0]) ? choices[0] : null;
        if (!choice) continue;
        const delta = isRecord(choice.delta) ? choice.delta : {};
        if (typeof delta.content === 'string') {
          text += delta.content;
        }
        if (typeof choice.finish_reason === 'string' && choice.finish_reason) {
          stop = choice.finish_reason;
        }
      }
      return { text, stop };
    } catch {
      return null;
    }
  }
}

/** Parse Anthropic upstream answer. */
export function parseAnthropicAnswer(rawText: string, stream: boolean): { text: string; stop: string } | null {
  if (!stream) {
    try {
      const obj = JSON.parse(rawText);
      if (!isRecord(obj)) return null;
      const content = obj.content;
      const text = blocksText(content);
      const stop = typeof obj.stop_reason === 'string' ? obj.stop_reason : 'end_turn';
      return { text, stop };
    } catch {
      return null;
    }
  } else {
    try {
      const frames = parseSseFrames(rawText);
      let text = '';
      let stop = 'end_turn';
      for (const frame of frames) {
        if (frame.event === 'ping') continue;
        if (!frame.data) continue;
        let data: unknown;
        try {
          data = JSON.parse(frame.data);
        } catch {
          continue;
        }
        if (!isRecord(data)) continue;
        if (frame.event === 'content_block_delta') {
          const delta = isRecord(data.delta) ? data.delta : {};
          if (delta.type === 'text_delta' && typeof delta.text === 'string') {
            text += delta.text;
          }
        } else if (frame.event === 'message_delta') {
          const delta = isRecord(data.delta) ? data.delta : {};
          if (typeof delta.stop_reason === 'string' && delta.stop_reason) {
            stop = delta.stop_reason;
          }
        }
      }
      return { text, stop };
    } catch {
      return null;
    }
  }
}

/** Parse Gemini upstream answer. */
export function parseGeminiAnswer(rawText: string, stream: boolean): { text: string; stop: string } | null {
  if (!stream) {
    try {
      const obj = JSON.parse(rawText);
      if (!isRecord(obj)) return null;
      const candidates = Array.isArray(obj.candidates) ? obj.candidates : [];
      const cand = isRecord(candidates[0]) ? candidates[0] : null;
      if (!cand) return null;
      const content = isRecord(cand.content) ? cand.content : {};
      const parts = Array.isArray(content.parts) ? content.parts : [];
      const text = geminiPartsText(parts);
      const finishReason = typeof cand.finishReason === 'string' ? cand.finishReason.toLowerCase() : 'stop';
      const stop = finishReason === 'stop' ? 'stop' : finishReason;
      return { text, stop };
    } catch {
      return null;
    }
  } else {
    try {
      const frames = parseSseFrames(rawText);
      let text = '';
      let stop = 'stop';
      for (const frame of frames) {
        if (!frame.data) continue;
        let chunk: unknown;
        try {
          chunk = JSON.parse(frame.data);
        } catch {
          continue;
        }
        if (!isRecord(chunk)) continue;
        const candidates = Array.isArray(chunk.candidates) ? chunk.candidates : [];
        const cand = isRecord(candidates[0]) ? candidates[0] : null;
        if (!cand) continue;
        const content = isRecord(cand.content) ? cand.content : {};
        const parts = Array.isArray(content.parts) ? content.parts : [];
        text += geminiPartsText(parts);
        if (typeof cand.finishReason === 'string' && cand.finishReason) {
          stop = cand.finishReason.toLowerCase();
        }
      }
      return { text, stop };
    } catch {
      return null;
    }
  }
}

/** Render OpenAI JSON response. */
export function openAIJson(text: string, usage: { input: number; output: number }, model: string): Json {
  return {
    id: 'chatcmpl-agg',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: text },
        finish_reason: 'stop',
        logprobs: null,
      },
    ],
    usage: {
      prompt_tokens: usage.input,
      completion_tokens: usage.output,
      total_tokens: usage.input + usage.output,
    },
  };
}

/** Render OpenAI SSE stream response. */
export function openAIStream(text: string, usage: { input: number; output: number }, model: string): string {
  const chunks: string[] = [];
  chunks.push(sseEvent(null, {
    id: 'chatcmpl-agg',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }],
  }));
  for (const piece of splitText(text)) {
    chunks.push(sseEvent(null, {
      id: 'chatcmpl-agg',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [{ index: 0, delta: { content: piece }, finish_reason: null }],
    }));
  }
  chunks.push(sseEvent(null, {
    id: 'chatcmpl-agg',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
  }));
  chunks.push(sseEvent(null, {
    id: 'chatcmpl-agg',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [],
    usage: {
      prompt_tokens: usage.input,
      completion_tokens: usage.output,
      total_tokens: usage.input + usage.output,
    },
  }));
  chunks.push('data: [DONE]\n\n');
  return chunks.join('');
}

/** Render Anthropic JSON response. */
export function anthropicJson(text: string, usage: { input: number; output: number }, model: string): Json {
  return {
    id: 'msg_agg',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text }],
    stop_reason: 'end_turn',
    stop_sequence: null,
    usage: {
      input_tokens: usage.input,
      output_tokens: usage.output,
    },
  };
}

/** Render Anthropic SSE stream response. */
export function anthropicStream(text: string, usage: { input: number; output: number }, model: string): string {
  const chunks: string[] = [];
  chunks.push(sseEvent('message_start', {
    type: 'message_start',
    message: {
      id: 'msg_agg',
      type: 'message',
      role: 'assistant',
      model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: usage.input, output_tokens: 0 },
    },
  }));
  chunks.push(sseEvent('content_block_start', {
    type: 'content_block_start',
    index: 0,
    content_block: { type: 'text', text: '' },
  }));
  for (const piece of splitText(text)) {
    chunks.push(sseEvent('content_block_delta', {
      type: 'content_block_delta',
      index: 0,
      delta: { type: 'text_delta', text: piece },
    }));
  }
  chunks.push(sseEvent('content_block_stop', {
    type: 'content_block_stop',
    index: 0,
  }));
  chunks.push(sseEvent('message_delta', {
    type: 'message_delta',
    delta: { stop_reason: 'end_turn', stop_sequence: null },
    usage: { output_tokens: usage.output },
  }));
  chunks.push(sseEvent('message_stop', {
    type: 'message_stop',
  }));
  return chunks.join('');
}
