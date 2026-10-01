type StreamMode = "chat" | "anthropic" | "gemini" | "responses";
export type FreshTreeStreamObserver = {
  start: (provider: string, stage: string) => void;
  chunk: (text: string) => void;
  complete: () => void;
};

/** Preserve provider text exactly; preview callbacks never turn drafts into saved nodes. */
export async function readTextEventStream(
  response: Response, mode: StreamMode, onText: (text: string) => void, signal: AbortSignal,
): Promise<string> {
  if (!response.ok) throw new Error(`Streaming API Error: ${response.status} - ${await response.text()}`);
  if (!response.headers.get("content-type")?.includes("text/event-stream")) {
    throw new Error("The provider did not return a text event stream.");
  }
  const reader = response.body?.getReader();
  if (!reader) throw new Error("The provider returned no stream body.");
  const decoder = new TextDecoder();
  let buffer = "", text = "", finished = false;
  const consume = (frame: string) => {
    const data = frame.split("\n").filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).replace(/^ /, "")).join("\n");
    if (!data) return;
    if (data.trim() === "[DONE]") { finished = true; return; }
    const event = JSON.parse(data);
    if (event.error || event.type === "error" || event.type === "response.failed") {
      throw new Error(`Provider stream failed: ${event.error?.message || event.response?.error?.message || "request failed"}`);
    }
    let delta = "";
    if (mode === "chat") {
      delta = event.choices?.[0]?.delta?.content || "";
    } else if (mode === "anthropic") {
      if (event.type === "content_block_delta" && event.delta?.type === "text_delta") delta = event.delta.text;
      if (event.type === "message_stop") finished = true;
    } else if (mode === "gemini") {
      delta = (event.candidates?.[0]?.content?.parts || [])
        .filter((part: any) => typeof part.text === "string" && !part.thought)
        .map((part: any) => part.text).join("");
      if (event.candidates?.[0]?.finishReason) finished = true;
    } else {
      if (event.type === "response.output_text.delta") delta = event.delta || "";
      if (event.type === "response.completed") finished = true;
      if (event.type === "response.incomplete") throw new Error("The research response was incomplete.");
    }
    if (typeof delta === "string" && delta) { text += delta; onText(delta); }
  };
  try {
    while (true) {
      signal.throwIfAborted();
      const result = await reader.read();
      buffer += result.done ? decoder.decode() : decoder.decode(result.value, { stream: true });
      // Normalize CRLF after joining chunks, including a CR/LF split across reads.
      buffer = buffer.replace(/\r\n/g, "\n");
      let boundary;
      while ((boundary = buffer.indexOf("\n\n")) >= 0) {
        consume(buffer.slice(0, boundary)); buffer = buffer.slice(boundary + 2);
      }
      if (result.done) break;
    }
    if (buffer.trim()) consume(buffer);
    signal.throwIfAborted();
    if (!finished) throw new Error("The provider stream ended before completion; draft text was not accepted.");
    if (!text.trim()) throw new Error("The provider returned no text.");
    return text;
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

const CHAT_PROVIDERS: Record<string, { url: string; model: string; key: string; maxTokens: number }> = {
  openai: { url: "https://api.openai.com/v1/chat/completions", model: "gpt-4o", key: "OPENAI_API_KEY", maxTokens: 16384 },
  perplexity: { url: "https://api.perplexity.ai/chat/completions", model: "sonar-pro", key: "PERPLEXITY_API_KEY", maxTokens: 16384 },
  grok: { url: "https://api.x.ai/v1/chat/completions", model: "grok-3-latest", key: "GROK_API_KEY", maxTokens: 16384 },
  deepseek: { url: "https://api.deepseek.com/chat/completions", model: "deepseek-chat", key: "DEEPSEEK_API_KEY", maxTokens: 8192 },
};

export async function streamFreshTreeCompletion(
  provider: string, prompt: string, signal: AbortSignal, onText: (text: string) => void,
): Promise<string> {
  signal.throwIfAborted();
  const chat = CHAT_PROVIDERS[provider];
  if (chat) {
    const response = await fetch(chat.url, {
      method: "POST", signal,
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${process.env[chat.key]}` },
      body: JSON.stringify({ model: chat.model, messages: [{ role: "user", content: prompt }], max_tokens: chat.maxTokens, stream: true }),
    });
    return readTextEventStream(response, "chat", onText, signal);
  }
  if (provider === "anthropic") {
    const response = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST", signal,
      headers: { "Content-Type": "application/json", "x-api-key": process.env.ANTHROPIC_API_KEY || "", "anthropic-version": "2023-06-01" },
      body: JSON.stringify({ model: "claude-sonnet-4-5", max_tokens: 16384, messages: [{ role: "user", content: prompt }], stream: true }),
    });
    return readTextEventStream(response, "anthropic", onText, signal);
  }
  if (provider === "gemini") {
    const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:streamGenerateContent?alt=sse&key=${encodeURIComponent(process.env.GEMINI_API_KEY || "")}`, {
      method: "POST", signal, headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ contents: [{ role: "user", parts: [{ text: prompt }] }], generationConfig: { maxOutputTokens: 16384 } }),
    });
    return readTextEventStream(response, "gemini", onText, signal);
  }
  throw new Error(`Unknown Fresh Tree streaming provider: ${provider}`);
}

export async function streamFreshTreeResearch(
  prompt: string, signal: AbortSignal, onText: (text: string) => void,
): Promise<string> {
  const response = await fetch("https://api.openai.com/v1/responses", {
    method: "POST", signal,
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${process.env.OPENAI_API_KEY}` },
    body: JSON.stringify({ model: "gpt-4o", tools: [{ type: "web_search_preview" }], input: prompt, stream: true }),
  });
  return readTextEventStream(response, "responses", onText, signal);
}