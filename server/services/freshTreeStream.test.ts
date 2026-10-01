import { test } from "node:test";
import assert from "node:assert/strict";
import { readTextEventStream } from "./freshTreeStream";

const signal = () => new AbortController().signal;
const response = (frames: string) => new Response(frames, { headers: { "Content-Type": "text/event-stream" } });
const frame = (value: unknown) => `data: ${JSON.stringify(value)}\n\n`;

test("text is delivered before the model completes, including split UTF-8 and CRLF frames", async () => {
  let controller!: ReadableStreamDefaultController<Uint8Array>;
  const body = new ReadableStream<Uint8Array>({ start(value) { controller = value; } });
  let notify!: () => void;
  const firstChunk = new Promise<void>((resolve) => { notify = resolve; });
  const chunks: string[] = [];
  let done = false;
  const output = readTextEventStream(new Response(body, { headers: { "Content-Type": "text/event-stream" } }), "chat",
    (text) => { chunks.push(text); notify(); }, signal()).then((text) => { done = true; return text; });
  const initial = new TextEncoder().encode(frame({ choices: [{ delta: { content: "1.0 Café" } }] }).replace(/\n/g, "\r\n"));
  for (const byte of initial) controller.enqueue(Uint8Array.of(byte));
  await firstChunk;
  assert.equal(done, false, "preview must not wait for the final response");
  assert.deepEqual(chunks, ["1.0 Café"]);
  controller.enqueue(new TextEncoder().encode(frame({ choices: [{ delta: { content: "\n1.1 Support" } }] }) + "data: [DONE]\n\n"));
  controller.close();
  assert.equal(await output, "1.0 Café\n1.1 Support");
});

test("Anthropic text deltas remain separate from events and finish at message_stop", async () => {
  const text = await readTextEventStream(response(
    frame({ type: "message_start", message: {} }) +
    frame({ type: "content_block_delta", delta: { type: "text_delta", text: "Thesis" } }) +
    frame({ type: "message_stop" }),
  ), "anthropic", () => {}, signal());
  assert.equal(text, "Thesis");
});

test("Gemini text is accumulated without exposing thought parts", async () => {
  const chunks: string[] = [];
  const text = await readTextEventStream(response(
    frame({ candidates: [{ content: { parts: [{ text: "private reasoning", thought: true }, { text: "First " }] } }] }) +
    frame({ candidates: [{ content: { parts: [{ text: "thesis" }] }, finishReason: "STOP" }] }),
  ), "gemini", (chunk) => chunks.push(chunk), signal());
  assert.equal(text, "First thesis");
  assert.deepEqual(chunks, ["First ", "thesis"]);
});

test("web research streams output text without changing its contents", async () => {
  const text = await readTextEventStream(response(
    frame({ type: "response.web_search_call.searching" }) +
    frame({ type: "response.output_text.delta", delta: '{"1.1":' }) +
    frame({ type: "response.output_text.delta", delta: '[{"text":"Fact","url":"https://example.org"}]}' }) +
    frame({ type: "response.completed" }),
  ), "responses", () => {}, signal());
  assert.deepEqual(JSON.parse(text), { "1.1": [{ text: "Fact", url: "https://example.org" }] });
});

test("interrupted streams expose drafts but never accept them as completed output", async () => {
  const chunks: string[] = [];
  await assert.rejects(readTextEventStream(response(frame({ choices: [{ delta: { content: "partial" } }] })),
    "chat", (chunk) => chunks.push(chunk), signal()), /ended before completion/);
  assert.deepEqual(chunks, ["partial"]);
});

test("provider error events reject text already streamed", async () => {
  await assert.rejects(readTextEventStream(response(
    frame({ type: "response.output_text.delta", delta: "draft" }) +
    frame({ type: "response.failed", response: { error: { message: "controlled provider failure" } } }),
  ), "responses", () => {}, signal()), /controlled provider failure/);
});

test("a JSON response is not misreported as a live stream", async () => {
  await assert.rejects(readTextEventStream(new Response('{"ok":true}', { headers: { "Content-Type": "application/json" } }),
    "chat", () => {}, signal()), /did not return a text event stream/);
});

test("an aborted call cannot accept completed text", async () => {
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(readTextEventStream(response(
    frame({ choices: [{ delta: { content: "text" } }] }) + "data: [DONE]\n\n",
  ), "chat", () => {}, controller.signal));
});