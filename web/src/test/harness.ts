import { vi } from "vitest";
import type { JsonRecord } from "../lib/api";

/**
 * An explicit HTTP envelope. A bare object is returned as a 200 body, so a real
 * payload that happens to contain a `status` field (like `/api/health`) is never
 * mistaken for an envelope.
 */
export type HttpEnvelope = { __http: true; status: number; body?: unknown };

export function http(status: number, body?: unknown): HttpEnvelope {
  return { __http: true, status, body };
}

function isEnvelope(value: unknown): value is HttpEnvelope {
  return Boolean(value) && typeof value === "object" && (value as HttpEnvelope).__http === true;
}

export type RouteHandler = unknown | ((init: RequestInit | undefined) => unknown);

/**
 * Install a fetch double that dispatches on the request path.
 *
 * `calls` records every request so a test can assert that an interaction issued
 * no network request at all — which is the whole point of the cost slider.
 */
export function mockFetch(routes: Record<string, RouteHandler>) {
  const calls: { url: string; method: string; body?: string }[] = [];

  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input.toString();
    calls.push({
      url,
      method: (init?.method ?? "GET").toUpperCase(),
      body: typeof init?.body === "string" ? init.body : undefined,
    });

    const key = Object.keys(routes).find((candidate) => url.startsWith(candidate));
    if (key === undefined) {
      return new Response(JSON.stringify({ detail: `No route double for ${url}` }), {
        status: 404,
        headers: { "Content-Type": "application/json" },
      });
    }

    const handler = routes[key];
    // Await the handler so a deferred promise genuinely holds the request open.
    const resolved =
      typeof handler === "function"
        ? await (handler as (init: RequestInit | undefined) => unknown)(init)
        : handler;
    const status = isEnvelope(resolved) ? resolved.status : 200;
    const body = isEnvelope(resolved) ? resolved.body : resolved;

    return new Response(JSON.stringify(body ?? {}), {
      status,
      headers: { "Content-Type": "application/json" },
    });
  });

  vi.stubGlobal("fetch", fetchMock);
  return { calls, fetchMock };
}

/** A controllable `EventSource` double for the run stream. */
export class FakeEventSource {
  static instances: FakeEventSource[] = [];
  readonly url: string;
  readyState = 0;
  onerror: ((event: Event) => void) | null = null;
  private listeners = new Map<string, ((event: Event) => void)[]>();

  constructor(url: string) {
    this.url = url;
    FakeEventSource.instances.push(this);
  }

  addEventListener(type: string, listener: (event: Event) => void) {
    const existing = this.listeners.get(type) ?? [];
    existing.push(listener);
    this.listeners.set(type, existing);
  }

  removeEventListener() {}

  close() {
    this.readyState = 2;
  }

  /** Deliver a named SSE event carrying JSON. */
  emit(type: string, data: unknown) {
    this.readyState = 1;
    for (const listener of this.listeners.get(type) ?? []) {
      listener(new MessageEvent(type, { data: JSON.stringify(data) }));
    }
  }

  /** Simulate a hard disconnect: the source closes and cannot self-heal. */
  fail() {
    this.readyState = 2;
    this.onerror?.(new Event("error"));
  }

  static reset() {
    FakeEventSource.instances = [];
  }

  static latest(): FakeEventSource | undefined {
    return FakeEventSource.instances[FakeEventSource.instances.length - 1];
  }
}

export function installEventSource() {
  FakeEventSource.reset();
  vi.stubGlobal("EventSource", FakeEventSource as unknown as typeof EventSource);
  return FakeEventSource;
}

/** A deferred promise, for holding a request open while asserting pending UI. */
export function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}
