// The App Server, as this browser reaches it. The token is the one /chat pairs,
// under the same storage keys, so pairing either page pairs both.

const OPERATOR_KEY = "fibsem-operator-token";
const PAIRED_KEY = "fibsem-paired-token";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
    readonly errorType?: string,
  ) {
    super(message);
  }
}

let token: string | null = null;
const listeners = new Set<() => void>();

function stored(store: () => Storage, key: string): string | null {
  try {
    return store().getItem(key);
  } catch {
    return null;
  }
}

/** The token from the URL fragment, this tab, or this browser's pairing. */
export function loadToken(): string | null {
  const fragment = new URLSearchParams(location.hash.slice(1)).get("token");
  if (fragment) history.replaceState(null, "", location.pathname);
  token = fragment || stored(() => sessionStorage, OPERATOR_KEY) || stored(() => localStorage, PAIRED_KEY);
  return token;
}

export function getToken() {
  return token;
}

/** A paired token stays in this browser; the operator token, in this tab only. */
export function setToken(value: string, kind: "paired" | "operator") {
  token = value;
  try {
    if (kind === "paired") localStorage.setItem(PAIRED_KEY, value);
    else sessionStorage.setItem(OPERATOR_KEY, value);
  } catch {
    // storage off: for this page load only
  }
  listeners.forEach((listener) => listener());
}

/** Why the token was last forgotten: refused by the server, or unpaired here. */
export let forgotten: "refused" | "unpaired" = "refused";

export function forgetToken(why: typeof forgotten = "refused") {
  forgotten = why;
  token = null;
  for (const store of [() => sessionStorage, () => localStorage]) {
    try {
      store().removeItem(OPERATOR_KEY);
      store().removeItem(PAIRED_KEY);
    } catch {
      // storage off
    }
  }
  listeners.forEach((listener) => listener());
}

export function onTokenChange(listener: () => void) {
  listeners.add(listener);
  return () => void listeners.delete(listener);
}

async function answer<T>(response: Response): Promise<T> {
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = body?.detail;
    const message =
      (typeof detail === "string" && detail) || detail?.message || detail?.error || response.statusText;
    throw new ApiError(response.status, message, detail?.error_type);
  }
  return body as T;
}

/** A call with this browser's token. A 401 forgets the token, so the app asks to pair again. */
export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
  });
  if (response.status === 401) forgetToken();
  return answer<T>(response);
}

/** The pairing routes, which take no token. */
export async function unauthed<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  return answer<T>(response);
}

/** What this browser calls itself in the paired list: "Firefox on Linux". */
export function browserName(): string {
  const ua = navigator.userAgent;
  const browser = /Edg\//.test(ua)
    ? "Edge"
    : /Firefox\//.test(ua)
      ? "Firefox"
      : /Chrome\//.test(ua)
        ? "Chrome"
        : /Safari\//.test(ua)
          ? "Safari"
          : "A browser";
  const os = /Windows/.test(ua)
    ? "Windows"
    : /Mac OS X/.test(ua)
      ? "macOS"
      : /Android/.test(ua)
        ? "Android"
        : /iPhone|iPad/.test(ua)
          ? "iOS"
          : /Linux/.test(ua)
            ? "Linux"
            : "";
  return os ? `${browser} on ${os}` : browser;
}
