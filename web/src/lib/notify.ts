import { useSyncExternalStore } from "react";

/** This browser's choices for being told when the agent needs an answer. */
export interface NoticePrefs {
  /** Toasts in the page; the tab title counts either way. */
  toasts: boolean;
  /** A failed action is a notice too, not only what needs an answer. */
  errors: boolean;
  /** A system notification while the tab is in the background. */
  system: boolean;
  sound: boolean;
}

const KEY = "fibsem-notices";
const DEFAULTS: NoticePrefs = { toasts: true, errors: true, system: false, sound: false };
const listeners = new Set<() => void>();
let cached: NoticePrefs | null = null;

function read(): NoticePrefs {
  if (cached) return cached;
  try {
    cached = { ...DEFAULTS, ...JSON.parse(localStorage.getItem(KEY) ?? "{}") };
  } catch {
    cached = DEFAULTS;
  }
  return cached!;
}

export function setNoticePrefs(change: Partial<NoticePrefs>) {
  cached = { ...read(), ...change };
  try {
    localStorage.setItem(KEY, JSON.stringify(cached));
  } catch {
    // Kept for this tab only.
  }
  for (const listener of listeners) listener();
}

export function useNoticePrefs(): NoticePrefs {
  return useSyncExternalStore((listener) => {
    listeners.add(listener);
    return () => listeners.delete(listener);
  }, read);
}

/** Whether this browser can show system notifications here: only on a secure
 * origin (localhost, or the SSH tunnel's). */
export const systemNotices = () => typeof Notification !== "undefined" && isSecureContext;

let lastChime = 0;

/** Two short notes, made here rather than loaded; once for news that lands together. */
export function chime() {
  if (Date.now() - lastChime < 2000) return;
  lastChime = Date.now();
  try {
    const audio = new AudioContext();
    for (const [i, hz] of [880, 1320].entries()) {
      const tone = audio.createOscillator();
      const gain = audio.createGain();
      const at = audio.currentTime + i * 0.16;
      tone.frequency.value = hz;
      gain.gain.setValueAtTime(0.0001, at);
      gain.gain.exponentialRampToValueAtTime(0.2, at + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.0001, at + 0.14);
      tone.connect(gain).connect(audio.destination);
      tone.start(at);
      tone.stop(at + 0.15);
    }
    setTimeout(() => void audio.close(), 600);
  } catch {
    // No sound in this browser.
  }
}
