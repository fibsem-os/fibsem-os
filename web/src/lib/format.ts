// Numbers as the Qt app shows them.

const trim = (value: number, digits: number) => String(+value.toPrecision(digits));

export function length(metres: number | null | undefined): string {
  if (metres == null) return "–";
  const size = Math.abs(metres);
  if (size >= 1e-3) return `${trim(metres * 1e3, 4)} mm`;
  if (size >= 1e-6 || size === 0) return `${trim(metres * 1e6, 4)} µm`;
  return `${trim(metres * 1e9, 4)} nm`;
}

export function current(amps: number | null | undefined): string {
  if (amps == null) return "–";
  if (amps >= 1e-6) return `${trim(amps * 1e6, 3)} µA`;
  if (amps >= 1e-9) return `${trim(amps * 1e9, 3)} nA`;
  return `${trim(amps * 1e12, 3)} pA`;
}

export function voltage(volts: number | null | undefined): string {
  if (volts == null) return "–";
  return volts >= 1000 ? `${trim(volts / 1000, 3)} kV` : `${trim(volts, 3)} V`;
}

export function angle(radians: number | null | undefined): string {
  if (radians == null) return "–";
  return `${((radians * 180) / Math.PI).toFixed(1)}°`;
}

export function seconds(value: number | null | undefined): string {
  if (value == null) return "–";
  if (value >= 1) return `${trim(value, 3)} s`;
  if (value >= 1e-3) return `${trim(value * 1e3, 3)} ms`;
  return `${trim(value * 1e6, 3)} µs`;
}

export function ago(iso: string | null | undefined, now = Date.now()): string {
  if (!iso) return "";
  const s = Math.max(0, (now - new Date(iso).getTime()) / 1000);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return `${Math.round(s / 86400)} d ago`;
}
