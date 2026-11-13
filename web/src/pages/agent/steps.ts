/** A step the harness printed as "Called <tool> <arguments>", in words:
 * "Took an SEM image", "Saved position 'demo-start'". Steps that already read
 * well (Wrote, Ran, Read, Searched) pass through. */

type Args = Record<string, unknown>;

export type Said = { verb: string; words: string; call?: string };

const str = (v: unknown) => (typeof v === "string" || typeof v === "number" ? String(v) : "");
const quoted = (v: unknown) => (str(v) ? `'${str(v)}'` : "");
const beam = (a: Args) => {
  const b = str(a.beam_type ?? a.beam).toUpperCase();
  return b.startsWith("ION") || b === "FIB" ? "FIB" : b.startsWith("ELEC") || b === "SEM" ? "SEM" : "";
};
const join = (...parts: string[]) => parts.filter(Boolean).join(" ");

// The calls the agent makes most (from the skills); any other tool reads as its
// name in words.
const WORDS: Record<string, (a: Args) => [string, string]> = {
  beam_acquire_image: (a) => [
    "Took",
    join(beam(a) === "SEM" ? "an SEM image" : beam(a) === "FIB" ? "a FIB image" : "an image", quoted(a.label)),
  ],
  beam_turn_on: (a) => ["Turned on", join(beam(a) || "the beam")],
  stage_stable_move: () => ["Moved", "the stage"],
  stage_vertical_move: () => ["Moved", "the stage vertically"],
  stage_move_absolute: () => ["Moved", "the stage to a position"],
  stage_move_to_saved_position: (a) => ["Moved", join("to", quoted(a.name) || "a saved position")],
  stage_save_position: (a) => ["Saved", join("position", quoted(a.name))],
  stage_get_position: () => ["Read", "the stage position"],
  milling_run_stage: (a) => ["Milled", quoted(a.name ?? a.stage) || "a stage"],
  milling_preview_patterns: () => ["Previewed", "the milling patterns"],
  milling_stop: () => ["Stopped", "milling"],
  item_measure_thickness: (a) => ["Measured", join("thickness", a.item ? `of ${quoted(a.item)}` : "")],
  experiment_message_operator: () => ["Messaged", "you"],
  experiment_add_note: () => ["Noted", "in the experiment"],
  experiment_read_file: (a) => ["Read", str(a.path)],
  experiment_crop_image: (a) => ["Cropped", str(a.path) || "an image"],
  app_get_status: () => ["Checked", "the microscope"],
  app_get_events: () => ["Read", "the events"],
  app_get_log: () => ["Read", "the log"],
  proposal_list: () => ["Read", "the proposals"],
  proposal_decide: (a) => ["Decided", join("proposal", str(a.proposal_id))],
  workflow_get_summary: () => ["Read", "the workflow summary"],
  grid_load: (a) => ["Loaded", join("grid", quoted(a.name ?? a.grid))],
};

// The argument that names what a tool works on, when nothing else says.
const MAIN = ["name", "path", "item", "task", "grid", "stage", "label"];

export function say(verb: string, text: string): Said {
  if (verb !== "Called") return { verb, words: text };
  const space = text.indexOf(" ");
  const tool = space < 0 ? text : text.slice(0, space);
  let args: Args = {};
  try {
    args = space < 0 ? {} : JSON.parse(text.slice(space + 1));
  } catch {
    // Clipped arguments: the words go without them.
  }
  const known = WORDS[tool];
  if (known) {
    const [v, w] = known(args);
    return { verb: v, words: w, call: text };
  }
  const key = MAIN.find((k) => str(args[k]));
  const words = tool.replace(/_/g, " ");
  return { verb: "Called", words: join(words, key ? `· ${str(args[key])}` : ""), call: text };
}

/** Seconds since *t*, as the running step's clock: "12 s", "3 min 05 s". */
export function since(t: string, now: number): string {
  const s = Math.max(0, Math.floor((now - Date.parse(t)) / 1000));
  if (s < 60) return `${s} s`;
  return `${Math.floor(s / 60)} min ${String(s % 60).padStart(2, "0")} s`;
}
