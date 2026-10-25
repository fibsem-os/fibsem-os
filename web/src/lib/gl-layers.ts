// Drawing layers on the GPU, as fibsem.fm.composite.composite_fm_layers draws them:
// each visible layer's (value - lo) / (hi - lo), clamped, raised to its gamma, times
// its colour and opacity, added together and clamped.

import type { Values } from "./planes";

export const MAX_LAYERS = 8;
// Textures kept on the GPU: a stack's channels at two sizes, with room to step back.
const KEPT = 32;

export interface DrawLayer {
  values: Values;
  width: number;
  height: number;
  clim: [number, number];
  gamma: number;
  tint: [number, number, number];
  opacity: number;
}

/** Where the image sits on the canvas: its top left and its scale, canvas pixels per image pixel. */
export interface Placement {
  x: number;
  y: number;
  scale: number;
}

const VERTEX = `#version 300 es
in vec2 a_corner;
uniform vec2 u_canvas;
uniform vec2 u_origin;
uniform vec2 u_size;
out vec2 v_uv;
void main() {
  v_uv = a_corner;
  vec2 at = u_origin + a_corner * u_size;
  gl_Position = vec4(at / u_canvas * 2.0 - 1.0, 0.0, 1.0);
  gl_Position.y = -gl_Position.y;
}`;

// One block per layer: GLSL ES 3.00 indexes an array of samplers only by a constant.
const FRAGMENT = `#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_values[${MAX_LAYERS}];
uniform int u_count;
uniform vec2 u_clim[${MAX_LAYERS}];
uniform float u_gamma[${MAX_LAYERS}];
uniform vec3 u_tint[${MAX_LAYERS}];
out vec4 colour;
float level(float v, vec2 clim, float gamma) {
  float n = clim.y > clim.x ? clamp((v - clim.x) / (clim.y - clim.x), 0.0, 1.0) : 0.0;
  return gamma != 1.0 ? pow(n, gamma) : n;
}
void main() {
  vec3 sum = vec3(0.0);
${Array.from(
  { length: MAX_LAYERS },
  (_, i) =>
    `  if (u_count > ${i}) sum += level(texture(u_values[${i}], v_uv).r, u_clim[${i}], u_gamma[${i}]) * u_tint[${i}];`,
).join("\n")}
  colour = vec4(clamp(sum, 0.0, 1.0), 1.0);
}`;

/** Halves a (h, w) plane by the mean of each 2 x 2 block, to the size WebGL asks of
 * the next mipmap level (half, rounded down); an odd last row or column is left out. */
function halve(values: Float32Array, w: number, h: number): [Float32Array, number, number] {
  const hw = Math.max(1, Math.floor(w / 2));
  const hh = Math.max(1, Math.floor(h / 2));
  const out = new Float32Array(hw * hh);
  for (let y = 0; y < hh; y++) {
    const y0 = Math.min(h - 1, 2 * y) * w;
    const y1 = Math.min(h - 1, 2 * y + 1) * w;
    for (let x = 0; x < hw; x++) {
      const x0 = Math.min(w - 1, 2 * x);
      const x1 = Math.min(w - 1, 2 * x + 1);
      out[y * hw + x] = (values[y0 + x0] + values[y0 + x1] + values[y1 + x0] + values[y1 + x1]) / 4;
    }
  }
  return [out, hw, hh];
}

interface Uploaded {
  texture: WebGLTexture;
  width: number;
  height: number;
}

/** A WebGL2 canvas that draws up to MAX_LAYERS layers; null where WebGL2 isn't there. */
export class LayerRenderer {
  private gl: WebGL2RenderingContext;
  private program: WebGLProgram;
  // the planes drawn lately, oldest first
  private textures = new Map<Values, Uploaded>();
  private smooth: boolean;

  static create(canvas: HTMLCanvasElement): LayerRenderer | null {
    const gl = canvas.getContext("webgl2", { preserveDrawingBuffer: true, antialias: false });
    if (!gl) return null;
    try {
      return new LayerRenderer(gl);
    } catch {
      return null;
    }
  }

  private constructor(gl: WebGL2RenderingContext) {
    this.gl = gl;
    // Without it a float texture can't be filtered, only sampled at its nearest pixel.
    this.smooth = !!gl.getExtension("OES_texture_float_linear");
    const program = gl.createProgram()!;
    for (const [type, source] of [
      [gl.VERTEX_SHADER, VERTEX],
      [gl.FRAGMENT_SHADER, FRAGMENT],
    ] as const) {
      const shader = gl.createShader(type)!;
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader) ?? "shader");
      gl.attachShader(program, shader);
    }
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program) ?? "link");
    this.program = program;
    gl.useProgram(program);
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), gl.STATIC_DRAW);
    const corner = gl.getAttribLocation(program, "a_corner");
    gl.enableVertexAttribArray(corner);
    gl.vertexAttribPointer(corner, 2, gl.FLOAT, false, 0, 0);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  }

  /** The values as a float texture with its halvings, so a plane shown smaller
   * than it is averages its pixels rather than skipping them. */
  private upload(layer: DrawLayer): Uploaded {
    const known = this.textures.get(layer.values);
    if (known) {
      this.textures.delete(layer.values);
      this.textures.set(layer.values, known);
      return known;
    }
    const gl = this.gl;
    const texture = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, texture);
    let level = 0;
    let [values, w, h] = [Float32Array.from(layer.values), layer.width, layer.height];
    for (;;) {
      gl.texImage2D(gl.TEXTURE_2D, level, gl.R32F, w, h, 0, gl.RED, gl.FLOAT, values);
      if (w === 1 && h === 1) break;
      [values, w, h] = halve(values, w, h);
      level++;
    }
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAX_LEVEL, level);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    const uploaded = { texture, width: layer.width, height: layer.height };
    this.textures.set(layer.values, uploaded);
    while (this.textures.size > KEPT) this.forget(this.textures.keys().next().value!);
    return uploaded;
  }

  /** Draws *layers* at *at* on a canvas *width* x *height* pixels; *pixelated*
   * shows each image pixel as a square when zoomed in. *clear* false draws over
   * what is there, as a sharper region over the whole. */
  draw(layers: DrawLayer[], at: Placement, width: number, height: number, pixelated: boolean, clear = true) {
    const gl = this.gl;
    gl.viewport(0, 0, width, height);
    if (clear) {
      gl.clearColor(0, 0, 0, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
    }
    const shown = layers.slice(0, MAX_LAYERS);
    if (!shown.length) return;
    const uniform = (name: string) => gl.getUniformLocation(this.program, name);
    const { width: iw, height: ih } = shown[0];
    gl.uniform2f(uniform("u_canvas"), width, height);
    gl.uniform2f(uniform("u_origin"), at.x, at.y);
    gl.uniform2f(uniform("u_size"), iw * at.scale, ih * at.scale);
    gl.uniform1i(uniform("u_count"), shown.length);
    const mag = this.smooth && !pixelated ? gl.LINEAR : gl.NEAREST;
    const min = this.smooth ? gl.LINEAR_MIPMAP_LINEAR : gl.NEAREST_MIPMAP_NEAREST;
    const uploaded = shown.map((layer) => this.upload(layer));
    shown.forEach((layer, i) => {
      const { texture } = uploaded[i];
      gl.activeTexture(gl.TEXTURE0 + i);
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, mag);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, min);
      gl.uniform1i(uniform(`u_values[${i}]`), i);
      gl.uniform2f(uniform(`u_clim[${i}]`), layer.clim[0], layer.clim[1]);
      gl.uniform1f(uniform(`u_gamma[${i}]`), layer.gamma);
      const [r, g, b] = layer.tint;
      gl.uniform3f(uniform(`u_tint[${i}]`), r * layer.opacity, g * layer.opacity, b * layer.opacity);
    });
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  /** Lets go of textures no longer drawn. */
  forget(values: Values) {
    const known = this.textures.get(values);
    if (known) this.gl.deleteTexture(known.texture);
    this.textures.delete(values);
  }
}
