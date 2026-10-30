// Browser/Node ancestry classifier for the Step 6/7 browser release bundle.
//
// Artifact contract (cloud Step 6 browser export, run my23-20261028-154934):
//   model/model.onnx          input  genotypes[N, feature_count] float32 (REF-allele count 0/1/2, NaN = missing)
//                             outputs probabilities[N, C], class_index[N]
//   model/markers.json        { genome_build: "37", sites: [[chrom, pos, ref, alt], ...] } in model column order
//   model/metadata.json       { classes, feature_count, input_name, probabilities_output, genome_build,
//                               minimum_called_fraction, maximum_incompatible_fraction, model_sha256, markers_sha256, run_id }
//   model/pca_reference.json  (optional) PC scores of reference people for the PCA plot
//
// The older scripts/export_onnx.py layout ({ n_markers, markers: ["chr:pos:ref:alt"] }, nested
// coverage_rules, pc_scores output) is still accepted; normalizeMeta() maps both onto one shape.
//
// Missing-value imputation happens inside the ONNX graph (fitted training means), so the
// encoder only has to produce 0/1/2 or NaN per marker in the saved marker order.

import { parse23andMe, genotypeClass } from "./parse23andme.js";

const ORT_VERSION = "1.22.0";
const ORT_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const DEFAULT_RULES = { min_called_fraction: 0.9, max_incompatible_fraction: 0.01 };

const state = {
  modelBase: "model/",
  ortBase: null,
  status: "idle", // idle | loading | ready | missing | error
  error: null,
  meta: null,
  markers: null,   // { chrom: string[], pos: Int32Array, ref: string[], alt: string[] }
  pcaRef: undefined, // undefined = not fetched yet, null = absent
  session: null,
  ort: null,
  loading: null,
};

/**
 * modelBase: directory holding model.onnx / markers.json / metadata.json (default "model/").
 * ortBase:   directory holding ort.wasm.min.mjs + ort-wasm-simd-threaded.{mjs,wasm}
 *            (default: jsDelivr CDN pinned to ORT_VERSION). `ortCdn` is accepted as an alias.
 */
export function configure({ modelBase, ortBase, ortCdn } = {}) {
  if (modelBase) state.modelBase = modelBase.endsWith("/") ? modelBase : `${modelBase}/`;
  const ob = ortBase || ortCdn;
  if (ob) state.ortBase = ob.endsWith("/") ? ob : `${ob}/`;
}

export function modelStatus() {
  return { status: state.status, error: state.error, meta: state.meta };
}

function docUrl(path) {
  return new URL(path, globalThis.location?.href ?? import.meta.url).href;
}

const hex = (buf) => [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");

async function fetchBytes(name, { optional = false, sha256 = null } = {}) {
  const res = await fetch(`${state.modelBase}${name}`, { cache: "no-cache" });
  if (res.status === 404 && optional) return null;
  if (!res.ok) throw new Error(`${name}: HTTP ${res.status}`);
  const bytes = await res.arrayBuffer();
  if (sha256 && globalThis.crypto?.subtle) {
    const actual = hex(await crypto.subtle.digest("SHA-256", bytes));
    if (actual !== sha256.toLowerCase()) throw new Error(`${name}: SHA-256 mismatch (expected ${sha256.slice(0, 12)}…, got ${actual.slice(0, 12)}…)`);
  }
  return bytes;
}

async function fetchJson(name, opts) {
  const bytes = await fetchBytes(name, opts);
  return bytes === null ? null : JSON.parse(new TextDecoder().decode(bytes));
}

/** Map the browser-export metadata and the export_onnx.py metadata onto one shape. */
function normalizeMeta(raw) {
  const sha = (k) => raw.artifacts?.[k]?.sha256 || null;
  const build = raw.required_build || (raw.genome_build ? `GRCh${String(raw.genome_build).replace(/^GRCh/i, "")}` : "GRCh37");
  return {
    ...raw,
    classes: raw.classes,
    n_markers: raw.n_markers ?? raw.feature_count ?? null,
    n_components: raw.n_components ?? null,
    required_build: build,
    coverage_rules: {
      min_called_fraction: raw.coverage_rules?.min_called_fraction ?? raw.minimum_called_fraction ?? DEFAULT_RULES.min_called_fraction,
      max_incompatible_fraction: raw.coverage_rules?.max_incompatible_fraction ?? raw.maximum_incompatible_fraction ?? DEFAULT_RULES.max_incompatible_fraction,
    },
    input: { name: raw.input?.name ?? raw.input_name ?? null },
    outputs: {
      probabilities: raw.outputs?.probabilities ?? raw.probabilities_output ?? null,
      label: raw.outputs?.label ?? raw.label_output ?? null,
      pcScores: raw.outputs?.pc_scores ?? raw.pc_scores_output ?? null,
    },
    model_sha256: raw.model_sha256 ?? sha("model.onnx"),
    markers_sha256: raw.markers_sha256 ?? sha("markers.json"),
    run_id: raw.run_id ?? null,
    model_version: raw.model_version ?? null,
    model_name: raw.model_name ?? null,
  };
}

/** Resolve to true when the model bundle is present and loaded, false when absent. */
export async function loadModel() {
  if (state.status === "ready") return true;
  if (state.status === "missing") return false;
  if (state.loading) return state.loading;
  state.status = "loading";
  state.loading = (async () => {
    try {
      const metaRes = await fetch(`${state.modelBase}metadata.json`, { cache: "no-cache" });
      if (metaRes.status === 404) { state.status = "missing"; return false; }
      if (!metaRes.ok) throw new Error(`metadata.json: HTTP ${metaRes.status}`);
      state.meta = normalizeMeta(await metaRes.json());

      const [markersJson, modelBytes, ort] = await Promise.all([
        fetchJson("markers.json", { sha256: state.meta.markers_sha256 }),
        fetchBytes("model.onnx", { sha256: state.meta.model_sha256 }),
        loadOrt(),
      ]);
      state.markers = parseMarkers(markersJson);
      const n = state.meta.n_markers;
      if (n != null && state.markers.pos.length !== n) {
        throw new Error(`markers.json has ${state.markers.pos.length} markers, metadata says ${n}`);
      }
      if (n == null) state.meta.n_markers = state.markers.pos.length;
      const mb = markersJson?.genome_build;
      if (mb != null && `GRCh${String(mb).replace(/^GRCh/i, "")}` !== state.meta.required_build) {
        throw new Error(`markers.json is build ${mb}, metadata requires ${state.meta.required_build}`);
      }
      state.ort = ort;
      state.session = await ort.InferenceSession.create(new Uint8Array(modelBytes), { executionProviders: ["wasm"] });
      state.status = "ready";
      return true;
    } catch (e) {
      state.status = "error";
      state.error = e.message || String(e);
      throw e;
    } finally {
      state.loading = null;
    }
  })();
  return state.loading;
}

/** PC coordinates of the reference people (train / test / v5 files), or null if not published. */
export async function loadPcaReference() {
  if (state.pcaRef !== undefined) return state.pcaRef;
  try {
    state.pcaRef = await fetchJson("pca_reference.json", { optional: true });
  } catch (e) {
    console.warn("pca_reference.json:", e.message);
    state.pcaRef = null;
  }
  return state.pcaRef;
}

// Uses the WASM-only ORT entry (ort.wasm.min.mjs), which pairs with
// ort-wasm-simd-threaded.{mjs,wasm}; the full ort.min.mjs build would request the
// *.jsep.* WebGPU binaries instead. Single-threaded to match the verified cloud export.
async function loadOrt() {
  if (globalThis.ort) return globalThis.ort;
  const base = state.ortBase ? docUrl(state.ortBase) : ORT_CDN;
  const mod = await import(/* @vite-ignore */ `${base}ort.wasm.min.mjs`);
  const ort = mod.default || mod;
  if (ort.env?.wasm) {
    ort.env.wasm.wasmPaths = base;
    ort.env.wasm.numThreads = 1;
  }
  return ort;
}

/**
 * Accepts the browser export {genome_build, sites:[[chrom,pos,ref,alt],…]}, export_onnx.py's
 * {markers:[...]} / a bare array of "chr:pos:ref:alt" strings, or columnar {chrom,pos,ref,alt}.
 */
function parseMarkers(json) {
  if (json && Array.isArray(json.chrom) && Array.isArray(json.pos)) {
    return { chrom: json.chrom.map(String), pos: Int32Array.from(json.pos), ref: json.ref, alt: json.alt };
  }
  const list = Array.isArray(json) ? json : json?.sites ?? json?.markers;
  if (!Array.isArray(list)) throw new Error("markers.json: unrecognised format");
  const n = list.length;
  const chrom = new Array(n), ref = new Array(n), alt = new Array(n);
  const pos = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const m = list[i];
    let c, p, r, a;
    if (typeof m === "string") [c, p, r, a] = m.split(":");
    else if (Array.isArray(m)) [c, p, r, a] = m;
    else { c = m.chrom ?? m.chromosome; p = m.pos ?? m.position; r = m.ref ?? m.counted_allele; a = m.alt; }
    if (m?.counted_allele && m.counted_allele !== r) throw new Error(`Marker ${c}:${p}: counted allele is not REF`);
    chrom[i] = String(c); pos[i] = Number(p); ref[i] = r; alt[i] = a;
  }
  return { chrom, pos, ref, alt };
}

/**
 * Build the model input for a parsed 23andMe file: REF-allele count (0/1/2) per marker,
 * NaN where the marker is absent, no-call, or carries alleles outside {REF, ALT}.
 * Returns { x, called, missing, incompatible, matched, nFeatures }.
 */
export function encodeDosages(parsed, markers = state.markers) {
  if (!markers) throw new Error("Model artifacts not loaded");
  const n = markers.pos.length;

  const lookup = new Map();
  for (let i = 0; i < parsed.nSnps; i++) {
    const c = parsed.chrom[i];
    if (c === "X" || c === "Y" || c === "MT") continue;
    const key = `${c}:${parsed.pos[i]}`;
    // Conflicting duplicate coordinates are treated as missing.
    if (lookup.has(key) && lookup.get(key) !== parsed.genotype[i]) lookup.set(key, "--");
    else lookup.set(key, parsed.genotype[i]);
  }

  const x = new Float32Array(n).fill(NaN);
  let called = 0, missing = 0, incompatible = 0;
  for (let j = 0; j < n; j++) {
    const g = lookup.get(`${markers.chrom[j]}:${markers.pos[j]}`);
    if (g === undefined || genotypeClass(g) === "nocall") { missing++; continue; }
    if (g.length !== 2) { incompatible++; continue; }
    const r = markers.ref[j], a = markers.alt[j];
    const g0 = g[0], g1 = g[1];
    if ((g0 !== r && g0 !== a) || (g1 !== r && g1 !== a)) { incompatible++; continue; }
    x[j] = (g0 === r ? 1 : 0) + (g1 === r ? 1 : 0);
    called++;
  }
  return { x, called, missing, incompatible, matched: called + incompatible, nFeatures: n };
}

/** Apply metadata.coverage_rules; returns null when the input is acceptable, else a reason string. */
export function coverageProblem(enc, rules = state.meta?.coverage_rules || DEFAULT_RULES) {
  const calledFraction = enc.called / enc.nFeatures;
  const incompatibleFraction = enc.matched ? enc.incompatible / enc.matched : 0;
  const minCalled = rules.min_called_fraction ?? DEFAULT_RULES.min_called_fraction;
  const maxIncompat = rules.max_incompatible_fraction ?? DEFAULT_RULES.max_incompatible_fraction;
  if (calledFraction < minCalled) return `only ${(calledFraction * 100).toFixed(1)}% of model markers called (minimum ${(minCalled * 100).toFixed(0)}%)`;
  if (incompatibleFraction > maxIncompat) return `${(incompatibleFraction * 100).toFixed(2)}% of matched markers have incompatible alleles (maximum ${(maxIncompat * 100).toFixed(1)}%)`;
  return null;
}

/**
 * Classify a parsed 23andMe file. Resolves to one of
 *   { available: false, status }                                 model bundle not published
 *   { available: true, status: "skipped", reason, coverage… }    failed the coverage / build rules
 *   { available: true, status: "classified", group, scores, pcScores, coverage… }
 */
export async function classifyParsed(parsed) {
  const ok = await loadModel();
  if (!ok) return { available: false, status: state.status };
  const { ort, session, meta } = state;

  const enc = encodeDosages(parsed);
  const coverage = {
    coverage: enc.called / enc.nFeatures,
    calledMarkers: enc.called,
    missingMarkers: enc.missing,
    incompatibleMarkers: enc.incompatible,
    matchedMarkers: enc.matched,
    nMarkers: enc.nFeatures,
  };
  const modelInfo = {
    version: meta.model_version,
    runId: meta.run_id,
    name: meta.model_name,
    classes: meta.classes,
    nMarkers: meta.n_markers,
    nComponents: meta.n_components,
    sha256: meta.model_sha256,
  };

  const requiredBuild = meta.required_build;
  if (parsed.build !== "Unknown" && parsed.build !== requiredBuild) {
    return { available: true, status: "skipped", reason: `genome build ${parsed.build}; model requires ${requiredBuild}`, ...coverage, model: modelInfo };
  }
  const problem = coverageProblem(enc);
  if (problem) return { available: true, status: "skipped", reason: problem, ...coverage, model: modelInfo };

  const inputName = meta.input.name || session.inputNames[0];
  const out = await session.run({ [inputName]: new ort.Tensor("float32", enc.x, [1, enc.nFeatures]) });
  const names = session.outputNames;
  const byName = (explicit, re, fallbackIdx) => out[explicit] || out[names.find((o) => re.test(o)) || names[fallbackIdx]];
  const probTensor = byName(meta.outputs.probabilities, /prob/i, 0);
  const probs = Array.from(probTensor.data);
  const classes = meta.classes;
  if (probs.length !== classes.length || !probs.every(Number.isFinite)) {
    throw new Error(`model returned ${probs.length} probabilities for ${classes.length} classes`);
  }
  const labelTensor = out[meta.outputs.label] || out.class_index || out.label_index;
  const pcTensor = out[meta.outputs.pcScores] || out.pc_scores
    || out[names.find((o) => /pc|pca|score/i.test(o) && !/prob|class|label/i.test(o))];
  const scores = Object.fromEntries(classes.map((c, i) => [c, probs[i]]));
  const idx = labelTensor ? Number(labelTensor.data[0]) : probs.indexOf(Math.max(...probs));

  return {
    available: true,
    status: "classified",
    group: classes[idx],
    scores,
    pcScores: pcTensor ? Array.from(pcTensor.data) : null,
    ...coverage,
    model: modelInfo,
  };
}

/** Convenience: classify from raw 23andMe text. */
export async function classifyAncestry(text, fileName = "") {
  return classifyParsed(parse23andMe(text, fileName));
}

export { parse23andMe };
