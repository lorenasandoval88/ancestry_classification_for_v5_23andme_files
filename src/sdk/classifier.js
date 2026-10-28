// Browser/Node ancestry classifier for the Step 6/7 browser release bundle.
//
// Artifact contract (produced by scripts/export_onnx.py or the cloud Step 6 exporter):
//   model/model.onnx          input  dosages[N, n_markers] float32 (REF-allele count 0/1/2, NaN = missing)
//                             outputs probabilities[N, C], pc_scores[N, K], label_index[N]
//   model/markers.json        { n_markers, counted_allele: "REF", markers: ["chr:pos:ref:alt", ...] }
//   model/metadata.json       { classes, n_markers, n_components, coverage_rules, artifacts, parity, ... }
//   model/pca_reference.json  (optional) PC scores of reference people for the PCA plot
//
// Missing-value imputation happens inside the ONNX graph (fitted training means), so the
// encoder only has to produce 0/1/2 or NaN per marker in the saved marker order.

import { parse23andMe, genotypeClass } from "./parse23andme.js";

const ORT_VERSION = "1.20.1";
const ORT_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const DEFAULT_RULES = { min_called_fraction: 0.9, max_incompatible_fraction: 0.01 };

const state = {
  modelBase: "model/",
  ortCdn: null,
  status: "idle", // idle | loading | ready | missing | error
  error: null,
  meta: null,
  markers: null,   // { chrom: string[], pos: Int32Array, ref: string[], alt: string[] }
  pcaRef: undefined, // undefined = not fetched yet, null = absent
  session: null,
  ort: null,
  loading: null,
};

export function configure({ modelBase, ortCdn } = {}) {
  if (modelBase) state.modelBase = modelBase.endsWith("/") ? modelBase : `${modelBase}/`;
  if (ortCdn) state.ortCdn = ortCdn;
}

export function modelStatus() {
  return { status: state.status, error: state.error, meta: state.meta };
}

async function fetchJson(name, { optional = false } = {}) {
  const res = await fetch(`${state.modelBase}${name}`, { cache: "no-cache" });
  if (res.status === 404 && optional) return null;
  if (!res.ok) throw new Error(`${name}: HTTP ${res.status}`);
  return res.json();
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
      state.meta = await metaRes.json();

      const [markersJson, ort] = await Promise.all([fetchJson("markers.json"), loadOrt()]);
      state.markers = parseMarkers(markersJson);
      const n = state.meta.n_markers;
      if (n != null && state.markers.pos.length !== n) {
        throw new Error(`markers.json has ${state.markers.pos.length} markers, metadata says ${n}`);
      }
      state.ort = ort;
      state.session = await ort.InferenceSession.create(`${state.modelBase}model.onnx`, { executionProviders: ["wasm"] });
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

async function loadOrt() {
  if (globalThis.ort) return globalThis.ort;
  const base = state.ortCdn || ORT_CDN;
  const mod = await import(/* @vite-ignore */ `${base}ort.min.mjs`);
  const ort = mod.default || mod;
  if (ort.env?.wasm) ort.env.wasm.wasmPaths = base;
  return ort;
}

/** Accepts {markers:[...]} / a bare array of "chr:pos:ref:alt" strings / columnar {chrom,pos,ref,alt}. */
function parseMarkers(json) {
  if (json && Array.isArray(json.chrom) && Array.isArray(json.pos)) {
    return { chrom: json.chrom.map(String), pos: Int32Array.from(json.pos), ref: json.ref, alt: json.alt };
  }
  const list = Array.isArray(json) ? json : json?.markers;
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
    version: meta.model_version || null,
    runId: meta.run_id || null,
    classes: meta.classes,
    nMarkers: meta.n_markers,
    nComponents: meta.n_components,
    sha256: meta.artifacts?.["model.onnx"]?.sha256 || null,
  };

  const requiredBuild = meta.required_build || "GRCh37";
  if (parsed.build !== "Unknown" && parsed.build !== requiredBuild) {
    return { available: true, status: "skipped", reason: `genome build ${parsed.build}; model requires ${requiredBuild}`, ...coverage, model: modelInfo };
  }
  const problem = coverageProblem(enc);
  if (problem) return { available: true, status: "skipped", reason: problem, ...coverage, model: modelInfo };

  const inputName = meta.input?.name || session.inputNames[0];
  const out = await session.run({ [inputName]: new ort.Tensor("float32", enc.x, [1, enc.nFeatures]) });
  const pick = (re, i) => out[session.outputNames.find((o) => re.test(o)) || session.outputNames[i]];
  const probs = Array.from(pick(/prob/i, 0).data);
  const pcTensor = out.pc_scores || out[session.outputNames.find((o) => /pc|pca|score/i.test(o) && !/prob/i.test(o))];
  const classes = meta.classes;
  const scores = Object.fromEntries(classes.map((c, i) => [c, probs[i]]));
  const idx = out.label_index ? Number(out.label_index.data[0]) : probs.indexOf(Math.max(...probs));

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
