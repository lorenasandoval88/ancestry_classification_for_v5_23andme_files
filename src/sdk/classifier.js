// Browser/Node ancestry classifier.
//
// Mirrors the training encoding in data/training_model/cloudModel/train_ancestry.py:
//   * features are the 335,632 selected markers, in the saved order (model/markers.tsv)
//   * dosage = PLINK `--export A` count of the REF allele (counted_allele) -> 0, 1 or 2
//   * missing markers -> training mean (SimpleImputer statistics, model/means.f32)
//   * then StandardScaler -> PCA -> LogisticRegression, exported to ONNX by scripts/export_onnx.py
//
// Artifacts are loaded lazily from `modelBase` (default "model/"). If model_meta.json
// is absent the classifier reports { available: false } so the UI can show "model pending".

import { parse23andMe, genotypeClass } from "./parse23andme.js";

const ORT_VERSION = "1.20.1";
const ORT_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

const state = {
  modelBase: "model/",
  status: "idle", // idle | loading | ready | missing | error
  error: null,
  meta: null,
  markers: null, // { chrom: string[], pos: Int32Array, ref: string[], alt: string[] }
  means: null,   // Float32Array
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

/** Resolve to true when the model artifacts are present and loaded, false when absent. */
export async function loadModel() {
  if (state.status === "ready") return true;
  if (state.status === "missing") return false;
  if (state.loading) return state.loading;
  state.status = "loading";
  state.loading = (async () => {
    try {
      const metaRes = await fetch(`${state.modelBase}model_meta.json`, { cache: "no-cache" });
      if (metaRes.status === 404) { state.status = "missing"; return false; }
      if (!metaRes.ok) throw new Error(`model_meta.json: HTTP ${metaRes.status}`);
      state.meta = await metaRes.json();

      const [markersText, meansBuf, ort] = await Promise.all([
        fetch(`${state.modelBase}markers.tsv`).then((r) => { if (!r.ok) throw new Error(`markers.tsv: HTTP ${r.status}`); return r.text(); }),
        fetch(`${state.modelBase}means.f32`).then((r) => { if (!r.ok) throw new Error(`means.f32: HTTP ${r.status}`); return r.arrayBuffer(); }),
        loadOrt(),
      ]);
      state.markers = parseMarkers(markersText);
      state.means = new Float32Array(meansBuf);
      const n = state.meta.n_features;
      if (state.markers.pos.length !== n || state.means.length !== n) {
        throw new Error(`Artifact size mismatch: meta=${n}, markers=${state.markers.pos.length}, means=${state.means.length}`);
      }
      state.ort = ort;
      state.session = await ort.InferenceSession.create(`${state.modelBase}ancestry_pipeline.onnx`, {
        executionProviders: ["wasm"],
      });
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

async function loadOrt() {
  if (globalThis.ort) return globalThis.ort;
  const base = state.ortCdn || ORT_CDN;
  const mod = await import(/* @vite-ignore */ `${base}ort.min.mjs`);
  const ort = mod.default || mod;
  if (ort.env?.wasm) ort.env.wasm.wasmPaths = base;
  return ort;
}

function parseMarkers(text) {
  const lines = text.split(/\r?\n/);
  const n = lines.length - 1 - (lines[lines.length - 1] === "" ? 1 : 0);
  const chrom = new Array(n), ref = new Array(n), alt = new Array(n);
  const pos = new Int32Array(n);
  let k = 0;
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i];
    if (!line) continue;
    const tab = line.indexOf("\t");
    const id = tab === -1 ? line : line.slice(0, tab);
    const counted = tab === -1 ? null : line.slice(tab + 1).trim();
    const [c, p, r, a] = id.split(":");
    if (counted && counted !== r) throw new Error(`Marker ${id}: counted allele ${counted} is not REF`);
    chrom[k] = c; pos[k] = Number(p); ref[k] = r; alt[k] = a; k++;
  }
  return { chrom, pos: pos.subarray(0, k), ref: ref.slice(0, k), alt: alt.slice(0, k) };
}

/**
 * Build the model feature vector for a parsed 23andMe file.
 * Returns { x: Float32Array, called, missing, incompatible }.
 */
export function encodeDosages(parsed, markers = state.markers, means = state.means) {
  if (!markers || !means) throw new Error("Model artifacts not loaded");
  const n = markers.pos.length;

  // chrom:pos -> genotype, autosomes only (the panel has chromosomes 1-22).
  const lookup = new Map();
  for (let i = 0; i < parsed.nSnps; i++) {
    const c = parsed.chrom[i];
    if (c === "X" || c === "Y" || c === "MT") continue;
    lookup.set(`${c}:${parsed.pos[i]}`, parsed.genotype[i]);
  }

  const x = new Float32Array(means); // start from the training means (upstream imputation)
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
  return { x, called, missing, incompatible, nFeatures: n };
}

/**
 * Classify a parsed 23andMe file. Resolves to
 *   { available: false } when the model is not exported yet, otherwise
 *   { available: true, group, scores, coverage, calledMarkers, missingMarkers, incompatibleMarkers, model }.
 */
export async function classifyParsed(parsed) {
  const ok = await loadModel();
  if (!ok) return { available: false, status: state.status };

  const enc = encodeDosages(parsed);
  const { ort, session, meta } = state;
  const input = new ort.Tensor("float32", enc.x, [1, enc.nFeatures]);
  const out = await session.run({ [meta.input_name || session.inputNames[0]]: input });

  const probName = session.outputNames.find((o) => /prob/i.test(o)) || session.outputNames[1];
  const labelName = session.outputNames.find((o) => /label/i.test(o)) || session.outputNames[0];
  const probs = Array.from(out[probName].data);
  const classes = meta.classes;
  const scores = Object.fromEntries(classes.map((c, i) => [c, probs[i]]));
  let group = out[labelName]?.data?.[0];
  if (typeof group !== "string") group = classes[probs.indexOf(Math.max(...probs))];

  return {
    available: true,
    group: String(group),
    scores,
    coverage: enc.called / enc.nFeatures,
    calledMarkers: enc.called,
    missingMarkers: enc.missing,
    incompatibleMarkers: enc.incompatible,
    model: {
      steps: meta.steps,
      classes,
      nFeatures: meta.n_features,
      sha256: meta.artifacts?.["ancestry_pipeline.onnx"]?.sha256 || null,
      trainingCohort: meta.training_cohort || null,
    },
  };
}

/** Convenience: classify from raw 23andMe text. */
export async function classifyAncestry(text, fileName = "") {
  return classifyParsed(parse23andMe(text, fileName));
}

export { parse23andMe };
