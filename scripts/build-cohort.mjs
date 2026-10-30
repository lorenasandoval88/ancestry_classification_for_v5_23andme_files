// Build data/reference_cohort.json: the 1000 Genomes people used to train/test the
// ancestry classifier, joined with the public population panel and, when present, the
// Step 6 test-set evaluation of the published model.
//
//   node scripts/build-cohort.mjs
//
// Inputs (data/training_model/cloudModel/2003train_501test_input/):
//   train2003.txt, test501.txt                       one sample ID per line
//   integrated_call_samples_v3.20130502.ALL.panel    sample  pop  super_pop  gender
//   (verified against population_panel_sha256 in the training summary)
//
// Optional (data/training_model/cloudModel/6_test_model/<run_id>/, the run matching model/metadata.json):
//   predictions.tsv      IID  population  actual_group  predicted_group  correct  prob_AFR … prob_SAS
//   test_summary.json    accuracy, balanced_accuracy, correct, selected_markers, run_id …
//   onnx_parity.json     max_probability_absolute_difference, same_predicted_labels, tolerance …
//   confusion_matrix.tsv actual_group  AFR  AMR  EAS  EUR  SAS

import { createHash } from "node:crypto";
import { existsSync, readFileSync, readdirSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const cloud = resolve(root, "data/training_model/cloudModel");
const inp = resolve(cloud, "2003train_501test_input");
const out = resolve(root, "data/reference_cohort.json");
const metadataPath = resolve(root, "model/metadata.json");

const POP_NAMES = {
  ACB: "African Caribbean in Barbados", ASW: "African Ancestry in Southwest US", BEB: "Bengali in Bangladesh",
  CDX: "Chinese Dai in Xishuangbanna", CEU: "Utah residents with Northern and Western European ancestry",
  CHB: "Han Chinese in Beijing", CHS: "Han Chinese South", CLM: "Colombian in Medellín", ESN: "Esan in Nigeria",
  FIN: "Finnish in Finland", GBR: "British in England and Scotland", GIH: "Gujarati Indians in Houston",
  GWD: "Gambian in Western Division", IBS: "Iberian populations in Spain", ITU: "Indian Telugu in the UK",
  JPT: "Japanese in Tokyo", KHV: "Kinh in Ho Chi Minh City", LWK: "Luhya in Webuye, Kenya", MSL: "Mende in Sierra Leone",
  MXL: "Mexican Ancestry in Los Angeles", PEL: "Peruvian in Lima", PJL: "Punjabi in Lahore", PUR: "Puerto Rican in Puerto Rico",
  STU: "Sri Lankan Tamil in the UK", TSI: "Toscani in Italia", YRI: "Yoruba in Ibadan, Nigeria",
};

const ids = (f) => readFileSync(resolve(inp, f), "utf8").split(/\r?\n/).map((s) => s.trim()).filter(Boolean);
const train = ids("train2003.txt");
const test = ids("test501.txt");
if (new Set(train).size !== train.length || new Set(test).size !== test.length) throw new Error("duplicate IDs in a split");
const overlap = train.filter((s) => test.includes(s));
if (overlap.length) throw new Error(`IDs in both splits: ${overlap.join(", ")}`);

const panelPath = resolve(inp, "integrated_call_samples_v3.20130502.ALL.panel");
const panelBuf = readFileSync(panelPath);
const panelSha = createHash("sha256").update(panelBuf).digest("hex");
const summaryFile = readdirSync(cloud).find((f) => f.endsWith("_training_summary.json"));
const summary = summaryFile ? JSON.parse(readFileSync(resolve(cloud, summaryFile), "utf8")) : {};
if (summary.population_panel_sha256 && summary.population_panel_sha256 !== panelSha) {
  throw new Error(`panel sha256 ${panelSha} != training summary ${summary.population_panel_sha256}`);
}

const panel = new Map();
for (const line of panelBuf.toString("utf8").split(/\r?\n/).slice(1)) {
  const f = line.trim().split(/\s+/);
  if (f.length >= 3) panel.set(f[0], { pop: f[1], superPop: f[2], sex: f[3] || null });
}

const rows = [];
for (const [split, list] of [["train", train], ["test", test]]) {
  for (const id of list) {
    const p = panel.get(id);
    if (!p) throw new Error(`${id} (${split}) not in population panel`);
    rows.push({ sample: id, split, pop: p.pop, popName: POP_NAMES[p.pop] || null, superPop: p.superPop, sex: p.sex });
  }
}

// ---------------------------------------------------------------------------
// Step 6 evaluation of the published model (optional)
// ---------------------------------------------------------------------------

const GROUPS = ["AFR", "AMR", "EAS", "EUR", "SAS"];
const readTsv = (path) => {
  const lines = readFileSync(path, "utf8").split(/\r?\n/).filter((l) => l.trim());
  const header = lines[0].split("\t");
  return lines.slice(1).map((l) => Object.fromEntries(header.map((h, i) => [h, l.split("\t")[i]])));
};

/** Pick the Step 6 results folder: the run named in model/metadata.json, else the only run present. */
function findStep6() {
  const dir = resolve(cloud, "6_test_model");
  if (!existsSync(dir)) return null;
  const runs = readdirSync(dir).filter((d) => statSync(join(dir, d)).isDirectory() && existsSync(join(dir, d, "predictions.tsv")));
  if (!runs.length) return null;
  const meta = existsSync(metadataPath) ? JSON.parse(readFileSync(metadataPath, "utf8")) : null;
  if (meta?.run_id) {
    if (!runs.includes(meta.run_id)) {
      console.warn(`Step 6 results present (${runs.join(", ")}) but none match model/metadata.json run ${meta.run_id}; skipping evaluation`);
      return null;
    }
    return { runId: meta.run_id, dir: join(dir, meta.run_id) };
  }
  if (runs.length > 1) throw new Error(`Several Step 6 runs under ${dir} and no model/metadata.json to pick one`);
  return { runId: runs[0], dir: join(dir, runs[0]) };
}

let evaluation = null;
const step6 = findStep6();
if (step6) {
  const preds = readTsv(join(step6.dir, "predictions.tsv"));
  const byId = new Map(rows.map((r) => [r.sample, r]));
  const testIds = new Set(rows.filter((r) => r.split === "test").map((r) => r.sample));
  if (preds.length !== testIds.size) throw new Error(`predictions.tsv has ${preds.length} rows, cohort has ${testIds.size} test people`);
  for (const p of preds) {
    const r = byId.get(p.IID);
    if (!r || r.split !== "test") throw new Error(`predictions.tsv: ${p.IID} is not a test person`);
    if (p.actual_group !== r.superPop) throw new Error(`predictions.tsv: ${p.IID} actual_group ${p.actual_group} != panel ${r.superPop}`);
    if (!GROUPS.includes(p.predicted_group)) throw new Error(`predictions.tsv: ${p.IID} unknown predicted_group ${p.predicted_group}`);
    r.predicted = p.predicted_group;
    r.probs = Object.fromEntries(GROUPS.map((g) => [g, Math.round(Number(p[`prob_${g}`]) * 1e4) / 1e4]));
    testIds.delete(p.IID);
  }
  if (testIds.size) throw new Error(`predictions.tsv is missing ${testIds.size} test people`);

  const readJson = (f) => existsSync(join(step6.dir, f)) ? JSON.parse(readFileSync(join(step6.dir, f), "utf8")) : null;
  const ts = readJson("test_summary.json"), parity = readJson("onnx_parity.json");
  const testRows = rows.filter((r) => r.split === "test");
  const correct = testRows.filter((r) => r.predicted === r.superPop).length;
  if (ts && ts.correct !== correct) throw new Error(`test_summary.json says ${ts.correct} correct, predictions.tsv gives ${correct}`);
  const confusion = {};
  for (const r of testRows) (confusion[r.superPop] ??= {})[r.predicted] = ((confusion[r.superPop] ??= {})[r.predicted] || 0) + 1;

  evaluation = {
    runId: step6.runId,
    source: `data/training_model/cloudModel/6_test_model/${step6.runId}/`,
    people: testRows.length,
    correct,
    accuracy: ts?.accuracy ?? correct / testRows.length,
    balancedAccuracy: ts?.balanced_accuracy ?? null,
    selectedMarkers: ts?.selected_markers ?? null,
    confusion,
    misclassified: testRows.filter((r) => r.predicted !== r.superPop).map((r) => ({ sample: r.sample, pop: r.pop, known: r.superPop, predicted: r.predicted })),
    parity: parity ? {
      passed: parity.same_predicted_labels === true && parity.max_probability_absolute_difference <= parity.tolerance,
      rows: parity.people_compared,
      max_abs_proba_diff: parity.max_probability_absolute_difference,
      tolerance: parity.tolerance,
      runtime: parity.runtime,
    } : null,
    note: ts?.note ?? null,
  };
}

const count = (filter) => rows.filter(filter).reduce((m, r) => { m[r.superPop] = (m[r.superPop] || 0) + 1; return m; }, {});
const summaryOut = {
  generatedAt: new Date().toISOString(),
  panel: { file: "integrated_call_samples_v3.20130502.ALL.panel", sha256: panelSha, verifiedAgainstTrainingSummary: !!summary.population_panel_sha256 },
  train: { n: train.length, byGroup: count((r) => r.split === "train") },
  test: { n: test.length, byGroup: count((r) => r.split === "test"), evaluation },
  trainingRun: summary.cohort_run_id ? { cohort: summary.cohort, cohortRunId: summary.cohort_run_id, trainingKeepSha256: summary.training_keep_sha256 } : null,
};

writeFileSync(out, JSON.stringify({ summary: summaryOut, rows }));
console.log(`Wrote ${out}`);
console.log(JSON.stringify({ ...summaryOut, test: { ...summaryOut.test, evaluation: evaluation ? { ...evaluation, confusion: "…" } : null } }, null, 2));
if (!evaluation) console.log("No Step 6 predictions found under data/training_model/cloudModel/6_test_model/ — test rows carry known labels only.");
