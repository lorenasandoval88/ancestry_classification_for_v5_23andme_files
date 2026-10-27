// Build data/reference_cohort.json: the 1000 Genomes people used to train/test the
// ancestry classifier, joined with the public population panel.
//
//   node scripts/build-cohort.mjs
//
// Inputs (data/training_model/cloudModel/2003train_501test_input/):
//   train2003.txt, test501.txt                       one sample ID per line
//   integrated_call_samples_v3.20130502.ALL.panel    sample  pop  super_pop  gender
//   (verified against population_panel_sha256 in the training summary)

import { createHash } from "node:crypto";
import { readFileSync, readdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const cloud = resolve(root, "data/training_model/cloudModel");
const inp = resolve(cloud, "2003train_501test_input");
const out = resolve(root, "data/reference_cohort.json");

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

const count = (filter) => rows.filter(filter).reduce((m, r) => { m[r.superPop] = (m[r.superPop] || 0) + 1; return m; }, {});
const summaryOut = {
  generatedAt: new Date().toISOString(),
  panel: { file: "integrated_call_samples_v3.20130502.ALL.panel", sha256: panelSha, verifiedAgainstTrainingSummary: !!summary.population_panel_sha256 },
  train: { n: train.length, byGroup: count((r) => r.split === "train") },
  test: { n: test.length, byGroup: count((r) => r.split === "test") },
  trainingRun: summary.cohort_run_id ? { cohort: summary.cohort, cohortRunId: summary.cohort_run_id, trainingKeepSha256: summary.training_keep_sha256 } : null,
};

writeFileSync(out, JSON.stringify({ summary: summaryOut, rows }));
console.log(`Wrote ${out}`);
console.log(JSON.stringify(summaryOut, null, 2));
