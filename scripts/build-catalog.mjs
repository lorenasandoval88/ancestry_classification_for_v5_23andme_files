// Build a slim, de-identified participant catalog for the web app from
// data/participantInfo_genotypeFiles_23me.json (one row per 23andMe file).
//
//   node scripts/build-catalog.mjs
//
// Output: data/participants_catalog.json
// Excluded on purpose: real_name, zip, and the storage-host finalUrl.

import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const src = resolve(root, "data/participantInfo_genotypeFiles_23me.json");
const out = resolve(root, "data/participants_catalog.json");

const participants = JSON.parse(readFileSync(src, "utf8"));

const survey = (p, re) => {
  const rows = p.profile?.google_survey_results?.[0];
  if (!Array.isArray(rows)) return null;
  const hit = rows.find(([q]) => re.test(String(q)));
  const v = hit?.[1]?.trim();
  return v && v !== "No response" ? v : null;
};

const rows = [];
for (const p of participants) {
  const self = {
    sex: survey(p, /^(sex\/gender|anatomical sex at birth|what is your gender)/i),
    race: survey(p, /^race\/ethnicity$|^what is your race/i),
    ethnicity: survey(p, /^what is your ethnicity/i),
    birthYear: survey(p, /^year of birth$/i),
    state: p.profile?.state || null,
  };
  for (const f of p.files || []) {
    const gb = f.genomeBuildFiles?.[0];
    const ccr = f.chipClassificationResult;
    const a = f.ancestryClassification;
    rows.push({
      participantId: p.id,
      profileUrl: p.profileUrl,
      fileName: f.filename || f.name || null,
      publishedDate: f.publishedDate || null,
      sizeMB: gb?.sizeMB ?? null,
      downloadUrl: f.downloadUrl || null,
      valid23File: !!f.valid23File,
      genomeBuild: f.genomeBuild ? `GRCh${f.genomeBuild}` : null,
      chip: f.chipClassification || null,
      chipSource: f.chipVersionSource || null,
      chipMatch: f.chipMatchPercentages || null,
      variantRows: ccr?.parsedVariantRows ?? null,
      uniqueCoords: ccr?.totalUniqueCoordinates ?? null,
      processingNote: f.processingNote || null,
      ancestry: a?.status === "classified" ? {
        group: a.predicted_group,
        calledFraction: a.called_fraction,
        calledMarkers: a.called_model_markers,
        modelMarkers: a.model_markers,
        model: a.model,
        runId: a.prediction_run_id,
      } : null,
      self,
    });
  }
}

const summary = {
  generatedAt: new Date().toISOString(),
  participants: participants.length,
  files: rows.length,
  valid23Files: rows.filter((r) => r.valid23File).length,
  classified: rows.filter((r) => r.ancestry).length,
  byGroup: rows.reduce((m, r) => { if (r.ancestry) m[r.ancestry.group] = (m[r.ancestry.group] || 0) + 1; return m; }, {}),
};

writeFileSync(out, JSON.stringify({ summary, rows }));
console.log(`Wrote ${out}`);
console.log(summary);
