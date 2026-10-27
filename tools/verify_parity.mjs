import { readFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import { createClassifier } from '../src/index.js';

export function verifyParity(model, reference, tolerance = 1e-8) {
  if (reference.modelVersion !== model.modelVersion || !reference.cases?.length) {
    throw new Error('Missing reference cases or mismatched model version');
  }
  const classifier = createClassifier(model);
  let maxError = 0;
  for (const [index, sample] of reference.cases.entries()) {
    if (sample.dosages.length !== model.markerCount) throw new Error('Invalid reference marker count');
    const records = model.markers.map((marker, i) => {
      const dosage = sample.dosages[i];
      if (![null, 0, 1, 2].includes(dosage)) throw new Error('Invalid reference dosage');
      return {
        id: marker.id, chromosome: marker.chromosome, position: marker.position,
        genotype: dosage === null ? '--' :
          dosage === 0 ? marker.otherAllele.repeat(2) :
            dosage === 1 ? marker.otherAllele + marker.effectAllele : marker.effectAllele.repeat(2)
      };
    }).reverse();
    const result = classifier.classify({ format: '23andMe', chip: 'v5', build: 'GRCh37', records });
    if (result.status !== 'classified' || result.predictedGroup !== sample.predictedGroup) {
      throw new Error(`Reference prediction mismatch in case ${index}: ${JSON.stringify(result)}`);
    }
    for (const group of model.classes) {
      const error = Math.abs(result.scores[group] - sample.scores[group]);
      if (!Number.isFinite(error) || error > tolerance) {
        throw new Error(`Reference score mismatch in case ${index}, ${group}: error ${error}`);
      }
      maxError = Math.max(maxError, error);
    }
  }
  return { cases: reference.cases.length, maxError };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  if (process.argv.length !== 4) {
    throw new Error('Usage: node tools/verify_parity.mjs model.json parity.json');
  }
  const [model, reference] = await Promise.all(process.argv.slice(2).map(async path =>
    JSON.parse(await readFile(path, 'utf8'))));
  console.log(verifyParity(model, reference));
}
