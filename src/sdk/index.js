// Public SDK entry point.
//
//   import { classifyAncestry, parse23andMe } from "ancestry-classification-for-v5-23andme-files";
//   const result = await classifyAncestry(await file.text());
//   result.group;     // "EUR"
//   result.scores;    // { AFR, AMR, EAS, EUR, SAS }
//   result.coverage;  // fraction of the 335,632 model markers called in the file

export { classifyAncestry, classifyParsed, encodeDosages, coverageProblem, loadModel, loadPcaReference, modelStatus, configure } from "./classifier.js";
export { parse23andMe, genotypeClass, inferChip, inferSex, CHROM_ORDER } from "./parse23andme.js";
