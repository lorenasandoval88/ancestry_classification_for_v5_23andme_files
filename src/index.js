const GROUPS = ['AFR', 'AMR', 'EAS', 'EUR', 'SAS'];
const BASE = /^[ACGT]$/;

function assert(condition, message) {
  if (!condition) throw new TypeError(message);
}

function vector(value, length, name, positive = false) {
  assert(Array.isArray(value) && value.length === length &&
    value.every(x => Number.isFinite(x) && (!positive || x > 0)),
  `Invalid model ${name}`);
}

function matrix(value, rows, columns, name) {
  assert(Array.isArray(value) && value.length === rows, `Invalid model ${name}`);
  value.forEach(row => vector(row, columns, name));
}

function chromosome(value) {
  return String(value).replace(/^chr/i, '').toUpperCase();
}

function validateModel(model) {
  assert(model && model.schemaVersion === 1, 'Unsupported model schema');
  assert(typeof model.modelVersion === 'string' && model.modelVersion.length > 0,
    'Missing model version');
  assert(model.chip === 'v5' && model.build === 'GRCh37', 'Unsupported model reference');
  const n = model.markers?.length;
  assert(n > 0 && model.markerCount === n, 'Invalid model marker count');
  const ids = new Set();
  for (const marker of model.markers) {
    assert(typeof marker.id === 'string' && marker.id.length > 0 && !ids.has(marker.id),
      'Invalid or duplicate model marker');
    ids.add(marker.id);
    assert(typeof marker.chromosome === 'string' &&
      /^(?:[1-9]|1[0-9]|2[0-2])$/.test(marker.chromosome) &&
      Number.isSafeInteger(marker.position) && marker.position > 0 &&
      BASE.test(marker.otherAllele) && BASE.test(marker.effectAllele) &&
      marker.otherAllele !== marker.effectAllele, 'Invalid model marker coordinates or alleles');
  }
  vector(model.imputerMeans, n, 'imputerMeans');
  assert(model.imputerMeans.every(x => x >= 0 && x <= 2), 'Invalid dosage means');
  vector(model.scalerMean, n, 'scalerMean');
  vector(model.scalerScale, n, 'scalerScale', true);
  vector(model.pcaMean, n, 'pcaMean');
  const k = model.pcaComponents?.length;
  assert(k === 10, 'Model must have 10 PCA components');
  matrix(model.pcaComponents, k, n, 'pcaComponents');
  vector(model.pcaScale, k, 'pcaScale', true);
  assert(Array.isArray(model.classes) && model.classes.length === 5 &&
    new Set(model.classes).size === 5 && model.classes.every(x => GROUPS.includes(x)),
  'Model must contain AFR, AMR, EAS, EUR and SAS');
  matrix(model.coefficients, 5, k, 'coefficients');
  vector(model.intercepts, 5, 'intercepts');
  assert(['softmax', 'ovr'].includes(model.probabilityMode), 'Invalid probability mode');
  assert(model.quality && ['minMatchRate', 'minCallRate'].every(key =>
    Number.isFinite(model.quality[key]) && model.quality[key] > 0 &&
    model.quality[key] <= 1), 'Invalid model quality thresholds');
}

function metadataValue(value) {
  return String(value ?? '').trim();
}

/**
 * Parse raw 23andMe tab-separated text. Chip version must be supplied explicitly
 * when absent from the header; a SNP count is not evidence of a chip version.
 */
export function parse23andMe(text, metadata = {}) {
  assert(typeof text === 'string', 'Expected 23andMe text');
  const records = [];
  let headerBuild;
  let headerChip;
  let is23andMe = metadata.format === '23andMe';
  for (const raw of text.replace(/^\uFEFF/, '').split(/\r?\n/)) {
    const line = raw.trim();
    if (!line) continue;
    if (line.startsWith('#')) {
      if (/23andMe/i.test(line)) is23andMe = true;
      for (const build of line.matchAll(/(?:build\s*|GRCh)(\d+)/gi)) {
        const found = `GRCh${build[1]}`;
        assert(!headerBuild || headerBuild === found, 'Conflicting reference builds');
        headerBuild = found;
      }
      const chipLabel = line.match(/\b(?:chip|platform)\b/i);
      const tokens = chipLabel
        ? line.slice(chipLabel.index + chipLabel[0].length).split(/[\s:=]+/).filter(Boolean)
        : [];
      if (tokens[0]?.toLowerCase() === 'version') tokens.shift();
      const chip = tokens[0]?.match(/^v?([1-9]\d*)$/i);
      if (chip) {
        const found = `v${chip[1]}`;
        assert(!headerChip || headerChip === found, 'Conflicting chip versions');
        headerChip = found;
      }
      continue;
    }
    const fields = line.split(/\s+/);
    if (fields[0].toLowerCase() === 'rsid') continue;
    assert(fields.length === 4, 'Expected rsid, chromosome, position and genotype columns');
    records.push({ id: fields[0], chromosome: fields[1],
      position: Number(fields[2]), genotype: fields[3] });
  }
  assert(!headerBuild || !metadata.build || headerBuild === metadata.build,
    'Metadata conflicts with reference build in file');
  assert(!headerChip || !metadata.chip || headerChip === metadata.chip,
    'Metadata conflicts with chip version in file');
  assert(is23andMe, 'Missing 23andMe provenance; supply confirmed format metadata if headers were removed');
  return {
    format: '23andMe',
    chip: headerChip ?? metadata.chip,
    build: headerBuild ?? metadata.build,
    records
  };
}

/** Adapt Diseasome/PGP parser output, including its extra unlabelled row index. */
export function fromDiseasome(parsed, metadata = {}) {
  assert(parsed && Array.isArray(parsed.cols) && Array.isArray(parsed.dt),
    'Expected Diseasome {cols, dt, meta}');
  const columns = ['rsid', 'chromosome', 'position', 'genotype'];
  const indices = columns.map(column => {
    assert(parsed.cols.filter(value => value === column).length === 1,
      `Missing or duplicate Diseasome column: ${column}`);
    return parsed.cols.indexOf(column);
  });
  assert(typeof parsed.meta === 'string', 'Expected Diseasome header comments');
  assert(parsed.meta.split(/\r?\n/).every(line => !line.trim() || line.trim().startsWith('#')),
    'Expected only comments in Diseasome meta');
  const upload = parse23andMe(parsed.meta, metadata);
  upload.records = parsed.dt.map(row => {
    assert(Array.isArray(row) && indices.every(index => index < row.length),
      'Invalid Diseasome genotype row');
    return { id: row[indices[0]], chromosome: row[indices[1]],
      position: row[indices[2]], genotype: row[indices[3]] };
  });
  return upload;
}

function normalizeUpload(input, metadata) {
  if (typeof input === 'string') {
    const trimmed = input.trimStart();
    if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
      input = JSON.parse(input);
    } else {
      return parse23andMe(input, metadata);
    }
  }
  assert(input && typeof input === 'object' && !Array.isArray(input), 'Expected an upload object');
  if ('cols' in input || 'dt' in input) return fromDiseasome(input, metadata);
  return input;
}

function failure(code, message, modelVersion, quality = {}) {
  return { status: 'rejected', modelVersion, reason: { code, message }, quality };
}

/**
 * Create once per exported model, then reuse for uploads. No network or storage
 * APIs are used. Invalid model artifacts throw; invalid uploads return rejections.
 */
export function createClassifier(artifact) {
  // Keep caller mutations from changing marker order or thresholds after validation.
  const model = structuredClone(artifact);
  validateModel(model);
  const markerIndex = new Map(model.markers.map((marker, index) => [marker.id, index]));

  function classify(input, metadata = {}) {
    let upload;
    try {
      upload = normalizeUpload(input, metadata);
    } catch (error) {
      return failure('INVALID_FILE', error.message, model.modelVersion);
    }
    if (upload.format !== '23andMe' || upload.chip !== 'v5' || upload.build !== 'GRCh37') {
      return failure('UNSUPPORTED_FILE', 'A confirmed 23andMe v5, GRCh37 upload is required',
        model.modelVersion);
    }
    if (!Array.isArray(upload.records) || upload.records.length === 0) {
      return failure('INVALID_FILE', 'No genotype records provided', model.modelVersion);
    }
    const dosages = new Float64Array(model.markerCount);
    dosages.fill(NaN);
    const seen = new Set();
    let matched = 0;
    let called = 0;
    let missing = 0;
    let alleleMismatches = 0;
    let coordinateMismatches = 0;
    for (const record of upload.records) {
      if (!record || typeof record.id !== 'string') {
        return failure('INVALID_FILE', 'Invalid genotype record', model.modelVersion);
      }
      const index = markerIndex.get(record.id);
      if (index === undefined) continue;
      if (seen.has(record.id)) {
        return failure('DUPLICATE_MARKER', 'Duplicate model SNP in upload', model.modelVersion);
      }
      seen.add(record.id);
      const marker = model.markers[index];
      if (chromosome(record.chromosome) !== marker.chromosome ||
        Number(record.position) !== marker.position) {
        coordinateMismatches++;
        continue;
      }
      matched++;
      const genotype = metadataValue(record.genotype).toUpperCase();
      if (['', '--', '00', 'NN'].includes(genotype)) {
        missing++;
        continue;
      }
      if (genotype.length !== 2 ||
        [...genotype].some(base => base !== marker.effectAllele && base !== marker.otherAllele)) {
        alleleMismatches++;
        continue;
      }
      dosages[index] = [...genotype].filter(base => base === marker.effectAllele).length;
      called++;
    }
    const quality = {
      expectedMarkers: model.markerCount, matchedMarkers: matched, calledMarkers: called,
      missingCalls: missing, absentMarkers: model.markerCount - seen.size,
      alleleMismatches, coordinateMismatches,
      matchRate: matched / model.markerCount,
      callRate: called / model.markerCount,
      imputedMarkers: model.markerCount - called
    };
    if (coordinateMismatches || alleleMismatches) {
      return failure('MARKER_MISMATCH', 'Model SNP coordinates or alleles do not match',
        model.modelVersion, quality);
    }
    if (quality.matchRate < model.quality.minMatchRate || quality.callRate < model.quality.minCallRate) {
      return failure('LOW_QUALITY', 'Insufficient matched or called model SNPs',
        model.modelVersion, quality);
    }
    const pcs = new Float64Array(10);
    for (let i = 0; i < model.markerCount; i++) {
      const dosage = Number.isNaN(dosages[i]) ? model.imputerMeans[i] : dosages[i];
      const centered = (dosage - model.scalerMean[i]) / model.scalerScale[i] - model.pcaMean[i];
      for (let j = 0; j < 10; j++) pcs[j] += centered * model.pcaComponents[j][i];
    }
    for (let j = 0; j < 10; j++) pcs[j] /= model.pcaScale[j];
    const logits = model.coefficients.map((weights, c) =>
      weights.reduce((sum, weight, j) => sum + weight * pcs[j], model.intercepts[c]));
    let probabilities;
    if (model.probabilityMode === 'softmax') {
      const max = Math.max(...logits);
      probabilities = logits.map(x => Math.exp(x - max));
    } else {
      // Log-sigmoid followed by softmax avoids underflow for extreme OVR logits.
      const logSigmoids = logits.map(x => -Math.max(0, -x) - Math.log1p(Math.exp(-Math.abs(x))));
      const max = Math.max(...logSigmoids);
      probabilities = logSigmoids.map(x => Math.exp(x - max));
    }
    const total = probabilities.reduce((a, b) => a + b, 0);
    probabilities = probabilities.map(x => x / total);
    if (!probabilities.every(Number.isFinite)) {
      return failure('NUMERICAL_ERROR', 'Model produced non-finite scores', model.modelVersion, quality);
    }
    const best = logits.indexOf(Math.max(...logits));
    return {
      status: 'classified', modelVersion: model.modelVersion,
      predictedGroup: model.classes[best],
      scores: Object.fromEntries(model.classes.map((group, i) => [group, probabilities[i]])),
      quality
    };
  }

  return Object.freeze({ classify, modelVersion: model.modelVersion });
}

/** Preserve existing consolidated fields; never attach raw genotypes. */
export function attachAncestry(consolidated, result) {
  assert(consolidated && typeof consolidated === 'object' && !Array.isArray(consolidated),
    'Expected consolidated data object');
  assert(result && ['classified', 'rejected'].includes(result.status), 'Expected classification result');
  return { ...consolidated, ancestry: result };
}
