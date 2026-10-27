# 23andMe ancestry classification SDK

Standalone, dependency-free ESM npm SDK for **local browser or Node.js inference**.
Accepts 23andMe text, genotype JSON, and Diseasome's parsed JSON. No genotype,
profile information, or prediction is transmitted or stored by the SDK.

**The trained model is not included.** The Google Cloud scripts, fitted weights,
335,632-marker manifest, and participant files were not provided in this repository.
Export your existing pipeline as described below; the SDK never substitutes a
demonstration model or fits a new one. The tests use synthetic data only.

## What happens after upload

1. Require confirmed **23andMe v5 / GRCh37** provenance.
2. Match SNP IDs, chromosomes, positions and alleles to the saved marker manifest;
   count copies of each saved **effect allele**, in training marker order.
3. Check quality, impute missing SNPs/calls using saved training means, then apply
   saved scaling and ten-component PCA (including whitening if fitted).
4. Apply saved logistic regression, returning AFR, AMR, EAS, EUR or SAS scores.
5. Attach the group, scores, quality metrics and model version to consolidated data.
   Rejections contain a reason and **no prediction or scores**.

These are model reference-group scores, **not ancestry mixture percentages**,
self-reported race/ethnicity, a diagnosis, or a measure of identity. Profile fields
(race, ethnicity, ZIP, etc.) are not classifier inputs. Validate the model on
held-out representative samples before deploying it.

## npm and browser use

From this checkout:

```sh
npm install
npm test
npm run check
npm pack --dry-run
```

In another app, install the checkout (`npm install /absolute/path/to/this/repository`)
or its `npm pack` tarball. The package name is
`ancestry_classification_for_23andme`; this repository does not publish to npm
automatically.

```js
import { createClassifier, attachAncestry } from 'ancestry_classification_for_23andme';

// Fetch only your trusted, versioned model asset; never send the genotype file.
const response = await fetch('/models/ancestry-v1.json');
if (!response.ok) throw new Error('Model unavailable');
const classifier = createClassifier(await response.json()); // reuse across users

// file is a browser File selected by the user; JSON and raw text are accepted.
// Confirm chip from reliable source metadata, NOT from SNP count or a guessed filename.
const result = classifier.classify(await file.text(), { chip: 'v5' });
const updatedUser = attachAncestry(consolidatedUser, result);
```

`classify` is synchronous. Invalid uploads return `{status: "rejected", reason:
{code, message}, modelVersion, quality}`; malformed/incompatible model artifacts
throw at `createClassifier`. A successful result has:

```js
{
  status: 'classified',
  predictedGroup: 'EUR',
  scores: { AFR: 0.01, AMR: 0.02, EAS: 0.01, EUR: 0.95, SAS: 0.01 },
  modelVersion: 'your-version',
  quality: { /* marker counts, matchRate, callRate, imputedMarkers, mismatches */ }
}
```

The numbers above illustrate the output format, not a real participant result.
`attachAncestry` returns a shallow copy with an `ancestry` field; other fields
are preserved, and raw genotype data is never added to the result. The host app
decides whether to persist that copy, with the user's consent.

For a local upload demo, run `python -m http.server 8000` from this checkout and
open **http://localhost:8000/examples/browser.html**. Select your exported model
and a genotype file. The demo has no network submission or persistence.

### Input formats

**Raw text:** standard four-column 23andMe files with comment headers. GRCh37 is
read from `build 37` / `GRCh37` headers. Most raw files do not identify their
chip, so provide independently confirmed `{chip: 'v5'}`. If chip/build cannot be
confirmed, the SDK rejects the file. Supplied metadata must not conflict with
headers. No liftover, chip-version guessing, strand flipping, or allele guessing
is performed.

**Canonical genotype JSON:** an object (or its JSON string) explicitly declaring
its reference and provenance:

```json
{
  "format": "23andMe",
  "chip": "v5",
  "build": "GRCh37",
  "records": [
    {"id": "rsExample", "chromosome": "1", "position": 12345, "genotype": "AG"}
  ]
}
```

This is a schema example, not a usable genome. Positions are 1-based GRCh37;
chromosomes may have a `chr` prefix. Calls must be two alleles from the model's
biallelic pair. `--`, `00`, `NN`, empty strings, or null represent missing calls.
Extra non-model SNPs are ignored. Duplicate model SNPs, coordinate conflicts,
unexpected alleles, indels, and haploid calls at model loci are rejected.
The model manifest supports biallelic autosomal SNPs only.

**Diseasome JSON:** `{cols, dt, meta}` from its PGP parser, also accepted as a
JSON string. `cols` must uniquely name `rsid`, `chromosome`, `position`, and
`genotype`; order is resolved by name. `dt` is an array of arrays. The parser's
extra row-index element is ignored; `meta` contains original comment headers.
The SDK does not import Diseasome's DOM-dependent app or cache internals.

```js
// Diseasome loaded entry: {user, parsed}; uploads can instead live in user._parsed.
const parsed = entry.parsed ?? entry.user._parsed;
const result = classifier.classify(parsed, { chip: confirmedChip });
const updatedEntry = {
  ...entry,
  user: attachAncestry(entry.user, result)
};
```

Preserve the selected file-specific `user.id` (which can include `#fileIndex`);
do not merge different genome files just because they share a participant ID.
Pass selected users' `_parsed` uploads as well as explicitly loaded users.
See Diseasome's [upload handler](https://github.com/episphere/diseasome/blob/77a15142993c50ce0d62fe831e7aa990b55b3f4d/src/app/displayUsers.js)
and [loaded-entry consolidation](https://github.com/episphere/diseasome/blob/77a15142993c50ce0d62fe831e7aa990b55b3f4d/src/app/calculatePrs.js).

For text/parsed JSON whose original headers have been removed, callers must
explicitly attest to provenance via `{format: '23andMe', chip: 'v5',
build: 'GRCh37'}`. Do not supply these values unless independently verified.
Metadata assertions cannot establish provenance cryptographically.
Participant/profile JSON containing only download links is **not genotype JSON**.

### Quality policy

The trusted model artifact specifies `minMatchRate` and `minCallRate`. Both rates
use **all model markers** as the denominator, so absent SNPs cannot inflate call
rate. Defaults in the exporter are 0.95; these are starting thresholds, **not
clinically validated cutoffs**. Establish appropriate thresholds from your
validation set. Missing/absent calls below the allowed limit use training means.
Any coordinate/allele mismatch at a model locus or duplicate model locus rejects
the upload, even if overall call rate is high.

Reason codes: `INVALID_FILE`, `UNSUPPORTED_FILE`, `DUPLICATE_MARKER`,
`MARKER_MISMATCH`, `LOW_QUALITY`, and `NUMERICAL_ERROR`. A valid reference-group
score does not detect all out-of-distribution samples; this SDK does not invent
a confidence cutoff or mixed-ancestry interpretation.

## Export the saved Python pipeline

Use your trusted training environment with its original compatible versions of
NumPy, scikit-learn and joblib. The exporter only reads **trusted local joblib
files**: pickle/joblib files can execute Python code, so never accept uploaded
models from end users.

The supported fitted pipeline, in order, is:

```
SimpleImputer(strategy="mean", missing_values=np.nan, add_indicator=False)
→ StandardScaler → PCA(n_components=10) → LogisticRegression
```

Other preprocessing, covariates, encodings, or estimators must be explicitly
converted and validated first; unsupported pipelines fail rather than being
silently approximated. Do not refit on the 881 uploads.

Prepare a JSON **array of all 335,632 markers in exact training-feature order**:

```json
[
  {
    "id": "rsExample",
    "chromosome": "1",
    "position": 12345,
    "otherAllele": "A",
    "effectAllele": "G"
  }
]
```

The example shows one entry only. `effectAllele` is the allele whose copies were
encoded as 0/1/2 **during training**; it is not necessarily ALT or minor allele.
Do not derive allele orientation from the new upload. Feature names, if present
in the fitted pipeline, must agree with the manifest order.

```sh
python tools/export_model.py \
  --pipeline /trusted/path/pipeline.joblib \
  --markers /trusted/path/markers.json \
  --model-version ancestry-v1 \
  --output /trusted/path/ancestry-v1.json \
  --parity-output /trusted/path/ancestry-v1.parity.json

node tools/verify_parity.mjs \
  /trusted/path/ancestry-v1.json /trusted/path/ancestry-v1.parity.json
```

If labels are encoded as numbers, supply `--class-map /trusted/path/classes.json`,
whose keys are the original saved class labels, e.g.
`{"0":"AFR","1":"AMR","2":"EAS","3":"EUR","4":"SAS"}` **only if that was your
training mapping**. Never assume a label mapping from this example.
The exporter defaults to 335,632 SNPs; `--expected-markers` allows an explicitly
different model size. Threshold options are `--min-match-rate` and
`--min-call-rate`.

The versioned schema stores ordered markers, imputer means, scaler means/scales,
PCA mean/components/whitening scale, original class order, logistic
coefficients/intercepts, probability mode (multinomial softmax or normalized OVR
sigmoid), and QC thresholds. These are fitted weights only, not participant data.
ONNX is not required for this linear pipeline.

The exporter also writes eight **synthetic**, non-participant reference probes;
the parity checker reverses their row order, reconstructs genotypes and compares
all five scores (absolute tolerance `1e-8`) and predicted labels against Python.
Do not deploy if parity fails. In addition, compare representative consented
held-out real files with your original Python parser and classifier to validate
marker orientation, missing-call semantics, and model accuracy end to end.
Synthetic probes alone cannot validate those training-data assumptions.

### Performance and privacy

A 335,632 × 10 PCA matrix is sizeable. Serve the model as a trusted versioned
static asset with HTTP compression, initialize once, and use a Web Worker for
parsing/inference in production so large files do not block Diseasome's UI.
This initial SDK parses in memory and retains a private model copy; budget for
the JSON, parsed rows, and model arrays on target devices. Gzipped/ZIP genotype
uploads must first be decompressed by the host; they are not parsed as text.

Keep participant genomes, profiles, joblib artifacts and validation files out of
source control. Host apps remain responsible for consent, access controls,
telemetry exclusions, and secure handling of any persisted results.

## Development

JavaScript tests and syntax check use only Node.js built-ins:

```sh
npm test
npm run check
```

With the Python export dependencies available in your environment:

```sh
python -m unittest discover -s test -p 'test_*.py' -v
```

The Python tests fit tiny synthetic pipelines and verify exported JavaScript
predictions for whitening and scaler configurations. No real model accuracy or
participant classifications are claimed by these tests.