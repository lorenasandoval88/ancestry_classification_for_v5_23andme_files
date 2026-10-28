# ancestry_classification_for_v5_23andme_files

**Live app:** https://lorenasandoval88.github.io/ancestry_classification_for_v5_23andme_files/

An **SDK (npm package) and web app** for classifying 23andMe raw-data files into a genetic
ancestry reference group. Upload or pass in a 23andMe v5 export and the classifier returns
the 1000 Genomes super-population that is genetically most similar to the profile:
**AFR, AMR, EAS, EUR, or SAS**, along with the classifier scores for each group.

The same model can be used two ways:

- **npm package (SDK)** – import the classifier into your own JavaScript/TypeScript project
  (Node.js or browser) and classify files programmatically.
- **Web app** – a browser-based interface where a user selects a 23andMe file and gets the
  reference-group prediction. Inference runs entirely in the browser; the genotype file is
  never uploaded to a server.

## What the classifier does

| Step | Action for a 23andMe file |
|------|---------------------------|
| Validate | Confirm a supported chip (23andMe v5) and genome build (GRCh37); record missing calls and incompatible alleles |
| Encode | Align the file to the fixed panel of **335,632 ancestry markers** (`chr:pos:ref:alt`, GRCh37) and count the **REF allele** at each: 0, 1 or 2 — the same encoding as PLINK `--export A` used in training |
| Impute | Fill markers that are absent, no-call, or carry alleles outside {REF, ALT} with the saved **training means** (the fitted `SimpleImputer` statistics). This happens upstream, in JavaScript |
| Classify | Run the exported ONNX graph — `StandardScaler → PCA(10) → LogisticRegression` — in the browser with `onnxruntime-web` |
| Report | Return the AFR / AMR / EAS / EUR / SAS assignment, per-group probabilities, marker coverage (called / missing / incompatible counts) and model checksum as separate fields |

New files never trigger retraining. Every stage uses a frozen marker order, allele coding,
preprocessing parameters, and model version.

## Model summary

| Resource | Status |
|----------|--------|
| Reference data | 1000 Genomes Phase 3 (2,504 people, 26 populations), restricted to 23andMe v5 positions on chromosomes 1–22, GRCh37 |
| Training cohort | 2,003 people (AFR 524 · AMR 272 · EAS 400 · EUR 408 · SAS 399) |
| Held-out test cohort | 501 people (AFR 137 · AMR 75 · EAS 104 · EUR 95 · SAS 90) |
| Relatedness audit | 15 published close-relative pairs were found among the training/test people |
| Ancestry model inputs | 335,632 markers (selected using training people only); 10 principal components; logistic regression C = 1.0 |
| Training-only cross-validation | 3-fold, balanced accuracy 0.995, accuracy 0.996 |
| Test accuracy | 500 of 501 reference labels correct |
| Personal-file application | 74 files from 72 participants (23andMe v5) |

### How the model was built (steps 1–7)

The reference model is produced by a seven-step cloud workflow; steps 1–6 run once to build and
test the model, step 7 applies the frozen model to personal files. Nothing in step 7 (or in this
web app) retrains or re-selects markers.

| Step | What it does |
|------|--------------|
| 1 Prepare reference genotypes | Read the Phase 3 reference files, match the verified 23andMe v5 positions on chromosomes 1–22, check genome build and alleles, and save one prepared PLINK genotype set per chromosome for all 2,504 people |
| 2 Verify the sample split | Confirm 2,003 unique training IDs and 501 unique test IDs, no overlap, every ID in the reference panel; save normalised lists and checksums (`train2003.txt`, `test501.txt`) |
| 3 Audit populations | Join IDs to the population panel and report training/test counts by population and broad reference group; record the relatedness audit (15 published close pairs) |
| 4 Marker QC and LD pruning | Using **only** the 2,003 training people, remove markers with excess missingness or very low allele frequency and LD-prune correlated markers; save one ordered panel with the counted (REF) allele per marker — 335,632 markers for the current model |
| 5 Train one PCA + classifier | Fit missing-value filling (mean), scaling, PCA (10 components) and logistic regression (C = 1.0) on the training people; save the full fitted pipeline, marker order and counted alleles, and a training summary with input hashes |
| 6 Test the frozen model | Apply the saved pipeline to the 501 test people and compare with the known 1000 Genomes labels: 500 / 501 correct, with per-group accuracy, a confusion table and coverage checks. Test people never influence marker selection, PCA or tuning |
| 7 Apply to v5 files | For compatible personal files (v5, GRCh37): match positions and alleles to the saved marker list, check usable-marker coverage, apply the same pipeline, and return the predicted group with quality checks and model version. Files with insufficient usable data receive a reason instead of an assignment |

This repository implements step 7 for the browser: the SDK reproduces the step-7 encoding
(verified to give the identical 332,847 called markers on the bundled demo file as the cloud
pipeline) and runs the exported pipeline with `onnxruntime-web`.

The trained scikit-learn pipeline (`SimpleImputer → StandardScaler → PCA → LogisticRegression`,
see `data/training_model/cloudModel/train_ancestry.py`) is exported to ONNX and checked for
agreement with the Python outputs so that the npm package and web app produce the same result
as the original model.

### Exporting the model for the browser

`model.joblib` can only be read by Python + scikit-learn, so the export runs once in the
training environment (sklearn 1.7.2) and the resulting artifacts are committed under `model/`:

```sh
pip install skl2onnx onnx onnxruntime numpy pandas joblib
python scripts/export_onnx.py \
  --model data/training_model/cloudModel/model.joblib \
  --markers "data/training_model/cloudModel/*_selected_markers.tsv" \
  --out model
```

| Artifact | Contents |
|----------|----------|
| `model/ancestry_pipeline.onnx` | `StandardScaler → PCA(10) → LogisticRegression` (imputer removed; zipmap disabled) |
| `model/markers.tsv` | `variant_id` + `counted_allele` (REF) in model feature order |
| `model/means.f32` | float32 little-endian training mean per marker, used for upstream imputation |
| `model/model_meta.json` | classes, feature count, steps, SHA-256 of every artifact, parity-check result |

Until these files exist the web app shows **"model pending"** and still parses/inspects files.
Pass `--parity X.npy` (an `(n, 335632)` float32 dosage matrix, e.g. the 501 test people) to
compare ONNX against scikit-learn on real data; the script falls back to synthetic rows.

## Installation (SDK)

```sh
npm install ancestry-classification-for-v5-23andme-files
```

```js
import { classifyAncestry } from "ancestry-classification-for-v5-23andme-files";

const text = await file.text(); // raw 23andMe v5 export
const result = await classifyAncestry(text);

result.group;              // e.g. "EUR"
result.scores;             // { AFR: 0.01, AMR: 0.03, EAS: 0.00, EUR: 0.95, SAS: 0.01 }
result.coverage;           // fraction of the 335,632 markers called in the file
result.calledMarkers;      // 332847
result.incompatibleMarkers;// markers whose alleles were not {REF, ALT} (filled with the mean)
result.model.sha256;       // checksum of ancestry_pipeline.onnx
```

Lower-level pieces are exported too: `parse23andMe(text)`, `encodeDosages(parsed)`,
`loadModel()`, `modelStatus()` and `configure({ modelBase })` to point at a different
artifact location. Source: `src/sdk/`.

## Web app

Open the web app, select your 23andMe raw-data file, and the page shows the closest
1000 Genomes reference group, the per-group scores, and marker coverage. Processing happens
locally in the browser.

The landing page lives in [`index.html`](./index.html) (styles in `src/css/`, logic in
`src/app/app.js`) and has three tabs:

1. **Upload & Classify** – drag-and-drop or browse for one or more `.txt` exports, or press **Try a demo file**
   to load the bundled real 23andMe v5 export (`data/demo_genotype_file/genome_23.txt`, 631,455 variants).
   The status line shows the cloud pipeline's result for that file (AMR, 332,847 markers called) so you can compare.
2. **Participants** – one searchable, filterable, sortable table combining three sources (Source filter):
   - **1000G train / test** – the 2,003 training and 501 held-out 1000 Genomes people from
     `data/reference_cohort.json`, with population code and name, sex and *known* super-population label
     (shown as an outlined badge, never as a prediction);
   - **PGP catalog** – 885 public PGP participants / 1,018 23andMe files from `data/participants_catalog.json`
     (chip version, build, variant count, file validity, self-reported sex / race / ethnicity, birth year, state,
     predicted reference group, called-marker fraction);
   - **Uploaded** – files you load in the Upload tab.
   Includes per-group counts for known vs. predicted labels, pagination and CSV export.
3. **Inspect Genotypes** – paginated genotype table with rsID / position search, chromosome and genotype-class filters, column sorting and CSV export.

### Data

| File | Contents |
|------|----------|
| `data/demo_genotype_file/genome_23.txt` | Real, anonymised 23andMe v5 export (GRCh37, 631,455 variants, 16 MB) used by the *Try a demo file* button. The SDK encoder yields 332,847 called model markers on it — identical to the cloud pipeline's record for the same file |
| `data/participantInfo_genotypeFiles_23me.json` | Full PGP participant / file catalog (12 MB) with chip classification, ancestry predictions and survey profiles |
| `data/participants_catalog.json` | Slim, de-identified per-file catalog generated from the above for the web app (no names, zip codes or storage URLs) |
| `data/training_model/cloudModel/2003train_501test_input/train2003.txt`, `test501.txt` | 1000 Genomes sample IDs of the 2,003-person training and 501-person test cohorts |
| `data/training_model/cloudModel/2003train_501test_input/integrated_call_samples_v3.20130502.ALL.panel` | Public Phase 3 population panel (sample, pop, super_pop, gender); SHA-256 `b4023dc6…` matches `population_panel_sha256` in the training summary |
| `data/reference_cohort.json` | Generated: the 2,504 reference people with split, population, super-population and sex, plus per-group counts (train: AFR 524 · AMR 272 · EAS 400 · EUR 408 · SAS 399; test: AFR 137 · AMR 75 · EAS 104 · EUR 95 · SAS 90) |
| `data/training_model/cloudModel/model.joblib` | Bundle written by `train_ancestry.py`: `{"pipeline": SimpleImputer → StandardScaler → PCA(10) → LogisticRegression(C=1.0), "markers", "classes", …}` (sklearn 1.7.2) |
| `data/training_model/cloudModel/train_ancestry.py` | Training script (PLINK `--export A` REF-allele dosages, grouped 3-fold CV, training-only fit) |
| `data/training_model/cloudModel/*_selected_markers.tsv` | The 335,632 model markers (`chr:pos:ref:alt`) and the counted (REF) allele |
| `data/training_model/cloudModel/*_training_summary.json` | Training provenance (cohort, CV accuracy 0.996, hashes) |
| `data/training_model/cloudModel/cloud_instructions_steps/*.docx` | Plan for the single-model seven-step cloud workflow summarised in *How the model was built* |
| `model/` (generated) | Browser artifacts from `scripts/export_onnx.py` — see *Exporting the model* above |

Regenerate the slim catalog and the reference cohort after updating the inputs:

```sh
npm run build:data      # = build:catalog + build:cohort
```

To run it locally, serve the repo root with any static server, e.g.

```sh
npx http-server -p 8080
# then open http://localhost:8080/
```

The Classify buttons call the SDK in `src/sdk/classifier.js`, which loads `model/` on demand.
Until the ONNX artifacts are exported, the Upload tab shows a *model artifacts not published yet*
badge and classification results read "model pending".

## Interpreting the result

- The output is a **genetic similarity to a reference group**, not an ancestry-mixture
  proportion. The five classifier scores should not be used as admixture weights.
- **AMR** denotes the admixed American reference populations of 1000 Genomes and should not
  be read as an Indigenous American label.
- Genetic group assignment is separate from self-reported race and ethnicity.
- A file that fails input checks returns the reason, and the classifier output is withheld.

## Related work

The classifier is also used as a building block for downstream work such as evaluating
polygenic-risk-score imputation (e.g. a FastImpute reproduction for PRS313 / PGS000004)
across reference groups. That analysis is separate from, and not required for, using this
package to classify a 23andMe file.

## References

1. 1000 Genomes Project Consortium. A global reference for human genetic variation.
   *Nature*. 2015;526:68-74. https://doi.org/10.1038/nature15393
2. Ge A, Balasubramanian J, Wu X, Kraft P, Almeida JS. FastImpute: Development and
   Validation of a Workflow for Open-source, Reference-Free Genotype Imputation Methods.
   *The Open Bioinformatics Journal*. 2026;19. https://doi.org/10.2174/0118750362421210250929110508
3. Scikit-learn developers. Model persistence (ONNX export).
   https://scikit-learn.org/stable/model_persistence.html
