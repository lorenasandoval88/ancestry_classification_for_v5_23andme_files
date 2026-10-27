# Step 5 — Fit four ancestry models using 1000 Genomes v5 SNPs

This package trains **four separate PCA plus broad-reference-group classifiers**.
It uses the two completed Step 4 marker-selection runs, the exact genotype
branches and the audited sample lists. **Training does not open any test
genotype file.** Step 6 will evaluate all four frozen models on the same 501
held-out people.

| Model | Training people | SNP panel | Main use |
| --- | ---: | --- | --- |
| `train-v5-2003-with` | 2,003 | Eligible VISAGE markers included | Primary |
| `train-v5-2003-without` | 2,003 | Eligible VISAGE markers excluded | Primary paired comparison |
| `train-v5-1998-with` | 1,998 | Eligible VISAGE markers included | Relatedness check |
| `train-v5-1998-without` | 1,998 | Eligible VISAGE markers excluded | Relatedness check |

The 2,003-person primary training cohort has **five relatives of test people**.
Step 6 should report full 501-person test accuracy and a diagnostic excluding
those five related test people; the 1,998-person runs have no known cross-split
relative pairs. Both Step 4 marker panels were derived separately using their
respective training cohort. Therefore, comparing 2,003 versus 1,998 also
reflects differences in selected SNPs. The clean paired comparison within
each cohort keeps the other selected SNPs fixed.

## Inputs

- With VISAGE training genotypes: `gs://ref1000g/ancestry/2_split_the_samples/results/chromosomes/chrN/train/`
- Without VISAGE training genotypes: `gs://ref1000g/ancestry/2b_without_visage/results/chromosomes/chrN/train/`
- Corrected 34-marker audit: `gs://ref1000g/ancestry/2a_visage_locus_audit/results/`
- Sample lists, family groups, population labels audit:
  `gs://ref1000g/ancestry/3_audit_pop_and_relatives/results/20261026-164720-2308/`
- Separate per-cohort selected marker lists and reports:
  `gs://ref1000g/ancestry/4_snp_qc_and_ld_prune/results/<EXACT_RUN_ID>/`
- The supplied Step 4 2,003-person report selects **335,666 with VISAGE**
  and **335,632 without**; all 34 matched VISAGE markers were retained.
  The 1,998-person totals must be read from its own completed result.
- Population panel verified against the Step 3 SHA-256:
  `gs://ref1000g/phase3/integrated_call_samples_v3.20130502.ALL.panel`

**Wait until Step 4 has completed both jobs and both result folders contain
`_SUCCESS.json`.** This package needs their actual result folder names. It
cannot know the final SNP counts until those results are available.

## Install and deploy from Cloud Shell

Upload `5_fit_pca_and_classifier_four_models.zip` to Cloud Shell, then:

```bash
unzip -o ~/5_fit_pca_and_classifier_four_models.zip -d ~/ancestry
cd ~/ancestry/5_fit_pca_and_classifier_four_models
ls -la

export COHORT_RUN_ID=20261026-164720-2308
export QC_RUN_2003="PASTE_EXACT_STEP4_TRAIN2003_RESULT_FOLDER"
export QC_RUN_1998="PASTE_EXACT_STEP4_TRAIN1998_RESULT_FOLDER"

bash deploy.sh
```

To find the exact Step 4 folder names:

```bash
gcloud storage ls gs://ref1000g/ancestry/4_snp_qc_and_ld_prune/results/
```

Choose the folder ending in `-train2003` for `QC_RUN_2003` and the one ending
in `-train1998` for `QC_RUN_1998`, **from the two successful runs you intend to
use**. Enter folder *names*, without the `gs://` prefix or trailing slash.
`deploy.sh` checks their `_SUCCESS.json` objects, copies the six source files
into a timestamped `source/` folder, builds the image with Cloud Build and
deploys four Cloud Run jobs. It uses the existing Artifact Registry repository
`phase3-download` and runtime account. No IAM changes are attempted.
Step 5 always defaults to `5_fit_pca_and_classifier_four_models/` for its outputs. It
ignores any old `ROOT` variable from earlier steps; to override its output
prefix explicitly, set `MODEL_ROOT`. If Step 4 results live elsewhere, set
`MARKER_QC_ROOT` to their common bucket prefix before deploying.

Deployment **does not run** the models. Execute one at a time to avoid the
cost and memory load of four simultaneous jobs:

```bash
gcloud run jobs execute train-v5-2003-with \
  --project=nih-nci-dceg-episphere-dev --region=us-east4 --async

gcloud run jobs execute train-v5-2003-without \
  --project=nih-nci-dceg-episphere-dev --region=us-east4 --async

gcloud run jobs execute train-v5-1998-with \
  --project=nih-nci-dceg-episphere-dev --region=us-east4 --async

gcloud run jobs execute train-v5-1998-without \
  --project=nih-nci-dceg-episphere-dev --region=us-east4 --async
```

Each job uses **one task, 8 vCPU, 32 GiB RAM, a 24-hour timeout, and zero
automatic retries**. The 32 GiB limit reflects the memory needed to export
genotypes, build a numerical training matrix and run PCA. Cloud Run temporary
files also consume allocated memory. These settings have **not** been measured
against your real Step 4 marker totals; monitor resources/cost and stop if a
job exceeds what you intend to spend. If interrupted, rerun the *same job*;
existing matching model artifacts are validated, but incomplete training
restarts from chromosome 1. Never start two copies of the same job concurrently.

## Training method

1. Check the Step 4 completion report, marker counts and selected-marker
   hashes. Check train cohort membership and group labels against Step 3.
2. Read the appropriate **training** genotype branch one chromosome at a
   time; export *only* the selected SNPs as REF-allele dosages (0/1/2).
   Confirm exported marker IDs, allele coding and all training sample IDs.
3. Use three-fold `StratifiedGroupKFold` so documented relatives within
   training remain in the same validation fold. Within each fold, fit
   missing-genotype imputation, scaling, PCA, and logistic classification
   on the training fold and score the validation fold. Choose the best of
   10 or 20 PCs and logistic regularization `C=0.1` or `C=1.0` by mean
   balanced validation accuracy, then overall accuracy.
4. Refit the chosen full pipeline using the complete corresponding training
   cohort. Save the classifier, transformations, ordered marker IDs and REF
   alleles, grouped CV scores, sample counts and source checksums.

CV scores **are not the 501-person test scores**. SNP QC and LD pruning in
Step 4 used the entire training cohort, which includes each CV fold's
validation people; this is disclosed in the report. The held-out 501-person
test set has not been used for SNP selection or model fitting.

The target is the five broad 1000 Genomes `super_pop` groups (AFR, AMR, EAS,
EUR and SAS), not individual ancestry mixture percentages. The 34 confirmed
VISAGE markers are eligible for the with branch but some can fail Step 4
quality or pruning checks. The actual number used appears in each model's
report. The separate coordinate-only `i710492` candidate was not among the
34 removed SNPs; it remains an ordinary candidate on both branches.

## Results and monitoring

The deploy script prints one timestamped `results/RUN_ID/` prefix:

```text
gs://ref1000g/ancestry/5_fit_pca_and_classifier_four_models/
├── source/RUN_ID/                       # exact code, Dockerfile, README, script
└── results/RUN_ID/
    ├── train2003_with_visage/
    ├── train2003_without_visage/
    ├── train1998_with_visage/
    └── train1998_without_visage/
```

Each finished model folder contains `model.joblib`, `selected_markers.tsv`,
`cv_scores.json`, `training_summary.json`, `input_snapshot.json`, and
`_SUCCESS.json`. `model.joblib` contains the fitted imputer, scaler, PCA and
classifier; **load it only from this trusted output location** and use the
matching scikit-learn container/version during later prediction.

```bash
gcloud run jobs executions list --job=train-v5-2003-with \
  --project=nih-nci-dceg-episphere-dev --region=us-east4

gcloud run jobs logs read train-v5-2003-with \
  --project=nih-nci-dceg-episphere-dev --region=us-east4 --limit=100

gcloud storage ls gs://ref1000g/ancestry/5_fit_pca_and_classifier_four_models/results/
```

If an execution fails, inspect its application logs and fix the cause before
starting it again. A successful Cloud Run execution alone should also have
the corresponding bucket `_SUCCESS.json` marker.

## Local checks used when creating this package

`python3 -m py_compile train_ancestry.py`, `python3 train_ancestry.py --self-test`,
`bash -n deploy.sh`, and ZIP integrity were run locally. The Dockerfile repeats
the synthetic model self-test during Cloud Build. Real input storage access,
deployment and 1000 Genomes model training have **not** been run here.

References: [PLINK 2 raw genotype export](https://www.cog-genomics.org/plink/2.0/data),
[scikit-learn grouped CV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html),
[scikit-learn avoiding leakage](https://scikit-learn.org/stable/common_pitfalls.html),
[Cloud Run Jobs memory limits](https://cloud.google.com/run/docs/configuring/jobs/memory-limits).
