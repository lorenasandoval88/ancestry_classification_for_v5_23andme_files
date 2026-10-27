# ancestry_classification_for_23andme

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
| Preprocess | Align the file to the fixed panel of **335,632 ancestry markers**, count the reference allele, and fill missing values with saved training means |
| Project | Apply the saved PCA (**10 principal components**) |
| Classify | Apply the saved classifier and return the AFR / AMR / EAS / EUR / SAS assignment with per-group scores |
| Report | Return the reference-group prediction, marker coverage, and model version as separate fields |

New files never trigger retraining. Every stage uses a frozen marker order, allele coding,
preprocessing parameters, and model version.

## Model summary

| Resource | Status |
|----------|--------|
| Reference data | 1000 Genomes Phase 3 (2,504 people, 26 populations) |
| Training cohort | 2,003 people |
| Held-out test cohort | 501 people |
| Ancestry model inputs | 335,632 markers; 10 principal components |
| Test accuracy | 500 of 501 reference labels correct |
| Personal-file application | 74 files from 72 participants (23andMe v5) |

The trained scikit-learn pipeline (preprocessing, PCA, and classifier) is exported to a
browser-compatible representation and checked for agreement with the Python outputs so that
the npm package and web app produce the same result as the original model.

## Installation (SDK)

```sh
npm install ancestry-classification-for-23andme
```

```js
import { classifyAncestry } from "ancestry-classification-for-23andme";

const text = await file.text(); // raw 23andMe v5 export
const result = await classifyAncestry(text);

console.log(result.group);    // e.g. "EUR"
console.log(result.scores);   // { AFR: 0.01, AMR: 0.03, EAS: 0.00, EUR: 0.95, SAS: 0.01 }
console.log(result.coverage); // fraction of the 335,632 markers found in the file
```

## Web app

Open the web app, select your 23andMe raw-data file, and the page shows the closest
1000 Genomes reference group, the per-group scores, and marker coverage. Processing happens
locally in the browser.

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
