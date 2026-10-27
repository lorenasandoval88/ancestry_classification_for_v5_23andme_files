#!/usr/bin/env python3
"""Fit one broad-group ancestry PCA/classifier using training genotypes only."""

import argparse
import base64
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import joblib
import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

STEP2 = os.environ.get("STEP2_RESULTS", "gs://ref1000g/ancestry/2_split_the_samples/results").rstrip("/")
STEP2B = os.environ.get("STEP2B_RESULTS", "gs://ref1000g/ancestry/2b_without_visage/results").rstrip("/")
STEP2A = os.environ.get("STEP2A_RESULTS", "gs://ref1000g/ancestry/2a_visage_locus_audit/results").rstrip("/")
STEP3 = os.environ.get("STEP3_RESULTS", "gs://ref1000g/ancestry/3_audit_pop_and_relatives/results").rstrip("/")
STEP4 = os.environ.get("STEP4_ROOT", "gs://ref1000g/ancestry/4_snp_qc_and_ld_prune").rstrip("/")
PANEL = os.environ.get("POP_PANEL", "gs://ref1000g/phase3/integrated_call_samples_v3.20130502.ALL.panel")
ROOT = os.environ.get("OUTPUT_ROOT", "gs://ref1000g/ancestry/5_fit_pca_and_classifier_four_models").rstrip("/")
COHORT_RUN_ID = os.environ.get("COHORT_RUN_ID", "")
RUN_2003 = os.environ.get("QC_RUN_2003", "")
RUN_1998 = os.environ.get("QC_RUN_1998", "")
COHORT = os.environ.get("TRAIN_COHORT", "")
BRANCH = os.environ.get("VISAGE_BRANCH", "")
RUN_ID = os.environ.get("OUTPUT_RUN_ID", "")


def command(args, capture=False):
    p = subprocess.run(args, text=True, stdout=subprocess.PIPE if capture else None,
                       stderr=subprocess.PIPE if capture else None)
    if p.returncode:
        raise RuntimeError(f"Command failed: {' '.join(map(str,args))}\n{p.stderr or ''}")
    return p.stdout if capture else None


def digest(path, algo="sha256"):
    h = hashlib.new(algo)
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.digest()


def sha(path):
    return digest(path).hex()


def info(uri):
    p = subprocess.run(["gcloud", "storage", "objects", "describe", uri, "--raw", "--format=json"],
                       capture_output=True, text=True)
    if p.returncode == 0:
        return json.loads(p.stdout)
    if "404" in p.stderr or "No URLs matched" in p.stderr:
        return None
    raise RuntimeError(f"Cannot inspect {uri}: {p.stderr.strip()}")


def require(uri):
    result = info(uri)
    if result is None:
        raise ValueError(f"Required object missing: {uri}")
    return result


def get(uri, path):
    require(uri)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    command(["gcloud", "storage", "cp", uri, str(path), "--quiet"])
    return Path(path)


def put(path, uri):
    local_size = str(Path(path).stat().st_size)
    local_md5 = base64.b64encode(digest(path, "md5")).decode()  # Transfer integrity, not security.
    previous = info(uri)
    if previous is None:
        command(["gcloud", "storage", "cp", str(path), uri, "--if-generation-match=0", "--quiet"])
        previous = require(uri)
    if str(previous.get("size")) != local_size or previous.get("md5Hash") != local_md5:
        raise ValueError(f"Existing output differs; use a fresh run ID: {uri}")


def lines(path):
    values = [x.strip() for x in Path(path).read_text().splitlines() if x.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"Empty or duplicate IDs: {path}")
    return values


def keep(path):
    data = Path(path).read_text().splitlines()
    if not data or data[0].strip() != "#IID":
        raise ValueError(f"Invalid keep header: {path}")
    return lines_without_header(data[1:])


def lines_without_header(data):
    values = [x.strip() for x in data if x.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("Missing or duplicate IDs")
    return values


def read_groups(path):
    with Path(path).open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows or set(rows[0]) != {"IID", "GROUP"} or len({r["IID"] for r in rows}) != len(rows):
        raise ValueError("Invalid cross-validation group file")
    return {r["IID"]: r["GROUP"] for r in rows}


def read_population(path):
    # The Phase 3 panel may be separated by tabs or other whitespace.
    entries = [(line_no, row.strip().split()) for line_no, row in enumerate(
        Path(path).read_text(encoding="utf-8-sig").splitlines(), 1) if row.strip()]
    if not entries:
        raise ValueError("Empty population panel")
    columns = {name.lower().lstrip("#"): idx for idx, name in enumerate(entries[0][1])}
    if not {"sample", "super_pop"} <= set(columns):
        raise ValueError("Population panel lacks sample/super_pop fields")
    labels = {}
    for line_no, fields in entries[1:]:
        if fields[0].startswith("#"):
            continue
        if len(fields) != len(entries[0][1]):
            raise ValueError(f"Population panel has malformed line {line_no}")
        sample, broad = fields[columns["sample"]], fields[columns["super_pop"]]
        if sample in labels or not sample or not broad:
            raise ValueError(f"Population panel has missing or repeated sample at line {line_no}")
        labels[sample] = broad
    if not labels:
        raise ValueError("Population panel has no people")
    return labels


def psam(path):
    source = Path(path).read_text().splitlines()
    cols = source[0].lstrip("#").split()
    ix = cols.index("IID")
    return lines_without_header([r.split()[ix] for r in source[1:] if r.strip()])


def parse_raw(path, expected, id_to_row, matrix, start):
    """Fill a sample-major float32 memmap and verify REF-counted allele headers."""
    with Path(path).open(encoding="ascii") as f:
        header = f.readline().split()
        if len(header) < 7 or header[:2] != ["FID", "IID"] or len(header[6:]) != len(expected):
            raise ValueError(f"Malformed PLINK raw header: {path}")
        actual = []
        for entry in header[6:]:
            variant, sep, allele = entry.rpartition("_")
            if not sep or len(variant.split(":")) != 4 or allele != variant.split(":")[2]:
                raise ValueError(f"REF allele/header mismatch: {entry}")
            actual.append(variant)
        if len(set(actual)) != len(actual) or set(actual) != set(expected):
            raise ValueError("PLINK exported SNPs differ from selected Step 4 marker list")
        seen = set()
        for source in f:
            meta = source.rstrip("\n").split(maxsplit=6)
            if len(meta) != 7 or meta[1] not in id_to_row or meta[1] in seen:
                raise ValueError("Invalid, missing, or repeated training IID in genotype export")
            seen.add(meta[1])
            values = np.fromstring(meta[6].replace("NA", "nan"), sep=" ", dtype=np.float32)
            if len(values) != len(actual) or np.isinf(values).any() or np.any((values < 0) | (values > 2)):
                raise ValueError(f"Bad genotype values for {meta[1]}")
            matrix[id_to_row[meta[1]], start:start + len(actual)] = values
        if seen != set(id_to_row):
            raise ValueError("Missing or extra training people in chromosome raw export")
    return [(v, v.split(":")[2]) for v in actual]


def make_folds(y, groups, n=3):
    folds = list(StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=2026).split(
        np.zeros(len(y)), y, groups))
    for training, validation in folds:
        if set(np.asarray(groups)[training]) & set(np.asarray(groups)[validation]):
            raise ValueError("A related training group crosses CV folds")
        if set(y[training]) != set(y):
            raise ValueError("A CV training fold lacks a broad population group")
    return folds


def cross_validate(X, y, groups):
    folds = make_folds(y, groups)
    choices = [(k, c) for k in (10, 20) for c in (0.1, 1.0)]
    results = {(k, c): [] for k, c in choices}
    for fold, (training, validation) in enumerate(folds, 1):
        imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
        a = imputer.fit_transform(np.array(X[training], dtype=np.float32))
        b = imputer.transform(np.array(X[validation], dtype=np.float32))
        scale = StandardScaler()
        a = scale.fit_transform(a)
        b = scale.transform(b)
        pca = PCA(n_components=20, svd_solver="randomized", random_state=2026)
        a = pca.fit_transform(a)
        b = pca.transform(b)
        for k, c in choices:
            clf = LogisticRegression(C=c, max_iter=1000)
            clf.fit(a[:, :k], y[training])
            predicted = clf.predict(b[:, :k])
            results[(k, c)].append({"fold": fold, "components": k, "C": c,
                                    "accuracy": float(accuracy_score(y[validation], predicted)),
                                    "balanced_accuracy": float(balanced_accuracy_score(y[validation], predicted)),
                                    "validation_samples": len(validation)})
        print(f"CV fold {fold}/{len(folds)} completed", flush=True)
    scores = []
    for (k, c), values in results.items():
        scores.append({"components": k, "C": c,
                       "mean_balanced_accuracy": float(np.mean([v["balanced_accuracy"] for v in values])),
                       "mean_accuracy": float(np.mean([v["accuracy"] for v in values])),
                       "folds": values})
    best = max(scores, key=lambda r: (r["mean_balanced_accuracy"], r["mean_accuracy"],
                                      -r["components"], -r["C"]))
    return best, scores


def fit_model(X, y, k, c):
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=k, svd_solver="randomized", random_state=2026)),
        ("classifier", LogisticRegression(C=c, max_iter=1000)),
    ])
    pipe.fit(X, y)
    return pipe


def synthetic_self_test():
    with tempfile.TemporaryDirectory() as folder:
        panel = Path(folder) / "sample.panel"
        panel.write_text("sample pop super_pop gender\nHG0001 GBR EUR male\nHG0002\tYRI\tAFR\tfemale\n")
        assert read_population(panel) == {"HG0001": "EUR", "HG0002": "AFR"}
    rng = np.random.default_rng(123)
    X = rng.integers(0, 3, size=(90, 28)).astype(np.float32)
    y = np.array(["AFR", "EAS", "EUR"] * 30)
    groups = [f"id{x}" for x in range(90)]
    best, results = cross_validate(X, y, groups)
    assert len(results) == 4 and best["components"] in (10, 20)
    model = fit_model(X, y, best["components"], best["C"])
    assert len(model.predict(X[:5])) == 5
    print("Synthetic model self-test passed")


def train():
    if (COHORT not in ("train2003", "train1998") or BRANCH not in ("with", "without") or
            not all((COHORT_RUN_ID, RUN_2003, RUN_1998, RUN_ID)) or
            any("/" in x for x in (COHORT_RUN_ID, RUN_2003, RUN_1998, RUN_ID))):
        raise ValueError("Set valid TRAIN_COHORT, VISAGE_BRANCH, COHORT_RUN_ID, QC_RUN_2003, QC_RUN_1998, OUTPUT_RUN_ID")
    qc_run = RUN_2003 if COHORT == "train2003" else RUN_1998
    qc_root = f"{STEP4}/results/{qc_run}"
    source = STEP2 if BRANCH == "with" else STEP2B
    dest = f"{ROOT}/results/{RUN_ID}/{COHORT}_{BRANCH}_visage"
    existing = info(f"{dest}/_SUCCESS.json")
    if existing:
        print(f"Output already complete: {dest}. Reuse it; do not retrain into this prefix.")
        return
    with tempfile.TemporaryDirectory(prefix="train-ancestry-") as temp:
        tmp = Path(temp)
        for uri in (f"{STEP2}/_SUCCESS.json", f"{STEP2B}/_SUCCESS.json",
                    f"{STEP2A}/_SUCCESS.json", f"{STEP3}/{COHORT_RUN_ID}/_SUCCESS.json",
                    f"{qc_root}/_SUCCESS.json", f"{source}/_SUCCESS.json"):
            require(uri)
        qc_info = json.loads(get(f"{qc_root}/_SUCCESS.json", tmp / "step4_success.json").read_text())
        qc_summary_path = get(f"{qc_root}/reports/marker_summary.json", tmp / "step4_summary.json")
        if not qc_info.get("complete") or qc_info.get("marker_summary_sha256") != sha(qc_summary_path):
            raise ValueError("Step 4 completion marker does not authenticate its marker summary")
        summary = json.loads(qc_summary_path.read_text())
        if (summary.get("selection_cohort") != COHORT or summary.get("cohort_run_id") != COHORT_RUN_ID
                or summary.get("train_samples_used") != int(COHORT.removeprefix("train"))
                or summary.get("test_samples_used_for_selection") != 0):
            raise ValueError("Wrong Step 4 cohort or Step 4 used held-out data")
        cohort_file = get(f"{STEP3}/{COHORT_RUN_ID}/audit_summary.json", tmp / "cohort.json")
        cohort_summary = json.loads(cohort_file.read_text())
        group_file = get(f"{STEP3}/{COHORT_RUN_ID}/{COHORT}_cv_groups.tsv", tmp / "groups.tsv")
        keep_file = get(f"{STEP3}/{COHORT_RUN_ID}/{COHORT}.keep", tmp / "training.keep")
        people = keep(keep_file)
        groups_by_id = read_groups(group_file)
        population_file = get(PANEL, tmp / "population.panel")
        if sha(population_file) != cohort_summary["input_sha256"]["phase3_population_panel"]:
            raise ValueError("Population labels differ from the Step 3 verified reference")
        labels_by_id = read_population(population_file)
        if (len(people) != int(COHORT.removeprefix("train")) or
                set(people) != set(groups_by_id) or not set(people) <= set(labels_by_id)):
            raise ValueError("Invalid training IDs, relatives groups, or population labels")
        visage_path = get(f"{STEP2A}/excluded_variant_ids.txt", tmp / "visage34.ids")
        visage = set(lines(visage_path))
        if len(visage) != 34:
            raise ValueError("Expected 34 audited VISAGE marker IDs")
        panels, global_marker_ids = [], set()
        for chrom in range(1, 23):
            folder = f"{qc_root}/chromosomes/chr{chrom}"
            selected_path = get(f"{folder}/{BRANCH}_markers.txt", tmp / f"chr{chrom}.{BRANCH}.ids")
            with_path = get(f"{folder}/with_markers.txt", tmp / f"chr{chrom}.with.ids") if BRANCH == "without" else selected_path
            without_path = get(f"{folder}/without_markers.txt", tmp / f"chr{chrom}.without.ids") if BRANCH == "with" else selected_path
            selected, with_ids, without_ids = lines(selected_path), lines(with_path), lines(without_path)
            if set(with_ids) - set(without_ids) != set(with_ids) & visage or not set(without_ids) <= set(with_ids):
                raise ValueError(f"Step 4 with/without panels disagree for chr{chrom}")
            if len(set(selected) & global_marker_ids) or any(not x.startswith(f"{chrom}:") for x in selected):
                raise ValueError(f"Duplicate or misplaced SNPs at chr{chrom}")
            global_marker_ids.update(selected)
            chrom_report = json.loads(get(f"{folder}/summary.json", tmp / f"chr{chrom}.summary.json").read_text())
            if len(selected) != chrom_report[f"ld_pruned_{BRANCH}"] or sha(selected_path) != chrom_report[f"{BRANCH}_sha256"]:
                raise ValueError(f"SNP count/hash differs from Step 4 chr{chrom} report")
            panels.append((chrom, selected, selected_path))
        count = sum(len(panel[1]) for panel in panels)
        if count != summary["counts"][f"ld_pruned_{BRANCH}"] or count < 20:
            raise ValueError("Genome-wide marker count differs from Step 4 report")
        matrix_path = tmp / "training.float32.dat"
        X = np.memmap(matrix_path, mode="w+", dtype=np.float32, shape=(len(people), count))
        row_ids = {sample: idx for idx, sample in enumerate(people)}
        order = []
        offset = 0
        for chrom, selected, marker_file in panels:
            chrom_dir = tmp / f"chr{chrom}"
            chrom_dir.mkdir()
            prefix = chrom_dir / f"chr{chrom}"
            input_prefix = f"{source}/chromosomes/chr{chrom}/train/chr{chrom}"
            for extension in ("pgen", "pvar.zst", "psam"):
                get(f"{input_prefix}.{extension}", Path(f"{prefix}.{extension}"))
            upstream_people = psam(Path(f"{prefix}.psam"))
            # Both input branches contain the original 2,003 training samples.
            if len(upstream_people) != 2003 or not set(people) <= set(upstream_people):
                raise ValueError(f"Unexpected upstream training sample IDs at chr{chrom}")
            export = chrom_dir / "selected"
            command(["plink2", "--pfile", str(prefix), "vzs", "--keep", str(keep_file),
                     "--extract", str(marker_file), "--export", "A", "--out", str(export)])
            order += parse_raw(Path(f"{export}.raw"), selected, row_ids, X, offset)
            offset += len(selected)
            shutil.rmtree(chrom_dir)
            print(f"chr{chrom}: loaded {len(selected)} training SNPs ({offset}/{count})", flush=True)
        X.flush()
        if len(order) != count or len(set(v for v, a in order)) != count or np.isnan(X).all(axis=0).any():
            raise ValueError("Genotype matrix has missing or duplicate columns")
        y = np.array([labels_by_id[s] for s in people])
        cv_groups = np.array([groups_by_id[s] for s in people])
        if len(set(y)) != 5:
            raise ValueError("Expected five broad reference groups")
        print("Running three-fold grouped training-only cross-validation", flush=True)
        best, cv = cross_validate(X, y, cv_groups)
        print("Fitting final training-only model", flush=True)
        model = fit_model(X, y, best["components"], best["C"])
        bundle = {"pipeline": model, "markers": order, "classes": sorted(set(y)),
                  "training_cohort": COHORT, "visage_branch": BRANCH,
                  "step4_run": qc_run, "step4_summary_sha256": sha(qc_summary_path),
                  "labels": "1000 Genomes Phase 3 super_pop",
                  "genotype_encoding": "PLINK 2 --export A; counted allele is REF"}
        model_path = tmp / "model.joblib"
        joblib.dump(bundle, model_path, compress=3)
        marker_path = tmp / "selected_markers.tsv"
        with marker_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["variant_id", "counted_allele"])
            writer.writerows(order)
        cv_path = tmp / "cv_scores.json"
        cv_path.write_text(json.dumps(cv, indent=2, sort_keys=True) + "\n")
        report = {"status": "trained; held-out test data not used", "cohort": COHORT, "branch": BRANCH,
                  "training_people": len(people), "test_people_used": 0, "selected_markers": count,
                  "visage_markers_in_panel": sum(v in visage for v, a in order),
                  "cv_method": "three-fold StratifiedGroupKFold; preprocessing and PCA refitted per fold",
                  "cv_limitation": "Step 4 SNP QC and pruning used the whole training cohort, including CV validation genotypes; the independent test set is untouched.",
                  "best_parameters": {"pca_components": best["components"], "logistic_C": best["C"]},
                  "cv_balanced_accuracy": best["mean_balanced_accuracy"], "cv_accuracy": best["mean_accuracy"],
                  "step4_summary_sha256": sha(qc_summary_path), "cohort_run_id": COHORT_RUN_ID,
                  "training_keep_sha256": sha(keep_file), "cv_groups_sha256": sha(group_file),
                  "population_panel_sha256": sha(population_file), "sklearn_version": sklearn.__version__,
                  "genotype_source": source, "no_heldout_genotypes_loaded": True}
        report_path = tmp / "training_summary.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        snapshot = tmp / "input_snapshot.json"
        snapshot.write_text(json.dumps({"qc_root": qc_root, "cohort": COHORT,
                                        "branch": BRANCH, "model_run": RUN_ID,
                                        "input_sha256": {"step4_summary": sha(qc_summary_path),
                                                         "training_keep": sha(keep_file),
                                                         "cv_groups": sha(group_file),
                                                         "visage_ids": sha(visage_path),
                                                         "population_panel": sha(population_file)}}, indent=2) + "\n")
        files = (model_path, marker_path, cv_path, report_path, snapshot)
        for path in files:
            put(path, f"{dest}/{path.name}")
        done = tmp / "_SUCCESS.json"
        done.write_text(json.dumps({"complete": True, "training_summary_sha256": sha(report_path),
                                    "model_sha256": sha(model_path),
                                    "marker_file_sha256": sha(marker_path)}) + "\n")
        put(done, f"{dest}/_SUCCESS.json")
        print(f"Trained {COHORT}/{BRANCH}: {count} SNPs, best {best['components']} PCs; results: {dest}/", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        synthetic_self_test()
    elif args.dry_run:
        print(json.dumps({"cohort": COHORT, "branch": BRANCH, "qc_run_2003": RUN_2003,
                          "qc_run_1998": RUN_1998, "test_genotypes_loaded": False}, indent=2))
    else:
        train()


if __name__ == "__main__":
    main()
