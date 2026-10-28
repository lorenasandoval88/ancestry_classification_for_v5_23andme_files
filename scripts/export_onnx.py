"""Reference exporter: fitted scikit-learn ancestry pipeline -> browser release bundle.

Defines the artifact contract that src/sdk/classifier.js consumes. The cloud Step 6
exporter should produce the same files (names, shapes, output tensors):

    model/model.onnx          hand-built graph: NaN-fill -> scale -> PCA -> logistic -> softmax
    model/markers.json        model marker order ("chr:pos:ref:alt"), counted allele = REF
    model/metadata.json       classes, sizes, coverage rules, checksums, parity result
    model/pca_reference.json  (optional) PC scores of train / test / v5 people for the PCA plot

Graph I/O
    input   dosages        float32 [N, n_markers]   REF-allele count 0/1/2, NaN = missing
    output  probabilities  float32 [N, n_classes]   softmax over classes (metadata.classes order)
    output  pc_scores      float32 [N, n_components] PCA coordinates (for plotting)
    output  label_index    int64   [N]              argmax of probabilities

Missing values are imputed *inside* the graph with the fitted SimpleImputer means, so the
browser sends NaN for markers that are absent, no-call, or allele-incompatible.

Usage (Python env with the trained model; sklearn 1.7.x):

    pip install onnx onnxruntime numpy joblib
    python scripts/export_onnx.py \
        --model  5_fit_pca_and_classifier/model.joblib \
        --out    model \
        [--markers selected_markers.tsv]              # cross-check vs. bundle marker list
        [--cohort data/reference_cohort.json]         # labels for the PCA reference
        [--train-matrix X_train.npy --train-ids train2003.txt]   # float32 [2003, n], NaN = missing
        [--test-matrix  X_test.npy  --test-ids  test501.txt]
        [--v5-matrix    X_v5.npy    --v5-ids    v5_files.txt]
        [--run-id my23-20261028-153511] [--model-version 1.0.1]

The parity check compares ONNX Runtime against the scikit-learn pipeline on the test
matrix when given (else on synthetic rows): identical labels and max |dp| <= 1e-4.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np

PROBA_TOL = 1e-4


# --------------------------------------------------------------------------- helpers

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_ids(path):
    ids = [x.strip() for x in Path(path).read_text().splitlines()]
    ids = [x for x in ids if x and not x.startswith("#")]
    if len(set(ids)) != len(ids):
        sys.exit(f"duplicate IDs in {path}")
    return ids


def unwrap(obj):
    """train_ancestry.py saves {"pipeline", "markers", "classes", ...}; also accept a bare Pipeline."""
    if isinstance(obj, dict):
        return obj["pipeline"], obj.get("markers"), obj
    return obj, None, {}


def get_step(pipe, cls_name):
    for _, est in pipe.steps:
        if type(est).__name__ == cls_name:
            return est
    return None


# --------------------------------------------------------------------------- ONNX graph

def build_onnx(means, scale_mean, scale_std, pca_mean, components, coef, intercept, opset=17):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    n = means.shape[0]
    k, c = components.shape[0], coef.shape[0]
    f32 = lambda a, name: numpy_helper.from_array(np.ascontiguousarray(a, dtype=np.float32), name)

    inits = [
        f32(means, "train_means"),
        f32(scale_mean, "scale_mean"),
        f32(scale_std, "scale_std"),
        f32(pca_mean, "pca_mean"),
        f32(components.T, "pca_components_T"),  # [n, k]
        f32(coef.T, "lr_coef_T"),                # [k, c]
        f32(intercept, "lr_intercept"),          # [c]
    ]
    nodes = [
        helper.make_node("IsNaN", ["dosages"], ["is_missing"], name="is_missing"),
        helper.make_node("Where", ["is_missing", "train_means", "dosages"], ["filled"], name="impute_mean"),
        helper.make_node("Sub", ["filled", "scale_mean"], ["centered"], name="scale_center"),
        helper.make_node("Div", ["centered", "scale_std"], ["scaled"], name="scale_divide"),
        helper.make_node("Sub", ["scaled", "pca_mean"], ["pca_centered"], name="pca_center"),
        helper.make_node("MatMul", ["pca_centered", "pca_components_T"], ["pc_scores"], name="pca_project"),
        helper.make_node("Gemm", ["pc_scores", "lr_coef_T", "lr_intercept"], ["logits"], name="logistic"),
        helper.make_node("Softmax", ["logits"], ["probabilities"], axis=1, name="softmax"),
        helper.make_node("ArgMax", ["probabilities"], ["label_index"], axis=1, keepdims=0, name="argmax"),
    ]
    graph = helper.make_graph(
        nodes, "ancestry_pca_logistic",
        inputs=[helper.make_tensor_value_info("dosages", TensorProto.FLOAT, ["N", n])],
        outputs=[
            helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, ["N", c]),
            helper.make_tensor_value_info("pc_scores", TensorProto.FLOAT, ["N", k]),
            helper.make_tensor_value_info("label_index", TensorProto.INT64, ["N"]),
        ],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)], producer_name="ancestry-export")
    model.ir_version = 8  # readable by onnxruntime-web 1.17+
    onnx.checker.check_model(model)
    return model


# --------------------------------------------------------------------------- PCA reference

def pca_block(pipe, X, ids, cohort, predicted_labels=None, classes=None):
    pre = pipe[:-1]
    scores = pre.transform(X).astype(np.float32)
    block = {"ids": ids, "scores": np.round(scores, 4).tolist()}
    if cohort:
        block["labels"] = [cohort.get(i, {}).get("superPop") for i in ids]
        block["pops"] = [cohort.get(i, {}).get("pop") for i in ids]
    if predicted_labels is not None:
        block["predicted"] = [classes[int(j)] for j in predicted_labels]
    return block


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="model")
    ap.add_argument("--markers", help="selected_markers.tsv to cross-check against the bundle")
    ap.add_argument("--cohort", help="data/reference_cohort.json (labels for PCA reference)")
    ap.add_argument("--train-matrix"); ap.add_argument("--train-ids")
    ap.add_argument("--test-matrix");  ap.add_argument("--test-ids")
    ap.add_argument("--v5-matrix");    ap.add_argument("--v5-ids")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--model-version", default=None)
    ap.add_argument("--min-called-fraction", type=float, default=0.90)
    ap.add_argument("--max-incompatible-fraction", type=float, default=0.01)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    pipe, bundle_markers, bundle = unwrap(joblib.load(args.model))
    imputer = get_step(pipe, "SimpleImputer")
    scaler = get_step(pipe, "StandardScaler")
    pca = get_step(pipe, "PCA")
    clf = pipe.steps[-1][1]
    if imputer is None or pca is None or type(clf).__name__ != "LogisticRegression":
        sys.exit(f"Unexpected pipeline steps: {[type(e).__name__ for _, e in pipe.steps]}")
    if getattr(imputer, "strategy", "mean") != "mean":
        sys.exit("SimpleImputer strategy must be 'mean'")

    means = np.asarray(imputer.statistics_, dtype=np.float32)
    n = means.shape[0]
    scale_mean = np.asarray(scaler.mean_, np.float32) if scaler is not None else np.zeros(n, np.float32)
    scale_std = np.asarray(scaler.scale_, np.float32) if scaler is not None else np.ones(n, np.float32)
    components = np.asarray(pca.components_, np.float32)
    pca_mean = np.asarray(pca.mean_, np.float32)
    coef = np.asarray(clf.coef_, np.float32)
    intercept = np.asarray(clf.intercept_, np.float32)
    classes = [str(c) for c in clf.classes_]
    if coef.shape[0] != len(classes):
        sys.exit("Binary/OvR logistic layout not supported; expected one coefficient row per class (multinomial)")

    # Marker list ----------------------------------------------------------------
    if bundle_markers is not None:
        markers = [str(v) for v, _ in bundle_markers]
        counted = [str(a) for _, a in bundle_markers]
    elif args.markers:
        rows = [l.split("\t") for l in Path(args.markers).read_text().splitlines()[1:] if l.strip()]
        markers, counted = [r[0] for r in rows], [r[1] for r in rows]
    else:
        sys.exit("No marker list in bundle; pass --markers")
    if args.markers and bundle_markers is not None:
        rows = [l.split("\t") for l in Path(args.markers).read_text().splitlines()[1:] if l.strip()]
        if [r[0] for r in rows] != markers:
            sys.exit("--markers order differs from the marker list inside the joblib bundle")
    if len(markers) != n:
        sys.exit(f"{len(markers)} markers but pipeline has {n} features")
    bad = [m for m, a in zip(markers, counted) if m.split(":")[2] != a]
    if bad:
        sys.exit(f"{len(bad)} markers whose counted allele is not REF (e.g. {bad[0]}); browser encoder assumes REF")

    # ONNX -----------------------------------------------------------------------
    import onnx
    model = build_onnx(means, scale_mean, scale_std, pca_mean, components, coef, intercept, args.opset)
    onnx_path = out / "model.onnx"
    onnx.save(model, onnx_path)

    markers_path = out / "markers.json"
    markers_path.write_text(json.dumps({"n_markers": n, "counted_allele": "REF", "format": "chr:pos:ref:alt", "markers": markers}, separators=(",", ":")))

    # Parity ---------------------------------------------------------------------
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    cohort = None
    if args.cohort:
        cohort = {r["sample"]: r for r in json.loads(Path(args.cohort).read_text())["rows"]}

    def load_matrix(p):
        X = np.load(p).astype(np.float32)
        if X.shape[1] != n:
            sys.exit(f"{p}: {X.shape[1]} columns but model has {n} markers")
        return X

    X_test = load_matrix(args.test_matrix) if args.test_matrix else None
    if X_test is not None:
        X_par, source = X_test, "test matrix"
    else:
        rng = np.random.default_rng(0)
        X_par = np.clip(np.round(means + rng.normal(0, 0.6, size=(32, n))), 0, 2).astype(np.float32)
        X_par[rng.random(X_par.shape) < 0.02] = np.nan
        source = "synthetic (2% NaN)"
    proba, pcs, labels = sess.run(None, {"dosages": X_par})
    ref_proba = pipe.predict_proba(X_par).astype(np.float32)
    ref_pcs = pipe[:-1].transform(X_par).astype(np.float32)
    ref_labels = np.array([classes.index(str(c)) for c in pipe.predict(X_par)])
    parity = {
        "source": source, "rows": int(X_par.shape[0]),
        "label_agreement": float(np.mean(labels == ref_labels)),
        "max_abs_proba_diff": float(np.max(np.abs(proba - ref_proba))),
        "max_abs_pc_diff": float(np.max(np.abs(pcs - ref_pcs))),
        "tolerance": PROBA_TOL,
    }
    parity["passed"] = parity["label_agreement"] == 1.0 and parity["max_abs_proba_diff"] <= PROBA_TOL
    print("parity:", json.dumps(parity))
    if not parity["passed"]:
        print("WARNING: ONNX differs from scikit-learn beyond tolerance", file=sys.stderr)

    # PCA reference ------------------------------------------------------------
    pca_ref = {"n_components": int(components.shape[0]),
               "explained_variance_ratio": [float(v) for v in getattr(pca, "explained_variance_ratio_", [])],
               "groups": {}}
    for key, mpath, ipath in (("train", args.train_matrix, args.train_ids), ("test", args.test_matrix, args.test_ids), ("v5_files", args.v5_matrix, args.v5_ids)):
        if not (mpath and ipath):
            continue
        X = X_test if (key == "test" and X_test is not None) else load_matrix(mpath)
        ids = read_ids(ipath)
        if len(ids) != X.shape[0]:
            sys.exit(f"{key}: {len(ids)} ids but matrix has {X.shape[0]} rows")
        pred = sess.run(["label_index"], {"dosages": X})[0] if key != "train" else None
        pca_ref["groups"][key] = pca_block(pipe, X, ids, cohort, pred, classes)
        print(f"pca_reference: {key} {X.shape[0]} people")
    pca_path = None
    if pca_ref["groups"]:
        pca_path = out / "pca_reference.json"
        pca_path.write_text(json.dumps(pca_ref, separators=(",", ":")))

    # Metadata -------------------------------------------------------------------
    import sklearn
    artifacts = [onnx_path, markers_path] + ([pca_path] if pca_path else [])
    meta = {
        "model_version": args.model_version or bundle.get("model_version"),
        "run_id": args.run_id or bundle.get("run_id"),
        "training_cohort": bundle.get("training_cohort"),
        "labels": bundle.get("labels", "1000 Genomes Phase 3 super_pop"),
        "classes": classes,
        "n_markers": n,
        "n_components": int(components.shape[0]),
        "logistic_C": float(getattr(clf, "C", float("nan"))),
        "genotype_encoding": "count of REF allele (0,1,2); NaN = missing, imputed in-graph with training means",
        "input": {"name": "dosages", "dtype": "float32", "shape": ["N", n]},
        "outputs": {"probabilities": ["N", len(classes)], "pc_scores": ["N", int(components.shape[0])], "label_index": ["N"]},
        "coverage_rules": {"min_called_fraction": args.min_called_fraction, "max_incompatible_fraction": args.max_incompatible_fraction,
                           "incompatible_denominator": "matched coordinates (called + incompatible)"},
        "required_build": "GRCh37",
        "supported_chips": ["v5"],
        "sklearn_version": sklearn.__version__,
        "onnx_opset": args.opset,
        "source_model_sha256": sha256(Path(args.model)),
        "artifacts": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in artifacts},
        "parity": parity,
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k not in ("artifacts",)}, indent=2))
    print("wrote", ", ".join(p.name for p in artifacts + [out / "metadata.json"]))


if __name__ == "__main__":
    main()
