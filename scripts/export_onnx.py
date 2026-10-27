"""Export the trained ancestry pipeline (cloudModel/model.joblib) for browser inference.

Imputation of missing markers is applied *upstream* in JavaScript using the saved
training means (the fitted SimpleImputer statistics), so the ONNX graph contains only
StandardScaler -> PCA -> LogisticRegression.

model.joblib is the bundle written by train_ancestry.py:
    {"pipeline": Pipeline(imputer, scale, pca, classifier), "markers": [(variant_id, counted_allele)], ...}

Run in the Python environment that trained the model (sklearn 1.7.2):

    pip install skl2onnx onnx onnxruntime numpy pandas joblib
    python scripts/export_onnx.py \
        --model data/training_model/cloudModel/model.joblib \
        --out model \
        [--markers data/training_model/cloudModel/*_selected_markers.tsv]  # cross-check against the bundle
        [--parity X_test.npy]                                               # optional (n, 335632) float32 dosages

Outputs (model/):
    ancestry_pipeline.onnx   StandardScaler -> PCA -> LogisticRegression (zipmap disabled)
    markers.tsv              variant_id<TAB>counted_allele, in model feature order
    means.f32                float32 little-endian training mean per marker
    model_meta.json          classes, n_features, steps, sha256 of every artifact
"""

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def unwrap(model):
    """train_ancestry.py saves {"pipeline": Pipeline, "markers": [(variant_id, counted_allele)], "classes": [...]}."""
    if isinstance(model, dict):
        return model["pipeline"], model.get("markers"), model
    return model, None, {}


def split_pipeline(model):
    """Return (means or None, Pipeline of post-imputation steps, step names)."""
    steps = list(model.steps) if isinstance(model, Pipeline) else [("model", model)]
    means = None
    kept = []
    for name, est in steps:
        cls = type(est).__name__
        if cls == "SimpleImputer":
            if getattr(est, "strategy", "mean") != "mean":
                sys.exit(f"Imputer strategy '{est.strategy}' is not 'mean'; export cannot move it upstream.")
            means = np.asarray(est.statistics_, dtype=np.float32)
            continue
        kept.append((name, est))
    return means, Pipeline(kept), [type(e).__name__ for _, e in kept]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--markers", help="selected_markers.tsv (glob allowed); defaults to the marker list inside the joblib bundle")
    ap.add_argument("--out", default="model")
    ap.add_argument("--means", help=".npy/.tsv/.csv of training means, if the joblib has no SimpleImputer")
    ap.add_argument("--parity", help="optional .npy (n, n_features) float32 dosage matrix for a parity check")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    bundle_or_pipe = joblib.load(args.model)
    model, bundle_markers, bundle = unwrap(bundle_or_pipe)
    means, pipe, step_names = split_pipeline(model)
    print("pipeline steps kept for ONNX:", step_names)

    if args.markers:
        marker_files = sorted(glob.glob(args.markers))
        if not marker_files:
            sys.exit(f"No marker file matches {args.markers}")
        markers = pd.read_csv(marker_files[0], sep="\t", dtype=str)
        if not {"variant_id", "counted_allele"} <= set(markers.columns):
            sys.exit("markers TSV must have columns variant_id and counted_allele")
        if bundle_markers is not None:
            inside = [(str(v), str(a)) for v, a in bundle_markers]
            if inside != list(map(tuple, markers[["variant_id", "counted_allele"]].to_numpy())):
                sys.exit("markers TSV does not match the marker order stored inside the joblib bundle")
    elif bundle_markers is not None:
        markers = pd.DataFrame(bundle_markers, columns=["variant_id", "counted_allele"], dtype=str)
    else:
        sys.exit("No marker list: pass --markers")
    bad = markers[markers.apply(lambda r: r["variant_id"].split(":")[2] != r["counted_allele"], axis=1)]
    if len(bad):
        sys.exit(f"{len(bad)} markers whose counted allele is not the REF allele; the JS encoder assumes REF counting")
    n_features = len(markers)
    print("markers:", n_features)

    if means is None:
        if not args.means:
            sys.exit("joblib has no SimpleImputer; pass --means with the training means used upstream.")
        p = Path(args.means)
        means = np.load(p) if p.suffix == ".npy" else pd.read_csv(p, sep=None, engine="python", header=None).to_numpy().ravel()
        means = np.asarray(means, dtype=np.float32)
    if means.shape[0] != n_features:
        sys.exit(f"means has {means.shape[0]} values but marker panel has {n_features}")

    first = pipe.steps[0][1]
    n_in = getattr(first, "n_features_in_", None)
    if n_in is not None and n_in != n_features:
        sys.exit(f"pipeline expects {n_in} features but marker panel has {n_features}")

    # --- ONNX export -------------------------------------------------------
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    clf = pipe.steps[-1][1]
    onx = convert_sklearn(
        pipe,
        initial_types=[("dosages", FloatTensorType([None, n_features]))],
        options={id(clf): {"zipmap": False}},
        target_opset=args.opset,
    )
    onnx_path = out / "ancestry_pipeline.onnx"
    onnx_path.write_bytes(onx.SerializeToString())

    markers_path = out / "markers.tsv"
    markers[["variant_id", "counted_allele"]].to_csv(markers_path, sep="\t", index=False)

    means_path = out / "means.f32"
    means.astype("<f4").tofile(means_path)

    # --- Parity check ------------------------------------------------------
    parity = None
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        in_name = sess.get_inputs()[0].name
        if args.parity:
            X = np.load(args.parity).astype(np.float32)
            X = np.where(np.isnan(X), means, X)
        else:
            rng = np.random.default_rng(0)
            X = np.clip(means + rng.normal(0, 0.5, size=(16, n_features)), 0, 2).astype(np.float32)
        label, proba = sess.run(None, {in_name: X})
        ref_label = pipe.predict(X)
        ref_proba = pipe.predict_proba(X)
        max_abs = float(np.max(np.abs(proba - ref_proba)))
        agree = float(np.mean(np.asarray(label).ravel() == np.asarray(ref_label).ravel()))
        parity = {"rows": int(X.shape[0]), "label_agreement": agree, "max_abs_proba_diff": max_abs,
                  "source": "parity file" if args.parity else "synthetic around means"}
        print("parity:", parity)
        if agree < 1.0 or max_abs > 1e-3:
            print("WARNING: ONNX output differs from scikit-learn beyond tolerance", file=sys.stderr)
    except ImportError:
        print("onnxruntime not installed; skipping parity check")

    meta = {
        "classes": [str(c) for c in clf.classes_],
        "n_features": n_features,
        "steps": step_names,
        "input_name": "dosages",
        "imputation": "upstream: replace missing/incompatible markers with means.f32 (SimpleImputer mean statistics)",
        "dosage": "count of the REF allele (counted_allele) in the 23andMe genotype: 0, 1 or 2",
        "training_cohort": bundle.get("training_cohort"),
        "visage_branch": bundle.get("visage_branch"),
        "labels": bundle.get("labels"),
        "sklearn_version": __import__("sklearn").__version__,
        "opset": args.opset,
        "source_model": Path(args.model).name,
        "source_model_sha256": sha256(Path(args.model)),
        "artifacts": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in (onnx_path, markers_path, means_path)},
        "parity": parity,
    }
    (out / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
