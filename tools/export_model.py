"""Export an already-fitted, trusted sklearn ancestry pipeline; never fits data."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


GROUPS = {"AFR", "AMR", "EAS", "EUR", "SAS"}


def export_model(pipeline, markers, version, class_map=None, min_match=0.95, min_call=0.95):
    if not isinstance(pipeline, Pipeline) or len(pipeline.steps) != 4:
        raise ValueError("Expected fitted Pipeline: SimpleImputer, StandardScaler, PCA, LogisticRegression")
    imputer, scaler, pca, classifier = [step for _, step in pipeline.steps]
    for step, expected in zip(
        [imputer, scaler, pca, classifier],
        [SimpleImputer, StandardScaler, PCA, LogisticRegression],
    ):
        if type(step) is not expected:
            raise ValueError(f"Unsupported pipeline step: expected {expected.__name__}")
    n = len(markers)
    if not n or any(step.n_features_in_ != n for step in [imputer, scaler, pca]):
        raise ValueError("Marker order/length must exactly match fitted input features")
    if hasattr(pipeline, "feature_names_in_"):
        if list(pipeline.feature_names_in_) != [marker["id"] for marker in markers]:
            raise ValueError("Marker IDs must match pipeline feature_names_in_ in order")
    if imputer.strategy != "mean" or imputer.add_indicator or not np.isnan(imputer.missing_values):
        raise ValueError("Only mean imputation of NaN without missing indicators is supported")
    if np.asarray(imputer.statistics_).shape != (n,) or not np.isfinite(imputer.statistics_).all():
        raise ValueError("All input SNPs must have fitted finite imputation means")
    if np.any((imputer.statistics_ < 0) | (imputer.statistics_ > 2)):
        raise ValueError("Imputation statistics must be effect-allele dosages between 0 and 2")
    if pca.components_.shape != (10, n) or classifier.coef_.shape != (5, 10):
        raise ValueError("Expected 10 PCA components and five logistic classes")
    classes = [str(x) for x in classifier.classes_]
    if class_map is not None:
        classes = [class_map[x] for x in classes]
    if len(set(classes)) != 5 or set(classes) != GROUPS:
        raise ValueError("Supply the saved class mapping to AFR, AMR, EAS, EUR and SAS")
    multi_class = getattr(classifier, "multi_class", "auto")
    if multi_class == "ovr" or classifier.solver == "liblinear":
        mode = "ovr"
    elif multi_class in ("auto", "deprecated", "multinomial"):
        mode = "softmax"
    else:
        raise ValueError("Unsupported logistic probability convention")
    if not version or not (0 < min_match <= 1 and 0 < min_call <= 1):
        raise ValueError("A model version and positive quality thresholds <= 1 are required")
    seen = set()
    for marker in markers:
        if not isinstance(marker.get("id"), str) or not marker["id"] or marker["id"] in seen:
            raise ValueError("Marker IDs must be unique nonempty strings")
        seen.add(marker["id"])
        if (marker.get("chromosome") not in {str(x) for x in range(1, 23)}
                or type(marker.get("position")) is not int or marker["position"] <= 0
                or marker.get("otherAllele") not in {"A", "C", "G", "T"}
                or marker.get("effectAllele") not in {"A", "C", "G", "T"}
                or marker["otherAllele"] == marker["effectAllele"]):
            raise ValueError("Expected biallelic autosomal GRCh37 marker definitions")
    artifact = {
        "schemaVersion": 1, "modelVersion": version, "chip": "v5", "build": "GRCh37",
        "markerCount": n, "markers": markers,
        "imputerMeans": imputer.statistics_.tolist(),
        "scalerMean": scaler.mean_.tolist() if scaler.with_mean else [0.0] * n,
        "scalerScale": scaler.scale_.tolist() if scaler.with_std else [1.0] * n,
        "pcaMean": pca.mean_.tolist(), "pcaComponents": pca.components_.tolist(),
        "pcaScale": np.maximum(np.sqrt(pca.explained_variance_), np.finfo(float).eps).tolist()
        if pca.whiten else [1.0] * 10,
        "classes": classes, "coefficients": classifier.coef_.tolist(),
        "intercepts": classifier.intercept_.tolist(), "probabilityMode": mode,
        "quality": {"minMatchRate": min_match, "minCallRate": min_call},
    }
    # Reject NaN/Infinity rather than writing JSON that JavaScript cannot load.
    json.dumps(artifact, allow_nan=False)
    return artifact


def parity_cases(pipeline, artifact):
    """Synthetic dosage-only probes; no participant data is exported."""
    n = artifact["markerCount"]
    rng = np.random.default_rng(42)
    dosages = rng.integers(0, 3, size=(8, n)).astype(float)
    dosages[0] = 0
    dosages[1] = 1
    dosages[2] = 2
    allowed_missing = int(n * (1 - artifact["quality"]["minCallRate"]) / 2)
    if allowed_missing:
        dosages[3, :allowed_missing] = np.nan
    scores = pipeline.predict_proba(dosages)
    predictions = pipeline.predict(dosages)
    original_classes = list(pipeline.steps[-1][1].classes_)
    return {
        "modelVersion": artifact["modelVersion"],
        "cases": [
            {
                "dosages": [None if np.isnan(x) else int(x) for x in row],
                "scores": dict(zip(artifact["classes"], probability.tolist())),
                "predictedGroup": artifact["classes"][original_classes.index(prediction)],
            }
            for row, probability, prediction in zip(dosages, scores, predictions)
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", required=True, help="Trusted local joblib Pipeline (pickle can execute code)")
    parser.add_argument("--markers", required=True, help="Ordered marker JSON; same dosage orientation as training")
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--parity-output", required=True, help="Synthetic Python reference predictions for JS verification")
    parser.add_argument("--class-map", help="JSON file mapping saved class labels to the five reference groups")
    parser.add_argument("--expected-markers", type=int, default=335632)
    parser.add_argument("--min-match-rate", type=float, default=0.95)
    parser.add_argument("--min-call-rate", type=float, default=0.95)
    args = parser.parse_args()
    markers = json.loads(Path(args.markers).read_text())
    if len(markers) != args.expected_markers:
        parser.error(f"Expected {args.expected_markers} markers, got {len(markers)}")
    class_map = json.loads(Path(args.class_map).read_text()) if args.class_map else None
    pipeline = joblib.load(args.pipeline)
    artifact = export_model(pipeline, markers, args.model_version, class_map,
                            args.min_match_rate, args.min_call_rate)
    parity = parity_cases(pipeline, artifact)
    Path(args.output).write_text(json.dumps(artifact, allow_nan=False, separators=(",", ":")))
    Path(args.parity_output).write_text(json.dumps(parity, allow_nan=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
