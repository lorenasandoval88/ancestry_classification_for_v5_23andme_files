import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tools.export_model import export_model, parity_cases


ROOT = Path(__file__).resolve().parents[1]


class ExportTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.x = rng.integers(0, 3, size=(100, 12)).astype(float)
        self.x[0, 0] = np.nan
        self.y = np.array(["AFR", "AMR", "EAS", "EUR", "SAS"] * 20)
        self.markers = [
            {"id": f"rs{i}", "chromosome": "1", "position": i + 1,
             "otherAllele": "A", "effectAllele": "G"}
            for i in range(12)
        ]

    def pipeline(self, whiten=False, with_mean=True, with_std=True):
        return make_pipeline(SimpleImputer(strategy="mean"),
                             StandardScaler(with_mean=with_mean, with_std=with_std),
                             PCA(n_components=10, whiten=whiten),
                             LogisticRegression(max_iter=1000)).fit(self.x, self.y)

    def test_python_javascript_parity(self):
        for whiten, center, scale in [(False, True, True), (True, True, True),
                                       (False, False, False), (True, False, True)]:
            with self.subTest(whiten=whiten, center=center, scale=scale):
                pipeline = self.pipeline(whiten, center, scale)
                artifact = export_model(pipeline, self.markers, "synthetic-only",
                                        min_match=0.8, min_call=0.8)
                reference = parity_cases(pipeline, artifact)
                self.assertIn(None, reference["cases"][3]["dosages"])
                with tempfile.TemporaryDirectory(dir="/tmp") as directory:
                    model_path = Path(directory) / "model.json"
                    parity_path = Path(directory) / "parity.json"
                    model_path.write_text(json.dumps(artifact, allow_nan=False))
                    parity_path.write_text(json.dumps(reference, allow_nan=False))
                    result = subprocess.run(
                        ["node", str(ROOT / "tools/verify_parity.mjs"),
                         str(model_path), str(parity_path)],
                        capture_output=True, text=True, check=True,
                    )
                    self.assertIn("cases: 8", result.stdout)

    def test_export_cli(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            folder = Path(directory)
            joblib.dump(self.pipeline(), folder / "pipeline.joblib")
            (folder / "markers.json").write_text(json.dumps(self.markers))
            import sys
            result = subprocess.run(
                [sys.executable, str(ROOT / "tools/export_model.py"),
                 "--pipeline", str(folder / "pipeline.joblib"),
                 "--markers", str(folder / "markers.json"),
                 "--model-version", "cli-test", "--expected-markers", "12",
                 "--output", str(folder / "model.json"),
                 "--parity-output", str(folder / "parity.json")],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            artifact = json.loads((folder / "model.json").read_text())
            self.assertEqual(artifact["markerCount"], 12)
            self.assertEqual(artifact["modelVersion"], "cli-test")

    def test_rejects_unsupported_steps_and_wrong_feature_count(self):
        pipeline = self.pipeline()
        with self.assertRaises(ValueError):
            export_model(pipeline, self.markers[:-1], "bad")
        with self.assertRaises(ValueError):
            export_model(make_pipeline(pipeline.steps[-1][1]), self.markers, "bad")
        pipeline.steps[0][1].strategy = "median"
        with self.assertRaises(ValueError):
            export_model(pipeline, self.markers, "bad")

    def test_class_mapping_preserves_saved_class_order(self):
        pipeline = self.pipeline()
        pipeline.steps[-1][1].classes_ = np.arange(5)
        mapping = dict(zip(map(str, range(5)), ["SAS", "EUR", "EAS", "AMR", "AFR"]))
        with self.assertRaises(ValueError):
            export_model(pipeline, self.markers, "bad")
        artifact = export_model(pipeline, self.markers, "mapped", mapping)
        self.assertEqual(artifact["classes"], list(mapping.values()))

    def test_rejects_invalid_markers_and_nonfinite_statistics(self):
        pipeline = self.pipeline()
        self.markers[0]["effectAllele"] = "A"
        with self.assertRaises(ValueError):
            export_model(pipeline, self.markers, "bad")
        self.markers[0]["effectAllele"] = "G"
        pipeline.steps[0][1].statistics_[0] = np.nan
        with self.assertRaises(ValueError):
            export_model(pipeline, self.markers, "bad")


if __name__ == "__main__":
    unittest.main()
