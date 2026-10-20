"""Synthetic tests for direct Bootstrap evaluation of eval.py prediction CSVs."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from statistical_analysis.bootstrap_predictions import (
    CLASSES,
    METRIC_COLUMNS,
    MetricEvaluator,
    compute_queue,
    load_prediction_csv,
    make_bootstrap_weights,
    run_analysis,
)


def write_prediction_csv(path, y, probabilities, ids=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y, dtype=int)
    p = np.asarray(probabilities, dtype=float)  # internal order: FTA, WI, EA, MI
    ids = list(ids or [f"patient_{i:03d}" for i in range(len(y))])
    # eval.py CSV columns are manuscript order: FTA, MI, EA, WI.
    frame = pd.DataFrame({
        "patient_id": ids,
        "label": y,
        "true_class": np.asarray(CLASSES, dtype=object)[y],
        "predicted_class": np.asarray(CLASSES, dtype=object)[p.argmax(axis=1)],
        "P_FTA": p[:, 0],
        "P_MI_FTC": p[:, 3],
        "P_EA_FTC": p[:, 2],
        "P_WI_FTC": p[:, 1],
        "n_images": 1,
    })
    frame.to_csv(path, index=False)
    return path


class BootstrapPredictionTests(unittest.TestCase):
    def test_probability_columns_are_reordered_to_model_label_order(self):
        with tempfile.TemporaryDirectory() as temp:
            y = np.arange(4)
            p = np.eye(4)
            path = write_prediction_csv(Path(temp) / "valid_predictions.csv", y, p)
            ids, actual_y, actual_p = load_prediction_csv(path)
            self.assertEqual(ids, [f"patient_{i:03d}" for i in range(4)])
            np.testing.assert_array_equal(actual_y, y)
            np.testing.assert_array_equal(actual_p, p)

    def test_duplicate_patient_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            y = np.arange(4)
            p = np.eye(4)
            path = write_prediction_csv(Path(temp) / "valid_predictions.csv", y, p, ["x", "x", "z", "w"])
            with self.assertRaisesRegex(ValueError, "duplicated patient_id"):
                load_prediction_csv(path)

    def test_inconsistent_true_class_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            y = np.arange(4)
            path = write_prediction_csv(Path(temp) / "valid_predictions.csv", y, np.eye(4))
            frame = pd.read_csv(path)
            frame.loc[0, "true_class"] = "MI-FTC"
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "true_class disagrees"):
                load_prediction_csv(path)

    def test_weighted_auc_and_metrics(self):
        y = np.tile(np.arange(4), 4)
        rng = np.random.default_rng(12)
        raw = rng.uniform(0.1, 1.0, (len(y), 4))
        p = raw / raw.sum(axis=1, keepdims=True)
        evaluator = MetricEvaluator(y, p)
        result = evaluator.calculate()
        from utils.metrics import compute_metrics_four_class
        core = compute_metrics_four_class(y, p.argmax(axis=1), p)
        self.assertAlmostEqual(result["ACC(For4Class)"], core["acc"])
        for i, name in enumerate(CLASSES):
            for metric in ("AUC", "ACC", "SEN", "SPE", "PPV", "NPV", "F1"):
                self.assertAlmostEqual(result[f"{name}vsRest: {metric}"], core["per_class"][i][metric.lower()], places=12)
        for metric in ("AUC", "SEN", "SPE", "PPV", "NPV", "F1"):
            self.assertAlmostEqual(result[f"Macro-{metric}"], core[f"macro_{metric.lower()}"] , places=12)

    def test_bootstrap_is_reproducible_and_preserves_stratum_sizes(self):
        y = np.repeat(np.arange(4), [3, 4, 5, 6])
        weights1 = make_bootstrap_weights(y, n_boot=25, seed=17)
        weights2 = make_bootstrap_weights(y, n_boot=25, seed=17)
        np.testing.assert_array_equal(weights1, weights2)
        for row in weights1:
            for c in range(4):
                self.assertEqual(int(row[y == c].sum()), int((y == c).sum()))
        p = np.eye(4)[y] * 0.6 + 0.1
        self.assertEqual(compute_queue(y, p, weights1), compute_queue(y, p, weights2))

    def test_direct_eval_directory_is_supported(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "results"
            output = Path(temp) / "bootstrap"
            y = np.repeat(np.arange(4), 3)
            p = np.eye(4)[y] * 0.6 + 0.1
            write_prediction_csv(root / "valid_predictions.csv", y, p)
            numeric = run_analysis(root, output, method_name="FTN-MCNet", n_boot=10, seed=3)
            self.assertEqual(set(numeric["method"]), {"FTN-MCNet"})
            self.assertEqual(set(numeric["cohort"]), {"valid"})
            self.assertTrue((output / "bootstrap_summary.csv").exists())

    def test_auc_ties_match_pairwise_definition(self):
        y = np.tile(np.arange(4), 4)
        raw = np.random.default_rng(4).integers(1, 5, (len(y), 4)).astype(float)
        p = raw / raw.sum(axis=1, keepdims=True)
        weights = np.tile(np.array([1, 2, 1, 3]), 4)
        result = MetricEvaluator(y, p).calculate(weights)
        for c, name in enumerate(CLASSES):
            positive = np.repeat(p[y == c, c], weights[y == c])
            negative = np.repeat(p[y != c, c], weights[y != c])
            expected = ((positive[:, None] > negative).astype(float)
                        + 0.5 * (positive[:, None] == negative)).mean()
            self.assertAlmostEqual(result[f"{name}vsRest: AUC"], expected, places=14)

    def test_multiple_methods_align_by_patient_id_and_write_aggregate_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "predictions"
            output = Path(temp) / "bootstrap"
            y = np.tile(np.arange(4), 4)
            rng = np.random.default_rng(5)
            raw = rng.uniform(0.1, 1, (len(y), 4))
            p = raw / raw.sum(axis=1, keepdims=True)
            ids = [f"p{i:02d}" for i in range(len(y))]
            write_prediction_csv(root / "FTN-MCNet" / "valid_predictions.csv", y, p, ids)
            order = rng.permutation(len(y))
            write_prediction_csv(root / "Baseline" / "valid_predictions.csv", y[order], p[order], [ids[i] for i in order])
            numeric = run_analysis(root, output, n_boot=20, seed=11)
            self.assertEqual(len(numeric), 2 * len(METRIC_COLUMNS))
            self.assertTrue((output / "bootstrap_metrics.csv").is_file())
            self.assertTrue((output / "bootstrap_summary.csv").is_file())
            self.assertTrue((output / "run_manifest.json").is_file())
            self.assertNotIn("patient_id", numeric.columns)
            self.assertEqual(set(numeric["method"]), {"FTN-MCNet", "Baseline"})
            # Results should align after input rows are shuffled.
            pivot = numeric.pivot(index="metric", columns="method", values="estimate")
            np.testing.assert_allclose(pivot["FTN-MCNet"], pivot["Baseline"], equal_nan=True)


if __name__ == "__main__":
    unittest.main()
