"""Patient-level metrics and stratified Bootstrap CIs from FTN-MCNet prediction CSVs.

The script consumes the CSV files produced by ``eval.py`` directly. It does not
convert predictions to Excel and does not retrain models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

N_BOOT = 1000
SEED = 20261006
ALPHA = 0.05
DEGENERATE_TOL = 1e-12

# This is the model's internal label order (label 0, 1, 2, 3).
CLASSES = ("FTA", "WI-FTC", "EA-FTC", "MI-FTC")
# eval.py writes probabilities in manuscript display order; reorder by class name.
PROBABILITY_COLUMNS = ("P_FTA", "P_MI_FTC", "P_EA_FTC", "P_WI_FTC")
MACRO_METRICS = ("AUC", "SEN", "SPE", "PPV", "NPV", "F1")
METRIC_COLUMNS = (
    ["ACC(For4Class)"]
    + [f"{c}vsRest: {m}" for c in CLASSES for m in ("ACC", "SEN", "SPE", "AUC")]
    + ["Macro-AUC", "Macro-SEN", "Macro-SPE"]
    + [f"{c}vsRest: {m}" for c in CLASSES for m in ("PPV", "NPV", "F1")]
    + ["Macro-PPV", "Macro-NPV", "Macro-F1"]
)


def divide(a, b):
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    return np.divide(a, b, out=np.full(a.shape, np.nan), where=b != 0)


def load_prediction_csv(path: Path, context: str = ""):
    """Read one eval.py CSV and return patient IDs, integer labels, class-order probabilities."""
    df = pd.read_csv(path, dtype={"patient_id": str})
    required = {"patient_id", "label", *PROBABILITY_COLUMNS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{context or path}: missing columns {sorted(missing)}")
    if df.columns.duplicated().any():
        raise ValueError(f"{context or path}: duplicated columns")

    ids_series = df["patient_id"]
    if ids_series.isna().any():
        raise ValueError(f"{context or path}: missing patient_id")
    ids = ids_series.astype(str).tolist()
    if any(not value.strip() for value in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{context or path}: empty or duplicated patient_id")

    raw_labels = pd.to_numeric(df["label"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(raw_labels).all() or not np.equal(raw_labels, np.floor(raw_labels)).all():
        raise ValueError(f"{context or path}: label must contain integer indices 0-3")
    y = raw_labels.astype(int)
    if not np.isin(y, np.arange(4)).all():
        raise ValueError(f"{context or path}: label must contain only 0, 1, 2, 3")

    # The CSV's columns are manuscript order: FTA, MI-FTC, EA-FTC, WI-FTC.
    # Hospital's evaluation order and model label indices are: FTA, WI-FTC, EA-FTC, MI-FTC.
    p = df.loc[:, list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError(f"{context or path}: probabilities must be finite and in [0, 1]")
    if not np.allclose(p.sum(axis=1), 1.0, rtol=0, atol=1e-6):
        raise ValueError(f"{context or path}: probability rows must sum to 1")

    # Reorder manuscript columns into internal label order [FTA, WI, EA, MI].
    p = p[:, [0, 3, 2, 1]]

    expected_true = np.asarray(CLASSES, dtype=object)[y]
    if "true_class" in df.columns:
        actual_true = df["true_class"].fillna("").astype(str).to_numpy()
        if not np.array_equal(actual_true, expected_true):
            bad = int(np.flatnonzero(actual_true != expected_true)[0])
            raise ValueError(
                f"{context or path}: true_class disagrees with label at patient_id={ids[bad]!r}"
            )
    if "predicted_class" in df.columns:
        expected_pred = np.asarray(CLASSES, dtype=object)[p.argmax(axis=1)]
        actual_pred = df["predicted_class"].fillna("").astype(str).to_numpy()
        if not np.array_equal(actual_pred, expected_pred):
            bad = int(np.flatnonzero(actual_pred != expected_pred)[0])
            raise ValueError(
                f"{context or path}: predicted_class disagrees with probabilities at patient_id={ids[bad]!r}"
            )
    if not all(np.any(y == c) for c in range(4)):
        raise ValueError(f"{context or path}: all four true classes are required for stratified Bootstrap")
    return ids, y, p


class MetricEvaluator:
    """Compute overall and one-vs-rest metrics, including tie-aware weighted AUC."""

    def __init__(self, y, probabilities):
        self.y = np.asarray(y, dtype=int)
        self.p = np.asarray(probabilities, dtype=float)
        self.pred = self.p.argmax(axis=1)
        self.score_groups = [np.unique(self.p[:, c], return_inverse=True)[1] for c in range(4)]

    def calculate(self, weights=None):
        y = self.y
        if weights is None:
            weights = np.ones(len(y), dtype=float)
        weights = np.asarray(weights, dtype=float)
        cm = np.bincount(y * 4 + self.pred, weights=weights, minlength=16).reshape(4, 4)
        tp = cm.diagonal()
        fn = cm.sum(axis=1) - tp
        fp = cm.sum(axis=0) - tp
        tn = cm.sum() - tp - fn - fp
        metric_values = {
            "ACC": divide(tp + tn, cm.sum()),
            "SEN": divide(tp, tp + fn),
            "SPE": divide(tn, tn + fp),
            "PPV": divide(tp, tp + fp),
            "NPV": divide(tn, tn + fn),
            "F1": divide(2 * tp, 2 * tp + fp + fn),
        }
        aucs = []
        for c, groups in enumerate(self.score_groups):
            pos = np.bincount(groups, weights=weights * (y == c))
            neg = np.bincount(groups, weights=weights * (y != c))
            concordant = np.sum(pos * (np.cumsum(neg) - neg + 0.5 * neg))
            aucs.append(float(divide(concordant, pos.sum() * neg.sum())))
        metric_values["AUC"] = np.asarray(aucs)
        result = {"ACC(For4Class)": float(tp.sum() / cm.sum()) if cm.sum() else float("nan")}
        for c, class_name in enumerate(CLASSES):
            for metric in ("ACC", "SEN", "SPE", "AUC", "PPV", "NPV", "F1"):
                result[f"{class_name}vsRest: {metric}"] = float(metric_values[metric][c])
        for metric in MACRO_METRICS:
            # Keep macro undefined if any of the four class-specific values is undefined.
            result[f"Macro-{metric}"] = float(np.mean(metric_values[metric]))
        return result


def make_bootstrap_weights(y, n_boot=N_BOOT, seed=SEED):
    """Create one shared set of stratified patient bootstrap weights for a cohort."""
    y = np.asarray(y, dtype=int)
    if n_boot < 2:
        raise ValueError("n_boot must be at least 2")
    strata = [np.flatnonzero(y == c) for c in range(4)]
    if any(len(s) == 0 for s in strata):
        raise ValueError("All four true classes are required")
    rng = np.random.default_rng(seed)
    weights = np.zeros((n_boot, len(y)), dtype=np.int32)
    for b in range(n_boot):
        sampled = np.concatenate([rng.choice(s, size=len(s), replace=True) for s in strata])
        weights[b] = np.bincount(sampled, minlength=len(y))
    return weights


def compute_queue(y, probabilities, bootstrap_weights, alpha=ALPHA):
    evaluator = MetricEvaluator(y, probabilities)
    points = evaluator.calculate()
    samples = np.empty((len(bootstrap_weights), len(METRIC_COLUMNS)), dtype=float)
    for b, weights in enumerate(bootstrap_weights):
        result = evaluator.calculate(weights)
        samples[b] = [result[key] for key in METRIC_COLUMNS]

    min_valid = math.ceil(len(bootstrap_weights) / 2)
    records = []
    for j, metric in enumerate(METRIC_COLUMNS):
        values = samples[:, j]
        values = values[np.isfinite(values)]
        point = points[metric]
        degenerate = bool(len(values) and np.ptp(values) < DEGENERATE_TOL)
        lower = upper = float("nan")
        flags = []
        if not np.isfinite(point):
            flags.append("undefined_point")
        elif len(values) < min_valid:
            flags.append("insufficient_valid_replicates")
        else:
            lower, upper = np.quantile(values, [alpha / 2, 1 - alpha / 2], method="linear")
            if degenerate:
                flags.append("degenerate_distribution")
            elif upper - lower < DEGENERATE_TOL:
                flags.append("zero_width_percentile_interval")
            if not lower <= point <= upper:
                flags.append("point_outside_percentile_interval")
        if len(values) < len(bootstrap_weights):
            flags.append("undefined_replicates_excluded")
        records.append({
            "metric": metric,
            "estimate": float(point),
            "lower": float(lower),
            "upper": float(upper),
            "valid_replicates": int(len(values)),
            "n_boot": int(len(bootstrap_weights)),
            "degenerate": degenerate,
            "status": ";".join(flags),
        })
    return records


def format_cell(record):
    point, lower, upper = (record[k] for k in ("estimate", "lower", "upper"))
    if not np.isfinite(point):
        return "NA"
    if not (np.isfinite(lower) and np.isfinite(upper)):
        return f"{point:.3f} (NA)"
    marker = "†" if record["degenerate"] else ""
    return f"{point:.3f} ({lower:.3f}–{upper:.3f}){marker}"


def _discover_methods(predictions_dir: Path, method_name: str):
    direct = sorted(predictions_dir.glob("*_predictions.csv"))
    subdirs = sorted(p for p in predictions_dir.iterdir() if p.is_dir() and any(p.glob("*_predictions.csv")))
    if direct and subdirs:
        raise ValueError(
            "Put either one method's prediction CSVs directly in --predictions-dir, "
            "or put each method in its own subdirectory; do not mix both layouts."
        )
    if direct:
        return {method_name: {p.name[:-len("_predictions.csv")]: p for p in direct}}
    if subdirs:
        methods = {}
        for folder in subdirs:
            files = sorted(folder.glob("*_predictions.csv"))
            methods[folder.name] = {p.name[:-len("_predictions.csv")]: p for p in files}
        return methods
    raise FileNotFoundError(f"No *_predictions.csv files found under {predictions_dir}")


def run_analysis(predictions_dir, output_dir, method_name="FTN-MCNet", n_boot=N_BOOT, seed=SEED, alpha=ALPHA):
    predictions_dir, output_dir = Path(predictions_dir), Path(output_dir)
    if not predictions_dir.is_dir():
        raise NotADirectoryError(predictions_dir)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    output_files = ("bootstrap_metrics.csv", "bootstrap_summary.csv", "run_manifest.json")
    if any((output_dir / name).exists() for name in output_files):
        raise FileExistsError("Output already exists; choose a new --output-dir to avoid overwriting results.")

    methods = _discover_methods(predictions_dir, method_name)
    method_names = sorted(methods)
    if not method_names:
        raise ValueError("No methods found")
    cohorts_by_method = {name: set(methods[name]) for name in method_names}
    first_cohorts = cohorts_by_method[method_names[0]]
    for name, cohorts in cohorts_by_method.items():
        if cohorts != first_cohorts:
            raise ValueError(f"Method {name!r} has cohorts {sorted(cohorts)}, expected {sorted(first_cohorts)}")
    cohorts = sorted(first_cohorts)

    loaded = {}
    source_hashes = {}
    references = {}
    for method in method_names:
        for cohort in cohorts:
            path = methods[method][cohort]
            context = f"{method}/{cohort}"
            source_hashes[str(path.relative_to(predictions_dir))] = hashlib.sha256(path.read_bytes()).hexdigest()
            ids, y, p = load_prediction_csv(path, context)
            if cohort not in references:
                references[cohort] = (ids, y.copy())
            else:
                ref_ids, ref_y = references[cohort]
                if set(ids) != set(ref_ids):
                    raise ValueError(f"{context}: patient_id set differs from other methods for this cohort")
                id_to_y = dict(zip(ids, y))
                if any(id_to_y[pid] != ref_y[i] for i, pid in enumerate(ref_ids)):
                    raise ValueError(f"{context}: true labels differ from other methods for this cohort")
                # Reorder each method to the reference patient order so bootstrap draws are paired.
                position = {pid: i for i, pid in enumerate(ids)}
                order = np.asarray([position[pid] for pid in ref_ids], dtype=int)
                y, p, ids = y[order], p[order], ref_ids
            loaded[(method, cohort)] = (ids, y, p)

    # A single shared set of bootstrap resamples per cohort is used for every method.
    weights_by_cohort = {}
    for cohort in cohorts:
        y = references[cohort][1]
        stable_offset = zlib.crc32(cohort.encode("utf-8"))
        weights_by_cohort[cohort] = make_bootstrap_weights(y, n_boot=n_boot, seed=(seed + stable_offset) % (2**32))

    all_records = []
    summary_rows = []
    for cohort in cohorts:
        for method in method_names:
            _, y, p = loaded[(method, cohort)]
            records = compute_queue(y, p, weights_by_cohort[cohort], alpha=alpha)
            for record in records:
                all_records.append({"method": method, "cohort": cohort, **record})
            summary_rows.append({
                "method": method,
                "cohort": cohort,
                **{record["metric"]: format_cell(record) for record in records},
            })
            print(f"[done] {method} / {cohort}", flush=True)

    # Ensure inputs were not changed while the analysis was running.
    for relative, digest in source_hashes.items():
        if hashlib.sha256((predictions_dir / relative).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Input prediction CSV changed during analysis; no outputs written")

    output_dir.mkdir(parents=True, exist_ok=True)
    numeric = pd.DataFrame(all_records)
    numeric.to_csv(output_dir / "bootstrap_metrics.csv", index=False, encoding="utf-8-sig", na_rep="NA")
    summary = pd.DataFrame(summary_rows, columns=["method", "cohort", *METRIC_COLUMNS])
    summary.to_csv(output_dir / "bootstrap_summary.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "n_boot": int(n_boot),
        "seed": int(seed),
        "alpha": float(alpha),
        "bootstrap": "class-stratified percentile bootstrap",
        "class_order": list(CLASSES),
        "probability_csv_order": ["FTA", "MI-FTC", "EA-FTC", "WI-FTC"],
        "probability_metric_order": list(CLASSES),
        "sampling_unit": "one unique patient_id row",
        "paired_resamples_across_methods": True,
        "cohorts": cohorts,
        "methods": method_names,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": source_hashes,
        "metric_rows": len(numeric),
        "undefined_estimates": int(numeric["estimate"].isna().sum()),
        "degenerate_intervals": int(numeric["degenerate"].sum()),
        "minimum_valid_replicates": math.ceil(n_boot / 2),
        "limitations": [
            "Class-stratified intervals condition on the observed class composition.",
            "Undefined bootstrap replicates are excluded per metric; the minimum-valid rule is not a coverage guarantee.",
            "Each patient_id must represent one independent patient-level unit.",
        ],
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return numeric


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", required=True, help="One method's CSV output directory, or a root containing one subdirectory per method")
    parser.add_argument("--output-dir", required=True, help="New output directory; existing result files are not overwritten")
    parser.add_argument("--method-name", default="FTN-MCNet", help="Method name when prediction CSVs are directly in --predictions-dir")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--alpha", type=float, default=ALPHA, help="Two-sided CI alpha; default 0.05")
    args = parser.parse_args()
    run_analysis(args.predictions_dir, args.output_dir, args.method_name, args.n_boot, args.seed, args.alpha)
    print(f"[output] {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
