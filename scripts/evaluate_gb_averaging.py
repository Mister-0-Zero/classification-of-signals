"""Compare single-sweep and rolling-average inputs on held-out groups."""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter_ns

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from threadpoolctl import threadpool_limits


WINDOWS = (1, 2, 3, 5, 7, 10)
CLASSES = ["AM", "FM", "Noise"]
SEED = 20261119


def average_dbm(traces):
    if len(traces) == 1:
        return traces[0]
    top = traces.max()
    power = 10 ** ((traces - top) / 10)
    return top + 10 * np.log10(power.mean(axis=0))


def load_recordings(data):
    recordings = []
    for label in CLASSES:
        for path in sorted((data / label).glob("**/samples/*.csv")):
            config = json.loads((path.parent.parent / "run_config.json").read_text(encoding="utf-8-sig"))
            if config["span_khz"] != 250:
                continue
            assert config["class_name"] == label and config["sweep_points"] == 501
            raw = pd.read_csv(path, usecols=["sweep_idx", "point_idx", "frequency_hz", "amplitude_dbm", "class_name"])
            assert set(raw["class_name"]) == {label}
            values = raw.pivot(index="sweep_idx", columns="point_idx", values="amplitude_dbm")
            assert np.array_equal(values.index, np.arange(50)) and np.array_equal(values.columns, np.arange(501))
            assert np.isfinite(values.to_numpy()).all()
            axis = raw.groupby("point_idx")["frequency_hz"].first().to_numpy()
            assert raw.groupby("point_idx")["frequency_hz"].nunique().eq(1).all()
            assert np.isclose(axis[-1] - axis[0], 250_000, atol=2) and (np.diff(axis) > 0).all()
            center_mhz = float(config["center_freq_mhz"])
            info = {"recording": path.relative_to(data).as_posix(), "label": label, "group": f"{label}:{center_mhz:.3f}"}
            recordings.append((info, axis, values.to_numpy(), center_mhz * 1_000_000))
    return recordings


def fold_mapping(recordings):
    groups = [[info["group"] for info, *_ in recordings if info["label"] == label] for label in CLASSES]
    am, fm, noise = [sorted(set(values)) for values in groups]
    assert tuple(map(len, (am, fm, noise))) == (9, 9, 10)
    station_order = np.random.default_rng(SEED).permutation(fm)
    noise_order = np.random.default_rng(SEED).permutation(noise)
    result = {group: fold for fold, pair in enumerate(zip(am, station_order, strict=True), 1) for group in pair}
    result.update({group: i % 9 + 1 for i, group in enumerate(noise_order)})
    return result


def examples(recordings, extract_features):
    folds = fold_mapping(recordings)
    by_window = {length: [] for length in WINDOWS}
    for info, axis, values, center in recordings:
        for end in range(max(WINDOWS) - 1, len(values)):
            base = info | {"end_sweep": end, "fold": folds[info["group"]]}
            for length, rows in by_window.items():
                merged = average_dbm(values[end - length + 1:end + 1])
                rows.append(base | extract_features(axis, merged, center))
    return {length: pd.DataFrame(rows) for length, rows in by_window.items()}


def train(template, frame, names):
    group_sizes = frame.groupby("group")["label"].transform("size")
    groups_per_class = frame.groupby("label")["group"].transform("nunique")
    weights = 1 / (group_sizes * groups_per_class)
    return clone(template).fit(frame[names], frame["label"], sample_weight=weights / weights.mean())


def evaluate(length, data, template, names):
    parts = []
    for fold, holdout in data.groupby("fold"):
        training = data.loc[data["fold"].ne(fold)]
        assert set(training["group"]).isdisjoint(holdout["group"])
        model = train(template, training, names)
        parts.append(holdout[["recording", "label", "group", "fold", "end_sweep"]].assign(
            window=length, predicted=model.predict(holdout[names])))
    prediction = pd.concat(parts, ignore_index=True)
    assert len(prediction) == len(data) and not prediction.duplicated(["recording", "end_sweep"]).any()
    actual, predicted = prediction["label"], prediction["predicted"]
    matrix = confusion_matrix(actual, predicted, labels=CLASSES)
    f1 = f1_score(actual, predicted, labels=CLASSES, average=None)
    result = {"window": length, "accuracy": accuracy_score(actual, predicted), "macro_f1": f1.mean(),
              **dict(zip(["f1_am", "f1_fm", "f1_noise"], f1, strict=True)),
              "signal_as_noise": int(matrix[:2, 2].sum()), "noise_as_signal": int(matrix[2, :2].sum()),
              "n_spectra": len(prediction)}
    print(f"Window {length}: Macro F1 {f1.mean():.4f}", flush=True)
    return result, prediction, train(template, data, names)


def latency(recordings, models, extract_features, names):
    rng = np.random.default_rng(SEED)
    samples = [(axis, values, center, int(end)) for _, axis, values, center in recordings
               for end in rng.choice(np.arange(max(WINDOWS) - 1, len(values)), 2, replace=False)]
    elapsed = {length: [] for length in WINDOWS}
    for repeat in range(4):
        indices = rng.permutation(len(samples))
        for index in indices:
            axis, values, center, end = samples[index]
            for length in rng.permutation(WINDOWS):
                ready = values[end - length + 1:end + 1]
                start = perf_counter_ns()
                features = extract_features(axis, average_dbm(ready), center)
                models[length].predict(pd.DataFrame([features], columns=names))
                if repeat:
                    elapsed[length].append((perf_counter_ns() - start) / 1_000_000)
    return pd.DataFrame([{"window": length, "timed_calls": len(values), "processing_p50_ms": np.median(values),
                          "processing_p95_ms": np.quantile(values, .95)}
                         for length, values in elapsed.items()])


def non_overlapping_scores(predictions):
    ends = predictions["end_sweep"].mod(max(WINDOWS)).eq(max(WINDOWS) - 1)
    rows = []
    for length, matching in predictions.loc[ends].groupby("window"):
        rows.append({"window": length, "spectra": len(matching),
                     "macro_f1": f1_score(matching["label"], matching["predicted"], labels=CLASSES, average="macro"),
                     "accuracy": accuracy_score(matching["label"], matching["predicted"])})
    return pd.DataFrame(rows)


def main(project, output, workers):
    sys.path.insert(0, str(project))
    from models.modulation.classifier import FEATURE_NAMES, extract_features

    output.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(project / "models/modulation/gradient_boosting.joblib")
    assert bundle["feature_names"] == FEATURE_NAMES
    template = bundle["model"]
    records = load_recordings(project / "data")
    assert len(records) == 97
    tables = examples(records, extract_features)
    keys = ["recording", "fold", "end_sweep", "label"]
    assert all(table[keys].equals(tables[1][keys]) for table in tables.values())
    print(f"{len(records)} recordings x {50 - max(WINDOWS) + 1} matched windows", flush=True)
    scores, predictions, models = [], [], {}
    with threadpool_limits(limits=1):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {pool.submit(evaluate, length, table, template, FEATURE_NAMES): length
                       for length, table in tables.items()}
            for future in as_completed(pending):
                metrics, predicted, model = future.result()
                scores.append(metrics)
                predictions.append(predicted)
                models[pending[future]] = model
        timings = latency(records, models, extract_features, FEATURE_NAMES)
    results = pd.DataFrame(scores).merge(timings, on="window", validate="one_to_one").sort_values("window")
    results.to_csv(output / "scores.csv", index=False)
    predictions = pd.concat(predictions).sort_values(["window", "recording", "end_sweep"])
    predictions.to_csv(output / "predictions.csv", index=False)
    non_overlapping_scores(predictions).to_csv(output / "non_overlapping_check.csv", index=False)
    print(results.round(5).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    main(args.project, args.output, args.workers)
