"""Evaluate the GB configuration with off-center FM included in the test."""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from evaluate_gb_averaging import average_dbm, fold_mapping, load_recordings, predict_window, summarize, train
from evaluate_mlp_shifted import load_shifted


def load_shifted_recordings(project):
    meta, traces = load_shifted(project, normalize=False)
    records = []
    for recording, rows in meta.groupby("recording", sort=False):
        assert np.array_equal(rows["sweep_idx"], np.arange(50))
        capture = pd.read_csv(project / "data" / recording, usecols=["sweep_idx", "point_idx", "frequency_hz"])
        axis = capture.loc[capture["sweep_idx"].eq(0)].sort_values("point_idx")["frequency_hz"].to_numpy()
        assert capture.groupby("point_idx")["frequency_hz"].nunique().eq(1).all()
        info = rows.iloc[0][["recording", "label", "group", "span_khz"]].to_dict()
        records.append((info, axis, traces[rows.index], rows.iloc[0]["center_mhz"] * 1_000_000))
    return records


def feature_table(records, folds, length, first_endpoint, extract_features, estimate_center=None):
    rows = []
    for info, axis, traces, center in records:
        fold = folds.get(info["group"], 0)
        for end in range(first_endpoint, len(traces)):
            spectrum = average_dbm(traces[end - length + 1:end + 1])
            candidate = center if estimate_center is None else estimate_center(axis, spectrum, center)
            features = extract_features(axis, spectrum, candidate)
            rows.append(info | {"fold": fold, "end_sweep": end} | features)
    return pd.DataFrame(rows)


def evaluate_fold(fold, training, centered, shifted, length, template, names):
    fit_rows = training.loc[training["fold"].ne(fold)]
    model = train(template, fit_rows, names)
    tests = pd.concat([centered.loc[centered["fold"].eq(fold)].assign(shifted=False),
                       shifted.loc[shifted["fold"].eq(fold)].assign(shifted=True)])
    assert set(fit_rows["group"]).isdisjoint(tests["group"])
    assert set(fit_rows["recording"]).isdisjoint(tests["recording"])
    predictions = predict_window(model, names, length, tests)
    return predictions.assign(shifted=tests["shifted"], span_khz=tests["span_khz"])


def evaluate(name, training, centered, shifted, length, template, names, pool):
    futures = [pool.submit(evaluate_fold, fold, training, centered, shifted, length, template, names)
               for fold in sorted(set(centered["fold"]) | set(shifted["fold"]))]
    predictions = pd.concat([future.result() for future in futures], ignore_index=True)
    assert len(predictions) == len(centered) + len(shifted)
    assert not predictions.duplicated(["recording", "end_sweep"]).any()
    counts = {}
    for suffix, mask in [
        ("all", predictions["label"].eq("FM")),
        ("centered", predictions["label"].eq("FM") & ~predictions["shifted"]),
        ("shifted", predictions["shifted"]),
    ]:
        fm = predictions.loc[mask, "predicted"]
        counts |= {f"fm_n_{suffix}": len(fm), f"fm_correct_{suffix}": int(fm.eq("FM").sum()),
                   f"fm_as_noise_{suffix}": int(fm.eq("Noise").sum())}
    score = {"model": name} | summarize(length, predictions) | counts
    print(f"{name}: Macro F1 {score['macro_f1']:.4f}", flush=True)
    return score, predictions.assign(model=name)


def main(project, output, workers):
    sys.path.insert(0, str(project))
    from models.modulation.classifier import FEATURE_NAMES, extract_features

    template = joblib.load(project / "models/modulation/gradient_boosting.joblib")["model"]
    baseline = load_recordings(project / "data")
    folds = fold_mapping(baseline)
    shifted = load_shifted_recordings(project)
    assert len(baseline) == 97 and len(shifted) == 15
    single = feature_table(baseline, folds, 1, 0, extract_features)
    single_shifted = feature_table(shifted, folds, 1, 0, extract_features)
    average = feature_table(baseline, folds, 7, 9, extract_features)
    average_shifted = feature_table(shifted, folds, 7, 9, extract_features)
    scenarios = [
        ("gb_single", single, single, single_shifted, 1),
        ("gb_single_to_avg7", single.loc[single["end_sweep"].ge(9)], average, average_shifted, 7),
        ("gb_avg7", average, average, average_shifted, 7),
    ]
    with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=workers) as pool:
        results = [evaluate(*scenario, template, FEATURE_NAMES, pool) for scenario in scenarios]
    output.mkdir(parents=True, exist_ok=True)
    scores, predictions = zip(*results, strict=True)
    pd.DataFrame(scores).to_csv(output / "scores.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(output / "predictions.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    main(args.project.resolve(), args.output, args.workers)
