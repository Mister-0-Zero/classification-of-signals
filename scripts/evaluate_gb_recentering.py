"""Test GB with a candidate frequency estimated from the whole captured span."""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter_ns

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from evaluate_gb_averaging import SEED, average_dbm, fold_mapping, load_recordings, train
from evaluate_gb_shifted import evaluate, feature_table, load_shifted_recordings


def candidate_center(axis, spectrum, midpoint):
    power = 10 ** ((spectrum - spectrum.max()) / 10)
    excess = np.maximum(power - np.median(power), 0)
    cumulative = np.cumsum(excess)
    total = cumulative[-1]
    if total <= 0:
        return midpoint
    return np.interp(.5 * total, cumulative, axis)


def latency(model, records, window, names, extract_features):
    rng = np.random.default_rng(SEED)
    cases = [(axis, traces[end-window+1:end+1], center) for _, axis, traces, center in records
             for end in rng.choice(np.arange(9, 50), size=2, replace=False)]
    elapsed = []
    for repeat in range(4):
        for index in rng.permutation(len(cases)):
            axis, traces, center = cases[index]
            start = perf_counter_ns()
            spectrum = average_dbm(traces)
            candidate = candidate_center(axis, spectrum, center)
            features = extract_features(axis, spectrum, candidate)
            model.predict_proba(pd.DataFrame([features], columns=names))
            if repeat:
                elapsed.append((perf_counter_ns() - start) / 1_000_000)
    return {"timed_calls": len(elapsed), "p50_ms": np.median(elapsed), "p95_ms": np.quantile(elapsed, .95)}


def main(project, output, workers, window):
    sys.path.insert(0, str(project))
    from models.modulation.classifier import FEATURE_NAMES, extract_features

    template = joblib.load(project / "models/modulation/gradient_boosting.joblib")["model"]
    records = load_recordings(project / "data")
    shifted_records = load_shifted_recordings(project)
    folds = fold_mapping(records)
    first_endpoint = 0 if window == 1 else 9
    training = feature_table(records, folds, window, first_endpoint, extract_features)
    centered = feature_table(records, folds, window, first_endpoint, extract_features, estimate_center=candidate_center)
    shifted = feature_table(shifted_records, folds, window, first_endpoint, extract_features,
                            estimate_center=candidate_center)
    assert (len(centered), len(shifted)) == (97 * (50 - first_endpoint), 15 * (50 - first_endpoint))
    with threadpool_limits(limits=1):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            score, predictions = evaluate("gb_power_median", training, centered, shifted, window,
                                          template, FEATURE_NAMES, pool, probabilities=True)
        fitted = train(template, training, FEATURE_NAMES)
        score.update(latency(fitted, records, window, FEATURE_NAMES, extract_features))
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([score]).to_csv(output / "scores.csv", index=False)
    predictions.to_csv(output / "predictions.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--window", type=int, choices=(1, 7), default=7)
    args = parser.parse_args()
    main(args.project.resolve(), args.output, args.workers, args.window)
