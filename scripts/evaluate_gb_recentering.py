"""Test GB with a candidate frequency estimated from the whole captured span."""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from evaluate_gb_averaging import fold_mapping, load_recordings
from evaluate_gb_shifted import evaluate, feature_table, load_shifted_recordings


def candidate_center(axis, spectrum, midpoint):
    power = 10 ** ((spectrum - spectrum.max()) / 10)
    excess = np.maximum(power - np.median(power), 0)
    cumulative = np.cumsum(excess)
    total = cumulative[-1]
    if total <= 0:
        return midpoint
    return np.interp(.5 * total, cumulative, axis)


def main(project, output, workers):
    sys.path.insert(0, str(project))
    from models.modulation.classifier import FEATURE_NAMES, extract_features

    template = joblib.load(project / "models/modulation/gradient_boosting.joblib")["model"]
    records = load_recordings(project / "data")
    shifted_records = load_shifted_recordings(project)
    folds = fold_mapping(records)
    training = feature_table(records, folds, 7, 9, extract_features)
    centered = feature_table(records, folds, 7, 9, extract_features, estimate_center=candidate_center)
    shifted = feature_table(shifted_records, folds, 7, 9, extract_features, estimate_center=candidate_center)
    assert (len(centered), len(shifted)) == (3977, 615)
    with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=workers) as pool:
        score, predictions = evaluate("gb_power_median", training, centered, shifted, 7,
                                      template, FEATURE_NAMES, pool)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([score]).to_csv(output / "scores.csv", index=False)
    predictions.to_csv(output / "predictions.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    main(args.project.resolve(), args.output, args.workers)
