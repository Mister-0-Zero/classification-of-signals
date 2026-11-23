"""Compare GB training with random frequency shifts on real held-out FM."""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from evaluate_gb_averaging import average_dbm, fold_mapping, load_recordings, summarize
from evaluate_gb_shifted import evaluate, feature_table, load_shifted_recordings


SHIFT_LIMITS_KHZ = ((250, 80), (400, 180))
AUGMENTATION_SEED = 20261123
FIRST_ENDPOINT = 9
WINDOW = 7
EDGE_WIDTH = 25


def edge_noise(power, distance_samples):
    # Extend only the outermost 12.5 kHz. Cap peaks in the extension
    # so that a localized edge spike is not repeated throughout the padding.
    cap = min(power.max(), np.median(power) * 10 ** .6)
    edge = np.minimum(power, cap)
    repeated = np.r_[edge, edge[::-1]]
    return np.interp(distance_samples, np.arange(len(repeated)), repeated, period=len(repeated))


def synthesize(axis, averaged_dbm, center, span_khz, shift_khz):
    relative = axis - center
    target = np.linspace(-span_khz * 500, span_khz * 500, axis.size)
    source = target - shift_khz * 1000
    top = averaged_dbm.max()
    power = 10 ** ((averaged_dbm - top) / 10)
    sampled = np.interp(source, relative, power)
    step = np.median(np.diff(relative))
    left = source < relative[0]
    right = source > relative[-1]
    sampled[left] = edge_noise(power[:EDGE_WIDTH], (relative[0] - source[left]) / step)
    sampled[right] = edge_noise(power[-EDGE_WIDTH:][::-1], (source[right] - relative[-1]) / step)
    return target + center, top + 10 * np.log10(sampled)


def shift_geometries(rng):
    for span, limit in SHIFT_LIMITS_KHZ:
        left, right = rng.uniform(0, limit, size=2)
        yield span, -left
        yield span, right


def augmented_features(recordings, folds, extract_features, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for info, axis, traces, center in recordings:
        metadata = info | {"fold": folds[info["group"]]}
        for end in range(FIRST_ENDPOINT, len(traces)):
            spectrum = average_dbm(traces[end - WINDOW + 1:end + 1])
            for span, offset in shift_geometries(rng):
                frequencies, levels = synthesize(axis, spectrum, center, span, offset)
                features = extract_features(frequencies, levels, center)
                rows.append(metadata | {"end_sweep": end, "span_khz": span, "offset_khz": offset} | features)
    return pd.DataFrame(rows)


def main(project, output, workers, seed):
    sys.path.insert(0, str(project))
    from models.modulation.classifier import FEATURE_NAMES, extract_features

    template = joblib.load(project / "models/modulation/gradient_boosting.joblib")["model"]
    records = load_recordings(project / "data")
    folds = fold_mapping(records)
    centered = feature_table(records, folds, WINDOW, FIRST_ENDPOINT, extract_features)
    shifted = feature_table(load_shifted_recordings(project), folds, WINDOW, FIRST_ENDPOINT, extract_features)
    synthetic = augmented_features(records, folds, extract_features, seed)
    assert (len(centered), len(shifted)) == (3977, 615)
    assert len(synthetic) == len(centered) * 2 * len(SHIFT_LIMITS_KHZ)
    assert shifted["label"].eq("FM").all()
    scenarios = [
        ("baseline", centered.iloc[:0]),
        ("shift_fm_only", synthetic.loc[synthetic["label"].eq("FM")]),
        ("shift_all_classes", synthetic),
    ]
    scores, predictions = [], []
    with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=workers) as pool:
        for name, additions in scenarios:
            training = pd.concat([centered, additions], ignore_index=True)
            score, predicted = evaluate(name, training, centered, shifted, WINDOW, template, FEATURE_NAMES, pool)
            central_score = summarize(WINDOW, predicted.loc[~predicted["shifted"]])
            score.update(macro_f1_centered=central_score["macro_f1"],
                         signal_as_noise_centered=central_score["signal_as_noise"],
                         synthetic_windows=len(additions), augmentation_seed=seed)
            scores.append(score)
            predictions.append(predicted)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(scores).to_csv(output / "scores.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(output / "predictions.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--seed", type=int, default=AUGMENTATION_SEED)
    args = parser.parse_args()
    main(args.project.resolve(), args.output, args.workers, args.seed)
