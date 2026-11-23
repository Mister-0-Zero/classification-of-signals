"""Compare GB and MLP probability combinations on aligned held-out sweeps."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_gb_confidence import KEYS, PROBABILITIES, rule_score


THRESHOLDS = (.3, .4, .5, .6, .7, .8, .9)
GB_WEIGHTS = (0, .25, .5, .75, 1)


def main(gb_path, mlp_path, output):
    gb = pd.read_csv(gb_path)
    mlp = pd.read_csv(mlp_path)
    assert gb["window"].eq(1).all() and mlp["window"].eq(1).all()
    assert len(gb) == len(mlp) == 5600
    keys = KEYS + ["shifted"]
    aligned = gb[keys].merge(mlp[keys + PROBABILITIES], on=keys, validate="one_to_one", how="left")
    gb_prob, mlp_prob = (table[PROBABILITIES].to_numpy() for table in (gb, aligned))
    for probabilities in (gb_prob, mlp_prob):
        assert np.isfinite(probabilities).all()
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=2e-7)
    gb_signal, mlp_signal = (p[:, :2].sum(axis=1) for p in (gb_prob, mlp_prob))
    mean = (gb_prob + mlp_prob) / 2
    scenarios = [
        ("weighted_mean", weight, weight * gb_prob + (1 - weight) * mlp_prob,
         weight * gb_signal + (1 - weight) * mlp_signal)
        for weight in GB_WEIGHTS
    ]
    scenarios.extend([
        ("either_signals", np.nan, mean, np.maximum(gb_signal, mlp_signal)),
        ("both_signal", np.nan, mean, np.minimum(gb_signal, mlp_signal)),
        ("mean_detect_mlp_modulation", .5, mlp_prob, (gb_signal + mlp_signal) / 2),
    ])
    rows = []
    for rule, gb_weight, scores, signal in scenarios:
        modulation = np.where(scores[:, 0] >= scores[:, 1], "AM", "FM")
        for threshold in THRESHOLDS:
            predictions = np.where(signal >= threshold, modulation, "Noise")
            rows.append({"gb_weight_for_detection_mean": gb_weight}
                        | rule_score(gb, rule, threshold, predictions))
    results = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "scores.csv", index=False)
    print(results[["rule", "gb_weight_for_detection_mean", "threshold", "macro_f1", "signal_as_noise",
                   "noise_as_signal", "shifted_correct"]].head(12).round(4).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gb", type=Path, required=True)
    parser.add_argument("--mlp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.gb, args.mlp, args.output)
