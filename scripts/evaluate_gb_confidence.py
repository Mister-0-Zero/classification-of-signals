"""Audit held-out GB probabilities and compare decision rules on the same windows."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from evaluate_gb_averaging import CLASSES, summarize


KEYS = ["recording", "end_sweep", "group", "fold", "label"]
PROBABILITIES = ["p_am", "p_fm", "p_noise"]
CONFIDENCE_BINS = [0, .5, .6, .7, .8, .9, .95, .99, 1]
SIGNAL_THRESHOLDS = [.3, .4, .5, .6, .7, .8, .9, .95, .99]
CASCADE_THRESHOLDS = [.5, .6, .7, .8, .9, .95, .99]


def audit(cases):
    actual, predicted = cases["label"], cases["predicted"]
    correct = actual.eq(predicted)
    outcome = np.select(
        [correct, actual.eq("Noise"), predicted.eq("Noise")],
        ["correct", "false_signal", "missed_signal"],
        default="wrong_modulation",
    )
    rows = cases.assign(correct=correct, outcome=outcome,
                        band=pd.cut(cases["confidence"], CONFIDENCE_BINS, include_lowest=True))
    bins = rows.groupby("band", observed=False).agg(n=("correct", "size"), correct=("correct", "sum"),
                                                    accuracy=("correct", "mean"), mean_score=("confidence", "mean"))
    kinds = rows.groupby("outcome").agg(n=("confidence", "size"), median_score=("confidence", "median"),
                                       confidence_ge_90=("confidence", lambda values: values.ge(.90).sum()),
                                       confidence_ge_99=("confidence", lambda values: values.ge(.99).sum()))
    auc = roc_auc_score(~correct, 1 - cases["confidence"])
    return bins, kinds, auc


def rule_score(cases, name, threshold, prediction, routed=None):
    original = cases["predicted"].eq(cases["label"])
    current = cases["label"].eq(prediction)
    changed = cases["predicted"].ne(prediction)
    shifted = cases["shifted"]
    return {"rule": name, "threshold": threshold, "routed": int(routed.sum()) if routed is not None else 0,
            "changed": int(changed.sum()), "fixed": int((~original & current).sum()),
            "broken": int((original & ~current).sum()),
            "shifted_correct": int((shifted & current).sum()),
            "shifted_as_noise": int((shifted & (prediction == "Noise")).sum())} | summarize(
                int(cases["window"].iloc[0]), cases.assign(predicted=prediction))


def decision_rules(cases, secondary):
    scores = [rule_score(cases, "gb_argmax", np.nan, cases["predicted"])]
    modulation = np.where(cases["p_am"].ge(cases["p_fm"]), "AM", "FM")
    for threshold in SIGNAL_THRESHOLDS:
        prediction = np.where(cases["p_am"] + cases["p_fm"] >= threshold, modulation, "Noise")
        scores.append(rule_score(cases, "signal_probability", threshold, prediction))
    if secondary is not None:
        alternatives = pd.concat([pd.read_csv(secondary / f"random_fm_{subset}.csv") for subset in ("centered", "shifted")])
        alternatives = alternatives[KEYS + ["predicted"]].rename(columns={"predicted": "secondary"})
        aligned = cases[KEYS].merge(alternatives, how="left", on=KEYS, validate="one_to_one")
        if len(alternatives) != len(cases) or aligned["secondary"].isna().any():
            raise ValueError("Secondary predictions must cover the same test windows and groups")
        for threshold in CASCADE_THRESHOLDS:
            routed = cases["confidence"].lt(threshold)
            prediction = np.where(routed, aligned["secondary"], cases["predicted"])
            scores.append(rule_score(cases, "defer_to_deep_sets", threshold, prediction, routed))
    return pd.DataFrame(scores)


def main(predictions, output, secondary, min_end_sweep=None):
    cases = pd.read_csv(predictions)
    if min_end_sweep is not None:
        cases = cases.loc[cases["end_sweep"].ge(min_end_sweep)].reset_index(drop=True)
    probabilities = cases[PROBABILITIES].to_numpy()
    assert np.isfinite(probabilities).all() and (probabilities >= 0).all() and (probabilities <= 1).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    np.testing.assert_array_equal(np.asarray(CLASSES)[probabilities.argmax(axis=1)], cases["predicted"])
    assert not cases.duplicated(KEYS).any()
    cases["confidence"] = probabilities.max(axis=1)
    bins, outcomes, auc = audit(cases)
    output.mkdir(parents=True, exist_ok=True)
    bins.to_csv(output / "confidence_bands.csv")
    outcomes.to_csv(output / "confidence_outcomes.csv")
    rules = decision_rules(cases, secondary)
    rules.to_csv(output / "rules.csv", index=False)
    print(f"Error detection ROC AUC from 1 - max(p): {auc:.4f}")
    print(outcomes.round(4).to_string())
    print(rules[["rule", "threshold", "routed", "fixed", "broken", "macro_f1", "signal_as_noise",
                 "noise_as_signal", "shifted_correct"]].round(4).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--secondary", type=Path, help="Deep Sets evaluation directory with random_fm predictions")
    parser.add_argument("--min-end-sweep", type=int)
    args = parser.parse_args()
    main(args.predictions, args.output, args.secondary, args.min_end_sweep)
