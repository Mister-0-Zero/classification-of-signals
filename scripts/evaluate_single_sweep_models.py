"""Compare one-sweep GB, neural classifiers, and confidence-gated fallback."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from evaluate_gb_averaging import CLASSES
from evaluate_gb_confidence import CASCADE_THRESHOLDS, KEYS, PROBABILITIES, rule_score


def load_neural(directory, architecture, dropout=None):
    parts = []
    for subset in ("centered", "shifted"):
        rows = pd.read_csv(directory / f"{subset}_predictions.csv")
        mask = rows["architecture"].eq(architecture)
        if dropout is not None:
            mask &= rows["dropout"].eq(dropout)
        parts.append(rows.loc[mask].rename(columns={"sweep_idx": "end_sweep"}))
    return pd.concat(parts, ignore_index=True)


def main(gb, mlp, cnn, output):
    cases = pd.read_csv(gb)
    assert len(cases) == 5600 and cases["window"].eq(1).all()
    assert cases["shifted"].sum() == 750 and not cases.duplicated(KEYS).any()
    confidence = cases[PROBABILITIES].max(axis=1)
    models = {
        "gb": cases["predicted"],
        "mlp": load_neural(mlp, "501-200-20-3", dropout=.1),
        "cnn_25_zones": load_neural(cnn, "late_dense_25"),
        "cnn_dilated": load_neural(cnn, "dilated_hybrid"),
    }
    scores, class_scores, matrices = [], [], []
    output.mkdir(parents=True, exist_ok=True)
    for name, predictions in models.items():
        if name == "gb":
            predicted = predictions
        else:
            alternatives = predictions[KEYS + ["predicted"]]
            matched = cases[KEYS].merge(alternatives, how="left", on=KEYS, validate="one_to_one")
            if len(alternatives) != len(cases) or matched["predicted"].isna().any():
                raise ValueError(f"{name}: forecasts do not match the GB holdout and groups")
            predicted = matched["predicted"]
            for threshold in CASCADE_THRESHOLDS:
                routed = confidence.lt(threshold)
                cascaded = np.where(routed, predicted, cases["predicted"])
                scores.append(rule_score(cases, f"gb_then_{name}", threshold, cascaded, routed))
        scores.append(rule_score(cases, name, np.nan, predicted))
        precision, recall, f1, support = precision_recall_fscore_support(
            cases["label"], predicted, labels=CLASSES, zero_division=0)
        class_scores.append(pd.DataFrame({"model": name, "label": CLASSES, "precision": precision,
                                          "recall": recall, "f1": f1, "support": support}))
        matrix = pd.DataFrame(confusion_matrix(cases["label"], predicted, labels=CLASSES),
                              index=pd.Index(CLASSES, name="actual"), columns=CLASSES)
        matrices.append(matrix.reset_index().assign(model=name))
    pd.DataFrame(scores).to_csv(output / "scores.csv", index=False)
    pd.concat(class_scores, ignore_index=True).to_csv(output / "class_scores.csv", index=False)
    pd.concat(matrices, ignore_index=True).to_csv(output / "confusion_matrices.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gb", type=Path, required=True, help="GB single-sweep predictions CSV")
    parser.add_argument("--mlp", type=Path, required=True, help="MLP wide-dropout evaluation directory")
    parser.add_argument("--cnn", type=Path, required=True, help="CNN architecture evaluation directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.gb, args.mlp, args.cnn, args.output)
