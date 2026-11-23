"""Evaluate single-sweep MLP changes and audit their held-out confidence."""

import argparse
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

from evaluate_gb_averaging import CLASSES, fold_mapping, load_recordings
from evaluate_gb_confidence import PROBABILITIES, audit, decision_rules, rule_score
from evaluate_gb_shift_augmentation import AUGMENTATION_SEED, shift_geometries, synthesize
from evaluate_mlp_architectures import evaluate_one
from evaluate_mlp_shifted import load_shifted
from evaluate_modulation_mlp import SEED, raw_input


class RandomFMShift:
    def __init__(self, records, labels):
        rng = np.random.default_rng(AUGMENTATION_SEED)
        self.lookup = np.full(len(labels), -1)
        self.lookup[labels.eq("FM").to_numpy()] = np.arange(labels.eq("FM").sum())
        self.variants = np.stack([
            np.stack([raw_input(synthesize(axis, trace, center, span, offset)[1])
                      for span, offset in shift_geometries(rng)])
            for info, axis, traces, center in records if info["label"] == "FM" for trace in traces
        ])
        assert len(self.variants) == labels.eq("FM").sum()

    def __call__(self, batch_inputs, batch_indices, mean, scale, rng):
        rows = self.lookup[batch_indices]
        active = (rows >= 0) & (rng.random(len(rows)) < .5)
        choices = rng.integers(self.variants.shape[1], size=active.sum())
        selected = self.variants[rows[active], choices]
        batch_inputs[active] = torch.from_numpy((selected - mean) / scale)


def timing(records, models):
    rng = np.random.default_rng(SEED)
    traces = [sweeps[index] for _, _, sweeps, _ in records
              for index in rng.choice(len(sweeps), size=2, replace=False)]
    elapsed = {name: [] for name in models}
    for state in models.values():
        state[0].eval()
    with torch.inference_mode():
        for repeat in range(4):
            for index in rng.permutation(len(traces)):
                for name in rng.permutation(list(models)):
                    model, _, _, _, _, _, mean, scale = models[name]
                    start = perf_counter_ns()
                    values = torch.from_numpy((raw_input(traces[index]) - mean) / scale)
                    model(values.unsqueeze(0)).softmax(dim=1).argmax(dim=1).item()
                    if repeat:
                        elapsed[name].append((perf_counter_ns() - start) / 1_000_000)
    return pd.DataFrame([{"model": name, "timed_calls": len(values), "p50_ms": np.median(values),
                          "p95_ms": np.quantile(values, .95)} for name, values in elapsed.items()])


def main(project, output):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    records = load_recordings(project / "data")
    folds = fold_mapping(records)
    meta = pd.DataFrame([info | {"fold": folds[info["group"]], "sweep_idx": index}
                         for info, _, traces, _ in records for index in range(len(traces))])
    inputs = np.stack([raw_input(trace) for _, _, traces, _ in records for trace in traces])
    stress, stress_inputs = load_shifted(project)
    stress["fold"] = stress["group"].map(folds).fillna(0).astype(int)
    assert (len(meta), len(stress)) == (4850, 750)
    augment = RandomFMShift(records, meta["label"])
    configurations = [
        ("wide_baseline", (200, 20), None, 0.0),
        ("wide_shift", (200, 20), augment, 0.0),
        ("compact_shift", (96, 32), augment, 0.0),
        ("compact_shift_miss_penalty", (96, 32), augment, 1.0),
    ]
    output.mkdir(parents=True, exist_ok=True)
    models, scores, bands, outcomes, rules, matrices = {}, [], [], [], [], []
    for name, hidden, shift, penalty in configurations:
        print(name, flush=True)
        _, centered, shifted, final, parameters = evaluate_one(
            hidden, meta, inputs, stress, stress_inputs, dropout=.1, probabilities=True,
            augment=shift, signal_penalty=penalty)
        cases = pd.concat([centered.assign(shifted=False), shifted.assign(shifted=True)], ignore_index=True)
        probabilities = cases[PROBABILITIES].to_numpy()
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=2e-7)
        cases = cases.assign(model=name, window=1, end_sweep=cases["sweep_idx"],
                             confidence=probabilities.max(axis=1))
        bins, kinds, roc_auc = audit(cases)
        scores.append({"model": name, "parameters": parameters, "error_roc_auc": roc_auc}
                      | rule_score(cases, "argmax", np.nan, cases["predicted"]))
        bands.append(bins.reset_index().assign(model=name))
        outcomes.append(kinds.reset_index().assign(model=name))
        decisions = decision_rules(cases, None).replace({"rule": {"gb_argmax": "mlp_argmax"}})
        rules.append(decisions.assign(model=name))
        matrix = confusion_matrix(cases["label"], cases["predicted"], labels=CLASSES)
        matrices.append(pd.DataFrame(matrix, columns=CLASSES).assign(actual=CLASSES, model=name))
        cases.to_csv(output / f"{name}_predictions.csv", index=False)
        print(scores[-1], flush=True)
        models[name] = final
    metrics = pd.DataFrame(scores).merge(timing(records, models), on="model", validate="one_to_one")
    metrics.to_csv(output / "scores.csv", index=False)
    pd.concat(bands, ignore_index=True).to_csv(output / "confidence_bands.csv", index=False)
    pd.concat(outcomes, ignore_index=True).to_csv(output / "confidence_outcomes.csv", index=False)
    pd.concat(rules, ignore_index=True).to_csv(output / "rules.csv", index=False)
    pd.concat(matrices, ignore_index=True).to_csv(output / "confusion_matrices.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.project, args.output)
