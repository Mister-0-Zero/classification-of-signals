"""Compare four raw-spectrum MLP shapes on centered and shifted held-out traces."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from evaluate_mlp_shifted import load_shifted, metrics_per_recording
from evaluate_modulation_mlp import CLASS_NAMES, SEED, benchmark, choose_epochs, fit, raw_input


ARCHITECTURES = [(32, 16), (16, 16), (32, 32), (32, 32, 16)]
WIDE_HIDDEN = (200, 20)
DROPOUT_LEVELS = (0.0, 0.1, 0.25, 0.4)


def predict_shifted(state, inputs):
    model, _, _, _, _, _, mean, scale = state
    model.eval()
    with torch.inference_mode():
        return model(torch.from_numpy((inputs - mean) / scale)).argmax(dim=1).numpy()


def prediction_rows(rows, state, inputs, architecture, probabilities):
    model, _, _, _, _, _, mean, scale = state
    model.eval()
    with torch.inference_mode():
        logits = model(torch.from_numpy((inputs - mean) / scale))
        result = rows.assign(architecture=architecture, predicted=np.asarray(CLASS_NAMES)[logits.argmax(dim=1).numpy()])
        if probabilities:
            result = result.assign(**{f"p_{label.lower()}": values for label, values in
                                      zip(CLASS_NAMES, logits.softmax(dim=1).numpy().T, strict=True)})
    return result


def evaluate_one(hidden, meta, inputs, stress_meta, stress_inputs, dropout=0.0, *,
                 probabilities=False, augment=None, signal_penalty=0.0):
    name = "-".join(map(str, (501, *hidden, 3)))
    centered, shifted, epochs = [], [], []
    options = {"augment": augment, "signal_penalty": signal_penalty}
    for fold in sorted(meta["fold"].unique()):
        train_idx = np.flatnonzero(meta["fold"].ne(fold))
        test_idx = np.flatnonzero(meta["fold"].eq(fold))
        cases = stress_meta.loc[stress_meta["fold"].eq(fold)]
        train_groups = set(meta.iloc[train_idx]["group"])
        assert train_groups.isdisjoint(set(meta.iloc[test_idx]["group"]) | set(cases["group"]))
        selected = choose_epochs(inputs, meta, train_idx, int(fold), dropout, hidden, **options)
        state = fit(inputs, meta, train_idx, dropout, selected, SEED + int(fold), hidden, **options)
        centered.append(prediction_rows(meta.iloc[test_idx], state, inputs[test_idx], name, probabilities)
                        .assign(best_epoch=selected))
        if not cases.empty:
            shifted.append(prediction_rows(cases, state, stress_inputs[cases.index], name, probabilities))
        epochs.append(selected)
    final = fit(inputs, meta, np.arange(len(meta)), dropout, int(np.median(epochs)), SEED, hidden, **options)
    unseen = stress_meta.loc[stress_meta["fold"].eq(0)]
    assert set(meta["group"]).isdisjoint(unseen["group"])
    if not unseen.empty:
        shifted.append(prediction_rows(unseen, final, stress_inputs[unseen.index], name, probabilities))
    centered, shifted = pd.concat(centered, ignore_index=True), pd.concat(shifted, ignore_index=True)
    assert (len(centered), len(shifted)) == (4850, 750)
    assert not centered.duplicated(["recording", "sweep_idx"]).any()
    assert not shifted.duplicated(["recording", "sweep_idx"]).any()
    parameters = sum(parameter.numel() for parameter in final[0].parameters())
    print(f"{name}, dropout {dropout:g}: {parameters} parameters, "
          f"epoch range {min(epochs)}-{max(epochs)}", flush=True)
    return name, centered, shifted, final, parameters


def summarize(name, centered, shifted, parameters):
    actual, estimated = centered["label"], centered["predicted"]
    matrix = confusion_matrix(actual, estimated, labels=CLASS_NAMES)
    f1 = f1_score(actual, estimated, labels=CLASS_NAMES, average=None)
    row = {"architecture": name, "parameters": parameters, "accuracy": accuracy_score(actual, estimated),
           "macro_f1": f1.mean(), **dict(zip(["f1_am", "f1_fm", "f1_noise"], f1, strict=True)),
           "signal_as_noise": int(matrix[:2, 2].sum()), "noise_as_signal": int(matrix[2, :2].sum())}
    for label, block in [("all", shifted), *shifted.groupby("span_khz")]:
        suffix = str(int(label)) if label != "all" else label
        right = int(block["predicted"].eq("FM").sum())
        row[f"shifted_fm_recall_{suffix}"] = right / len(block)
        row[f"shifted_fm_correct_{suffix}"] = right
        row[f"shifted_fm_as_noise_{suffix}"] = int(block["predicted"].eq("Noise").sum())
    return row


def main(project, output, wide_dropout=False):
    sys.path.insert(0, str(project))
    from models.modulation.classifier import FEATURE_NAMES, ModulationClassifier, extract_features
    from scripts.evaluate_gb_averaging import fold_mapping, load_recordings

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    recordings = load_recordings(project / "data")
    folds = fold_mapping(recordings)
    rows, inputs = [], []
    for info, _, levels, _ in recordings:
        for index, trace in enumerate(levels):
            rows.append(info | {"fold": folds[info["group"]], "sweep_idx": index})
            inputs.append(raw_input(trace))
    meta = pd.DataFrame(rows)
    x = np.stack(inputs)
    stress, stress_x = load_shifted(project)
    stress["fold"] = stress["group"].map(folds).fillna(0).astype(int)
    results, centered_predictions, shifted_predictions, models = [], [], [], {}
    configurations = ([(WIDE_HIDDEN, value) for value in DROPOUT_LEVELS] if wide_dropout
                      else [(hidden, 0.0) for hidden in ARCHITECTURES])
    for architecture, dropout in configurations:
        name, centered, shifted, state, parameters = evaluate_one(architecture, meta, x, stress, stress_x, dropout)
        results.append(summarize(name, centered, shifted, parameters) | {"dropout": dropout})
        if wide_dropout:
            centered = centered.assign(dropout=dropout)
            shifted = shifted.assign(dropout=dropout)
        centered_predictions.append(centered)
        shifted_predictions.append(shifted)
        models[(name, dropout)] = state
    timing = benchmark(recordings, models, extract_features, FEATURE_NAMES, ModulationClassifier()._model)
    timing = timing.rename(columns={"representation": "architecture"})
    scores = pd.DataFrame(results).merge(timing, on=["architecture", "dropout"], validate="one_to_one")
    if not wide_dropout:
        scores = scores.drop(columns="dropout")
    scores.to_csv(output / "scores.csv", index=False)
    centered = pd.concat(centered_predictions, ignore_index=True)
    shifted = pd.concat(shifted_predictions, ignore_index=True)
    centered.to_csv(output / "centered_predictions.csv", index=False, encoding="utf-8-sig")
    shifted.to_csv(output / "shifted_predictions.csv", index=False, encoding="utf-8-sig")
    grouping = ["architecture"] + (["dropout"] if wide_dropout else []) + ["span_khz", "group"]
    metrics_per_recording(shifted, grouping).to_csv(
        output / "shifted_by_station.csv", index=False, encoding="utf-8-sig")
    print(scores.round(5).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wide-dropout", action="store_true",
                        help="Compare dropout levels for the 501-200-20-3 network")
    args = parser.parse_args()
    main(args.project, args.output, args.wide_dropout)
