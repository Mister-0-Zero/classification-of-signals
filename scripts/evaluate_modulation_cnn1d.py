"""Compare compact 1D CNNs with regional versus global pooling."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn

from evaluate_mlp_architectures import predict_shifted, summarize
from evaluate_mlp_shifted import load_shifted, metrics_per_recording
from evaluate_modulation_mlp import (
    CLASS_NAMES, SEED, benchmark, predict, raw_input, sample_weights, targets, train_epoch, validation_split,
)


POOL_SIZES = (5, 1)
MAX_EPOCHS = 40
PATIENCE = 8


class SpectrumCNN(nn.Module):
    def __init__(self, regions):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, stride=2, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(regions),
            nn.Flatten(),
            nn.Linear(32 * regions, len(CLASS_NAMES)),
        )

    def forward(self, data):
        return self.layers(data.unsqueeze(1))


def training_state(x, meta, indices, regions, seed):
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    mean = x[indices].mean()
    scale = max(x[indices].std(), np.float32(1e-6))
    inputs = torch.from_numpy((x - mean) / scale)
    model = SpectrumCNN(regions)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    return model, optimizer, generator, inputs, targets(meta), sample_weights(meta.iloc[indices]), mean, scale


def choose_epochs(x, meta, training, fold, regions):
    inside, validation = validation_split(meta, training, fold)
    state = training_state(x, meta, inside, regions, SEED + fold)
    actual = targets(meta).numpy()[validation]
    best_score, best_epoch = -float("inf"), 1
    for epoch in range(1, MAX_EPOCHS + 1):
        train_epoch(state, inside)
        estimated = predict(state, validation)
        macro = f1_score(actual, estimated, labels=[0, 1, 2], average="macro")
        missed = ((actual != 2) & (estimated == 2)).sum() / (actual != 2).sum()
        utility = macro - missed
        if utility > best_score + 1e-8:
            best_score, best_epoch = utility, epoch
        if epoch - best_epoch >= PATIENCE:
            break
    return best_epoch


def fit(x, meta, training, regions, epochs, seed):
    state = training_state(x, meta, training, regions, seed)
    for _ in range(epochs):
        train_epoch(state, training)
    return state


def evaluate(regions, x, meta, shifted, shifted_inputs):
    name = f"CNN1D_pool{regions}"
    centered_results, shifted_results, epochs = [], [], []
    classes = np.asarray(CLASS_NAMES)
    for fold in sorted(meta["fold"].unique()):
        train_idx = np.flatnonzero(meta["fold"].ne(fold))
        test_idx = np.flatnonzero(meta["fold"].eq(fold))
        shifted_test = shifted.loc[shifted["fold"].eq(fold)]
        train_groups = set(meta.iloc[train_idx]["group"])
        assert train_groups.isdisjoint(set(meta.iloc[test_idx]["group"]) | set(shifted_test["group"]))
        best_epoch = choose_epochs(x, meta, train_idx, int(fold), regions)
        state = fit(x, meta, train_idx, regions, best_epoch, SEED + int(fold))
        centered_results.append(meta.iloc[test_idx].assign(
            architecture=name, best_epoch=best_epoch, predicted=classes[predict(state, test_idx)]))
        if not shifted_test.empty:
            shifted_results.append(shifted_test.assign(
                architecture=name, predicted=classes[predict_shifted(state, shifted_inputs[shifted_test.index])]))
        epochs.append(best_epoch)
        print(f"{name} | fold {fold}: epoch {best_epoch}", flush=True)
    final = fit(x, meta, np.arange(len(meta)), regions, int(np.median(epochs)), SEED)
    unseen = shifted.loc[shifted["fold"].eq(0)]
    assert set(meta["group"]).isdisjoint(unseen["group"])
    if not unseen.empty:
        shifted_results.append(unseen.assign(
            architecture=name, predicted=classes[predict_shifted(final, shifted_inputs[unseen.index])]))
    centered = pd.concat(centered_results, ignore_index=True)
    off_center = pd.concat(shifted_results, ignore_index=True)
    assert (len(centered), len(off_center)) == (4850, 750)
    assert not centered.duplicated(["recording", "sweep_idx"]).any()
    assert not off_center.duplicated(["recording", "sweep_idx"]).any()
    params = sum(parameter.numel() for parameter in final[0].parameters())
    print(f"{name}: {params} parameters; best epochs {epochs}", flush=True)
    return name, centered, off_center, final, params


def main(project, output, training_threads):
    sys.path.insert(0, str(project))
    from models.modulation.classifier import FEATURE_NAMES, ModulationClassifier, extract_features
    from scripts.evaluate_gb_averaging import fold_mapping, load_recordings

    torch.set_num_threads(training_threads)
    torch.set_num_interop_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    recordings = load_recordings(project / "data")
    folds = fold_mapping(recordings)
    rows, inputs = [], []
    for info, _, levels, _ in recordings:
        for i, trace in enumerate(levels):
            rows.append(info | {"sweep_idx": i, "fold": folds[info["group"]]})
            inputs.append(raw_input(trace))
    x, meta = np.stack(inputs), pd.DataFrame(rows)
    shifted, shifted_inputs = load_shifted(project)
    shifted["fold"] = shifted["group"].map(folds).fillna(0).astype(int)
    assert x.shape == (4850, 501) and shifted_inputs.shape == (750, 501)
    scores, centered_results, shifted_results, models = [], [], [], {}
    for pool in POOL_SIZES:
        name, centered, stress, model, params = evaluate(pool, x, meta, shifted, shifted_inputs)
        scores.append(summarize(name, centered, stress, params))
        centered_results.append(centered)
        shifted_results.append(stress)
        models[(name, 0.0)] = model
    torch.set_num_threads(1)
    timing = benchmark(recordings, models, extract_features, FEATURE_NAMES, ModulationClassifier()._model)
    timing = timing.rename(columns={"representation": "architecture"}).drop(columns="dropout")
    results = pd.DataFrame(scores).merge(timing, on="architecture", validate="one_to_one")
    results.to_csv(output / "scores.csv", index=False)
    stress = pd.concat(shifted_results, ignore_index=True)
    pd.concat(centered_results, ignore_index=True).to_csv(
        output / "centered_predictions.csv", index=False, encoding="utf-8-sig")
    stress.to_csv(output / "shifted_predictions.csv", index=False, encoding="utf-8-sig")
    metrics_per_recording(stress, ["architecture", "span_khz", "group"]).to_csv(
        output / "shifted_by_station.csv", index=False, encoding="utf-8-sig")
    print(results.round(5).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-threads", type=int, default=4)
    args = parser.parse_args()
    main(args.project, args.output, args.training_threads)
