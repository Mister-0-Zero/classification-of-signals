"""Compare small feature and raw-spectrum MLPs with grouped validation."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn


CLASS_NAMES = ["AM", "FM", "Noise"]
SEED = 20261119
MAX_EPOCHS = 60
PATIENCE = 10
BATCH_SIZE = 128


class ModulationMLP(nn.Sequential):
    def __init__(self, width, dropout, hidden=(32, 16)):
        layers = []
        incoming = width
        for size in hidden:
            layers.extend((nn.Linear(incoming, size), nn.ReLU()))
            incoming = size
        layers.extend((nn.Dropout(dropout), nn.Linear(incoming, len(CLASS_NAMES))))
        super().__init__(*layers)


def raw_input(amplitude):
    return np.asarray(amplitude - amplitude.max(), dtype=np.float32)


def collect(recordings, folds, extract_features, feature_names):
    rows, features, raw = [], [], []
    for info, axis, data, center in recordings:
        for index, sweep in enumerate(data):
            values = extract_features(axis, sweep, center)
            features.append([values[name] for name in feature_names])
            raw.append(raw_input(sweep))
            rows.append(info | {"fold": folds[info["group"]], "sweep_idx": index})
    metadata = pd.DataFrame(rows)
    assert metadata.groupby("group")["fold"].nunique().eq(1).all()
    assert metadata.groupby("recording").size().eq(50).all()
    inputs = {"features_10": np.asarray(features, dtype=np.float32), "raw_501": np.stack(raw)}
    assert all(np.isfinite(x).all() for x in inputs.values())
    return metadata, inputs


def standardize(x, inside):
    mean = x[inside].mean(axis=0)
    scale = x[inside].std(axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return torch.from_numpy((x - mean) / scale), mean, scale


def sample_weights(meta):
    group_count = meta.groupby("group")["label"].transform("size")
    class_groups = meta.groupby("label")["group"].transform("nunique")
    weights = 1 / (group_count * class_groups)
    return torch.from_numpy((weights / weights.mean()).to_numpy(dtype=np.float32))


def targets(meta):
    mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    return torch.from_numpy(meta["label"].map(mapping).to_numpy(dtype=np.int64, copy=True))


def training_state(x, meta, indices, dropout, seed, hidden=(32, 16)):
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    inputs, mean, scale = standardize(x, indices)
    model = ModulationMLP(x.shape[1], dropout, hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    y = targets(meta)
    weights = sample_weights(meta.iloc[indices])
    return model, optimizer, generator, inputs, y, weights, mean, scale


def train_epoch(state, indices, augment=None, rng=None, signal_penalty=0.0):
    model, optimizer, generator, x, y, weights, mean, scale = state
    model.train()
    for local in torch.randperm(len(indices), generator=generator).split(BATCH_SIZE):
        batch = indices[local.numpy()]
        optimizer.zero_grad(set_to_none=True)
        values = x[batch]
        if augment is not None:
            augment(values, batch, mean, scale, rng)
        logits = model(values)
        losses = nn.functional.cross_entropy(logits, y[batch], reduction="none")
        if signal_penalty:
            signal = y[batch].ne(CLASS_NAMES.index("Noise"))
            losses = losses + signal_penalty * logits.softmax(dim=1)[:, 2] * signal
        loss = (losses * weights[local]).sum() / weights[local].sum()
        loss.backward()
        optimizer.step()


def predict(state, indices):
    model, _, _, x, *_ = state
    model.eval()
    with torch.inference_mode():
        return model(x[indices]).argmax(dim=1).numpy()


def validation_split(meta, training, fold):
    rng = np.random.default_rng(SEED + fold)
    allowed = meta.iloc[training]
    held = []
    for label in CLASS_NAMES:
        groups = sorted(allowed.loc[allowed["label"].eq(label), "group"].unique())
        assert len(groups) >= 4
        held.extend(rng.choice(groups, size=2, replace=False))
    val_mask = allowed["group"].isin(held).to_numpy()
    inside, validation = training[~val_mask], training[val_mask]
    assert set(meta.iloc[inside]["group"]).isdisjoint(meta.iloc[validation]["group"])
    assert set(meta.iloc[validation]["label"]) == set(CLASS_NAMES)
    return inside, validation


def choose_epochs(x, meta, training, fold, dropout, hidden=(32, 16), *, augment=None, signal_penalty=0.0):
    inside, validation = validation_split(meta, training, fold)
    state = training_state(x, meta, inside, dropout, SEED + fold, hidden)
    actual = targets(meta).numpy()[validation]
    best_score, best_epoch = -float("inf"), 1
    rng = np.random.default_rng(SEED + fold) if augment is not None else None
    for epoch in range(1, MAX_EPOCHS + 1):
        train_epoch(state, inside, augment, rng, signal_penalty)
        estimated = predict(state, validation)
        macro = f1_score(actual, estimated, labels=[0, 1, 2], average="macro")
        missed = ((actual != 2) & (estimated == 2)).sum() / (actual != 2).sum()
        utility = macro - missed
        if utility > best_score + 1e-8:
            best_score, best_epoch = utility, epoch
        if epoch - best_epoch >= PATIENCE:
            break
    return best_epoch


def fit(x, meta, training, dropout, epochs, seed, hidden=(32, 16), *, augment=None, signal_penalty=0.0):
    state = training_state(x, meta, training, dropout, seed, hidden)
    rng = np.random.default_rng(seed) if augment is not None else None
    for _ in range(epochs):
        train_epoch(state, training, augment, rng, signal_penalty)
    return state


def evaluate(x, meta, dropout, representation):
    outputs, epochs = [], []
    for fold in sorted(meta["fold"].unique()):
        inside = np.flatnonzero(meta["fold"].ne(fold))
        outside = np.flatnonzero(meta["fold"].eq(fold))
        assert set(meta.iloc[inside]["group"]).isdisjoint(meta.iloc[outside]["group"])
        assert set(meta.iloc[inside]["recording"]).isdisjoint(meta.iloc[outside]["recording"])
        best_epoch = choose_epochs(x, meta, inside, int(fold), dropout)
        model = fit(x, meta, inside, dropout, best_epoch, SEED + int(fold))
        outputs.append(meta.iloc[outside].assign(
            representation=representation, dropout=dropout, best_epoch=best_epoch,
            predicted=np.asarray(CLASS_NAMES)[predict(model, outside)],
        ))
        epochs.append(best_epoch)
    estimate = pd.concat(outputs, ignore_index=True)
    assert len(estimate) == len(meta) and not estimate.duplicated(["recording", "sweep_idx"]).any()
    actual, prediction = estimate["label"], estimate["predicted"]
    f1 = f1_score(actual, prediction, labels=CLASS_NAMES, average="macro")
    print(f"{representation} | dropout {dropout}: Macro F1 {f1:.4f}, epochs {epochs}", flush=True)
    final = fit(x, meta, np.arange(len(meta)), dropout, int(np.median(epochs)), SEED)
    return estimate, final


def summarize(estimated):
    rows, matrices = [], []
    for (name, dropout), frame in estimated.groupby(["representation", "dropout"]):
        actual, predicted = frame["label"], frame["predicted"]
        matrix = confusion_matrix(actual, predicted, labels=CLASS_NAMES)
        f1 = f1_score(actual, predicted, labels=CLASS_NAMES, average=None)
        row = {"representation": name, "dropout": dropout, "accuracy": accuracy_score(actual, predicted),
               "macro_f1": f1.mean(), **dict(zip(["f1_am", "f1_fm", "f1_noise"], f1, strict=True)),
               "signal_as_noise": int(matrix[:2, 2].sum()), "noise_as_signal": int(matrix[2, :2].sum())}
        rows.append(row)
        matrices.extend({"representation": name, "dropout": dropout, "actual": label,
                         **dict(zip(CLASS_NAMES, counts, strict=True))}
                        for label, counts in zip(CLASS_NAMES, matrix, strict=True))
    return pd.DataFrame(rows), pd.DataFrame(matrices)


def benchmark(recordings, models, extract_features, feature_names, gb):
    rng = np.random.default_rng(SEED)
    traces = [(axis, sweeps[index], center) for _, axis, sweeps, center in recordings
              for index in rng.choice(len(sweeps), size=2, replace=False)]
    timings = {key: [] for key in [*models, ("gb_reference", 0.0)]}
    for state in models.values():
        state[0].eval()
    with torch.inference_mode():
        for repeat in range(4):
            for index in rng.permutation(len(traces)):
                axis, levels, center = traces[index]
                for model_index in rng.permutation(len(timings)):
                    key = list(timings)[model_index]
                    start = perf_counter_ns()
                    if key[0] == "gb_reference":
                        values = extract_features(axis, levels, center)
                        gb.predict(pd.DataFrame([values], columns=feature_names))
                    else:
                        model, _, _, _, _, _, mean, scale = models[key]
                        if key[0] == "features_10":
                            values = extract_features(axis, levels, center)
                            data = np.asarray([values[name] for name in feature_names], dtype=np.float32)
                        else:
                            data = raw_input(levels)
                        inputs = torch.from_numpy((data - mean) / scale).unsqueeze(0)
                        model(inputs).argmax(dim=1).item()
                    if repeat:
                        timings[key].append((perf_counter_ns() - start) / 1_000_000)
    return pd.DataFrame([
        {"representation": kind, "dropout": dropout, "n_measurements": len(values),
         "p50_ms": np.median(values), "p95_ms": np.quantile(values, .95)}
        for (kind, dropout), values in timings.items()
    ])


def main(project, output):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(project))
    from models.modulation.classifier import FEATURE_NAMES, ModulationClassifier, extract_features
    from scripts.evaluate_gb_averaging import fold_mapping, load_recordings

    output.mkdir(parents=True, exist_ok=True)
    recordings = load_recordings(project / "data")
    assert len(recordings) == 97
    meta, inputs = collect(recordings, fold_mapping(recordings), extract_features, FEATURE_NAMES)
    assert len(meta) == 4850
    predictions, models = [], {}
    for representation, x in inputs.items():
        for dropout in (0.0, 0.1):
            estimated, state = evaluate(x, meta, dropout, representation)
            predictions.append(estimated)
            models[(representation, dropout)] = state
    estimated = pd.concat(predictions, ignore_index=True)
    scores, matrices = summarize(estimated)
    timings = benchmark(recordings, models, extract_features, FEATURE_NAMES, ModulationClassifier()._model)
    scores.to_csv(output / "scores.csv", index=False)
    timings.to_csv(output / "latency.csv", index=False)
    matrices.to_csv(output / "confusion.csv", index=False)
    estimated.to_csv(output / "predictions.csv", index=False, encoding="utf-8-sig")
    (output / "protocol.json").write_text(
        json.dumps({"torch": torch.__version__, "threads": torch.get_num_threads(),
                    "seed": SEED, "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "batch_size": BATCH_SIZE}, indent=2),
        encoding="utf-8",
    )
    print(scores.round(5).to_string(index=False))
    print(timings.round(5).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.project, args.output)
