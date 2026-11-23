"""Compare Deep Sets with the same CNN fed a mean of seven analyzer sweeps."""

import argparse
import sys
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn

from evaluate_mlp_shifted import load_shifted, metrics_per_recording
from evaluate_modulation_mlp import CLASS_NAMES, SEED, sample_weights, targets, validation_split


WINDOW = 7
ENDPOINTS = np.arange(9, 50)
FIRST_WINDOW = ENDPOINTS[0] - WINDOW + 1
MAX_EPOCHS = 32
PATIENCE = 7
BATCH_RECORDINGS = 3
MODES = ("cnn_mean_spectrum", "deep_sets")


class Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv1d(1, 16, 9, padding=4), nn.ReLU(),
            nn.Conv1d(16, 24, 7, padding=3), nn.ReLU(),
            nn.AvgPool1d(2),
            nn.Conv1d(24, 32, 7, padding=3), nn.ReLU(),
            nn.AdaptiveAvgPool1d(25), nn.Flatten(),
            nn.Linear(800, 64), nn.ReLU(),
        )

    def forward(self, spectra):
        return super().forward(spectra.unsqueeze(1))


class WindowClassifier(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        if mode == "mlp_mean_spectrum":
            self.encoder = nn.Sequential(nn.Linear(501, 200), nn.ReLU(), nn.Linear(200, 20),
                                         nn.ReLU(), nn.Dropout(.1))
            self.output = nn.Linear(20, len(CLASS_NAMES))
        else:
            self.encoder = Encoder()
            self.output = nn.Linear(64, len(CLASS_NAMES))

    def forward(self, series):
        encoded = self.encoder(series.flatten(0, 1)).unflatten(0, series.shape[:2])
        if self.mode == "deep_sets":
            # Share the representation of each sweep between overlapping windows.
            encoded = encoded.unfold(1, WINDOW, 1)[:, FIRST_WINDOW:].mean(dim=-1)
        return self.output(encoded)

    def predict_burst(self, spectra):
        encoded = self.encoder(spectra)
        if self.mode == "deep_sets":
            encoded = encoded.mean(dim=0, keepdim=True)
        return self.output(encoded).argmax(dim=1).item()


def relative_levels(spectra):
    return (spectra - spectra.max(axis=-1, keepdims=True)).astype(np.float32)


def prepare(raw, average_dbm):
    mean_windows = np.stack([
        np.stack([average_dbm(trace[end - WINDOW + 1:end + 1]) for end in ENDPOINTS])
        for trace in raw
    ])
    return {"cnn_mean_spectrum": relative_levels(mean_windows), "deep_sets": relative_levels(raw)}


def training_state(mode, inputs, meta, indices, seed):
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    mean = inputs[indices].mean()
    scale = max(inputs[indices].std(), np.float32(1e-6))
    data = torch.from_numpy((inputs - mean) / scale)
    model = WindowClassifier(mode)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    return model, optimizer, generator, data, targets(meta), sample_weights(meta.iloc[indices]), mean, scale


def train_epoch(state, indices, augmented=None, augmentation_rng=None):
    model, optimizer, generator, data, labels, weights, _, _ = state
    model.train()
    shifted = {} if augmented is None else {
        index: augmented[index][augmentation_rng.integers(len(augmented[index]))]
        for index in indices if index in augmented and augmentation_rng.random() < .5
    }
    for local in torch.randperm(len(indices), generator=generator).split(BATCH_RECORDINGS):
        batch = indices[local.numpy()]
        values = data[batch]
        for position, index in enumerate(batch):
            if index in shifted:
                values[position] = (shifted[index] - state[-2]) / state[-1]
        optimizer.zero_grad(set_to_none=True)
        logits = model(values)
        expected = labels[batch, None].expand(-1, logits.shape[1])
        loss = nn.functional.cross_entropy(logits.flatten(0, 1), expected.reshape(-1), reduction="none")
        loss = loss.reshape(len(batch), -1).mean(dim=1)
        loss = (loss * weights[local]).sum() / weights[local].sum()
        loss.backward()
        optimizer.step()


def predict(state, records):
    model = state[0]
    model.eval()
    with torch.inference_mode():
        return np.concatenate([model(batch).argmax(dim=2).numpy()
                               for batch in records.split(BATCH_RECORDINGS)], axis=0)


def choose_epochs(mode, inputs, meta, training, fold, augmented=None):
    inside, validation = validation_split(meta, training, fold)
    state = training_state(mode, inputs, meta, inside, SEED + fold)
    actual = np.repeat(targets(meta).numpy()[validation], len(ENDPOINTS))
    best_score, best_epoch = -float("inf"), 1
    augmentation_rng = np.random.default_rng(SEED + fold)
    for epoch in range(1, MAX_EPOCHS + 1):
        train_epoch(state, inside, augmented, augmentation_rng)
        predicted = predict(state, state[3][validation]).reshape(-1)
        macro = f1_score(actual, predicted, labels=[0, 1, 2], average="macro")
        missed = ((actual != 2) & (predicted == 2)).sum() / (actual != 2).sum()
        utility = macro - missed
        if utility > best_score + 1e-8:
            best_score, best_epoch = utility, epoch
        if epoch - best_epoch >= PATIENCE:
            break
    return best_epoch


def fit(mode, inputs, meta, training, epochs, seed, augmented=None):
    state = training_state(mode, inputs, meta, training, seed)
    augmentation_rng = np.random.default_rng(seed)
    for _ in range(epochs):
        train_epoch(state, training, augmented, augmentation_rng)
    return state


def attach(meta, predictions, mode):
    assert predictions.shape == (len(meta), len(ENDPOINTS))
    rows = meta.loc[meta.index.repeat(len(ENDPOINTS))].reset_index(drop=True)
    return rows.assign(model=mode, end_sweep=np.tile(ENDPOINTS, len(meta)),
                       predicted=np.asarray(CLASS_NAMES)[predictions.reshape(-1)])


def predict_shifted(state, data):
    return predict(state, torch.from_numpy((data - state[-2]) / state[-1]))


def evaluate(mode, train_x, test_x, meta, offcenter, augmented=None):
    ordinary, shifted, epochs = [], [], []
    for fold in sorted(meta["fold"].unique()):
        train = np.flatnonzero(meta["fold"].ne(fold))
        held = np.flatnonzero(meta["fold"].eq(fold))
        stress = np.flatnonzero(offcenter["fold"].eq(fold))
        assert set(meta.iloc[train]["group"]).isdisjoint(set(meta.iloc[held]["group"]) |
                                                       set(offcenter.iloc[stress]["group"]))
        epoch = choose_epochs(mode, train_x, meta, train, int(fold), augmented)
        state = fit(mode, train_x, meta, train, epoch, SEED + int(fold), augmented)
        ordinary.append(attach(meta.iloc[held], predict(state, state[3][held]), mode))
        if stress.size:
            shifted.append(attach(offcenter.iloc[stress], predict_shifted(state, test_x[stress]), mode))
        epochs.append(epoch)
        print(f"{mode} | fold {fold}: selected epoch {epoch}", flush=True)
    final = fit(mode, train_x, meta, np.arange(len(meta)), int(np.median(epochs)), SEED, augmented)
    unseen = np.flatnonzero(offcenter["fold"].eq(0))
    assert set(meta["group"]).isdisjoint(offcenter.iloc[unseen]["group"])
    if unseen.size:
        shifted.append(attach(offcenter.iloc[unseen], predict_shifted(final, test_x[unseen]), mode))
    ordinary, shifted = pd.concat(ordinary, ignore_index=True), pd.concat(shifted, ignore_index=True)
    assert (len(ordinary), len(shifted)) == (3977, 615)
    assert not ordinary.duplicated(["recording", "end_sweep"]).any()
    assert not shifted.duplicated(["recording", "end_sweep"]).any()
    return ordinary, shifted, final


def scores(mode, centered, shifted, params):
    combined = pd.concat([centered, shifted], ignore_index=True)
    actual, pred = combined["label"], combined["predicted"]
    matrix = confusion_matrix(actual, pred, labels=CLASS_NAMES)
    f1 = f1_score(actual, pred, labels=CLASS_NAMES, average=None)
    out = {"model": mode, "parameters": params, "accuracy": accuracy_score(actual, pred), "macro_f1": f1.mean(),
           **dict(zip(["f1_am", "f1_fm", "f1_noise"], f1, strict=True)),
           "signal_as_noise": int(matrix[:2, 2].sum()), "noise_as_signal": int(matrix[2, :2].sum())}
    for suffix, part in [("all", combined), ("centered", centered)]:
        fm = part.loc[part["label"].eq("FM"), "predicted"]
        out |= {f"fm_correct_{suffix}": int(fm.eq("FM").sum()), f"fm_n_{suffix}": len(fm)}
    for span, part in [("all", shifted), *shifted.groupby("span_khz")]:
        suffix = str(int(span)) if span != "all" else span
        correct = int(part["predicted"].eq("FM").sum())
        out |= {f"shifted_correct_{suffix}": correct, f"shifted_n_{suffix}": len(part),
                f"shifted_as_noise_{suffix}": int(part["predicted"].eq("Noise").sum())}
    return out


def latency(raw, models, average_dbm):
    rng = np.random.default_rng(SEED)
    windows = [record[end - WINDOW + 1:end + 1] for record in raw
               for end in rng.choice(ENDPOINTS, size=2, replace=False)]
    elapsed = {key: [] for key in models}
    for model, *_ in models.values():
        model.eval()
    with torch.inference_mode():
        for repeat in range(4):
            for i in rng.permutation(len(windows)):
                for mode in rng.permutation(list(models)):
                    model, _, _, _, _, _, mean, scale = models[mode]
                    start = perf_counter_ns()
                    x = windows[i] if mode == "deep_sets" else average_dbm(windows[i])[None, :]
                    data = torch.from_numpy((relative_levels(x) - mean) / scale)
                    model.predict_burst(data)
                    if repeat:
                        elapsed[mode].append((perf_counter_ns() - start) / 1_000_000)
    return pd.DataFrame([{"model": mode, "timed_calls": len(values), "p50_ms": np.median(values),
                          "p95_ms": np.quantile(values, .95)} for mode, values in elapsed.items()])


def main(project, output, training_threads):
    sys.path.insert(0, str(project))
    from scripts.evaluate_gb_averaging import average_dbm, fold_mapping, load_recordings

    torch.set_num_threads(training_threads)
    torch.set_num_interop_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    records = load_recordings(project / "data")
    folds = fold_mapping(records)
    meta = pd.DataFrame([info | {"fold": folds[info["group"]]} for info, *_ in records])
    raw = np.stack([levels for _, _, levels, _ in records])
    shifted_meta, shifted_raw = load_shifted(project, normalize=False)
    metadata_columns = ["recording", "group", "label", "span_khz", "offset_khz", "station_name"]
    offcenter = shifted_meta[metadata_columns].drop_duplicates().reset_index(drop=True)
    offcenter["fold"] = offcenter["group"].map(folds).fillna(0).astype(int)
    matrices = []
    for name in offcenter["recording"]:
        members = shifted_meta.loc[shifted_meta["recording"].eq(name)]
        assert np.array_equal(members["sweep_idx"], np.arange(50))
        matrices.append(shifted_raw[members.index])
    assert len(meta) == 97 and len(offcenter) == 15
    train_inputs, stress_inputs = prepare(raw, average_dbm), prepare(np.stack(matrices), average_dbm)
    centered, stress, table, models = [], [], [], {}
    for mode in MODES:
        ordinary, shifted, final = evaluate(mode, train_inputs[mode], stress_inputs[mode], meta, offcenter)
        params = sum(parameter.numel() for parameter in final[0].parameters())
        table.append(scores(mode, ordinary, shifted, params))
        print(f"{mode}: Macro F1 {table[-1]['macro_f1']:.4f}, shifted FM "
              f"{table[-1]['shifted_correct_all']}/{table[-1]['shifted_n_all']}", flush=True)
        pd.DataFrame(table).to_csv(output / "progress_scores.csv", index=False)
        centered.append(ordinary)
        stress.append(shifted)
        models[mode] = final
    torch.set_num_threads(1)
    metrics = pd.DataFrame(table).merge(latency(raw, models, average_dbm), on="model", validate="one_to_one")
    metrics.to_csv(output / "scores.csv", index=False)
    pd.concat(centered, ignore_index=True).to_csv(output / "centered_predictions.csv", index=False, encoding="utf-8-sig")
    stress = pd.concat(stress, ignore_index=True)
    stress.to_csv(output / "shifted_predictions.csv", index=False, encoding="utf-8-sig")
    metrics_per_recording(stress, ["model", "span_khz", "group"]).to_csv(
        output / "shifted_by_station.csv", index=False, encoding="utf-8-sig")
    print(metrics.round(5).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-threads", type=int, default=4)
    args = parser.parse_args()
    main(args.project, args.output, args.training_threads)
