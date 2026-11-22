"""Train raw-spectrum MLPs with bounded signal shifts on training groups only."""

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
    BATCH_SIZE, CLASS_NAMES, MAX_EPOCHS, PATIENCE, SEED, predict, raw_input,
    targets, training_state, validation_split,
)


ARCHITECTURES = [(32, 16), (32, 32)]
SHIFTS = (-40, -20, 20, 40)
SHIFT_PROBABILITY = .5
EDGE_POWER_LIMIT = .05


def prepare_shifts(levels, labels, mean, scale):
    """Translate sampled power levels without wrapping. Estimate signal support from excess power."""
    relative = levels - levels.max(axis=1, keepdims=True)
    power = 10 ** (relative / 10)
    background = np.median(np.concatenate([power[:, :40], power[:, -40:]], axis=1), axis=1)
    excess = np.maximum(power - background[:, None], 0)
    total = excess.sum(axis=1)
    center_peak = levels.argmax(axis=1)
    noise = labels == "Noise"
    choices, allowed = [], []
    for amount in SHIFTS:
        width = abs(amount)
        if amount > 0:
            destination = np.concatenate([levels[:, 1:width + 1][:, ::-1], levels[:, :-width]], axis=1)
            outgoing = excess[:, -width:]
            fill_source = excess[:, 1:width + 1]
        else:
            destination = np.concatenate([levels[:, width:], levels[:, -width - 1:-1][:, ::-1]], axis=1)
            outgoing = excess[:, :width]
            fill_source = excess[:, -width - 1:-1]
        retains_peak = (0 <= center_peak + amount) & (center_peak + amount < levels.shape[1])
        safe_edges = (total > 0) & (outgoing.sum(axis=1) <= EDGE_POWER_LIMIT * total)
        clean_fill = fill_source.sum(axis=1) <= EDGE_POWER_LIMIT * total
        possible = noise | (retains_peak & safe_edges & clean_fill)
        raw = (destination - destination.max(axis=1, keepdims=True)).astype(np.float32)
        transformed = (raw - mean) / scale
        assert np.isfinite(transformed).all()
        allowed.append(possible)
        choices.append(transformed)
    return torch.from_numpy(np.stack(choices, axis=1)), np.column_stack(allowed)


def augmentation_state(inputs, levels, meta, train, hidden, seed):
    original = training_state(inputs, meta, train, 0.0, seed, hidden)
    shifts, allowed = prepare_shifts(levels[train], meta.iloc[train]["label"].to_numpy(), *original[-2:])
    assert shifts.shape[:2] == (len(train), len(SHIFTS))
    return original, shifts, allowed, np.random.default_rng(seed + 343_277)


def train_epoch(state, train):
    original, choices, allowed, rng = state
    model, optimizer, generator, inputs, targets_all, weights, _, _ = original
    model.train()
    change = rng.random(len(train)) < SHIFT_PROBABILITY
    random_rank = np.where(allowed, rng.random(allowed.shape), -1)
    selected = random_rank.argmax(axis=1)
    enabled = change & allowed.any(axis=1)
    for local in torch.randperm(len(train), generator=generator).split(BATCH_SIZE):
        positions = local.numpy()
        batch = train[positions]
        vectors = inputs[batch].clone()
        mask = enabled[positions]
        replacement = positions[mask]
        if replacement.size:
            vectors[torch.from_numpy(mask)] = choices[replacement, selected[replacement]]
        optimizer.zero_grad(set_to_none=True)
        losses = nn.functional.cross_entropy(model(vectors), targets_all[batch], reduction="none")
        loss = (losses * weights[local]).sum() / weights[local].sum()
        loss.backward()
        optimizer.step()
    return int(enabled.sum())


def choose_epochs(x, raw, meta, training, fold, hidden):
    inside, validation = validation_split(meta, training, fold)
    state = augmentation_state(x, raw, meta, inside, hidden, SEED + fold)
    actual = targets(meta).numpy()[validation]
    best_score, best_epoch = -float("inf"), 1
    for epoch in range(1, MAX_EPOCHS + 1):
        train_epoch(state, inside)
        estimate = predict(state[0], validation)
        macro = f1_score(actual, estimate, labels=[0, 1, 2], average="macro")
        miss_rate = ((actual != 2) & (estimate == 2)).sum() / (actual != 2).sum()
        if macro - miss_rate > best_score + 1e-8:
            best_score, best_epoch = macro - miss_rate, epoch
        if epoch - best_epoch >= PATIENCE:
            break
    return best_epoch


def fit_augmented(inputs, raw, meta, training, hidden, epochs, seed):
    state = augmentation_state(inputs, raw, meta, training, hidden, seed)
    augmented = sum(train_epoch(state, training) for _ in range(epochs))
    return state[0], augmented / (len(training) * epochs)


def evaluate_one(hidden, meta, inputs, raw, stress_meta, stress_inputs):
    name = "-".join(map(str, (501, *hidden, 3)))
    centered, shifted, epochs, rates = [], [], [], []
    names = np.asarray(CLASS_NAMES)
    for fold in sorted(meta["fold"].unique()):
        train = np.flatnonzero(meta["fold"].ne(fold))
        test = np.flatnonzero(meta["fold"].eq(fold))
        stress = stress_meta.loc[stress_meta["fold"].eq(fold)]
        assert set(meta.iloc[train]["group"]).isdisjoint(set(meta.iloc[test]["group"]) | set(stress["group"]))
        selected = choose_epochs(inputs, raw, meta, train, int(fold), hidden)
        state, rate = fit_augmented(inputs, raw, meta, train, hidden, selected, SEED + int(fold))
        centered.append(meta.iloc[test].assign(architecture=name, best_epoch=selected,
                                              predicted=names[predict(state, test)]))
        if not stress.empty:
            shifted.append(stress.assign(architecture=name, predicted=names[
                predict_shifted(state, stress_inputs[stress.index])]))
        epochs.append(selected)
        rates.append(rate)
    final, _ = fit_augmented(inputs, raw, meta, np.arange(len(meta)), hidden, int(np.median(epochs)), SEED)
    unseen = stress_meta.loc[stress_meta["fold"].eq(0)]
    assert set(meta["group"]).isdisjoint(unseen["group"])
    if not unseen.empty:
        shifted.append(unseen.assign(architecture=name, predicted=names[
            predict_shifted(final, stress_inputs[unseen.index])]))
    centered, shifted = pd.concat(centered, ignore_index=True), pd.concat(shifted, ignore_index=True)
    assert len(centered) == 4850 and len(shifted) == 750
    assert not centered.duplicated(["recording", "sweep_idx"]).any()
    assert not shifted.duplicated(["recording", "sweep_idx"]).any()
    params = sum(parameter.numel() for parameter in final[0].parameters())
    print(f"{name}: mean augmented training fraction {np.mean(rates):.3f}, epoch range {min(epochs)}-{max(epochs)}",
          flush=True)
    return name, centered, shifted, params, np.mean(rates)


def check_reference(plain, augmented):
    keys = ["architecture", "recording", "sweep_idx", "fold", "group", "label"]
    joined = plain.merge(augmented[keys], on=keys, validate="one_to_one")
    assert len(joined) == len(plain) == len(augmented)


def main(project, output, baseline):
    sys.path.insert(0, str(project))
    from scripts.evaluate_gb_averaging import fold_mapping, load_recordings

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    traces = load_recordings(project / "data")
    fold_map = fold_mapping(traces)
    meta, levels = [], []
    for desc, axis, recording, _ in traces:
        np.testing.assert_allclose(np.diff(axis), 500, rtol=0, atol=1e-3)
        meta.extend(desc | {"fold": fold_map[desc["group"]], "sweep_idx": i} for i in range(len(recording)))
        levels.extend(recording)
    meta, levels = pd.DataFrame(meta), np.stack(levels)
    inputs = np.stack([raw_input(sweep) for sweep in levels])
    stress, stress_inputs = load_shifted(project)
    stress["fold"] = stress["group"].map(fold_map).fillna(0).astype(int)
    reference = {name: pd.read_csv(baseline / f"{name}_predictions.csv") for name in ("centered", "shifted")}
    rows, centered_results, shifted_results = [], [], []
    for hidden in ARCHITECTURES:
        name, centered, shifted, params, rate = evaluate_one(hidden, meta, inputs, levels, stress, stress_inputs)
        before_center = reference["centered"].loc[reference["centered"]["architecture"].eq(name)]
        before_shift = reference["shifted"].loc[reference["shifted"]["architecture"].eq(name)]
        check_reference(before_center, centered)
        check_reference(before_shift, shifted)
        rows.append(summarize(name, before_center, before_shift, params) | {"augmented": False, "aug_fraction": 0.0})
        rows.append(summarize(name, centered, shifted, params) | {"augmented": True, "aug_fraction": rate})
        centered_results.extend([before_center.assign(augmented=False), centered.assign(augmented=True)])
        shifted_results.extend([before_shift.assign(augmented=False), shifted.assign(augmented=True)])
    scores = pd.DataFrame(rows)
    scores.to_csv(output / "scores.csv", index=False)
    centered = pd.concat(centered_results, ignore_index=True)
    shifted = pd.concat(shifted_results, ignore_index=True)
    centered.to_csv(output / "centered_predictions.csv", index=False, encoding="utf-8-sig")
    shifted.to_csv(output / "shifted_predictions.csv", index=False, encoding="utf-8-sig")
    metrics_per_recording(shifted, ["architecture", "augmented", "span_khz", "group"]).to_csv(
        output / "shifted_by_station.csv", index=False, encoding="utf-8-sig")
    print(scores.round(5).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    main(args.project, args.output, args.baseline)
