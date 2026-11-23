"""Evaluate random FM shifts for MLP, 1D CNN and Deep Sets on seven traces."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluate_gb_averaging import average_dbm, fold_mapping, load_recordings
from evaluate_gb_shift_augmentation import AUGMENTATION_SEED, shift_geometries, synthesize
from evaluate_gb_shifted import load_shifted_recordings
from evaluate_modulation_deep_sets import MODES, evaluate, latency, prepare, scores


MLP = "mlp_mean_spectrum"
ARCHITECTURES = (MLP, *MODES)
GEOMETRY_DRAWS = 4


def model_input(inputs, mode):
    # MLP and CNN receive the exact same averaged spectrum.
    return inputs["cnn_mean_spectrum" if mode == MLP else mode]


def synthetic_inputs(records, modes):
    rng = np.random.default_rng(AUGMENTATION_SEED)
    views = {mode: {} for mode in modes}
    for index, (info, axis, traces, center) in enumerate(records):
        if info["label"] != "FM":
            continue
        transformed = [
            np.stack([synthesize(axis, trace, center, span, offset)[1] for trace in traces])
            for _ in range(GEOMETRY_DRAWS) for span, offset in shift_geometries(rng)
        ]
        inputs = prepare(np.stack(transformed), average_dbm)
        for mode in modes:
            views[mode][index] = torch.from_numpy(model_input(inputs, mode))
    return views


def main(project, output, mode, training_threads, with_baseline):
    torch.set_num_threads(training_threads)
    torch.set_num_interop_threads(1)
    records = load_recordings(project / "data")
    shifted_records = load_shifted_recordings(project)
    folds = fold_mapping(records)
    meta = pd.DataFrame([info | {"fold": folds[info["group"]]} for info, *_ in records])
    shifted = pd.DataFrame([info | {"fold": folds.get(info["group"], 0)} for info, *_ in shifted_records])
    raw = np.stack([levels for _, _, levels, _ in records])
    stress = np.stack([levels for _, _, levels, _ in shifted_records])
    train_x = model_input(prepare(raw, average_dbm), mode)
    test_x = model_input(prepare(stress, average_dbm), mode)
    variants = [("baseline", None)] if mode == MLP or with_baseline else []
    variants.append(("random_fm", synthetic_inputs(records, [mode])[mode]))
    output.mkdir(parents=True, exist_ok=True)
    rows, models = [], {}
    for variant, augmentation in variants:
        print(f"{mode} | {variant}", flush=True)
        centered, offcenter, final = evaluate(mode, train_x, test_x, meta, shifted, augmentation)
        params = sum(parameter.numel() for parameter in final[0].parameters())
        row = scores(mode, centered, offcenter, params) | {"variant": variant}
        rows.append(row)
        models[variant] = final
        centered.to_csv(output / f"{variant}_centered.csv", index=False)
        offcenter.to_csv(output / f"{variant}_shifted.csv", index=False)
        pd.DataFrame(rows).to_csv(output / "scores.csv", index=False)
        print(f"{mode} | {variant}: Macro F1 {row['macro_f1']:.4f}", flush=True)

    torch.set_num_threads(1)
    timings = [latency(raw, {mode: final}, average_dbm).assign(variant=variant)
               for variant, final in models.items()]
    pd.concat(timings, ignore_index=True).to_csv(output / "latency.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=ARCHITECTURES, required=True)
    parser.add_argument("--training-threads", type=int, default=4)
    parser.add_argument("--with-baseline", action="store_true", help="Also retrain the CNN or Deep Sets control")
    args = parser.parse_args()
    main(args.project, args.output, args.model, args.training_threads, args.with_baseline)
