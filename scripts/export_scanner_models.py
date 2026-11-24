"""Fit deployment weights on the curated recordings. Usage: python scripts/export_scanner_models.py DATA_DIR"""
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.modulation.classifier import FEATURE_NAMES, extract_features
from evaluate_fast_mlp import RandomFMShift
from evaluate_gb_averaging import load_recordings, train
from evaluate_gb_shifted import feature_table
from evaluate_modulation_mlp import SEED, raw_input, fit
from evaluate_modulation_deep_sets import relative_levels, choose_epochs, fit as fit_sets
from evaluate_neural_shift_augmentation import synthetic_inputs


def export_net(state, path):
    model = state[0].eval()
    weights = {key: value.detach().numpy() for key, value in model.state_dict().items()}
    np.savez_compressed(path, mean=state[-2], scale=state[-1], **weights)


def main(data):
    torch.set_num_threads(4)
    records = load_recordings(data)
    if len(records) != 97:
        raise ValueError("Expected the 97 curated training series")
    out = ROOT / "models/modulation"
    template = joblib.load(out / "gradient_boosting.joblib")["model"]
    for window, first in [(1, 0), (7, 9)]:
        frame = feature_table(records, {}, window, first, extract_features)
        model = train(template, frame, FEATURE_NAMES)
        joblib.dump(model, out / f"scanner_gb{window}.joblib", compress=3)
    meta = pd.DataFrame([info for info, *_ in records])
    traces = np.stack([levels for _, _, levels, _ in records])
    sweeps = meta.loc[meta.index.repeat(traces.shape[1])].reset_index(drop=True)
    inputs = np.stack([raw_input(trace) for series in traces for trace in series])
    # Median epoch across the nine compact_shift outer folds.
    state = fit(inputs, sweeps, np.arange(len(inputs)), .1, 18, SEED, (96, 32),
                augment=RandomFMShift(records, sweeps.label))
    export_net(state, out / "scanner_mlp.npz")
    inputs = relative_levels(traces)
    augmented = synthetic_inputs(records, ["deep_sets"])["deep_sets"]
    indices = np.arange(len(records))
    # Select deployment training duration on a grouped internal validation split.
    epochs = choose_epochs("deep_sets", inputs, meta, indices, 1, augmented)
    print("Deep Sets epochs:", epochs, flush=True)
    state = fit_sets("deep_sets", inputs, meta, indices, epochs, SEED, augmented)
    export_net(state, out / "scanner_sets.npz")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
