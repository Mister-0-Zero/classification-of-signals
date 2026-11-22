"""Compare frequency-resolution, receptive-field and pooling changes for 1D CNNs."""

import argparse
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch import nn

from evaluate_mlp_architectures import summarize
from evaluate_mlp_shifted import load_shifted, metrics_per_recording
from evaluate_modulation_cnn1d import evaluate
from evaluate_modulation_mlp import CLASS_NAMES, benchmark, raw_input


ARCHITECTURES = ("resolution_501", "late_hybrid", "dilated_hybrid", "late_dense_25")


class RefinedCNN(nn.Module):
    def __init__(self, regions, *, architecture):
        super().__init__()
        self.architecture = architecture
        if architecture == "resolution_501":
            # Preserve the original two kernels and head while removing early decimation.
            self.trunk = nn.Sequential(
                nn.Conv1d(1, 16, 9, padding=4), nn.ReLU(),
                nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(),
            )
            self.regional = nn.AdaptiveAvgPool1d(regions)
            self.head = nn.Linear(32 * regions, len(CLASS_NAMES))
        else:
            dilation_mid, dilation_last = (3, 9) if architecture == "dilated_hybrid" else (1, 1)
            self.trunk = nn.Sequential(
                nn.Conv1d(1, 16, 9, padding=4), nn.ReLU(),
                nn.Conv1d(16, 24, 7, dilation=dilation_mid, padding=3 * dilation_mid), nn.ReLU(),
                nn.AvgPool1d(2),
                nn.Conv1d(24, 32, 7, dilation=dilation_last, padding=3 * dilation_last), nn.ReLU(),
            )
            if architecture == "late_dense_25":
                self.regional = nn.AdaptiveAvgPool1d(25)
                self.head = nn.Sequential(nn.Linear(32 * 25, 64), nn.ReLU(), nn.Linear(64, len(CLASS_NAMES)))
            else:
                self.regional = nn.AdaptiveAvgPool1d(regions)
                self.head = nn.Sequential(nn.Linear(32 * (regions + 2), 32), nn.ReLU(),
                                          nn.Linear(32, len(CLASS_NAMES)))

    def forward(self, raw):
        signals = self.trunk(raw.unsqueeze(1))
        pieces = self.regional(signals).flatten(1)
        if self.architecture in ("late_hybrid", "dilated_hybrid"):
            pieces = torch.cat((pieces, signals.mean(dim=-1), signals.amax(dim=-1)), dim=1)
        return self.head(pieces)


def write_results(output, results, ordinary, shifted):
    predictions = pd.concat(ordinary, ignore_index=True)
    stress = pd.concat(shifted, ignore_index=True)
    matrices = []
    for name, cases in predictions.groupby("architecture"):
        confusion = confusion_matrix(cases["label"], cases["predicted"], labels=CLASS_NAMES)
        matrices.extend({"architecture": name, "actual": label,
                         **dict(zip(CLASS_NAMES, values, strict=True))}
                        for label, values in zip(CLASS_NAMES, confusion, strict=True))
        hits = results["architecture"].eq(name)
        results.loc[hits, "am_as_fm"] = confusion[0, 1]
        results.loc[hits, "noise_as_fm"] = confusion[2, 1]
    results.to_csv(output / "scores.csv", index=False)
    pd.DataFrame(matrices).to_csv(output / "confusion.csv", index=False)
    predictions.to_csv(output / "centered_predictions.csv", index=False, encoding="utf-8-sig")
    stress.to_csv(output / "shifted_predictions.csv", index=False, encoding="utf-8-sig")
    metrics_per_recording(stress, ["architecture", "span_khz", "group"]).to_csv(
        output / "shifted_by_station.csv", index=False, encoding="utf-8-sig")
    print(results.round(5).to_string(index=False))


def main(project, output, training_threads):
    sys.path.insert(0, str(project))
    from models.modulation.classifier import FEATURE_NAMES, ModulationClassifier, extract_features
    from scripts.evaluate_gb_averaging import fold_mapping, load_recordings

    torch.set_num_threads(training_threads)
    torch.set_num_interop_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    recordings = load_recordings(project / "data")
    folds = fold_mapping(recordings)
    meta, x = [], []
    for info, _, levels, _ in recordings:
        for sweep, trace in enumerate(levels):
            meta.append(info | {"sweep_idx": sweep, "fold": folds[info["group"]]})
            x.append(raw_input(trace))
    meta, x = pd.DataFrame(meta), np.stack(x)
    stress, shifted_inputs = load_shifted(project)
    stress["fold"] = stress["group"].map(folds).fillna(0).astype(int)
    assert x.shape == (4850, 501) and shifted_inputs.shape == (750, 501)
    scores, ordinary, off_center, models = [], [], [], {}
    for architecture in ARCHITECTURES:
        factory = partial(RefinedCNN, architecture=architecture)
        name, centered, shifted, model, params = evaluate(
            5, x, meta, stress, shifted_inputs, model_factory=factory, name=architecture)
        scores.append(summarize(name, centered, shifted, params))
        ordinary.append(centered)
        off_center.append(shifted)
        models[(name, 0.0)] = model
        # Checkpoint each completed architecture so an interruption preserves its results.
        pd.DataFrame(scores).to_csv(output / "progress_scores.csv", index=False)
        print(f"{name}: F1={scores[-1]['macro_f1']:.4f}, "
              f"shifted FM {scores[-1]['shifted_fm_correct_all']}/750", flush=True)
    torch.set_num_threads(1)
    timing = benchmark(recordings, models, extract_features, FEATURE_NAMES, ModulationClassifier()._model)
    timing = timing.rename(columns={"representation": "architecture"}).drop(columns="dropout")
    results = pd.DataFrame(scores).merge(timing, on="architecture", validate="one_to_one")
    write_results(output, results, ordinary, off_center)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-threads", type=int, default=4)
    args = parser.parse_args()
    main(args.project, args.output, args.training_threads)
