"""Audit the raw-spectrum MLP and measure transfer to off-center FM."""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_shifted(project, *, normalize=True):
    rows, raw = [], []
    base = project / "data"
    for path in sorted((base / "FM_dop").glob("**/samples/*.csv")):
        config_file = path.parent.parent / "run_config.json"
        settings = json.loads(config_file.read_text(encoding="utf-8-sig"))
        station = float(re.search(r"station_frequency=([0-9.]+) MHz", settings["note"])[1])
        midpoint = float(settings["center_freq_mhz"])
        capture = pd.read_csv(path, usecols=["class_name", "sweep_idx", "point_idx", "amplitude_dbm"])
        assert set(capture["class_name"]) == {"FM"} == {settings["class_name"]}
        spectrum = capture.pivot(index="sweep_idx", columns="point_idx", values="amplitude_dbm")
        assert np.array_equal(spectrum.index, np.arange(50))
        assert np.array_equal(spectrum.columns, np.arange(501))
        traces = spectrum.to_numpy(dtype=float)
        assert np.isfinite(traces).all()
        shifted = (traces - traces.max(axis=1, keepdims=True)).astype(np.float32) if normalize else traces
        raw.extend(shifted)
        for idx in spectrum.index:
            rows.append({"recording": path.relative_to(base).as_posix(), "sweep_idx": idx,
                         "group": f"FM:{station:.3f}", "station_name": settings["source_name"],
                         "center_mhz": midpoint, "offset_khz": round((station - midpoint) * 1000),
                         "span_khz": settings["span_khz"], "label": "FM"})
    return pd.DataFrame(rows), np.stack(raw)


def metrics_per_recording(predictions, groups):
    summaries = []
    for group, cases in predictions.groupby(groups, dropna=False):
        group = group if isinstance(group, tuple) else (group,)
        outcomes = cases["predicted"].value_counts()
        summaries.append(dict(zip(groups, group, strict=True)) | {
            "n": len(cases), "correct": int(cases["label"].eq(cases["predicted"]).sum()),
            "pred_am": int(outcomes.get("AM", 0)), "pred_fm": int(outcomes.get("FM", 0)),
            "pred_noise": int(outcomes.get("Noise", 0)),
        })
    results = pd.DataFrame(summaries)
    results["errors"] = results["n"] - results["correct"]
    results["recall"] = results["correct"] / results["n"]
    return results.sort_values(["errors", "n"], ascending=False)


def accuracy_on_shifts(centered, stress):
    audit = []
    for span, block in [("all", stress), *stress.groupby("span_khz")]:
        row = {"span_khz": span, "n": len(block), "fm_correct": int(block["predicted"].eq("FM").sum()),
               "as_am": int(block["predicted"].eq("AM").sum()), "as_noise": int(block["predicted"].eq("Noise").sum())}
        row["fm_recall"] = row["fm_correct"] / row["n"]
        row["fm_f1_positive_only"] = 2 * row["fm_correct"] / (row["n"] + row["fm_correct"])
        audit.append(row)
    ref = centered.loc[centered["label"].eq("FM") & centered["group"].isin(stress["group"])]
    return pd.DataFrame(audit), metrics_per_recording(ref, ["group"])


def main(project, original_predictions, output):
    sys.path.insert(0, str(project))
    from scripts.evaluate_gb_averaging import fold_mapping, load_recordings
    from scripts.evaluate_modulation_mlp import CLASS_NAMES, SEED, choose_epochs, fit, predict, raw_input

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    records = load_recordings(project / "data")
    folds = fold_mapping(records)
    rows, arrays = [], []
    for info, _, matrix, _ in records:
        for sweep, trace in enumerate(matrix):
            rows.append(info | {"fold": folds[info["group"]], "sweep_idx": sweep})
            arrays.append(raw_input(trace))
    metadata, inputs = pd.DataFrame(rows), np.stack(arrays)
    predictions = pd.read_csv(original_predictions)
    reference = predictions.loc[predictions["representation"].eq("raw_501") & predictions["dropout"].eq(0)]
    keys = ["recording", "sweep_idx", "label", "group", "fold"]
    original = metadata.merge(reference[keys + ["best_epoch", "predicted"]], on=keys, validate="one_to_one")
    assert len(original) == len(metadata) == 4850
    shifted, shifted_inputs = load_shifted(project)
    assert len(shifted) == 750
    shifted["fold"] = shifted["group"].map(folds).fillna(0).astype(int)
    shifted["predicted"] = ""
    for fold, test in shifted.groupby("fold"):
        train_idx = np.flatnonzero(original["fold"].ne(fold)) if fold else np.arange(len(original))
        assert set(original.iloc[train_idx]["group"]).isdisjoint(test["group"])
        if fold:
            chosen = original.loc[original["fold"].eq(fold), "best_epoch"].unique()
            assert len(chosen) == 1
            epochs = int(chosen.item())
            assert choose_epochs(inputs, metadata, train_idx, int(fold), 0.0) == epochs
        else:
            epochs = int(original.groupby("fold")["best_epoch"].first().median())
        state = fit(inputs, metadata, train_idx, 0., epochs, SEED + int(fold))
        if fold:
            control = np.flatnonzero(original["fold"].eq(fold))
            np.testing.assert_array_equal(np.asarray(CLASS_NAMES)[predict(state, control)],
                                          original.iloc[control]["predicted"])
        model, _, _, _, _, _, mean, scale = state
        model.eval()
        values = torch.from_numpy((shifted_inputs[test.index] - mean) / scale)
        with torch.inference_mode():
            estimates = model(values).argmax(dim=1).numpy()
        shifted.loc[test.index, "predicted"] = np.asarray(CLASS_NAMES)[estimates]
    assert set(shifted["predicted"]) <= set(CLASS_NAMES)
    assert not shifted.duplicated(["recording", "sweep_idx"]).any()
    centered_runs = metrics_per_recording(original, ["label", "group", "recording"])
    shifted_runs = metrics_per_recording(shifted, ["group", "station_name", "span_khz", "offset_khz", "recording"])
    scores, matched = accuracy_on_shifts(original, shifted)
    by_station = metrics_per_recording(shifted, ["group"]).merge(
        matched[["group", "n", "correct", "recall"]], on="group", how="left", suffixes=("_shifted", "_centered"))
    for name, table in [("centered_errors_by_recording", centered_runs), ("shifted_errors_by_recording", shifted_runs),
                        ("shifted_scores", scores), ("same_fm_stations", by_station), ("shifted_predictions", shifted)]:
        table.to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig")
    print("Centered worst recordings")
    print(centered_runs.head(10).to_string(index=False))
    print("Shifted totals")
    print(scores.to_string(index=False))
    print("Shifted worst recordings")
    print(shifted_runs.head(10).to_string(index=False))
    print("Same stations")
    print(by_station.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--centered-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.project, args.centered_predictions, args.output)
