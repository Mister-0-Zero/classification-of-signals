from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"

CLASS_NAMES = ("AM", "FM", "Noise")
DEFAULT_EXCLUDED_GROUPS = frozenset({"Noise_101_9", "FM_88_3", "FM_95_2"})

AggregateMode = Literal["mean", "median", "none"]


@dataclass(frozen=True)
class DatasetGroup:
    """One measured signal group, for example AM_102_3 or Noise_93_4."""

    class_name: str
    group_name: str
    run_dir: Path
    sample_csv: Path
    metadata_csv: Path | None
    run_config_json: Path | None


def normalize_classes(classes: str | Iterable[str] | None) -> tuple[str, ...]:
    if classes is None:
        return CLASS_NAMES
    if isinstance(classes, str):
        classes = (classes,)

    normalized = tuple(classes)
    unknown = sorted(set(normalized) - set(CLASS_NAMES))
    if unknown:
        raise ValueError(f"Unknown class names: {unknown}. Expected one of {CLASS_NAMES}.")
    return normalized


def find_dataset_groups(
    data_root: str | Path = DATA_ROOT,
    classes: str | Iterable[str] | None = None,
    excluded_groups: Iterable[str] = DEFAULT_EXCLUDED_GROUPS,
) -> list[DatasetGroup]:
    """Find available dataset groups.

    By default the contaminated Noise_101_9 group is excluded.
    The somnitelno directory is ignored because it is not part of the main dataset.
    """

    data_root = Path(data_root)
    selected_classes = normalize_classes(classes)
    excluded = set(excluded_groups)
    groups: list[DatasetGroup] = []

    for class_name in selected_classes:
        class_dir = data_root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Class directory not found: {class_dir}")

        for group_dir in sorted(p for p in class_dir.iterdir() if p.is_dir()):
            group_name = group_dir.name
            if group_name in excluded:
                continue

            run_dirs = sorted(p for p in group_dir.iterdir() if p.is_dir())
            if not run_dirs:
                continue

            for run_dir in run_dirs:
                samples_dir = run_dir / "samples"
                sample_files = sorted(samples_dir.glob("*.csv")) if samples_dir.exists() else []
                if not sample_files:
                    raise FileNotFoundError(f"No sample CSV files found in: {samples_dir}")
                if len(sample_files) > 1:
                    raise ValueError(f"Expected one sample CSV in {samples_dir}, got {len(sample_files)}")

                groups.append(
                    DatasetGroup(
                        class_name=class_name,
                        group_name=group_name,
                        run_dir=run_dir,
                        sample_csv=sample_files[0],
                        metadata_csv=run_dir / "metadata.csv" if (run_dir / "metadata.csv").exists() else None,
                        run_config_json=run_dir / "run_config.json" if (run_dir / "run_config.json").exists() else None,
                    )
                )

    return groups


def load_signal_dataset(
    data_root: str | Path = DATA_ROOT,
    classes: str | Iterable[str] | None = None,
    excluded_groups: Iterable[str] = DEFAULT_EXCLUDED_GROUPS,
    add_relative_frequency: bool = True,
) -> pd.DataFrame:
    """Load signal samples into one long-format DataFrame.

    Returned rows correspond to measured spectrum points. For the current data:
    16 groups * 50 sweeps * 501 points = 400800 rows by default.
    """

    groups = find_dataset_groups(
        data_root=data_root,
        classes=classes,
        excluded_groups=excluded_groups,
    )
    frames: list[pd.DataFrame] = []

    for group in groups:
        frame = pd.read_csv(group.sample_csv)
        frame["class_name"] = group.class_name
        frame["sample_group_name"] = group.group_name
        frame["run_id"] = group.run_dir.name
        frame["sample_csv"] = str(group.sample_csv)

        if add_relative_frequency:
            center_hz = frame["frequency_hz"].mean()
            frame["relative_frequency_hz"] = frame["frequency_hz"] - center_hz

        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def load_all_dataset(
    data_root: str | Path = DATA_ROOT,
    excluded_groups: Iterable[str] = DEFAULT_EXCLUDED_GROUPS,
) -> pd.DataFrame:
    """Load the combined AM + FM + Noise dataset."""

    return load_signal_dataset(data_root=data_root, classes=None, excluded_groups=excluded_groups)


def load_class_dataset(
    class_name: str,
    data_root: str | Path = DATA_ROOT,
    excluded_groups: Iterable[str] = DEFAULT_EXCLUDED_GROUPS,
) -> pd.DataFrame:
    """Load only one signal class: AM, FM, or Noise."""

    return load_signal_dataset(data_root=data_root, classes=class_name, excluded_groups=excluded_groups)


def load_am_dataset(
    data_root: str | Path = DATA_ROOT,
    excluded_groups: Iterable[str] = DEFAULT_EXCLUDED_GROUPS,
) -> pd.DataFrame:
    return load_class_dataset("AM", data_root=data_root, excluded_groups=excluded_groups)


def load_fm_dataset(
    data_root: str | Path = DATA_ROOT,
    excluded_groups: Iterable[str] = DEFAULT_EXCLUDED_GROUPS,
) -> pd.DataFrame:
    return load_class_dataset("FM", data_root=data_root, excluded_groups=excluded_groups)


def load_noise_dataset(
    data_root: str | Path = DATA_ROOT,
    excluded_groups: Iterable[str] = DEFAULT_EXCLUDED_GROUPS,
) -> pd.DataFrame:
    return load_class_dataset("Noise", data_root=data_root, excluded_groups=excluded_groups)


def load_datasets_by_class(
    data_root: str | Path = DATA_ROOT,
    excluded_groups: Iterable[str] = DEFAULT_EXCLUDED_GROUPS,
) -> dict[str, pd.DataFrame]:
    """Load separate DataFrames for AM, FM, and Noise."""

    return {
        class_name: load_class_dataset(
            class_name=class_name,
            data_root=data_root,
            excluded_groups=excluded_groups,
        )
        for class_name in CLASS_NAMES
    }


def make_sweep_examples(
    data: pd.DataFrame,
    window_size: int = 1,
    step: int | None = None,
    aggregate: AggregateMode = "mean",
    value_column: str = "amplitude_dbm",
    frequency_column: str = "relative_frequency_hz",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Convert long-format rows into model-ready spectrum examples.

    window_size=1 uses each sweep as a separate example.
    window_size=3 with aggregate="mean" averages every 3 consecutive sweeps.
    If step is None, non-overlapping windows are used.
    """

    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if step is None:
        step = window_size
    if step < 1:
        raise ValueError("step must be >= 1")
    if aggregate not in {"mean", "median", "none"}:
        raise ValueError("aggregate must be one of: mean, median, none")
    if frequency_column not in data.columns:
        raise ValueError(f"Missing frequency column: {frequency_column}")

    feature_rows: list[np.ndarray] = []
    labels: list[str] = []
    meta_rows: list[dict[str, object]] = []

    group_columns = ["class_name", "sample_group_name", "run_id", "sample_id"]
    for keys, group_df in data.groupby(group_columns, sort=True):
        class_name, group_name, run_id, sample_id = keys
        available_sweeps = sorted(group_df["sweep_idx"].unique())

        for start_idx in range(0, len(available_sweeps) - window_size + 1, step):
            selected_sweeps = available_sweeps[start_idx : start_idx + window_size]
            window_df = group_df[group_df["sweep_idx"].isin(selected_sweeps)]

            pivot = window_df.pivot_table(
                index="point_idx",
                columns="sweep_idx",
                values=value_column,
                aggfunc="mean",
            ).sort_index()

            if pivot.shape[1] != window_size:
                continue

            if aggregate == "mean":
                values = pivot.mean(axis=1).to_numpy(dtype=np.float32)
            elif aggregate == "median":
                values = pivot.median(axis=1).to_numpy(dtype=np.float32)
            else:
                values = pivot.to_numpy(dtype=np.float32).T.reshape(-1)

            feature_rows.append(values)
            labels.append(class_name)
            meta_rows.append(
                {
                    "class_name": class_name,
                    "sample_group_name": group_name,
                    "run_id": run_id,
                    "sample_id": sample_id,
                    "start_sweep_idx": int(selected_sweeps[0]),
                    "end_sweep_idx": int(selected_sweeps[-1]),
                    "window_size": window_size,
                    "aggregate": aggregate,
                }
            )

    if not feature_rows:
        return np.empty((0, 0), dtype=np.float32), np.array([], dtype=object), pd.DataFrame()

    return np.vstack(feature_rows), np.array(labels, dtype=object), pd.DataFrame(meta_rows)


if __name__ == "__main__":
    dataset = load_all_dataset()
    print("Loaded rows:", len(dataset))
    print("Groups:", dataset["sample_group_name"].nunique())
    print(dataset.groupby("class_name")["sample_group_name"].nunique())
