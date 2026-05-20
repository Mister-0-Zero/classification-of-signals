from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from mainFunction import DEFAULT_EXCLUDED_GROUPS, load_all_dataset


RANDOM_STATE = 42
WINDOW_SIZE = 5


@dataclass(frozen=True)
class PreparedDataset:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    train_meta: pd.DataFrame
    val_meta: pd.DataFrame
    feature_table: pd.DataFrame
    label_encoder: LabelEncoder


def add_relative_frequency(
    df: pd.DataFrame,
    group_col: str = "sample_group_name",
    frequency_col: str = "frequency_hz",
    output_col: str = "relative_frequency_hz",
) -> pd.DataFrame:
    """Add frequency relative to the center of each measured dataset."""

    result = df.copy()
    centers = result.groupby(group_col)[frequency_col].transform("mean")
    result[output_col] = result[frequency_col] - centers
    return result


def assign_sweep_windows(
    df: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
    step: int | None = None,
) -> pd.DataFrame:
    """Assign non-overlapping or sliding sweep-window ids inside each dataset."""

    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if step is None:
        step = window_size
    if step < 1:
        raise ValueError("step must be >= 1")

    frames: list[pd.DataFrame] = []
    group_cols = ["class_name", "sample_group_name", "run_id", "sample_id"]

    for _, group_df in df.groupby(group_cols, sort=True):
        group_df = group_df.copy()
        sweeps = sorted(group_df["sweep_idx"].unique())
        window_by_sweep: dict[int, int] = {}
        window_bounds: dict[int, tuple[int, int]] = {}

        window_id = 0
        for start_pos in range(0, len(sweeps) - window_size + 1, step):
            selected = sweeps[start_pos : start_pos + window_size]
            for sweep_idx in selected:
                # Non-overlapping windows are the default. For sliding windows,
                # rows are intentionally assigned to the first matching window.
                window_by_sweep.setdefault(int(sweep_idx), window_id)
            window_bounds[window_id] = (int(selected[0]), int(selected[-1]))
            window_id += 1

        group_df = group_df[group_df["sweep_idx"].isin(window_by_sweep)].copy()
        group_df["window_id"] = group_df["sweep_idx"].map(window_by_sweep)
        group_df["window_start_sweep"] = group_df["window_id"].map(lambda x: window_bounds[int(x)][0])
        group_df["window_end_sweep"] = group_df["window_id"].map(lambda x: window_bounds[int(x)][1])
        frames.append(group_df)

    if not frames:
        return pd.DataFrame(columns=list(df.columns) + ["window_id", "window_start_sweep", "window_end_sweep"])

    return pd.concat(frames, ignore_index=True)


def _safe_mode(values: pd.Series) -> float:
    modes = values.mode()
    return float(modes.iloc[0]) if not modes.empty else float("nan")


def compute_window_statistics(
    df: pd.DataFrame,
    amplitude_col: str = "amplitude_dbm",
) -> pd.DataFrame:
    """Compute mathematical statistics for each sweep window."""

    group_cols = [
        "class_name",
        "sample_group_name",
        "run_id",
        "sample_id",
        "window_id",
        "window_start_sweep",
        "window_end_sweep",
    ]

    stats = (
        df.groupby(group_cols)[amplitude_col]
        .agg(
            amp_count="count",
            amp_mean="mean",
            amp_std="std",
            amp_var="var",
            amp_min="min",
            amp_q25=lambda x: x.quantile(0.25),
            amp_median="median",
            amp_q75=lambda x: x.quantile(0.75),
            amp_max="max",
            amp_skew="skew",
            amp_mode=_safe_mode,
        )
        .reset_index()
    )

    stats["amp_range"] = stats["amp_max"] - stats["amp_min"]
    stats["amp_iqr"] = stats["amp_q75"] - stats["amp_q25"]
    stats["amp_rms"] = (
        df.assign(_amp_sq=df[amplitude_col] ** 2)
        .groupby(group_cols)["_amp_sq"]
        .mean()
        .pow(0.5)
        .to_numpy()
    )
    stats["amp_abs_mean"] = (
        df.assign(_amp_abs=df[amplitude_col].abs())
        .groupby(group_cols)["_amp_abs"]
        .mean()
        .to_numpy()
    )

    return stats


def _moving_average(values: np.ndarray, window: int = 21) -> np.ndarray:
    if window <= 1:
        return values.astype(float, copy=True)
    window = min(window, len(values))
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(values.astype(float), (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _count_prominent_peaks(values: np.ndarray) -> int:
    if len(values) < 3:
        return 0
    threshold = float(values.mean() + values.std())
    peak_mask = (values[1:-1] > values[:-2]) & (values[1:-1] > values[2:]) & (values[1:-1] > threshold)
    return int(peak_mask.sum())


def _spectrum_shape_features(group_df: pd.DataFrame) -> dict[str, float]:
    spectrum = (
        group_df.groupby("point_idx", sort=True)
        .agg(
            amplitude_dbm=("amplitude_dbm", "mean"),
            relative_frequency_hz=("relative_frequency_hz", "mean"),
        )
        .reset_index(drop=True)
    )
    y = spectrum["amplitude_dbm"].to_numpy(dtype=float)
    f = spectrum["relative_frequency_hz"].to_numpy(dtype=float)

    weights = y - y.min() + 1e-6
    weights_sum = weights.sum()
    centroid = float(np.sum(f * weights) / weights_sum)
    bandwidth = float(np.sqrt(np.sum(((f - centroid) ** 2) * weights) / weights_sum))

    prob = weights / weights_sum
    entropy = float(-(prob * np.log(prob + 1e-12)).sum() / np.log(len(prob)))

    smooth = _moving_average(y, window=21)
    residual = y - smooth

    center_mask = np.abs(f) <= 50_000
    edge_mask = np.abs(f) >= 80_000
    left_mask = f < 0
    right_mask = f > 0

    center_mean = float(y[center_mask].mean()) if center_mask.any() else float("nan")
    edge_mean = float(y[edge_mask].mean()) if edge_mask.any() else float("nan")
    left_mean = float(y[left_mask].mean()) if left_mask.any() else float("nan")
    right_mean = float(y[right_mask].mean()) if right_mask.any() else float("nan")
    slope = float(np.polyfit(f, y, deg=1)[0]) if len(np.unique(f)) > 1 else 0.0

    return {
        "shape_centroid_hz": centroid,
        "shape_bandwidth_hz": bandwidth,
        "shape_entropy": entropy,
        "shape_peak_count": _count_prominent_peaks(y),
        "shape_peak_to_mean": float(y.max() - y.mean()),
        "shape_peak_to_median": float(y.max() - np.median(y)),
        "shape_total_variation": float(np.abs(np.diff(y)).sum()),
        "shape_mean_abs_diff": float(np.abs(np.diff(y)).mean()),
        "shape_smooth_std": float(smooth.std()),
        "shape_smooth_range": float(smooth.max() - smooth.min()),
        "shape_residual_std": float(residual.std()),
        "shape_structure_ratio": float(smooth.std() / (y.std() + 1e-9)),
        "shape_center_mean": center_mean,
        "shape_edge_mean": edge_mean,
        "shape_center_minus_edge": center_mean - edge_mean,
        "shape_center_edge_ratio": center_mean / (abs(edge_mean) + 1e-9),
        "shape_left_mean": left_mean,
        "shape_right_mean": right_mean,
        "shape_left_right_diff": left_mean - right_mean,
        "shape_linear_slope": slope,
    }


def compute_window_shape_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute features that preserve coarse spectrum shape inside each window."""

    group_cols = [
        "class_name",
        "sample_group_name",
        "run_id",
        "sample_id",
        "window_id",
        "window_start_sweep",
        "window_end_sweep",
    ]
    rows = []
    for keys, group_df in df.groupby(group_cols, sort=True):
        row = dict(zip(group_cols, keys))
        row.update(_spectrum_shape_features(group_df))
        rows.append(row)
    return pd.DataFrame(rows)


def compute_window_features(df: pd.DataFrame) -> pd.DataFrame:
    """Combine amplitude distribution stats with coarse spectral-shape features."""

    group_cols = [
        "class_name",
        "sample_group_name",
        "run_id",
        "sample_id",
        "window_id",
        "window_start_sweep",
        "window_end_sweep",
    ]
    stats = compute_window_statistics(df)
    shape = compute_window_shape_features(df)
    return stats.merge(shape, on=group_cols, how="inner")


def make_stat_feature_table(
    data_root=None,
    window_size: int = WINDOW_SIZE,
    step: int | None = None,
    excluded_groups: Iterable[str] = DEFAULT_EXCLUDED_GROUPS,
) -> pd.DataFrame:
    """Load raw rows, normalize frequency, group by windows, and compute stats."""

    df = load_all_dataset(data_root=data_root, excluded_groups=excluded_groups) if data_root else load_all_dataset(
        excluded_groups=excluded_groups
    )
    df = add_relative_frequency(df)
    windowed = assign_sweep_windows(df, window_size=window_size, step=step)
    return compute_window_features(windowed)


def split_train_val_by_group(
    feature_table: pd.DataFrame,
    val_groups_per_class: dict[str, int] | None = None,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by source dataset group, not by individual windows.

    Default for the current 16 groups:
    AM: 1 validation group, FM: 2 validation groups, Noise: 1 validation group.
    """

    if val_groups_per_class is None:
        val_groups_per_class = {"AM": 1, "FM": 2, "Noise": 1}

    rng = np.random.default_rng(random_state)
    val_groups: set[str] = set()

    group_table = feature_table[["class_name", "sample_group_name"]].drop_duplicates()
    for class_name, class_groups in group_table.groupby("class_name", sort=True):
        groups = class_groups["sample_group_name"].to_numpy()
        n_val = val_groups_per_class.get(class_name, 1)
        if not 1 <= n_val < len(groups):
            raise ValueError(f"Invalid validation group count for {class_name}: {n_val}")
        val_groups.update(rng.choice(groups, size=n_val, replace=False).tolist())

    is_val = feature_table["sample_group_name"].isin(val_groups)
    train_df = feature_table.loc[~is_val].reset_index(drop=True)
    val_df = feature_table.loc[is_val].reset_index(drop=True)
    return train_df, val_df


def prepare_train_val_datasets(
    window_size: int = WINDOW_SIZE,
    step: int | None = None,
    excluded_groups: Iterable[str] = DEFAULT_EXCLUDED_GROUPS,
    random_state: int = RANDOM_STATE,
) -> PreparedDataset:
    """Prepare X/y train and validation datasets for baseline models."""

    feature_table = make_stat_feature_table(
        window_size=window_size,
        step=step,
        excluded_groups=excluded_groups,
    )
    train_df, val_df = split_train_val_by_group(feature_table, random_state=random_state)

    meta_cols = [
        "class_name",
        "sample_group_name",
        "run_id",
        "sample_id",
        "window_id",
        "window_start_sweep",
        "window_end_sweep",
    ]
    feature_cols = [col for col in feature_table.columns if col not in meta_cols]

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["class_name"])
    y_val = label_encoder.transform(val_df["class_name"])

    return PreparedDataset(
        X_train=train_df[feature_cols],
        X_val=val_df[feature_cols],
        y_train=y_train,
        y_val=y_val,
        train_meta=train_df[meta_cols],
        val_meta=val_df[meta_cols],
        feature_table=feature_table,
        label_encoder=label_encoder,
    )


def train_logistic_regression(
    prepared: PreparedDataset,
    max_iter: int = 5000,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Train Logistic Regression on statistical amplitude features."""

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=max_iter,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )
    model.fit(prepared.X_train, prepared.y_train)
    return model


def train_random_forest(
    prepared: PreparedDataset,
    n_estimators: int = 300,
    random_state: int = RANDOM_STATE,
) -> RandomForestClassifier:
    """Train Random Forest on statistical amplitude features."""

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=random_state,
        min_samples_leaf=2,
        n_jobs=1,
    )
    model.fit(prepared.X_train, prepared.y_train)
    return model


def train_gradient_boosting(
    prepared: PreparedDataset,
    random_state: int = RANDOM_STATE,
) -> GradientBoostingClassifier:
    """Train Gradient Boosting on statistical amplitude features."""

    model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=2,
        random_state=random_state,
    )
    model.fit(prepared.X_train, prepared.y_train)
    return model


def train_mlp_on_features(
    prepared: PreparedDataset,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Train a compact MLP on engineered statistical and shape features."""

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(32, 16),
                    activation="relu",
                    alpha=1e-3,
                    learning_rate_init=1e-3,
                    max_iter=2000,
                    early_stopping=True,
                    validation_fraction=0.2,
                    random_state=random_state,
                ),
            ),
        ]
    )
    model.fit(prepared.X_train, prepared.y_train)
    return model


def evaluate_model(model: Pipeline, prepared: PreparedDataset) -> dict[str, object]:
    pred = model.predict(prepared.X_val)
    class_names = prepared.label_encoder.classes_

    return {
        "accuracy": accuracy_score(prepared.y_val, pred),
        "macro_f1": f1_score(prepared.y_val, pred, average="macro"),
        "classification_report": classification_report(
            prepared.y_val,
            pred,
            target_names=class_names,
            zero_division=0,
        ),
        "confusion_matrix": pd.DataFrame(
            confusion_matrix(prepared.y_val, pred),
            index=[f"true_{name}" for name in class_names],
            columns=[f"pred_{name}" for name in class_names],
        ),
        "predictions": prepared.val_meta.assign(
            y_true=prepared.label_encoder.inverse_transform(prepared.y_val),
            y_pred=prepared.label_encoder.inverse_transform(pred),
        ),
    }


def compare_baseline_models(prepared: PreparedDataset) -> tuple[pd.DataFrame, dict[str, object]]:
    """Train and compare several baseline models on the same split."""

    models = {
        "logistic_regression": train_logistic_regression(prepared),
        "random_forest": train_random_forest(prepared),
        "gradient_boosting": train_gradient_boosting(prepared),
        "mlp_features": train_mlp_on_features(prepared),
    }

    rows = []
    metrics_by_model = {}
    for name, model in models.items():
        metrics = evaluate_model(model, prepared)
        metrics_by_model[name] = metrics
        rows.append(
            {
                "model": name,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
            }
        )

    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False), metrics_by_model


if __name__ == "__main__":
    prepared_data = prepare_train_val_datasets(window_size=WINDOW_SIZE)
    comparison, metrics_by_model = compare_baseline_models(prepared_data)

    print("Feature table shape:", prepared_data.feature_table.shape)
    print("Train shape:", prepared_data.X_train.shape)
    print("Validation shape:", prepared_data.X_val.shape)
    print("Validation groups:")
    print(prepared_data.val_meta[["class_name", "sample_group_name"]].drop_duplicates().to_string(index=False))
    print("\nBaseline comparison:")
    print(comparison.to_string(index=False))

    for model_name, metrics in metrics_by_model.items():
        print(f"\n=== {model_name} ===")
        print("Accuracy:", round(metrics["accuracy"], 4))
        print("Macro-F1:", round(metrics["macro_f1"], 4))
        print("\nClassification report:")
        print(metrics["classification_report"])
        print("Confusion matrix:")
        print(metrics["confusion_matrix"].to_string())
