from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from baseline_logreg import RANDOM_STATE, WINDOW_SIZE, split_train_val_by_group
from mainFunction import load_all_dataset, make_sweep_examples


def add_frequency_bin_features(
    X: np.ndarray,
    bin_sizes: tuple[int, ...] = (5, 10, 25, 50),
    stats: tuple[str, ...] = ("mean",),
) -> np.ndarray:
    """Append multi-scale local features over neighboring frequency points.

    Example for bin_size=10: each 501-point spectrum is split into consecutive
    10-point chunks, and selected statistics of each chunk are appended.
    """

    features = [X.astype(np.float32)]
    n_points = X.shape[1]

    for bin_size in bin_sizes:
        if bin_size < 1:
            raise ValueError("bin sizes must be positive")
        bin_features = []
        for start in range(0, n_points, bin_size):
            stop = min(start + bin_size, n_points)
            chunk = X[:, start:stop]
            if "mean" in stats:
                bin_features.append(chunk.mean(axis=1))
            if "std" in stats:
                bin_features.append(chunk.std(axis=1))
            if "min" in stats:
                bin_features.append(chunk.min(axis=1))
            if "max" in stats:
                bin_features.append(chunk.max(axis=1))
            if "range" in stats:
                bin_features.append(chunk.max(axis=1) - chunk.min(axis=1))
            if "energy" in stats:
                bin_features.append(np.sqrt((chunk**2).mean(axis=1)))
            if "slope" in stats:
                x_axis = np.arange(chunk.shape[1], dtype=np.float32)
                x_centered = x_axis - x_axis.mean()
                denom = float((x_centered**2).sum())
                if denom == 0.0:
                    bin_features.append(np.zeros(len(chunk), dtype=np.float32))
                else:
                    y_centered = chunk - chunk.mean(axis=1, keepdims=True)
                    bin_features.append((y_centered @ x_centered) / denom)

        if bin_features:
            features.append(np.stack(bin_features, axis=1).astype(np.float32))

    return np.concatenate(features, axis=1)


def prepare_spectrum_vectors(
    window_size: int = WINDOW_SIZE,
    aggregate: str = "mean",
    add_binned_features: bool = False,
    bin_sizes: tuple[int, ...] = (5, 10, 25, 50),
    bin_stats: tuple[str, ...] = ("mean",),
    random_state: int = RANDOM_STATE,
):
    """Prepare averaged spectrum vectors for MLP classification."""

    df = load_all_dataset()
    X, y_text, meta = make_sweep_examples(df, window_size=window_size, aggregate=aggregate)

    # Reuse the same group-level split logic as baseline_logreg.
    split_table = meta[["class_name", "sample_group_name"]].copy()
    split_table["row_idx"] = np.arange(len(meta))
    train_meta_idx, val_meta_idx = split_train_val_by_group(split_table, random_state=random_state)

    train_idx = train_meta_idx["row_idx"].to_numpy()
    val_idx = val_meta_idx["row_idx"].to_numpy()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)

    return {
        "X_train": add_frequency_bin_features(X[train_idx], bin_sizes=bin_sizes, stats=bin_stats)
        if add_binned_features
        else X[train_idx],
        "X_val": add_frequency_bin_features(X[val_idx], bin_sizes=bin_sizes, stats=bin_stats)
        if add_binned_features
        else X[val_idx],
        "y_train": y[train_idx],
        "y_val": y[val_idx],
        "train_meta": meta.iloc[train_idx].reset_index(drop=True),
        "val_meta": meta.iloc[val_idx].reset_index(drop=True),
        "label_encoder": label_encoder,
    }


def train_logreg_spectrum(prepared: dict, random_state: int = RANDOM_STATE) -> Pipeline:
    """Train Logistic Regression on spectrum-vector features."""

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )
    model.fit(prepared["X_train"], prepared["y_train"])
    return model


def train_mlp_spectrum(prepared: dict, random_state: int = RANDOM_STATE) -> Pipeline:
    """Train compact MLP on 501-point averaged spectrum vectors."""

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    alpha=1e-2,
                    learning_rate_init=1e-3,
                    max_iter=5000,
                    early_stopping=False,
                    random_state=random_state,
                ),
            ),
        ]
    )
    model.fit(prepared["X_train"], prepared["y_train"])
    return model


def evaluate_spectrum_model(model: Pipeline, prepared: dict) -> dict[str, object]:
    pred = model.predict(prepared["X_val"])
    class_names = prepared["label_encoder"].classes_

    return {
        "accuracy": accuracy_score(prepared["y_val"], pred),
        "macro_f1": f1_score(prepared["y_val"], pred, average="macro"),
        "classification_report": classification_report(
            prepared["y_val"],
            pred,
            target_names=class_names,
            zero_division=0,
        ),
        "confusion_matrix": pd.DataFrame(
            confusion_matrix(prepared["y_val"], pred),
            index=[f"true_{name}" for name in class_names],
            columns=[f"pred_{name}" for name in class_names],
        ),
        "predictions": prepared["val_meta"].assign(
            y_true=prepared["label_encoder"].inverse_transform(prepared["y_val"]),
            y_pred=prepared["label_encoder"].inverse_transform(pred),
        ),
    }


def evaluate_mlp(model: Pipeline, prepared: dict) -> dict[str, object]:
    return evaluate_spectrum_model(model, prepared)


def compare_spectrum_feature_sets(window_size: int = WINDOW_SIZE) -> pd.DataFrame:
    """Compare raw averaged spectrum with multi-scale binned spectrum features."""

    rows = []
    experiments = {
        "raw_501": prepare_spectrum_vectors(window_size=window_size, add_binned_features=False),
        "raw_501_plus_bin_mean": prepare_spectrum_vectors(
            window_size=window_size,
            add_binned_features=True,
            bin_sizes=(5, 10, 25, 50),
            bin_stats=("mean",),
        ),
        "raw_501_plus_bin_stats": prepare_spectrum_vectors(
            window_size=window_size,
            add_binned_features=True,
            bin_sizes=(5, 10, 25, 50),
            bin_stats=("mean", "std", "min", "max", "range", "energy", "slope"),
        ),
    }

    for feature_set_name, prepared in experiments.items():
        models = {
            "logreg": train_logreg_spectrum(prepared),
            "mlp": train_mlp_spectrum(prepared),
        }
        for model_name, model in models.items():
            metrics = evaluate_spectrum_model(model, prepared)
            rows.append(
                {
                    "feature_set": feature_set_name,
                    "model": model_name,
                    "n_features": prepared["X_train"].shape[1],
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                }
            )

    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False)


if __name__ == "__main__":
    print("Feature-set comparison:")
    print(compare_spectrum_feature_sets(window_size=WINDOW_SIZE).to_string(index=False))

    prepared_data = prepare_spectrum_vectors(
        window_size=WINDOW_SIZE,
        aggregate="mean",
        add_binned_features=True,
        bin_stats=("mean", "std", "min", "max", "range", "energy", "slope"),
    )
    model = train_mlp_spectrum(prepared_data)
    metrics = evaluate_mlp(model, prepared_data)

    print("Train shape:", prepared_data["X_train"].shape)
    print("Validation shape:", prepared_data["X_val"].shape)
    print("Validation groups:")
    print(prepared_data["val_meta"][["class_name", "sample_group_name"]].drop_duplicates().to_string(index=False))
    print("\nAccuracy:", round(metrics["accuracy"], 4))
    print("Macro-F1:", round(metrics["macro_f1"], 4))
    print("\nClassification report:")
    print(metrics["classification_report"])
    print("Confusion matrix:")
    print(metrics["confusion_matrix"].to_string())
