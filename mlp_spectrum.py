from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from baseline_logreg import RANDOM_STATE, WINDOW_SIZE, split_train_val_by_group
from mainFunction import load_all_dataset, make_sweep_examples


def prepare_spectrum_vectors(
    window_size: int = WINDOW_SIZE,
    aggregate: str = "mean",
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
        "X_train": X[train_idx],
        "X_val": X[val_idx],
        "y_train": y[train_idx],
        "y_val": y[val_idx],
        "train_meta": meta.iloc[train_idx].reset_index(drop=True),
        "val_meta": meta.iloc[val_idx].reset_index(drop=True),
        "label_encoder": label_encoder,
    }


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


def evaluate_mlp(model: Pipeline, prepared: dict) -> dict[str, object]:
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


if __name__ == "__main__":
    prepared_data = prepare_spectrum_vectors(window_size=WINDOW_SIZE, aggregate="mean")
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
