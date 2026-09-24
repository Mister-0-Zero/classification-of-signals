from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

from baseline_logreg import (
    RANDOM_STATE,
    WINDOW_SIZE,
    PreparedDataset,
    make_stat_feature_table,
    split_train_val_by_group,
    train_gradient_boosting,
    train_logistic_regression,
    train_mlp_on_features,
    train_random_forest,
)
from cnn2d_spectrum import (
    AugmentationConfig,
    Spectrum2DDataset,
    _predict,
    _require_torch,
    build_cnn2d_model,
    train_cnn2d,
)
from mainFunction import load_all_dataset, make_sweep_examples
from mlp_spectrum import train_logreg_spectrum, train_mlp_spectrum


N_SPLITS = 4
CLASS_NAMES = ("AM", "FM", "Noise")


@dataclass(frozen=True)
class Fold:
    number: int
    train_groups: frozenset[str]
    test_groups: frozenset[str]


def build_folds(meta: pd.DataFrame, labels: np.ndarray) -> list[Fold]:
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    groups = meta["sample_group_name"].to_numpy()
    folds = []
    for number, (train_idx, test_idx) in enumerate(splitter.split(meta, labels, groups), start=1):
        folds.append(
            Fold(
                number=number,
                train_groups=frozenset(groups[train_idx]),
                test_groups=frozenset(groups[test_idx]),
            )
        )
    return folds


def fold_mask(meta: pd.DataFrame, groups: frozenset[str]) -> np.ndarray:
    return meta["sample_group_name"].isin(groups).to_numpy()


def prediction_frame(
    model_name: str,
    representation: str,
    fold: Fold,
    meta: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    encoder: LabelEncoder,
) -> pd.DataFrame:
    result = meta.copy()
    result["fold"] = fold.number
    result["model"] = model_name
    result["representation"] = representation
    result["y_true"] = encoder.inverse_transform(y_true)
    result["y_pred"] = encoder.inverse_transform(y_pred)
    return result


def score_predictions(
    model_name: str,
    representation: str,
    fold: Fold,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, object]:
    return {
        "model": model_name,
        "representation": representation,
        "fold": fold.number,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "test_objects": len(y_true),
        "test_groups": ", ".join(sorted(fold.test_groups)),
    }


def evaluate_feature_models(
    folds: list[Fold],
    encoder: LabelEncoder,
) -> tuple[list[dict[str, object]], list[pd.DataFrame]]:
    feature_table = make_stat_feature_table(window_size=WINDOW_SIZE)
    meta_cols = [
        "class_name",
        "sample_group_name",
        "run_id",
        "sample_id",
        "window_id",
        "window_start_sweep",
        "window_end_sweep",
    ]
    feature_cols = [column for column in feature_table.columns if column not in meta_cols]
    y_all = encoder.transform(feature_table["class_name"])
    meta = feature_table[meta_cols]
    builders: dict[str, Callable[[PreparedDataset], object]] = {
        "Logistic Regression": train_logistic_regression,
        "Random Forest": train_random_forest,
        "Gradient Boosting": train_gradient_boosting,
        "MLP (features)": train_mlp_on_features,
    }
    scores = []
    predictions = []

    for fold in folds:
        train_mask = fold_mask(meta, fold.train_groups)
        test_mask = fold_mask(meta, fold.test_groups)
        prepared = PreparedDataset(
            X_train=feature_table.loc[train_mask, feature_cols],
            X_val=feature_table.loc[test_mask, feature_cols],
            y_train=y_all[train_mask],
            y_val=y_all[test_mask],
            train_meta=meta.loc[train_mask].reset_index(drop=True),
            val_meta=meta.loc[test_mask].reset_index(drop=True),
            feature_table=feature_table,
            label_encoder=encoder,
        )
        for model_name, builder in builders.items():
            model = builder(prepared)
            pred = model.predict(prepared.X_val)
            scores.append(score_predictions(model_name, "computed features", fold, prepared.y_val, pred))
            predictions.append(
                prediction_frame(
                    model_name,
                    "computed features",
                    fold,
                    prepared.val_meta,
                    prepared.y_val,
                    pred,
                    encoder,
                )
            )
    return scores, predictions


def evaluate_vector_models(
    folds: list[Fold],
    encoder: LabelEncoder,
) -> tuple[list[dict[str, object]], list[pd.DataFrame]]:
    df = load_all_dataset()
    X, y_text, meta = make_sweep_examples(df, window_size=WINDOW_SIZE, aggregate="mean")
    y_all = encoder.transform(y_text)
    builders = {
        "Logistic Regression": train_logreg_spectrum,
        "MLP (vector)": train_mlp_spectrum,
    }
    scores = []
    predictions = []

    for fold in folds:
        train_mask = fold_mask(meta, fold.train_groups)
        test_mask = fold_mask(meta, fold.test_groups)
        prepared = {
            "X_train": X[train_mask],
            "X_val": X[test_mask],
            "y_train": y_all[train_mask],
            "y_val": y_all[test_mask],
            "train_meta": meta.loc[train_mask].reset_index(drop=True),
            "val_meta": meta.loc[test_mask].reset_index(drop=True),
            "label_encoder": encoder,
        }
        for model_name, builder in builders.items():
            model = builder(prepared)
            pred = model.predict(prepared["X_val"])
            scores.append(score_predictions(model_name, "averaged vector (501)", fold, prepared["y_val"], pred))
            predictions.append(
                prediction_frame(
                    model_name,
                    "averaged vector (501)",
                    fold,
                    prepared["val_meta"],
                    prepared["y_val"],
                    pred,
                    encoder,
                )
            )
    return scores, predictions


def normalize_from_train(X_train: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = float(X_train.mean())
    std = float(X_train.std() + 1e-6)
    return tuple((array - mean) / std for array in (X_train, *arrays))


def select_cnn_epoch(
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    encoder: LabelEncoder,
    outer_train_mask: np.ndarray,
    fold: Fold,
    epochs: int,
    patience: int,
) -> int:
    outer_train_meta = meta.loc[outer_train_mask].copy()
    outer_train_meta["row_idx"] = np.flatnonzero(outer_train_mask)
    inner_train, inner_val = split_train_val_by_group(
        outer_train_meta,
        val_groups_per_class={"AM": 1, "FM": 1, "Noise": 1},
        random_state=RANDOM_STATE + fold.number,
    )
    train_idx = inner_train["row_idx"].to_numpy()
    val_idx = inner_val["row_idx"].to_numpy()
    X_train, X_val = normalize_from_train(X[train_idx], X[val_idx])
    prepared = {
        "X_train": X_train[:, None, :, :],
        "X_val": X_val[:, None, :, :],
        "y_train": y[train_idx],
        "y_val": y[val_idx],
        "train_meta": meta.iloc[train_idx].reset_index(drop=True),
        "val_meta": meta.iloc[val_idx].reset_index(drop=True),
        "label_encoder": encoder,
    }
    model, _ = train_cnn2d(
        prepared,
        epochs=epochs,
        patience=patience,
        random_state=RANDOM_STATE + fold.number,
    )
    best = max(
        model.training_history,
        key=lambda row: (row["val_macro_f1"], -row["val_loss"]),
    )
    return int(best["epoch"])


def train_cnn_fixed_epochs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    num_classes: int,
    epochs: int,
    random_state: int,
):
    torch, nn, DataLoader, _ = _require_torch()
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_cnn2d_model(num_classes).to(device)
    train_ds = Spectrum2DDataset(
        X_train,
        y_train,
        augment=AugmentationConfig(enabled=True),
        random_state=random_state,
    )
    generator = torch.Generator()
    generator.manual_seed(random_state)
    loader = DataLoader(train_ds, batch_size=16, shuffle=True, generator=generator)
    class_counts = np.bincount(y_train, minlength=num_classes)
    weights = class_counts.sum() / np.maximum(class_counts, 1)
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        scheduler.step()
    return model, device


def evaluate_cnn_model(
    folds: list[Fold],
    encoder: LabelEncoder,
    epochs: int,
    patience: int,
) -> tuple[list[dict[str, object]], list[pd.DataFrame]]:
    df = load_all_dataset()
    X_flat, y_text, meta = make_sweep_examples(df, window_size=WINDOW_SIZE, aggregate="none")
    X = X_flat.reshape(len(X_flat), WINDOW_SIZE, 501).astype(np.float32)
    y_all = encoder.transform(y_text)
    scores = []
    predictions = []

    for fold in folds:
        outer_train_mask = fold_mask(meta, fold.train_groups)
        test_mask = fold_mask(meta, fold.test_groups)
        best_epoch = select_cnn_epoch(
            X,
            y_all,
            meta,
            encoder,
            outer_train_mask,
            fold,
            epochs=epochs,
            patience=patience,
        )
        X_train, X_test = normalize_from_train(X[outer_train_mask], X[test_mask])
        model, device = train_cnn_fixed_epochs(
            X_train[:, None, :, :],
            y_all[outer_train_mask],
            len(encoder.classes_),
            epochs=best_epoch,
            random_state=RANDOM_STATE + fold.number,
        )
        pred = _predict(model, device, X_test[:, None, :, :])
        row = score_predictions("2D CNN", "matrix (5 x 501)", fold, y_all[test_mask], pred)
        row["selected_epochs"] = best_epoch
        scores.append(row)
        predictions.append(
            prediction_frame(
                "2D CNN",
                "matrix (5 x 501)",
                fold,
                meta.loc[test_mask].reset_index(drop=True),
                y_all[test_mask],
                pred,
                encoder,
            )
        )
    return scores, predictions


def summarize_results(scores: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_name, representation), group in scores.groupby(["model", "representation"], sort=False):
        selected = predictions[
            (predictions["model"] == model_name) & (predictions["representation"] == representation)
        ]
        rows.append(
            {
                "model": model_name,
                "representation": representation,
                "fold_accuracy_mean": group["accuracy"].mean(),
                "fold_accuracy_std": group["accuracy"].std(ddof=1),
                "fold_macro_f1_mean": group["macro_f1"].mean(),
                "fold_macro_f1_std": group["macro_f1"].std(ddof=1),
                "oof_accuracy": accuracy_score(selected["y_true"], selected["y_pred"]),
                "oof_macro_f1": f1_score(selected["y_true"], selected["y_pred"], average="macro", zero_division=0),
            }
        )
    return pd.DataFrame(rows).sort_values("oof_macro_f1", ascending=False).reset_index(drop=True)


def print_diagnostics(summary: pd.DataFrame, scores: pd.DataFrame, predictions: pd.DataFrame) -> None:
    print("\nCross-validation summary:")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nFold metrics:")
    print(scores.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    for row in summary.itertuples(index=False):
        selected = predictions[
            (predictions["model"] == row.model) & (predictions["representation"] == row.representation)
        ]
        print(f"\n=== {row.model} | {row.representation} ===")
        print(classification_report(selected["y_true"], selected["y_pred"], labels=list(CLASS_NAMES), zero_division=0))
        matrix = confusion_matrix(selected["y_true"], selected["y_pred"], labels=list(CLASS_NAMES))
        print(
            pd.DataFrame(
                matrix,
                index=[f"true_{name}" for name in CLASS_NAMES],
                columns=[f"pred_{name}" for name in CLASS_NAMES],
            ).to_string()
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grouped cross-validation of spectrum classifiers.")
    parser.add_argument(
        "--skip-cnn",
        action="store_true",
        help="Run classical and MLP models only; skip the slower nested 2D CNN evaluation.",
    )
    parser.add_argument("--cnn-epochs", type=int, default=350, help="Maximum epochs for internal CNN selection.")
    parser.add_argument("--cnn-patience", type=int, default=80, help="Early-stopping patience during CNN selection.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "cross_validation",
        help="Directory for CSV output files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_all_dataset()
    _, y_text, meta = make_sweep_examples(df, window_size=WINDOW_SIZE, aggregate="mean")
    encoder = LabelEncoder().fit(np.array(CLASS_NAMES, dtype=object))
    folds = build_folds(meta, encoder.transform(y_text))

    print("Fold test groups:")
    for fold in folds:
        test_meta = meta[meta["sample_group_name"].isin(fold.test_groups)]
        class_counts = test_meta["class_name"].value_counts().reindex(CLASS_NAMES, fill_value=0)
        print(
            f"fold={fold.number}: groups={', '.join(sorted(fold.test_groups))}; "
            f"objects=AM:{class_counts['AM']}, FM:{class_counts['FM']}, Noise:{class_counts['Noise']}"
        )

    score_rows, prediction_frames = evaluate_feature_models(folds, encoder)
    vector_scores, vector_predictions = evaluate_vector_models(folds, encoder)
    score_rows.extend(vector_scores)
    prediction_frames.extend(vector_predictions)

    if not args.skip_cnn:
        cnn_scores, cnn_predictions = evaluate_cnn_model(
            folds,
            encoder,
            epochs=args.cnn_epochs,
            patience=args.cnn_patience,
        )
        score_rows.extend(cnn_scores)
        prediction_frames.extend(cnn_predictions)

    scores = pd.DataFrame(score_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = summarize_results(scores, predictions)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    scores.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    print_diagnostics(summary, scores, predictions)
    print(f"\nCSV results saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
