from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder

from baseline_logreg import RANDOM_STATE, WINDOW_SIZE, split_train_val_by_group
from mainFunction import load_all_dataset, make_sweep_examples


def _require_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed. Install torch first: pip install torch"
        ) from exc
    return torch, nn, DataLoader, Dataset


@dataclass(frozen=True)
class AugmentationConfig:
    enabled: bool = True
    shift_prob: float = 0.70
    max_shift_points: int = 25
    crop_prob: float = 0.60
    max_crop_points: int = 60
    noise_prob: float = 0.25
    noise_std: float = 0.03


def _shift_frequency_axis(x: np.ndarray, shift: int, fill_value: float = 0.0) -> np.ndarray:
    """Shift spectrum along frequency axis while preserving fixed width."""

    if shift == 0:
        return x

    result = np.full_like(x, fill_value)
    if shift > 0:
        result[..., shift:] = x[..., :-shift]
    else:
        result[..., :shift] = x[..., -shift:]
    return result


def _crop_frequency_axis(
    x: np.ndarray,
    left_crop: int = 0,
    right_crop: int = 0,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Simulate left/right span crop by masking edge frequency columns."""

    result = x.copy()
    if left_crop > 0:
        result[..., :left_crop] = fill_value
    if right_crop > 0:
        result[..., -right_crop:] = fill_value
    return result


class Spectrum2DDataset:
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        augment: AugmentationConfig | None = None,
        random_state: int = RANDOM_STATE,
    ):
        torch, _, _, TorchDataset = _require_torch()
        self.torch = torch
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.augment = augment or AugmentationConfig(enabled=False)
        self.rng = np.random.default_rng(random_state)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = self.X[idx].copy()

        if self.augment.enabled:
            if self.rng.random() < self.augment.shift_prob:
                shift = int(self.rng.integers(-self.augment.max_shift_points, self.augment.max_shift_points + 1))
                x = _shift_frequency_axis(x, shift=shift, fill_value=0.0)

            if self.rng.random() < self.augment.crop_prob:
                mode = self.rng.choice(["left", "right", "both"])
                if mode == "left":
                    left_crop = int(self.rng.integers(5, self.augment.max_crop_points + 1))
                    right_crop = 0
                elif mode == "right":
                    left_crop = 0
                    right_crop = int(self.rng.integers(5, self.augment.max_crop_points + 1))
                else:
                    left_crop = int(self.rng.integers(5, self.augment.max_crop_points // 2 + 1))
                    right_crop = int(self.rng.integers(5, self.augment.max_crop_points // 2 + 1))
                x = _crop_frequency_axis(x, left_crop=left_crop, right_crop=right_crop, fill_value=0.0)

            if self.rng.random() < self.augment.noise_prob:
                x = x + self.rng.normal(0.0, self.augment.noise_std, size=x.shape).astype(np.float32)

        return self.torch.tensor(x, dtype=self.torch.float32), self.torch.tensor(self.y[idx], dtype=self.torch.long)


def prepare_cnn2d_arrays(window_size: int = WINDOW_SIZE, random_state: int = RANDOM_STATE) -> dict[str, object]:
    """Prepare tensors for 2D-CNN: N x 1 x window_size x 501."""

    df = load_all_dataset()
    X_flat, y_text, meta = make_sweep_examples(df, window_size=window_size, aggregate="none")
    X = X_flat.reshape(len(X_flat), window_size, 501).astype(np.float32)

    split_table = meta[["class_name", "sample_group_name"]].copy()
    split_table["row_idx"] = np.arange(len(meta))
    train_meta_idx, val_meta_idx = split_train_val_by_group(split_table, random_state=random_state)

    train_idx = train_meta_idx["row_idx"].to_numpy()
    val_idx = val_meta_idx["row_idx"].to_numpy()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text).astype(np.int64)

    X_train = X[train_idx]
    X_val = X[val_idx]

    mean = float(X_train.mean())
    std = float(X_train.std() + 1e-6)
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    return {
        "X_train": X_train[:, None, :, :],
        "X_val": X_val[:, None, :, :],
        "y_train": y[train_idx],
        "y_val": y[val_idx],
        "train_meta": meta.iloc[train_idx].reset_index(drop=True),
        "val_meta": meta.iloc[val_idx].reset_index(drop=True),
        "label_encoder": label_encoder,
        "normalization_mean": mean,
        "normalization_std": std,
    }


def build_cnn2d_model(num_classes: int):
    torch, nn, _, _ = _require_torch()

    class Spectrum2DCNN(nn.Module):
        def __init__(self, n_classes: int):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=(3, 15), padding=(1, 7)),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, 2)),
                nn.Dropout2d(0.10),
                nn.Conv2d(16, 32, kernel_size=(3, 11), padding=(1, 5)),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, 2)),
                nn.Dropout2d(0.15),
                nn.Conv2d(32, 64, kernel_size=(3, 7), padding=(1, 3)),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 24)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.35),
                nn.Linear(64 * 24, 128),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, n_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    return Spectrum2DCNN(num_classes)


def _predict(model, device, X: np.ndarray) -> np.ndarray:
    torch, _, _, _ = _require_torch()
    model.eval()
    with torch.no_grad():
        x = torch.tensor(X, dtype=torch.float32, device=device)
        return model(x).argmax(dim=1).cpu().numpy()


def train_cnn2d(
    prepared: dict[str, object],
    epochs: int = 350,
    batch_size: int = 16,
    lr: float = 8e-4,
    weight_decay: float = 5e-4,
    patience: int = 80,
    augmentation: AugmentationConfig | None = None,
    random_state: int = RANDOM_STATE,
):
    torch, nn, DataLoader, _ = _require_torch()
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_cnn2d_model(num_classes=len(prepared["label_encoder"].classes_)).to(device)

    train_ds = Spectrum2DDataset(
        prepared["X_train"],
        prepared["y_train"],
        augment=augmentation or AugmentationConfig(enabled=True),
        random_state=random_state,
    )
    generator = torch.Generator()
    generator.manual_seed(random_state)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)

    class_counts = np.bincount(prepared["y_train"], minlength=len(prepared["label_encoder"].classes_))
    class_weights = class_counts.sum() / np.maximum(class_counts, 1)
    class_weights = class_weights / class_weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    x_val = torch.tensor(prepared["X_val"], dtype=torch.float32, device=device)
    y_val = torch.tensor(prepared["y_val"], dtype=torch.long, device=device)

    best_state = None
    best_macro_f1 = -1.0
    best_loss = float("inf")
    stale_epochs = 0
    history = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += float(loss.item()) * len(xb)
        scheduler.step()
        train_loss /= len(train_ds)

        model.eval()
        with torch.no_grad():
            logits = model(x_val)
            val_loss = float(criterion(logits, y_val).item())
            pred = logits.argmax(dim=1).cpu().numpy()
            macro_f1 = float(f1_score(prepared["y_val"], pred, average="macro"))
            acc = float(accuracy_score(prepared["y_val"], pred))

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": acc,
                "val_macro_f1": macro_f1,
            }
        )

        improved = (macro_f1 > best_macro_f1 + 1e-6) or (
            abs(macro_f1 - best_macro_f1) <= 1e-6 and val_loss < best_loss
        )
        if improved:
            best_macro_f1 = macro_f1
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.training_history = history
    model.best_macro_f1 = best_macro_f1
    model.best_val_loss = best_loss
    return model, device


def evaluate_cnn2d(model, device, prepared: dict[str, object]) -> dict[str, object]:
    pred = _predict(model, device, prepared["X_val"])
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
    try:
        prepared_data = prepare_cnn2d_arrays(window_size=WINDOW_SIZE)
        model, device = train_cnn2d(prepared_data)
        metrics = evaluate_cnn2d(model, device, prepared_data)
    except RuntimeError as exc:
        print(exc)
        raise SystemExit(1)

    print("Train shape:", prepared_data["X_train"].shape)
    print("Validation shape:", prepared_data["X_val"].shape)
    print("Validation groups:")
    print(prepared_data["val_meta"][["class_name", "sample_group_name"]].drop_duplicates().to_string(index=False))
    print("\nDevice:", device)
    print("Best macro-F1:", round(model.best_macro_f1, 4))
    print("Accuracy:", round(metrics["accuracy"], 4))
    print("Macro-F1:", round(metrics["macro_f1"], 4))
    print("\nClassification report:")
    print(metrics["classification_report"])
    print("Confusion matrix:")
    print(metrics["confusion_matrix"].to_string())
