from __future__ import annotations

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
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed. Install torch to train the CNN, for example: "
            "pip install torch --index-url https://download.pytorch.org/whl/cu121 "
            "or install the CPU build from pytorch.org."
        ) from exc
    return torch, nn, DataLoader, TensorDataset


def prepare_cnn_arrays(window_size: int = WINDOW_SIZE, random_state: int = RANDOM_STATE) -> dict[str, object]:
    """Prepare 5 x 501 sweep-window tensors for CNN classification."""

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

    # Normalize using train statistics only.
    mean = float(X_train.mean())
    std = float(X_train.std() + 1e-6)
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    # CNN input: N, C, H, W = N, 1, sweep_window, frequency_points.
    return {
        "X_train": X_train[:, None, :, :],
        "X_val": X_val[:, None, :, :],
        "y_train": y[train_idx],
        "y_val": y[val_idx],
        "train_meta": meta.iloc[train_idx].reset_index(drop=True),
        "val_meta": meta.iloc[val_idx].reset_index(drop=True),
        "label_encoder": label_encoder,
    }


def build_cnn_model(num_classes: int):
    torch, nn, _, _ = _require_torch()

    class SpectrumCNN(nn.Module):
        def __init__(self, n_classes: int):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=(3, 9), padding=(1, 4)),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, 2)),
                nn.Conv2d(16, 32, kernel_size=(3, 9), padding=(1, 4)),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, 2)),
                nn.Conv2d(32, 64, kernel_size=(3, 7), padding=(1, 3)),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 16)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.25),
                nn.Linear(64 * 16, 64),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(64, n_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    return SpectrumCNN(num_classes)


def train_cnn(
    prepared: dict[str, object],
    epochs: int = 200,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    random_state: int = RANDOM_STATE,
):
    torch, nn, DataLoader, TensorDataset = _require_torch()
    torch.manual_seed(random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_cnn_model(num_classes=len(prepared["label_encoder"].classes_)).to(device)

    train_ds = TensorDataset(
        torch.tensor(prepared["X_train"], dtype=torch.float32),
        torch.tensor(prepared["y_train"], dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_loss = float("inf")

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            x_val = torch.tensor(prepared["X_val"], dtype=torch.float32, device=device)
            y_val = torch.tensor(prepared["y_val"], dtype=torch.long, device=device)
            val_loss = float(criterion(model(x_val), y_val).item())
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, device


def evaluate_cnn(model, device, prepared: dict[str, object]) -> dict[str, object]:
    torch, _, _, _ = _require_torch()
    model.eval()
    with torch.no_grad():
        x_val = torch.tensor(prepared["X_val"], dtype=torch.float32, device=device)
        pred = model(x_val).argmax(dim=1).cpu().numpy()

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
        prepared_data = prepare_cnn_arrays(window_size=WINDOW_SIZE)
        cnn_model, cnn_device = train_cnn(prepared_data)
        metrics = evaluate_cnn(cnn_model, cnn_device, prepared_data)
    except RuntimeError as exc:
        print(exc)
        raise SystemExit(1)

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
