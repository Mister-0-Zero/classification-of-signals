"""CPU inference for SignalScope. Only the exporter needs PyTorch."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax

from .classifier import CLASSES, FEATURE_NAMES, extract_features

ASSETS = Path(__file__).parent


def dense(x, weights, name):
    return x @ weights[name + ".weight"].T + weights[name + ".bias"]


def convolution(x, weights, name):
    filters = weights[name + ".weight"]
    kernel = filters.shape[-1]
    padded = np.pad(x, ((0, 0), (0, 0), (kernel // 2, kernel // 2)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel, axis=-1)
    values = np.einsum("bctk,ock->bot", windows, filters, optimize=True)
    return np.maximum(values + weights[name + ".bias"][None, :, None], 0)


def neural_logits(traces, weights, mode):
    relative = (traces - traces.max(axis=1, keepdims=True)).astype(np.float32)
    x = (relative - weights["mean"]) / weights["scale"]
    if mode == "fast":
        for layer in ("0", "2"):
            x = np.maximum(dense(x, weights, layer), 0)
        return dense(x, weights, "5")[0]
    x = convolution(x[:, None, :], weights, "encoder.0")
    x = convolution(x, weights, "encoder.2")
    x = (x[:, :, :500:2] + x[:, :, 1:500:2]) * .5
    x = convolution(x, weights, "encoder.5")
    x = x.reshape(7, 32, 25, 10).mean(axis=-1).reshape(7, 800)
    x = np.maximum(dense(x, weights, "encoder.9"), 0)
    return dense(x.mean(axis=0), weights, "output")


class ScannerClassifier:
    def __init__(self, mode):
        self.sweeps = {"fast": 1, "quality": 7}[mode]
        self.mode = mode
        self.gb = joblib.load(ASSETS / f"scanner_gb{self.sweeps}.joblib")
        if list(self.gb.classes_) != CLASSES or list(self.gb.feature_names_in_) != FEATURE_NAMES:
            raise ValueError("Несовместимая модель классификатора")
        asset = "scanner_mlp.npz" if mode == "fast" else "scanner_sets.npz"
        with np.load(ASSETS / asset, allow_pickle=False) as weights:
            self.weights = dict(weights)

    def predict(self, spectrum):
        spectrum.validate()
        traces = np.asarray(spectrum.sweeps_dbm)
        axis = np.asarray(spectrum.frequency_hz)
        if (spectrum.amplitude_unit != "DBM" or traces.shape != (self.sweeps, 501)
                or not np.isclose(axis[-1] - axis[0], 250000, atol=2)):
            return "Неизвестно", "Нужны измерения в дБм, 501 точка и полоса 250 кГц"
        peak = traces.max()
        averaged = peak + 10 * np.log10((10 ** ((traces - peak) / 10)).mean(axis=0))
        power = 10 ** ((averaged - averaged.max()) / 10)
        cumulative = np.cumsum(np.maximum(power - np.median(power), 0))
        center = (np.interp(cumulative[-1] * .5, cumulative, axis)
                  if cumulative[-1] else spectrum.settings.center_hz)
        try:
            features = extract_features(axis, averaged, center)
        except ValueError as exc:
            return "Неизвестно", str(exc)
        gb = self.gb.predict_proba(pd.DataFrame([features], columns=FEATURE_NAMES))[0]
        if self.mode == "fast":
            scores = .75 * softmax(neural_logits(traces, self.weights, "fast")) + .25 * gb
            index = scores[:2].argmax() if scores[:2].sum() >= .30 else 2
            origin = "MLP + GB"
        elif gb.max() < .90:
            index = neural_logits(traces, self.weights, "quality").argmax()
            origin = "GB → Deep Sets"
        else:
            index = gb.argmax()
            origin = "GB (7 развёрток)"
        return ("АМ", "ЧМ", "Шум")[index], origin
