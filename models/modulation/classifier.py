"""Inference for the frozen single-spectrum AM/FM/Noise classifier."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


CONTRAST_BANDS = (
    ("contrast_3_8_db", 2, 3, 8),
    ("contrast_8_18_db", 6, 8, 18),
    ("contrast_18_35_db", 6, 18, 35),
    ("contrast_35_65_db", 6, 35, 65),
    ("contrast_65_100_db", 6, 65, 100),
)
FEATURE_NAMES = [
    "bandwidth_50_khz", "bandwidth_90_khz", "bandwidth_50_90_ratio",
    "central_power_fraction", "mirror_asymmetry",
    *[name for name, *_ in CONTRAST_BANDS],
]
CLASSES = ["AM", "FM", "Noise"]
MODEL_PATH = Path(__file__).with_name("gradient_boosting.joblib")


def extract_features(frequency_hz, amplitude_dbm, candidate_hz):
    """Build the ten training features from one trace and its estimated center."""
    frequencies = np.asarray(frequency_hz, dtype=float)
    amplitude = np.asarray(amplitude_dbm, dtype=float)
    if (frequencies.ndim != 1 or frequencies.size < 2 or amplitude.shape != frequencies.shape
            or not np.isfinite(frequencies).all() or not np.isfinite(amplitude).all()
            or not np.isfinite(candidate_hz) or np.any(np.diff(frequencies) <= 0)):
        raise ValueError("Use finite levels in dBm and strictly increasing frequencies in Hz")
    relative_khz = (frequencies - candidate_hz) / 1000
    start = int(np.ceil(max(relative_khz[0], -125)))
    end = int(np.floor(min(relative_khz[-1], 125)))
    if end - start < 40:
        raise ValueError("Less than 40 kHz of measured spectrum in the candidate window")

    grid = np.arange(start, end + 1, dtype=float)
    normalized_power = 10 ** ((amplitude - amplitude.max()) / 10)
    power = np.interp(grid, relative_khz, normalized_power)
    total = power.sum()
    if total <= 0:
        raise ValueError("No positive power in the candidate window")
    cumulative = (np.cumsum(power) - power / 2) / total
    q05, q25, center, q75, q95 = np.interp([.05, .25, .5, .75, .95], cumulative, grid)
    b50, b90 = q75 - q25, q95 - q05
    offset = np.abs(grid - center)

    smoothed = gaussian_filter1d(power, sigma=1.5, mode="nearest")
    floor = smoothed.max() * 1e-6
    radius = int(min(center - start, end - center, 100))
    distances = np.arange(1, radius + 1)
    left = np.interp(center - distances, grid, smoothed)
    right = np.interp(center + distances, grid, smoothed)
    asymmetry = np.abs(left - right).sum() / max((left + right).sum(), floor)
    features = {
        "bandwidth_50_khz": float(b50),
        "bandwidth_90_khz": float(b90),
        "bandwidth_50_90_ratio": float(b50 / max(b90, 1e-9)),
        "central_power_fraction": float(power[offset <= 6].sum() / total),
        "mirror_asymmetry": float(asymmetry),
    }
    for name, core, near, far in CONTRAST_BANDS:
        zone = smoothed[(offset >= near) & (offset <= far)]
        if not zone.size:
            raise ValueError(f"The measured spectrum does not cover the zone for {name}")
        peak = smoothed[offset <= core].max()
        features[name] = float(10 * np.log10((peak + floor) / (zone.mean() + floor)))
    return features


class ModulationClassifier:
    """Load the repository's own model once, then reuse it for multiple sweeps."""

    def __init__(self):
        artifact = joblib.load(MODEL_PATH)
        self._model = artifact["model"]
        if (artifact["feature_names"] != FEATURE_NAMES
                or list(self._model.feature_names_in_) != FEATURE_NAMES
                or list(self._model.classes_) != CLASSES):
            raise ValueError("The model artifact does not match this feature extractor")

    def predict(self, frequency_hz, amplitude_dbm, candidate_hz):
        """Return AM, FM or Noise for the given spectrum."""
        features = extract_features(frequency_hz, amplitude_dbm, candidate_hz)
        frame = pd.DataFrame([features], columns=FEATURE_NAMES)
        return str(self._model.predict(frame)[0])
