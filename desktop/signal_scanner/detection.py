from __future__ import annotations

from .domain import Detection, Spectrum


class ThresholdDetector:
    """Noise-relative baseline. No ground-truth modulation or ML confidence is exposed."""

    def detect(self, spectrum: Spectrum, threshold_db: float = 8.0) -> list[Detection]:
        spectrum.validate()
        y = spectrum.mean_dbm
        ordered = sorted(y)
        floor = ordered[int((len(ordered) - 1) * .2)]
        indices = [i for i, value in enumerate(y) if value >= floor + threshold_db]
        if not indices:
            return []
        # Merge short gaps using physical frequency distance, not a fixed pixel count.
        spacing = spectrum.settings.span_hz / (spectrum.settings.points - 1)
        gap = max(2, int(max(spectrum.settings.rbw_hz * 2, 45_000) / spacing))
        groups = [[indices[0]]]
        for index in indices[1:]:
            if index - groups[-1][-1] <= gap:
                groups[-1].append(index)
            else:
                groups.append([index])
        results = []
        for group in groups:
            peak = max(group, key=lambda i: y[i])
            first, last = group[0], group[-1]
            results.append(Detection(
                spectrum.frequency_hz[peak], spectrum.frequency_hz[first],
                spectrum.frequency_hz[last], y[peak], y[peak] - floor,
                amplitude_unit=spectrum.amplitude_unit,
            ))
        return results
