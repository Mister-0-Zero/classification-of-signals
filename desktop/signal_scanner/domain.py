from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import math


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class AnalyzerSettings:
    center_hz: float = 100_100_000
    span_hz: float = 1_000_000
    rbw_hz: float = 10_000
    vbw_hz: float = 10_000
    points: int = 1001
    detector: str = "POS"

    def validate(self) -> None:
        values = (self.center_hz, self.span_hz, self.rbw_hz, self.vbw_hz)
        if not all(math.isfinite(x) and x > 0 for x in values):
            raise ValueError("Частота, span, RBW и VBW должны быть положительными числами.")
        if self.center_hz <= self.span_hz / 2:
            raise ValueError("Нижняя граница окна должна быть выше 0 Гц.")
        if not 101 <= self.points <= 10001:
            raise ValueError("Количество точек: от 101 до 10001.")
        if self.rbw_hz > self.span_hz or self.vbw_hz > self.span_hz:
            raise ValueError("RBW и VBW не должны превышать span.")
        if self.detector not in {"POS", "SAMPLE", "RMS"}:
            raise ValueError("Неизвестный детектор.")


@dataclass(frozen=True)
class Spectrum:
    settings: AnalyzerSettings
    frequency_hz: tuple[float, ...]
    sweeps_dbm: tuple[tuple[float, ...], ...]
    captured_at: str
    simulated: bool = True

    @property
    def mean_dbm(self) -> tuple[float, ...]:
        n = len(self.sweeps_dbm)
        return tuple(sum(row[i] for row in self.sweeps_dbm) / n
                     for i in range(len(self.frequency_hz)))


@dataclass(frozen=True)
class Detection:
    center_hz: float
    start_hz: float
    stop_hz: float
    peak_dbm: float
    prominence_db: float
    modulation: str = "Неизвестно"
    origin: str = "Пороговый детектор; модель не подключена"


@dataclass(frozen=True)
class ScanPlan:
    start_hz: float
    stop_hz: float
    settings: AnalyzerSettings
    overlap: float = 0.15
    continuous: bool = False
    refine: bool = True
    sweeps: int = 2
    threshold_db: float = 8.0

    def windows(self) -> tuple[tuple[float, float], ...]:
        if not (math.isfinite(self.start_hz) and math.isfinite(self.stop_hz)
                and 0 < self.start_hz < self.stop_hz):
            raise ValueError("Начальная частота должна быть положительной и меньше конечной.")
        if not 0 <= self.overlap <= 0.5:
            raise ValueError("Перекрытие должно быть от 0 до 50%.")
        if not 1 <= self.sweeps <= 50:
            raise ValueError("Количество срезов: от 1 до 50.")
        if not 1 <= self.threshold_db <= 40:
            raise ValueError("Порог обнаружения: от 1 до 40 дБ над фоном.")
        width = min(self.settings.span_hz, self.stop_hz - self.start_hz)
        replace(self.settings, center_hz=self.start_hz + width / 2, span_hz=width).validate()
        if max(self.settings.rbw_hz, self.settings.vbw_hz) > width:
            raise ValueError("Выбранный диапазон уже RBW или VBW.")
        count = math.ceil(max(0, self.stop_hz - self.start_hz - width)
                          / (width * (1 - self.overlap))) + 1
        if count > 20000:
            raise ValueError("Более 20000 окон за проход. Увеличьте span или сузьте диапазон.")
        return tuple((min(self.start_hz + i * width * (1 - self.overlap),
                          self.stop_hz - width) + width / 2, width)
                     for i in range(count))


@dataclass(frozen=True)
class CaptureLabels:
    signal_count: str
    modulation: str
    comment: str = ""
    source_name: str = ""
    settings_source: str = "application"

    def validate(self) -> None:
        if self.signal_count not in {"none", "one", "multiple"}:
            raise ValueError("Укажите количество сигналов.")
        if self.modulation not in {"AM", "FM", "mixed", "unknown", "none"}:
            raise ValueError("Укажите модуляцию.")
        if self.signal_count == "none" and self.modulation != "none":
            raise ValueError("Для пустого окна модуляция должна быть «Нет».")
        if self.signal_count != "none" and self.modulation == "none":
            raise ValueError("Для сигнала выберите модуляцию или «Неизвестно».")
        if self.signal_count == "one" and self.modulation == "mixed":
            raise ValueError("Смешанная модуляция предназначена для нескольких сигналов.")


PRESETS = {
    "Быстрый": (2_000_000, 30_000, 30_000, 1),
    "Сбалансированный": (1_000_000, 10_000, 10_000, 2),
    "Подробный": (250_000, 3_000, 3_000, 5),
}
