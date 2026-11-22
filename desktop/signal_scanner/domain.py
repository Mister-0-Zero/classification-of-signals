from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
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
        if not 101 <= self.points <= 32001:
            raise ValueError("Количество точек: от 101 до 32001.")
        if self.rbw_hz > self.span_hz or self.vbw_hz > self.span_hz:
            raise ValueError("RBW и VBW не должны превышать span.")
        if self.detector not in {"POS", "SAMPLE", "RMS", "NEG", "APE", "AVER", "QPE"}:
            raise ValueError("Неизвестный детектор.")


@dataclass(frozen=True)
class Spectrum:
    settings: AnalyzerSettings
    frequency_hz: tuple[float, ...]
    sweeps_dbm: tuple[tuple[float, ...], ...]
    captured_at: str
    simulated: bool = True
    measurement_metadata: dict = field(default_factory=dict)
    # Legacy attribute names *_dbm are retained for existing callers. The unit
    # is explicit, and hardware values are never silently converted.
    amplitude_unit: str = "DBM"

    def validate(self) -> None:
        self.settings.validate()
        if self.amplitude_unit not in {"DBM", "DBUV", "DBMV", "DBUA", "DBPW"}:
            raise ValueError("Неподдерживаемая единица уровня.")
        if type(self.simulated) is not bool:
            raise ValueError("simulated must be an explicit boolean")
        n = self.settings.points
        if len(self.frequency_hz) != n or not self.sweeps_dbm:
            raise ValueError("Число точек оси/настроек не совпало или нет срезов.")
        if any(len(row) != n for row in self.sweeps_dbm):
            raise ValueError("Число точек трассы не совпало с осью; обрезка запрещена.")
        if not all(math.isfinite(v) for v in self.frequency_hz):
            raise ValueError("Некорректная частотная ось.")
        if not all(math.isfinite(v) for row in self.sweeps_dbm for v in row):
            raise ValueError("Некорректные уровни трассы.")
        minima = self.measurement_metadata.get("autopeak_min_sweeps")
        if minima is not None and (len(minima) != len(self.sweeps_dbm)
                                   or any(len(row) != n for row in minima)
                                   or any(not math.isfinite(v) for row in minima for v in row)):
            raise ValueError("Некорректный второй столбец AUTOPEAK.")
        start = self.settings.center_hz - self.settings.span_hz / 2
        step = self.settings.span_hz / (n - 1)
        if any(not math.isclose(v, start + i * step, rel_tol=1e-9, abs_tol=1e-3)
               for i, v in enumerate(self.frequency_hz)):
            raise ValueError("Частотная ось не соответствует фактическим настройкам.")

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
    amplitude_unit: str = "DBM"


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
    observation_id: str = ""
    session_id: str = ""
    label_certainty: str = "certain"

    @property
    def signal_present(self) -> bool | None:
        if self.label_certainty == "uncertain" or self.signal_count == "unknown":
            return None
        return self.signal_count != "none"

    @property
    def training_eligible(self) -> bool:
        return self.signal_present is not None

    def validate(self) -> None:
        if self.label_certainty not in {"certain", "uncertain"}:
            raise ValueError("Неизвестный признак уверенности разметки.")
        if self.signal_count not in {"none", "one", "multiple", "unknown"}:
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
