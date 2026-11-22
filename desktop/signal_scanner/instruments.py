from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
import math
import random
import threading

from .domain import AnalyzerSettings, Spectrum, utc_now


class Cancelled(Exception):
    pass


class Analyzer(ABC):
    """Single-owner device interface. Future transports must read actual settings."""

    @property
    @abstractmethod
    def simulated(self) -> bool: ...

    def snapshot_state(self):
        return self.read_settings()

    def restore_state(self, state) -> None:
        self.apply_settings(state)

    def series_snapshot(self):
        raise NotImplementedError("Этот адаптер не поддерживает случайные серии.")

    def apply_variant(self, variant, initial):
        raise NotImplementedError("Этот адаптер не поддерживает случайные серии.")

    def restore_series_state(self, state, *, spectral=True):
        raise NotImplementedError("Этот адаптер не поддерживает случайные серии.")

    @abstractmethod
    def read_settings(self) -> AnalyzerSettings: ...

    @abstractmethod
    def apply_settings(self, settings: AnalyzerSettings) -> AnalyzerSettings: ...

    @abstractmethod
    def capture(self, sweeps: int, cancel: threading.Event) -> Spectrum: ...

    @abstractmethod
    def release_local(self) -> None: ...


@dataclass(frozen=True)
class DemoSignal:
    center_hz: float
    bandwidth_hz: float
    peak_dbm: float
    modulation: str
    intermittent: bool = False


class Simulator(Analyzer):
    """Synthetic spectra, not a physical model or trained classifier."""

    simulated = True

    def __init__(self, seed: int = 42, delay_scale: float = 1.0):
        self.settings = AnalyzerSettings()
        self.random = random.Random(seed)
        self.delay_scale = delay_scale
        self.pass_number = 1
        self.scene = "Эфир"
        self.world = [
            DemoSignal(14_200_000, 18_000, -44, "AM"),
            DemoSignal(27_125_000, 25_000, -54, "AM"),
            DemoSignal(40_680_000, 40_000, -60, "FM"),
            DemoSignal(68_500_000, 70_000, -55, "FM"),
            DemoSignal(87_500_000, 150_000, -43, "FM"),
            DemoSignal(89_100_000, 120_000, -52, "FM"),
            DemoSignal(95_600_000, 130_000, -58, "FM", True),
            DemoSignal(100_100_000, 140_000, -44, "FM"),
            DemoSignal(104_700_000, 80_000, -48, "AM"),
            DemoSignal(145_500_000, 30_000, -57, "FM"),
            DemoSignal(172_300_000, 100_000, -49, "AM"),
        ]

    def read_settings(self) -> AnalyzerSettings:
        return self.settings

    def apply_settings(self, settings: AnalyzerSettings) -> AnalyzerSettings:
        settings.validate()
        self.settings = settings
        return settings

    def series_snapshot(self):
        return {"settings": asdict(self.settings),
                "trace_mode": getattr(self, "series_trace_mode", "WRIT"),
                "sweep_count": getattr(self, "series_sweep_count", 0)}

    def apply_variant(self, variant, initial):
        settings = AnalyzerSettings(**variant["settings"])
        if (settings.center_hz, settings.span_hz, settings.points) != (initial.center_hz, initial.span_hz, initial.points):
            raise ValueError("Серия не должна менять частотное окно или число точек.")
        self.series_trace_mode = variant["trace_mode"]
        self.series_sweep_count = variant["sweep_count"]
        return self.apply_settings(settings)

    def restore_series_state(self, state, *, spectral=True):
        self.apply_settings(AnalyzerSettings(**state["settings"]))
        self.series_trace_mode = state["trace_mode"]
        self.series_sweep_count = state["sweep_count"]

    def set_scene(self, name: str) -> None:
        if name not in {"Эфир", "Нет сигнала", "Один AM", "Один FM", "Несколько", "Слабый сигнал"}:
            raise ValueError("Неизвестная демонстрационная сцена.")
        self.scene = name

    def signals(self) -> list[DemoSignal]:
        s = self.settings
        if self.scene == "Эфир":
            return [v for v in self.world if not v.intermittent or self.pass_number % 2]
        if self.scene == "Нет сигнала":
            return []
        if self.scene in {"Один AM", "Один FM", "Слабый сигнал"}:
            return [DemoSignal(s.center_hz, min(s.span_hz * .25, 120_000),
                               -82 if self.scene == "Слабый сигнал" else -48,
                               "AM" if self.scene == "Один AM" else "FM")]
        return [DemoSignal(s.center_hz - s.span_hz * .22, s.span_hz * .1, -48, "AM"),
                DemoSignal(s.center_hz + s.span_hz * .19, s.span_hz * .17, -56, "FM")]

    def capture(self, sweeps: int, cancel: threading.Event) -> Spectrum:
        s = self.settings
        s.validate()
        if not 1 <= sweeps <= 50:
            raise ValueError("Количество срезов: от 1 до 50.")
        axis = tuple(s.center_hz - s.span_hz / 2 + s.span_hz * i / (s.points - 1)
                     for i in range(s.points))
        candidates = [v for v in self.signals()
                      if abs(v.center_hz - s.center_hz) <= s.span_hz / 2 + v.bandwidth_hz]
        matrix = []
        for sweep in range(sweeps):
            if cancel.is_set():
                raise Cancelled()
            if cancel.wait(self.delay_scale * min(.25, .025 + 80 / s.rbw_hz)):
                raise Cancelled()
            row = []
            for frequency in axis:
                noise = -94 + 10 * math.log10(s.rbw_hz / 10_000)
                noise += self.random.gauss(0, 1.5 if s.vbw_hz >= s.rbw_hz else .7)
                power = 10 ** (noise / 10)
                for signal in candidates:
                    if signal.modulation == "AM":
                        width = max(s.rbw_hz * .6, signal.bandwidth_hz * .025)
                        for offset, attenuation in ((0, 0), (-signal.bandwidth_hz * .35, 7),
                                                    (signal.bandwidth_hz * .35, 7)):
                            z = (frequency - signal.center_hz - offset) / width
                            power += 10 ** ((signal.peak_dbm - attenuation) / 10) * math.exp(-.5 * z * z)
                    else:
                        width = max(s.rbw_hz, signal.bandwidth_hz * .3)
                        z = (frequency - signal.center_hz) / width
                        ripple = 1 + .12 * math.sin(frequency / 7000 + sweep)
                        power += 10 ** (signal.peak_dbm / 10) * math.exp(-.5 * z ** 4) * ripple
                row.append(10 * math.log10(max(power, 1e-30)))
            matrix.append(tuple(row))
        return Spectrum(s, axis, tuple(matrix), utc_now())

    def release_local(self) -> None:
        """No hardware lock exists in the simulator."""


def create_analyzer(transport: str, address: str, seed: int = 42) -> Analyzer:
    if transport == "Имитация":
        return Simulator(seed)
    from .fsl import FslAnalyzer, SocketTransport
    if not address.strip():
        raise ValueError("Укажите текущий IP / VISA-адрес прибора.")
    if transport == "TCP-сокет":
        return FslAnalyzer(address, transport_factory=lambda: SocketTransport(address), diagnostics_only=True)
    if transport in {"PyVISA", "PyVISA (системная VISA)"}:
        return FslAnalyzer(address, backend="" if transport.endswith("VISA)") else "@py")
    raise ValueError("Неизвестный способ подключения.")
