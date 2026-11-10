from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from .domain import Spectrum


class SpectrumPlot(QWidget):
    def __init__(self, title: str, panorama: bool = False):
        super().__init__()
        self.title = title
        self.panorama = panorama
        self.start = 87e6
        self.stop = 108e6
        self.values: list[float | None] = [None] * 2048
        self.current: Spectrum | None = None
        self.setMinimumHeight(200 if panorama else 170)
        self.setMouseTracking(True)
        self.hover_text = ""

    def reset_band(self, start: float, stop: float) -> None:
        self.start, self.stop = start, stop
        self.values = [None] * 2048
        self.current = None
        self.update()

    def add(self, spectrum: Spectrum) -> None:
        if not self.panorama:
            self.current = spectrum
            self.start = spectrum.frequency_hz[0]
            self.stop = spectrum.frequency_hz[-1]
        else:
            for freq, level in zip(spectrum.frequency_hz, spectrum.mean_dbm):
                if self.start <= freq <= self.stop:
                    index = min(len(self.values) - 1, int((freq - self.start)
                                / (self.stop - self.start) * (len(self.values) - 1)))
                    previous = self.values[index]
                    self.values[index] = level if previous is None else max(previous, level)
        self.update()

    def mouseMoveEvent(self, event) -> None:
        width = max(self.width() - 84, 1)
        fraction = min(1, max(0, (event.position().x() - 60) / width))
        freq = (self.start + fraction * (self.stop - self.start)) / 1e6
        self.hover_text = f"{freq:.5f} МГц"
        self.update()

    def leaveEvent(self, event) -> None:
        self.hover_text = ""
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#101e2e"))
        area = QRectF(60, 42, max(1, self.width() - 84), max(1, self.height() - 78))
        painter.setFont(QFont("Segoe UI", 9))
        painter.setPen(QColor("#d5e3f1"))
        painter.drawText(QRectF(16, 8, self.width() - 32, 24), self.title)
        painter.setPen(QColor("#7692aa"))
        painter.drawText(QRectF(self.width() - 160, 8, 140, 24), Qt.AlignmentFlag.AlignRight, self.hover_text)
        low, high = -110.0, -20.0
        for n in range(7):
            x = area.left() + n * area.width() / 6
            painter.setPen(QPen(QColor("#24364a"), 1))
            painter.drawLine(QPointF(x, area.top()), QPointF(x, area.bottom()))
            painter.setPen(QColor("#90a7be"))
            frequency = (self.start + n * (self.stop - self.start) / 6) / 1e6
            painter.drawText(QRectF(x - 40, area.bottom() + 6, 80, 20), Qt.AlignmentFlag.AlignCenter,
                             f"{frequency:.3f}")
        for level in (-110, -90, -70, -50, -30):
            y = area.bottom() - (level - low) / (high - low) * area.height()
            painter.setPen(QPen(QColor("#24364a"), 1))
            painter.drawLine(QPointF(area.left(), y), QPointF(area.right(), y))
            painter.setPen(QColor("#90a7be"))
            painter.drawText(QRectF(5, y - 8, 48, 18), Qt.AlignmentFlag.AlignRight, str(level))
        painter.drawText(QRectF(6, 25, 48, 18), "dBm")
        painter.drawText(QRectF(self.width() - 55, 25, 45, 18), "МГц")
        points = self.values if self.panorama else (self.current.mean_dbm if self.current else [])
        if not points or all(v is None for v in points):
            painter.setPen(QColor("#70869c"))
            painter.drawText(area, Qt.AlignmentFlag.AlignCenter, "Ожидание измерений · имитация")
            return
        path = QPainterPath()
        connected = False
        for index, value in enumerate(points):
            if value is None or not math.isfinite(value):
                connected = False
                continue
            x = area.left() + index / max(1, len(points) - 1) * area.width()
            y = area.bottom() - (max(low, min(high, value)) - low) / (high - low) * area.height()
            if connected:
                path.lineTo(x, y)
            else:
                path.moveTo(x, y)
                # A bin may be isolated during a wide scan; render it without bridging unknown bins.
                path.lineTo(x + .4, y)
            connected = True
        painter.setPen(QPen(QColor("#52dfc3" if self.panorama else "#64b8ff"), 1.5))
        painter.drawPath(path)
