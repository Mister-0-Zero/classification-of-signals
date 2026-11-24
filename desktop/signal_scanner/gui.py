from __future__ import annotations

from dataclasses import asdict, replace
import csv
import json
from pathlib import Path
import threading
import uuid

from PySide6.QtCore import QThread, QTimer, Qt, QUrl, Signal
from PySide6.QtGui import QAction, QColor, QDesktopServices, QKeySequence, QPalette
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QListWidget, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QScrollArea, QSpinBox, QSplitter, QStackedWidget, QTableWidget,
    QTableWidgetItem, QTextBrowser, QVBoxLayout, QWidget,
)

from .domain import AnalyzerSettings, CaptureLabels, PRESETS, ScanPlan
from .help_text import HELP_HTML
from .instruments import Analyzer, Cancelled, Simulator, create_analyzer
from .plot import SpectrumPlot
from .services import Control, collect, scan
from .capture_series import SeriesPlan, collect_series
from .storage import Repository, application_home


STYLE = """
QWidget { font-family: 'Segoe UI'; font-size: 10pt; color: #203249; }
QMainWindow, QStackedWidget { background: #f2f5f9; }
QLabel#brand { color: white; font-size: 21pt; font-weight: 700; }
QLabel#brandSub { color: #9eafc4; font-size: 9pt; }
QWidget#sidebar { background: #152439; }
QListWidget#navigation { background: transparent; border: none; color: #b7c7dc; outline: none; }
QListWidget#navigation::item { padding: 15px 10px; margin: 4px 0; border-radius: 7px; }
QListWidget#navigation::item:selected { background: #28415d; color: #78efd4; }
QLabel#title { font-size: 22pt; font-weight: 650; color: #152d46; }
QLabel#subtitle { color: #63788e; }
QLabel#banner { background: #fff0cf; color: #785215; border: 1px solid #ead59f; padding: 9px 14px; border-radius: 6px; }
QLabel#badge { background: #e0e8f0; padding: 7px 12px; border-radius: 6px; color: #40566b; }
QGroupBox { background: white; border: 1px solid #dbe3ed; border-radius: 8px; margin-top: 12px; padding: 16px 10px 10px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 14px; padding: 0 5px; }
QPushButton { background: white; border: 1px solid #cbd6e2; border-radius: 6px; padding: 8px 14px; }
QPushButton:hover { background: #eaf0f7; }
QPushButton:disabled { color: #99a7b7; background: #edf1f5; }
QPushButton#primary { background: #126c67; color: white; border: none; font-weight: 600; }
QPushButton#primary:hover { background: #19837c; }
QPushButton#danger { color: #a84439; border-color: #ddbbb7; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit { background: white; border: 1px solid #cbd6e2; border-radius: 4px; padding: 5px; selection-background-color: #b7e7de; }
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled { background: #eff2f6; color: #8391a0; }
QTableWidget { background: white; border: 1px solid #dbe3ed; border-radius: 6px; gridline-color: #edf1f6; alternate-background-color: #f5f8fb; selection-background-color: #d6eee9; selection-color: #163e3b; }
QHeaderView::section { background: #e9eff6; border: none; padding: 8px 5px; font-weight: 600; }
QProgressBar { border: none; background: #dfe7f0; border-radius: 4px; text-align: center; min-height: 18px; }
QProgressBar::chunk { background: #62bba6; border-radius: 4px; }
QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #aabccc; border-radius: 3px; background: white; }
QCheckBox::indicator:checked { background: #168275; border: 1px solid #126c67; image: none; }
QTextBrowser { border: 1px solid #dbe3ed; background: white; padding: 20px; }
QScrollArea { border: none; background: transparent; }
QStatusBar { background: #e6edf4; color: #41566e; }
"""


def configure_application(app: QApplication) -> None:
    app.setStyle("Fusion")
    palette = QPalette()
    for role, color in ((QPalette.ColorRole.Window, "#f2f5f9"),
                        (QPalette.ColorRole.WindowText, "#203249"),
                        (QPalette.ColorRole.Base, "#ffffff"),
                        (QPalette.ColorRole.AlternateBase, "#f5f8fb"),
                        (QPalette.ColorRole.Text, "#203249"),
                        (QPalette.ColorRole.Button, "#ffffff"),
                        (QPalette.ColorRole.ButtonText, "#203249"),
                        (QPalette.ColorRole.Highlight, "#c4e8df"),
                        (QPalette.ColorRole.HighlightedText, "#163e3b")):
        palette.setColor(role, QColor(color))
    app.setPalette(palette)
    app.setStyleSheet(STYLE)


def number(value: float, minimum: float, maximum: float, decimals: int = 3) -> QDoubleSpinBox:
    widget = QDoubleSpinBox()
    widget.setRange(minimum, maximum)
    widget.setDecimals(decimals)
    widget.setValue(value)
    widget.setKeyboardTracking(False)
    widget.setMinimumWidth(105)
    return widget


def combo(items: list[str]) -> QComboBox:
    widget = QComboBox()
    widget.addItems(items)
    return widget


def button(text: str, callback, kind: str = "") -> QPushButton:
    widget = QPushButton(text)
    if kind:
        widget.setObjectName(kind)
    widget.clicked.connect(callback)
    return widget


class SettingsEditor(QGroupBox):
    def __init__(self, title: str, *, center: bool = True):
        super().__init__(title)
        form = QFormLayout(self)
        self.center = number(100.1, .001, 6000, 6)
        self.span = number(1, .001, 1000, 6)
        self.rbw = number(10, .001, 1_000_000, 3)
        self.vbw = number(10, .001, 1_000_000, 3)
        self.points = QSpinBox()
        self.points.setRange(101, 10001)
        self.points.setValue(1001)
        self.detector = combo(["POS", "SAMPLE", "RMS", "NEG", "APE", "AVER", "QPE"])
        if center:
            form.addRow("Центр, МГц", self.center)
        form.addRow("Span, МГц", self.span)
        form.addRow("RBW, кГц", self.rbw)
        form.addRow("VBW, кГц", self.vbw)
        form.addRow("Точек", self.points)
        form.addRow("Детектор", self.detector)

    def settings(self) -> AnalyzerSettings:
        return AnalyzerSettings(self.center.value() * 1e6, self.span.value() * 1e6,
                                self.rbw.value() * 1e3, self.vbw.value() * 1e3,
                                self.points.value(), self.detector.currentText())

    def load(self, settings: AnalyzerSettings) -> None:
        self.center.setValue(settings.center_hz / 1e6)
        self.span.setValue(settings.span_hz / 1e6)
        self.rbw.setValue(settings.rbw_hz / 1e3)
        self.vbw.setValue(settings.vbw_hz / 1e3)
        self.points.setValue(settings.points)
        self.detector.setCurrentText(settings.detector)


class Job(QThread):
    event = Signal(object)
    error = Signal(str)

    def __init__(self, operation):
        super().__init__()
        self.operation = operation
        self.control = Control()

    def run(self) -> None:
        try:
            self.operation(self.control, self.event.emit)
        except Cancelled:
            self.event.emit({"type": "preview_stopped"})
        except Exception as error:
            self.error.emit(str(error))


class MainWindow(QMainWindow):
    def __init__(self, home: Path | None = None):
        super().__init__()
        self.setWindowTitle("SignalScope — сканер спектра")
        self.resize(1440, 980)
        self.setMinimumSize(1080, 740)
        self.home = home or application_home()
        self.capture_session_id = uuid.uuid4().hex
        self.preferences = Repository(self.home)
        self.analyzer: Analyzer | None = None
        self.job: Job | None = None
        self.job_kind = ""
        self.rows: list[dict] = []
        self.last_run = ""
        self.close_requested = False
        self.session_start = 87e6
        self.session_stop = 108e6
        self._build()
        self._restore_preferences()
        self._set_busy(False)
        self.nav.setCurrentRow(0)
        self.statusBar().showMessage("Выберите «Подключение» → «Имитация» → «Подключить».")

    def _build(self) -> None:
        shell = QWidget()
        layout = QHBoxLayout(shell)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(200)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(18, 24, 18, 20)
        brand = QLabel("SignalScope")
        brand.setObjectName("brand")
        side.addWidget(brand)
        sub = QLabel("ЛАБОРАТОРИЯ СПЕКТРА\nВКР · версия 0.1")
        sub.setObjectName("brandSub")
        side.addWidget(sub)
        side.addSpacing(25)
        self.nav = QListWidget()
        self.nav.setObjectName("navigation")
        self.nav.addItems(["Сканер", "Сбор данных", "Подключение", "Справка"])
        self.nav.currentRowChanged.connect(self._navigate)
        side.addWidget(self.nav)
        footer = QLabel("ИМИТАЦИЯ / LAN FSL\nИсточник: «Подключение»\nМодели не подключены")
        footer.setObjectName("brandSub")
        side.addWidget(footer)
        layout.addWidget(sidebar)
        main = QVBoxLayout()
        main.setContentsMargins(24, 18, 24, 18)
        header = QHBoxLayout()
        self.title = QLabel("Сканер")
        self.title.setObjectName("title")
        header.addWidget(self.title)
        header.addStretch()
        self.connection_badge = QLabel("Не подключено")
        self.connection_badge.setObjectName("badge")
        header.addWidget(self.connection_badge)
        main.addLayout(header)
        banner = QLabel("SignalScope · Пороговый детектор. Классификация АМ, ЧМ и шума при уточнении. Тип источника указан во вкладке «Подключение».")
        banner.setObjectName("banner")
        banner.setWordWrap(True)
        main.addWidget(banner)
        self.stack = QStackedWidget()
        main.addWidget(self.stack, 1)
        layout.addLayout(main, 1)
        self.setCentralWidget(shell)
        self._build_scanner()
        self._build_capture()
        self._build_connection()
        help_page = QTextBrowser()
        help_page.setHtml(HELP_HTML)
        self.stack.addWidget(help_page)
        self.statusBar().setSizeGripEnabled(True)
        enter = QAction(self)
        enter.setShortcut(QKeySequence("Return"))
        enter.triggered.connect(self._enter_capture)
        self.addAction(enter)

    def _navigate(self, index: int) -> None:
        if index >= 0:
            self.stack.setCurrentIndex(index)
            self.title.setText(["Сканер", "Сбор данных", "Подключение", "Справка"][index])

    def _build_scanner(self) -> None:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 12, 0, 0)
        split = QSplitter()
        left = QWidget()
        left.setMinimumWidth(250)
        left.setMaximumWidth(310)
        form_column = QVBoxLayout(left)
        form_column.setContentsMargins(0, 0, 12, 0)
        self.scan_form = QWidget()
        controls = QVBoxLayout(self.scan_form)
        controls.setContentsMargins(0, 0, 0, 0)
        group = QGroupBox("Диапазон и проход")
        form = QFormLayout(group)
        self.scan_start = number(87, .001, 6000, 4)
        self.scan_stop = number(108, .001, 6000, 4)
        self.profile = combo([*PRESETS, "Ручной"])
        self.profile.setCurrentText("Сбалансированный")
        self.profile.currentTextChanged.connect(self._preset_changed)
        self.scan_mode = combo(["Однократный", "Непрерывный"])
        self.scan_depth = combo(["Обзор с уточнением", "Только обзор"])
        self.classification = combo(["Быстрый · 1 развёртка", "Основной · 7 развёрток"])
        self.scan_depth.currentIndexChanged.connect(self.classification.setDisabled)
        self.scan_sweeps = QSpinBox()
        self.scan_sweeps.setRange(1, 50)
        self.scan_sweeps.setValue(2)
        self.overlap = number(15, 0, 50, 0)
        self.threshold = number(8, 1, 40, 1)
        for label, widget in [("От, МГц", self.scan_start), ("До, МГц", self.scan_stop),
                              ("Профиль", self.profile), ("Проход", self.scan_mode),
                              ("Обработка", self.scan_depth), ("Классификация", self.classification), ("Срезов обзора", self.scan_sweeps),
                              ("Перекрытие, %", self.overlap), ("Порог, дБ", self.threshold)]:
            form.addRow(label, widget)
        self.threshold.setToolTip("Минимальное превышение над оценкой шумового фона, дБ")
        controls.addWidget(group)
        self.scan_settings = SettingsEditor("Параметры обзора", center=False)
        controls.addWidget(self.scan_settings)
        self.log_checkbox = QCheckBox("Сохранять журнал находок")
        self.log_checkbox.setChecked(True)
        self.raw_checkbox = QCheckBox("Сохранять все спектры сканера")
        controls.addWidget(self.log_checkbox)
        controls.addWidget(self.raw_checkbox)
        form_column.addWidget(self.scan_form)
        self.scan_start_button = button("Начать сканирование", self.start_scan, "primary")
        self.scan_start_button.setObjectName("primary")
        self.scan_start_button.setAccessibleName("Начать сканирование")
        form_column.addWidget(self.scan_start_button)
        actions = QHBoxLayout()
        self.pause_button = button("Пауза", self.toggle_pause)
        self.stop_button = button("Стоп", self.stop, "danger")
        actions.addWidget(self.pause_button)
        actions.addWidget(self.stop_button)
        form_column.addLayout(actions)
        form_column.addStretch()
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left)
        left_scroll.setMinimumWidth(285)
        left_scroll.setMaximumWidth(335)
        split.addWidget(left_scroll)
        right = QWidget()
        results = QVBoxLayout(right)
        results.setContentsMargins(8, 0, 0, 0)
        self.metrics = QLabel("Проход —    ·    Обнаружений 0    ·    Время 0,0 с")
        self.metrics.setObjectName("subtitle")
        results.addWidget(self.metrics)
        self.progress = QProgressBar()
        self.progress.setFormat("Диапазон ещё не проверен")
        results.addWidget(self.progress)
        self.panorama = SpectrumPlot("Панорама · текущий проход · max по частотным ячейкам", panorama=True)
        results.addWidget(self.panorama, 2)
        self.detail_plot = SpectrumPlot("Последнее отдельное измерение · среднее срезов")
        results.addWidget(self.detail_plot, 1)
        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(["Центр, МГц", "Полоса, кГц", "Пик, ед. прибора", "Над фоном, дБ",
                                             "Модуляция", "Состояние", "Уточнение", "Последнее, UTC", "Наблюдений"])
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.cellDoubleClicked.connect(self.inspect_finding)
        self.table.setMinimumHeight(180)
        results.addWidget(self.table, 2)
        bottom = QHBoxLayout()
        self.export_button = button("Экспорт находок CSV", self.export_findings)
        bottom.addWidget(self.export_button)
        bottom.addStretch()
        self.open_run_button = button("Открыть папку записи", self.open_last_run)
        bottom.addWidget(self.open_run_button)
        results.addLayout(bottom)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        outer.addWidget(split)
        self.stack.addWidget(page)
        self._preset_changed(self.profile.currentText())

    def _build_capture(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 12, 0, 0)
        hint = QLabel("Выберите число сигналов → «Сделать снимок». Настройки считаются с прибора; серия выполнится автоматически.")
        hint.setObjectName("subtitle")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        columns = QHBoxLayout()
        self.capture_form = QWidget()
        form_layout = QHBoxLayout(self.capture_form)
        form_layout.setContentsMargins(0, 0, 0, 0)
        first = QVBoxLayout()
        self.capture_settings = SettingsEditor("Считанные настройки прибора")
        first.addWidget(self.capture_settings)
        source_group = QGroupBox("Источник настроек")
        source_form = QFormLayout(source_group)
        self.settings_source = combo(["Настройки из приложения", "Считать ручные настройки"])
        self.settings_source.currentIndexChanged.connect(self._capture_source_changed)
        self.settings_source.setCurrentIndex(1)
        self.settings_source.hide()
        source_form.addRow(QLabel("Снимок всегда использует текущие настройки прибора."))
        self.read_panel_button = button("Считать с панели", self.read_front_panel)
        source_form.addRow(self.read_panel_button)
        self.capture_trace_info = QLabel("Параметры Trace 1 появятся после чтения панели.")
        self.capture_trace_info.setWordWrap(True)
        source_form.addRow(self.capture_trace_info)
        source_form.addRow(QLabel("Каждый срез — новая последовательность sweep.\nMax/Min Hold и Average начинают накопление заново."))
        first.addWidget(source_group)
        first.addStretch()
        form_layout.addLayout(first, 1)
        second = QVBoxLayout()
        labels = QGroupBox("Описание наблюдения")
        form = QFormLayout(labels)
        self.scene = combo(["Эфир", "Нет сигнала", "Один AM", "Один FM", "Несколько", "Слабый сигнал"])
        self.count_combo = combo(["Один", "Несколько", "Нет сигнала", "Сомнительно один", "Сомнительно несколько"])
        self.mod_combo = combo(["FM", "AM", "Смешанные", "Неизвестно", "Нет"])
        self.mod_combo.setCurrentText("Неизвестно")
        self.mod_combo.hide()  # modulation is outside the collection task
        self.observation_id = QLineEdit()
        self.observation_id.setPlaceholderText("Общая группа для повторов одного наблюдения")
        self.observation_id.setToolTip("Одинаковое значение связывает повторные записи и варианты при разделении данных. Пустое — новая группа для каждой записи.")
        self.source_name = QLineEdit()
        self.source_name.setPlaceholderText("Например, лабораторный генератор")
        self.comment = QPlainTextEdit()
        self.comment.setPlaceholderText("Условия, замечания, предполагаемые частоты…")
        self.comment.setMaximumHeight(110)
        self.capture_sweeps = QSpinBox()
        self.capture_sweeps.setRange(1, 50)
        self.capture_sweeps.setValue(1)
        self.variants_checkbox = QCheckBox("После снимка собрать случайную серию")
        self.variants_checkbox.setChecked(True)
        self.variants_checkbox.setToolTip("Исходный снимок → случайные варианты → повтор исходного. Центр, span и число точек сохраняются.")
        self.variant_count = QSpinBox()
        self.variant_count.setRange(10, 20)
        self.variant_count.setValue(15)
        self.count_combo.currentTextChanged.connect(self._count_changed)
        form.addRow("Сцена имитации", self.scene)
        form.addRow("Сигналов в окне", self.count_combo)
        form.addRow(self.variants_checkbox)
        form.addRow("Случайных вариантов", self.variant_count)
        for label, widget in [
                              ("Название источника", self.source_name), ("Группа наблюдения", self.observation_id),
                              ("Срезов / запись", self.capture_sweeps), ("Комментарий", self.comment)]:
            form.addRow(label, widget)
        ranges = QLabel("RBW: 0,002–0,2 × span; VBW: 0,1–1 × RBW.\nTrace: Clear/Write, Average, Max Hold. Детектор: POS, RMS, Sample.\nSweep Count сохраняется; при 0 для Average/Max Hold — 5.\nВарианты получают метку наблюдения с признаком проверки.")
        ranges.setWordWrap(True)
        form.addRow(ranges)
        second.addWidget(labels)
        second.addStretch()
        form_layout.addLayout(second, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.capture_form)
        columns.addWidget(scroll)
        layout.addLayout(columns, 1)
        self.capture_plot = SpectrumPlot("Последняя запись / предпросмотр")
        self.capture_plot.setMinimumHeight(240)
        layout.addWidget(self.capture_plot, 1)
        row = QHBoxLayout()
        self.preview_button = button("Предпросмотр", self.preview_capture)
        self.record_button = button("Сделать снимок · Enter", self.start_capture, "primary")
        self.capture_stop = button("Остановить запись", self.stop, "danger")
        row.addWidget(self.preview_button)
        row.addWidget(self.record_button)
        row.addWidget(self.capture_stop)
        row.addStretch()
        self.capture_status = QLabel("Данные ещё не записаны")
        row.addWidget(self.capture_status)
        layout.addLayout(row)
        self.stack.addWidget(page)

    def _build_connection(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 12, 0, 0)
        self.connection_form = QWidget()
        columns = QHBoxLayout(self.connection_form)
        columns.setContentsMargins(0, 0, 0, 0)
        group = QGroupBox("Соединение")
        form = QFormLayout(group)
        self.transport = combo(["Имитация", "TCP-сокет", "PyVISA", "PyVISA (системная VISA)"])
        self.address = QLineEdit()
        self.address.setPlaceholderText("IP прибора или TCPIP0::<IP>::inst0::INSTR")
        self.seed = QSpinBox()
        self.seed.setRange(0, 2_000_000_000)
        self.seed.setValue(42)
        self.sim_speed = number(1, .1, 5, 1)
        self.connect_button = button("Подключить", self.toggle_connection, "primary")
        form.addRow("Способ", self.transport)
        form.addRow("IP / VISA-адрес", self.address)
        form.addRow("Seed имитации", self.seed)
        form.addRow("Задержка имитации, ×", self.sim_speed)
        form.addRow(self.connect_button)
        warning = QLabel("PyVISA: R&S FSL-3 по LAN/VXI-11. Подключение только читает настройки.\nTCP: только диагностика, возврат ручного управления не проверен.\nРезультат аппаратной проверки отображается в сведениях ниже.")
        warning.setWordWrap(True)
        form.addRow(warning)
        self.connection_info = QLabel("Прибор не подключён")
        self.connection_info.setWordWrap(True)
        form.addRow(self.connection_info)
        columns.addWidget(group, 1)
        self.panel = SettingsEditor("Передняя панель · имитация")
        self.panel.layout().addRow(button("Применить на панели", self.apply_front_panel))
        panel_hint = QLabel("Отдельное состояние прибора.\nПри сборе выберите «Считать ручные настройки».")
        panel_hint.setWordWrap(True)
        self.panel.layout().addRow(panel_hint)
        columns.addWidget(self.panel, 1)
        layout.addWidget(self.connection_form)
        self.saving_form = QGroupBox("Хранение данных")
        save = QFormLayout(self.saving_form)
        self.data_path = QLineEdit(str(self.home / "data"))
        path_row = QHBoxLayout()
        path_row.addWidget(self.data_path, 1)
        path_row.addWidget(button("Выбрать…", self.choose_folder))
        save.addRow("Рабочая папка", path_row)
        label = QLabel("Имитация: simulation/. Прибор: measurements/. Метки задаёт оператор.")
        label.setWordWrap(True)
        save.addRow(label)
        layout.addWidget(self.saving_form)
        layout.addStretch()
        self.stack.addWidget(page)

    def _preset_changed(self, name: str) -> None:
        if not hasattr(self, "scan_settings"):
            return
        if name in PRESETS:
            span, rbw, vbw, sweeps = PRESETS[name]
            self.scan_settings.load(replace(self.scan_settings.settings(), span_hz=span,
                                             rbw_hz=rbw, vbw_hz=vbw))
            self.scan_sweeps.setValue(sweeps)
        self.scan_settings.setEnabled(name == "Ручной")

    def _count_changed(self, value: str) -> None:
        self.mod_combo.setEnabled(value != "Нет сигнала")
        if value == "Нет сигнала":
            self.mod_combo.setCurrentText("Нет")
        elif self.mod_combo.currentText() == "Нет":
            self.mod_combo.setCurrentText("Неизвестно")

    def _set_busy(self, busy: bool) -> None:
        self.scan_form.setEnabled(not busy)
        self.capture_form.setEnabled(not busy)
        self.connection_form.setEnabled(not busy)
        self.saving_form.setEnabled(not busy)
        connected = self.analyzer is not None and not getattr(self.analyzer, "diagnostics_only", False) and not getattr(self.analyzer, "acquisition_blocked", False)
        self.panel.setEnabled(not busy and isinstance(self.analyzer, Simulator))
        self.read_panel_button.setEnabled(self.analyzer is not None and not busy)
        self.scene.setEnabled(not busy and isinstance(self.analyzer, Simulator))
        self.scan_start_button.setEnabled(connected and not busy)
        self.preview_button.setEnabled(connected and not busy)
        self.record_button.setEnabled(connected and not busy)
        self.stop_button.setEnabled(busy)
        self.capture_stop.setEnabled(busy)
        self.pause_button.setEnabled(busy and self.job_kind == "scan")
        self.open_run_button.setEnabled(bool(self.last_run))
        self.export_button.setEnabled(bool(self.rows) and not busy)
        if not busy:
            self.scan_settings.setEnabled(self.profile.currentText() == "Ручной")
            self._count_changed(self.count_combo.currentText())
            self._capture_source_changed()

    def _capture_source_changed(self, *args):
        self.capture_settings.setEnabled(False)

    def _guard(self) -> bool:
        if self.job is not None:
            self.statusBar().showMessage("Дождитесь завершения текущей операции или нажмите «Стоп».")
            return False
        if self.analyzer is None:
            self.nav.setCurrentRow(2)
            self.statusBar().showMessage("Сначала подключите имитацию или анализатор.")
            return False
        return True

    def _error(self, message: str) -> None:
        self.statusBar().showMessage(message)
        QMessageBox.warning(self, "SignalScope", message)

    def toggle_connection(self) -> None:
        if self.job:
            return
        if self.analyzer:
            self.analyzer.release_local()
            self.analyzer = None
            self.connection_badge.setText("Не подключено")
            self.connection_info.setText("Прибор отключён")
            self.connect_button.setText("Подключить")
        else:
            try:
                analyzer = create_analyzer(self.transport.currentText(), self.address.text(), self.seed.value())
                if isinstance(analyzer, Simulator):
                    analyzer.delay_scale = self.sim_speed.value()
                    settings = analyzer.read_settings()
                    info = "Симулятор. Сеть не используется."
                else:
                    identity, settings, metadata = analyzer.probe()
                    info = identity + f"\nЦентр {settings.center_hz/1e6:.6f} МГц; span {settings.span_hz/1e3:g} кГц"
                    info += f"\nRBW/VBW {settings.rbw_hz:g}/{settings.vbw_hz:g} Гц; {settings.points} точек; {settings.detector}"
                    mode = metadata["trace_mode"]
                    trace_name = {"WRIT": "Clear/Write", "WRITE": "Clear/Write", "AVER": "Average", "AVERAGE": "Average",
                                  "MAXH": "Max Hold", "MAXHOLD": "Max Hold", "MINH": "Min Hold", "VIEW": "View"}.get(mode.upper(), mode)
                    info += f"\nTrace 1: {trace_name}; Sweep Count: {metadata['sweep_count']}"
                    info += f"\nЗапуск: {'непрерывный' if metadata['continuous'] in ('1', 'ON') else 'однократный'}; trigger: {metadata['trigger']}; единицы: {metadata['amplitude_unit']}"
                    info += "\nНастройки считаны без reset. Частоты и единицы проверяются по файлу трассы при сборе."
                    try:
                        report = json.loads((self.repository().root / "fsl_check_latest.json").read_text(encoding="utf-8"))
                        if report.get("identity") == identity and report.get("hardware_acquisition_verified"):
                            info += f"\nБазовый сбор проверен на этом FSL: {report['sweeps']} × {report['points']} точек; {report['amplitude_unit']}. Настройки восстановлены."
                    except (OSError, ValueError, KeyError):
                        pass
                    if analyzer.diagnostics_only:
                        info += "\nТолько чтение TCP. Возврат Local не проверен. Для записи выберите PyVISA."
                self.analyzer = analyzer
                self.panel.load(settings)
                self.capture_settings.load(settings)
                self.capture_trace_info.setText("Имитация" if analyzer.simulated else
                    f"Trace 1: {metadata['trace_mode']}; Sweep Count: {metadata['sweep_count']}; {metadata['amplitude_unit']}\n"
                    f"Trigger: {metadata['trigger']}; Continuous: {metadata['continuous']}")
                if not analyzer.simulated:
                    self.settings_source.setCurrentIndex(1)
                self.connection_badge.setText("Имитация подключена" if analyzer.simulated else "FSL подключён")
                self.connection_info.setText(info)
                self.connect_button.setText("Отключить")
                self.statusBar().showMessage("Подключение проверено. См. сведения о приборе.")
            except Exception as error:
                self._error(str(error))
        self._set_busy(False)

    def repository(self) -> Repository:
        text = self.data_path.text().strip()
        if not text:
            raise ValueError("Укажите папку сохранения данных.")
        return Repository(Path(text))

    def _start_job(self, kind: str, operation) -> None:
        if self.job:
            return
        self.job_kind = kind
        self.job = Job(operation)
        self.job.event.connect(self._event)
        self.job.error.connect(self._error)
        self.job.finished.connect(self._job_finished)
        self._set_busy(True)
        self.job.start()

    def start_scan(self) -> None:
        if not self._guard():
            return
        try:
            plan = ScanPlan(self.scan_start.value() * 1e6, self.scan_stop.value() * 1e6,
                            self.scan_settings.settings(), self.overlap.value() / 100,
                            self.scan_mode.currentText() == "Непрерывный",
                            self.scan_depth.currentText() == "Обзор с уточнением",
                            self.scan_sweeps.value(), self.threshold.value(),
                            "fast" if self.classification.currentIndex() == 0 else "quality")
            plan.windows()
            repository = self.repository()
            save_log, save_raw = self.log_checkbox.isChecked(), self.raw_checkbox.isChecked()
            if isinstance(self.analyzer, Simulator):
                self.analyzer.set_scene("Эфир")
            analyzer = self.analyzer
            self.session_start, self.session_stop = plan.start_hz, plan.stop_hz
            self.rows = []
            self.last_run = ""
            self._render_rows()
            self.progress.setValue(0)
            self.pause_button.setText("Пауза")
            self.statusBar().showMessage("Сканирование в режиме имитации…" if self.analyzer.simulated else "Сканирование прибором…")
            self._start_job("scan", lambda control, emit: scan(analyzer, plan, repository, control, emit,
                                                                save_log=save_log, save_raw=save_raw))
        except Exception as error:
            self._error(str(error))

    def toggle_pause(self) -> None:
        if self.job and self.job_kind == "scan":
            if self.job.control.paused.is_set():
                self.job.control.paused.clear()
                self.pause_button.setText("Пауза")
                self.statusBar().showMessage("Сканирование продолжено.")
            else:
                self.job.control.paused.set()
                self.pause_button.setText("Продолжить")
                self.statusBar().showMessage("Пауза запрошена. Текущее измерение завершится.")

    def stop(self) -> None:
        if self.job:
            self.job.control.cancel.set()
            self.statusBar().showMessage("Остановка…")

    def _labels(self) -> CaptureLabels:
        choice = self.count_combo.currentText()
        count = {"Один": "one", "Несколько": "multiple", "Нет сигнала": "none",
                 "Сомнительно один": "one", "Сомнительно несколько": "multiple"}[choice]
        certainty = "uncertain" if choice.startswith("Сомнительно") else "certain"
        modulation = "none" if count == "none" else "unknown"
        return CaptureLabels(count, modulation, self.comment.toPlainText(), self.source_name.text(),
                             "manual" if self.settings_source.currentIndex() == 1 else "application",
                             self.observation_id.text().strip(), self.capture_session_id, certainty)

    def start_capture(self) -> None:
        if not self._guard():
            return
        try:
            labels = self._labels()
            labels.validate()
            manual = True
            settings = AnalyzerSettings()
            settings.validate()
            repository = self.repository()
            scene = self.scene.currentText()
            if isinstance(self.analyzer, Simulator):
                self.analyzer.set_scene(scene)
            analyzer = self.analyzer
            sweeps = self.capture_sweeps.value()
            variants = self.variants_checkbox.isChecked()
            self.capture_status.setText("Запись…")
            if variants:
                plan = SeriesPlan(variants=self.variant_count.value())
                self._start_job("capture", lambda control, emit: collect_series(
                    analyzer, repository, control, emit, labels, sweeps, plan, scene))
            else:
                self._start_job("capture", lambda control, emit: collect(analyzer, repository, control, emit,
                    labels, settings, sweeps, read_manual=True, variants=False, scene=scene))
        except Exception as error:
            self._error(str(error))

    def _enter_capture(self) -> None:
        if self.stack.currentIndex() == 1 and not isinstance(QApplication.focusWidget(), QPlainTextEdit):
            self.start_capture()

    def preview_capture(self) -> None:
        if not self._guard():
            return
        try:
            settings = AnalyzerSettings() if self.settings_source.currentIndex() == 1 else self.capture_settings.settings()
            settings.validate()
            if isinstance(self.analyzer, Simulator):
                self.analyzer.set_scene(self.scene.currentText())
            self._preview(settings, "capture")
        except Exception as error:
            self._error(str(error))

    def _preview(self, settings: AnalyzerSettings, target: str) -> None:
        analyzer = self.analyzer
        manual = target == "capture" and self.settings_source.currentIndex() == 1
        def operation(control, emit):
            initial = analyzer.snapshot_state()
            changed = False
            primary = None
            try:
                if not manual:
                    changed = True
                    analyzer.apply_settings(settings)
                spectrum = analyzer.capture(2, control.cancel)
                emit({"type": "preview", "spectrum": spectrum, "target": target})
            except BaseException as error:
                primary = error
                raise
            finally:
                from .services import finish_device
                summary = {}
                error = finish_device(analyzer, initial, changed, summary)
                if error and primary is None:
                    raise error
        self._start_job("preview", operation)

    def inspect_finding(self, row: int, column: int) -> None:
        if not self._guard() or row >= len(self.rows):
            return
        if isinstance(self.analyzer, Simulator):
            self.analyzer.set_scene("Эфир")
        settings = replace(self.analyzer.read_settings(), center_hz=self.rows[row]["center_hz"],
                           span_hz=250_000, rbw_hz=3000, vbw_hz=3000)
        self._preview(settings, "scanner")

    def apply_front_panel(self) -> None:
        if self._guard():
            try:
                self.analyzer.apply_settings(self.panel.settings())
                self.statusBar().showMessage("Настройки имитированной передней панели применены.")
            except Exception as error:
                self._error(str(error))

    def read_front_panel(self) -> None:
        if self._guard():
            analyzer = self.analyzer
            def operation(control, emit):
                settings = analyzer.read_settings()
                emit({"type": "panel_read", "settings": settings,
                      "metadata": dict(getattr(analyzer, "last_metadata", {}))})
            self.statusBar().showMessage("Чтение ручных настроек…")
            self._start_job("read_settings", operation)

    def _event(self, event: dict) -> None:
        kind = event["type"]
        if kind == "panel_read":
            self.capture_settings.load(event["settings"])
            meta = event["metadata"]
            if meta:
                info = f"Trace 1: {meta['trace_mode']}; Sweep Count: {meta['sweep_count']}; {meta['amplitude_unit']}"
                info += f"\nTrigger: {meta['trigger']}; Continuous: {meta['continuous']}"
                self.capture_trace_info.setText(info)
            self.statusBar().showMessage("Ручные настройки и параметры Trace 1 считаны.")
            return
        if "rows" in event:
            self.rows = event["rows"]
            self._render_rows()
        if event.get("run"):
            self.last_run = event["run"]
            self.open_run_button.setEnabled(True)
        if kind == "pass_start":
            self.panorama.reset_band(self.session_start, self.session_stop)
            self.progress.setValue(0)
            self.progress.setFormat(f"Проход {event['pass']}: 0 / {event['total']} окон")
        if kind in {"overview", "detail"}:
            self.detail_plot.add(event["spectrum"])
            if kind == "overview":
                self.panorama.add(event["spectrum"])
                self.progress.setValue(round(100 * event["index"] / event["total"]))
                self.progress.setFormat(f"Проход {event['pass']}: {event['index']} / {event['total']} окон · %p%")
            self.metrics.setText(f"Проход {event['pass']}    ·    Находок {len(self.rows)}    ·    Время {event['elapsed']:.1f} с    ·    {'Уточнение' if kind == 'detail' else 'Обзор'}")
        if kind == "scan_finished":
            status = {"complete": "завершено", "stopped": "остановлено", "error": "ошибка"}[event["status"]]
            self.statusBar().showMessage(f"Сканирование {status}. Полных проходов: {event['passes']}; измерений: {event['captures']}; время: {event['elapsed']:.1f} с.")
            if event["status"] != "complete":
                self.progress.setFormat(self.progress.format() + " · проход может быть неполным")
        if kind in {"capture", "preview"}:
            target = self.detail_plot if event.get("target") == "scanner" else self.capture_plot
            target.title = "Последнее измерение · " + ("синтетические данные" if event["spectrum"].simulated else "прибор FSL")
            target.add(event["spectrum"])
            if kind == "capture":
                self.capture_status.setText(f"Записано {event['index']} / {event['total']}")
            else:
                self.statusBar().showMessage("Предпросмотр готов. Файлы не сохранялись.")
        if kind == "capture_finished":
            status = {"complete": "Готово", "stopped": "Остановлено", "error": "Ошибка"}[event["status"]]
            self.capture_status.setText(f"{status}: {event['captured']} записей")
            self.statusBar().showMessage(f"{status}. Папка: {event['run']}")

    def _render_rows(self) -> None:
        self.table.setRowCount(len(self.rows))
        for r, row in enumerate(self.rows):
            values = [f"{row['center_hz'] / 1e6:.5f}", f"{(row['stop_hz'] - row['start_hz']) / 1e3:.1f}",
                      f"{row['peak_dbm']:.1f} {row.get('amplitude_unit', 'DBM')}", f"{row['prominence_db']:.1f}", row["modulation"],
                      row["status"], "Да" if row["refined"] else "Нет",
                      row["last_seen"][11:19], str(row["observations"])]
            for c, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(row["origin"] + ("\nСинтетическое измерение" if row["simulated"] else "\nИзмерение прибором"))
                if c == 5:
                    item.setForeground(QColor("#147566" if row["status"] == "Обнаружен" else "#8b7252"))
                self.table.setItem(r, c, item)
        self.export_button.setEnabled(bool(self.rows) and self.job is None)

    def _job_finished(self) -> None:
        job = self.job
        self.job = None
        self.job_kind = ""
        if job:
            job.deleteLater()
        self.pause_button.setText("Пауза")
        self._set_busy(False)
        if isinstance(self.analyzer, Simulator):
            self.panel.load(self.analyzer.read_settings())
        if self.close_requested:
            self.close()

    def choose_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Папка для данных", self.data_path.text())
        if path:
            self.data_path.setText(path)

    def open_last_run(self) -> None:
        if self.last_run and Path(self.last_run).is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.last_run))

    def export_findings(self) -> None:
        if not self.rows:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Экспорт находок", str(self.home / "findings.csv"), "CSV (*.csv)")
        if not path:
            return
        try:
            with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
                writer.writeheader()
                writer.writerows(self.rows)
            self.statusBar().showMessage(f"Находки сохранены: {path}")
        except OSError as error:
            self._error(str(error))

    def _restore_preferences(self) -> None:
        prefs = self.preferences.load_preferences()
        try:
            self.data_path.setText(str(prefs.get("data_path", self.data_path.text())))
            self.address.setText(str(prefs.get("address", self.address.text())))
            self.transport.setCurrentText(str(prefs.get("transport", "Имитация")))
            self.seed.setValue(int(prefs.get("seed", 42)))
            self.sim_speed.setValue(float(prefs.get("delay_scale", 1)))
            self.scan_start.setValue(float(prefs.get("scan_start_mhz", 87)))
            self.scan_stop.setValue(float(prefs.get("scan_stop_mhz", 108)))
            self.profile.setCurrentText(str(prefs.get("profile", "Сбалансированный")))
            self.log_checkbox.setChecked(bool(prefs.get("save_log", True)))
            self.raw_checkbox.setChecked(bool(prefs.get("save_raw", False)))
            if isinstance(prefs.get("scan_settings"), dict):
                settings = AnalyzerSettings(**prefs["scan_settings"])
                settings.validate()
                self.scan_settings.load(settings)
            if isinstance(prefs.get("capture_settings"), dict):
                settings = AnalyzerSettings(**prefs["capture_settings"])
                settings.validate()
                self.capture_settings.load(settings)
            self.scan_mode.setCurrentText(str(prefs.get("scan_mode", "Однократный")))
            self.scan_depth.setCurrentText(str(prefs.get("scan_depth", "Обзор с уточнением")))
            self.overlap.setValue(float(prefs.get("overlap_percent", 15)))
            self.threshold.setValue(float(prefs.get("threshold_db", 8)))
            self.scan_sweeps.setValue(int(prefs.get("scan_sweeps", 2)))
            self.capture_sweeps.setValue(int(prefs.get("capture_sweeps", 1)) if "random_series" in prefs else 1)
            self.variant_count.setValue(int(prefs.get("variant_count", 15)))
            self.variants_checkbox.setChecked(bool(prefs.get("random_series", True)))
        except (TypeError, ValueError):
            self.statusBar().showMessage("Часть сохранённых настроек повреждена; использованы значения по умолчанию.")

    def closeEvent(self, event) -> None:
        if self.job:
            self.close_requested = True
            self.stop()
            event.ignore()
            return
        try:
            self.preferences.save_preferences({
                "data_path": self.data_path.text(), "address": self.address.text(),
                "transport": self.transport.currentText(), "seed": self.seed.value(),
                "delay_scale": self.sim_speed.value(), "scan_start_mhz": self.scan_start.value(),
                "scan_stop_mhz": self.scan_stop.value(), "profile": self.profile.currentText(),
                "save_log": self.log_checkbox.isChecked(), "save_raw": self.raw_checkbox.isChecked(),
                "scan_settings": asdict(self.scan_settings.settings()),
                "capture_settings": asdict(self.capture_settings.settings()),
                "scan_mode": self.scan_mode.currentText(), "scan_depth": self.scan_depth.currentText(),
                "overlap_percent": self.overlap.value(), "threshold_db": self.threshold.value(),
                "scan_sweeps": self.scan_sweeps.value(), "capture_sweeps": self.capture_sweeps.value(),
                "variant_count": self.variant_count.value(), "random_series": self.variants_checkbox.isChecked(),
            })
        except OSError as error:
            self.statusBar().showMessage(f"Не удалось сохранить настройки: {error}")
        if self.analyzer:
            self.analyzer.release_local()
        event.accept()


def main() -> int:
    import sys
    app = QApplication(sys.argv)
    configure_application(app)
    window = MainWindow()
    window.show()
    return app.exec()
