"""R&S FSL adapter. Hardware behavior must be validated on the laboratory instrument.

Command reference: FSL Operating Manual 1300.2519.12-12 (FW 2.00), chapters 5/6.
VXI-11 local control is a VISA interface operation, never an SCPI '&GTL' string.
"""
from __future__ import annotations

from contextlib import contextmanager
from importlib.metadata import version
import math
import re
import socket
import threading
import time
import uuid

from .fsl_export import parse_trace_export

from .domain import AnalyzerSettings, Spectrum, utc_now
from .instruments import Analyzer, Cancelled


class InstrumentError(RuntimeError):
    def __init__(self, message, *, command=None, completed_sweeps=0):
        super().__init__(message)
        self.context = {"command": command, "completed_sweeps": completed_sweeps}


class SocketTransport:
    """Bounded ASCII framing; diagnostics only until local control is verified."""
    def __init__(self, address, timeout_s=5):
        if "::" in address:
            raise ValueError("Для TCP укажите IP:порт; VISA INSTR не является TCP-сокетом.")
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)(?::([0-9]+))?", address.strip())
        if not match:
            raise ValueError("Укажите IP или имя прибора, например 192.168.10.2:5025.")
        host, port = match[1], int(match[2] or 5025)
        if not 1 <= port <= 65535:
            raise ValueError("Недопустимый TCP-порт.")
        self.socket = socket.create_connection((host, port), timeout=timeout_s)
        self.socket.settimeout(timeout_s)
        self.pending = bytearray()
        self.timeout_s = timeout_s

    def write(self, command):
        self.socket.sendall((command + "\n").encode("ascii"))

    def query(self, command):
        self.write(command)
        deadline = time.monotonic() + self.timeout_s
        while b"\n" not in self.pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Таймаут ответа на {command}")
            self.socket.settimeout(remaining)
            chunk = self.socket.recv(65536)
            if not chunk:
                raise ConnectionError(f"Соединение закрыто до конца ответа на {command}")
            self.pending.extend(chunk)
            if len(self.pending) > 4_000_000:
                raise ValueError("Слишком длинный SCPI-ответ.")
        line, _, rest = self.pending.partition(b"\n")
        self.pending = bytearray(rest)
        return line.decode("ascii").strip()

    def local(self):
        # No verified raw TCP go-to-local mechanism in the cited FSL manual.
        pass

    def close(self):
        self.socket.close()


class VisaTransport:
    def __init__(self, address, backend="@py", timeout_s=5):
        try:
            import pyvisa
        except ImportError as error:
            raise RuntimeError("Установите desktop/requirements-lan.txt в окружение приложения.") from error
        address = address.strip()
        if re.fullmatch(r"[A-Za-z0-9_.-]+", address):
            address = f"TCPIP0::{address}::inst0::INSTR"
        if not re.fullmatch(r"TCPIP[0-9]*::[A-Za-z0-9_.-]+::inst0::INSTR", address, re.I):
            raise ValueError("Для FSL нужен VXI-11: TCPIP0::<IP>::inst0::INSTR. VISA SOCKET здесь не поддерживается.")
        self.manager = pyvisa.ResourceManager(backend)
        try:
            self.instrument = self.manager.open_resource(address, open_timeout=int(timeout_s * 1000))
            self.instrument.timeout = int(timeout_s * 1000)
            self.instrument.read_termination = "\n"
            self.instrument.write_termination = "\n"
        except BaseException:
            self.manager.close()
            raise
        self.local_operation = pyvisa.constants.RENLineOperation.deassert_gtl
        self.backend = backend

    def write(self, command):
        self.instrument.write(command)

    def query(self, command):
        return self.instrument.query(command).strip()

    def query_file(self, command):
        # IEEE definite-length block, including embedded newlines in file data.
        return bytes(self.instrument.query_binary_values(command, datatype="B", container=bytes))

    def set_timeout(self, seconds):
        self.instrument.timeout = int(seconds * 1000)

    def local(self):
        import pyvisa
        try:
            self.instrument.control_ren(self.local_operation)
        except pyvisa.errors.VisaIOError as error:
            if not (self.backend == "@py" and version("pyvisa-py") == "0.8.1"
                    and error.error_code == pyvisa.constants.StatusCode.error_nonsupported_operation):
                raise
            # Isolated compatibility bridge for the pinned backend: protocol-level RPC,
            # not an unverified SCPI string or a modification to the installed library.
            session = self.manager.visalib.sessions[self.instrument.session]
            result = session.interface.device_local(session.link, 0, session.lock_timeout,
                                                    int(self.instrument.timeout))
            if result != 0:
                raise InstrumentError(f"VXI-11 device_local отклонён: код {result}")

    def close(self):
        try:
            self.instrument.close()
        finally:
            self.manager.close()


class FslAnalyzer(Analyzer):
    simulated = False
    frequency_limits_hz = (9000, 3e9)

    # Query every essential field; a missing/unsupported response rejects acquisition.
    QUERIES = {
        "instrument_mode": "INST?", "start_hz": "FREQ:STAR?", "stop_hz": "FREQ:STOP?",
        "center_hz": "FREQ:CENT?", "span_hz": "FREQ:SPAN?", "rbw_hz": "BAND:RES?",
        "vbw_hz": "BAND:VID?", "points": "SWE:POIN?", "detector": "DET?",
        "rbw_auto": "BAND:RES:AUTO?", "vbw_auto": "BAND:VID:AUTO?",
        "detector_auto": "DET:AUTO?", "trace_mode": "DISP:TRAC1:MODE?",
        "frequency_axis": "DISP:TRAC1:X:SPAC?", "amplitude_unit": "UNIT:POW?",
        "sweep_time_s": "SWE:TIME?", "sweep_time_auto": "SWE:TIME:AUTO?",
        "sweep_count": "SWE:COUN?", "continuous": "INIT:CONT?", "trigger": "TRIG:SOUR?",
        "reference_level": "DISP:TRAC1:Y:RLEV?", "attenuation_db": "INP:ATT?",
        "attenuation_auto": "INP:ATT:AUTO?", "data_format": "FORM?",
        "gated": "SWE:EGAT?", "signal_track": "CALC:MARK:FUNC:STR?",
        "sweep_mode": "SWE:MODE?",
        "trace_math": "CALC:MATH:STAT?", "statistics_apd": "CALC:STAT:APD?",
        "statistics_ccdf": "CALC:STAT:CCDF?", "reference_offset_db": "DISP:TRAC1:Y:RLEV:OFFS?",
    }
    def __init__(self, address, *, backend="@py", transport_factory=None, diagnostics_only=False):
        self.address = address
        self.backend = backend
        self.diagnostics_only = diagnostics_only
        self.factory = transport_factory or (lambda: VisaTransport(address, backend))
        self.real_transport = transport_factory is None
        self.identity = ""
        self.last_metadata = {}
        self.last_command = None
        self._settings_modified = False
        self._series_settings_modified = False
        self.acquisition_blocked = False
        self.preexisting_errors = []

    @contextmanager
    def session(self):
        transport = self.factory()
        primary = None
        try:
            yield transport
        except BaseException as error:
            primary = error
            raise
        finally:
            errors = []
            for operation in (transport.local, transport.close):
                try:
                    operation()
                except Exception as error:
                    errors.append(str(error))
            if errors:
                message = "Не подтверждён возврат ручного управления / закрытие соединения: " + "; ".join(errors)
                if primary is not None:
                    primary.add_note(message)
                else:
                    raise InstrumentError(message)

    def query(self, transport, command):
        self.last_command = command
        if getattr(transport, "desynchronized", False):
            raise InstrumentError("Соединение рассинхронизировано; требуется новый сеанс", command=command)
        try:
            result = transport.query(command).strip().strip('"')
            if not result:
                raise ValueError("пустой ответ")
            return result
        except Exception as error:
            transport.desynchronized = True
            raise InstrumentError(f"Ошибка чтения {command}: {error}", command=command) from error

    def checked_write(self, transport, command):
        self.last_command = command
        if getattr(transport, "desynchronized", False):
            raise InstrumentError("Запись в рассинхронизированное соединение запрещена", command=command)
        transport.write(command)
        response = self.query(transport, "SYST:ERR?")
        if int(response.split(",", 1)[0]) != 0:
            raise InstrumentError(f"Прибор отклонил {command}: {response}", command=command)

    def read_error_queue(self, transport):
        errors = []
        for _ in range(32):
            response = self.query(transport, "SYST:ERR?")
            if int(response.split(",", 1)[0]) == 0:
                return errors
            errors.append(response)
        raise InstrumentError("Очередь ошибок FSL не заканчивается после 32 ответов")

    def metadata(self, transport):
        raw = {name: self.query(transport, query) for name, query in self.QUERIES.items()
               if name != "frequency_axis"}
        raw["frequency_axis"] = "PENDING_EXPORT"
        self.last_metadata = raw
        return raw

    BASIC_FIELDS = ("instrument_mode", "start_hz", "stop_hz", "center_hz", "span_hz",
                    "rbw_hz", "vbw_hz", "points", "detector", "trace_mode",
                    "sweep_count", "continuous", "trigger", "amplitude_unit")

    def read_basic(self, transport):
        raw = {name: self.query(transport, self.QUERIES[name]) for name in self.BASIC_FIELDS}
        raw["frequency_axis"] = "PENDING_EXPORT"
        self.last_metadata = raw
        return raw

    def export_trace(self, transport):
        if not hasattr(transport, "query_file"):
            raise InstrumentError("Нужен транспорт PyVISA с поддержкой чтения файла трассы.")
        filename = "C:\\R_S\\Instr\\user\\ss_" + uuid.uuid4().hex + ".ASC"
        primary, created = None, False
        try:
            self.checked_write(transport, f"MMEM:STOR:TRAC 1,'{filename}'")
            created = True
            if self.query(transport, "*OPC?") != "1":
                raise InstrumentError("Экспорт трассы не завершён")
            self.last_command = f"MMEM:DATA? '{filename}'"
            try:
                data = transport.query_file(self.last_command)
            except Exception as error:
                transport.desynchronized = True
                raise InstrumentError(f"Ошибка чтения файла трассы: {error}", command=self.last_command) from error
            return parse_trace_export(data)
        except BaseException as error:
            primary = error
            raise
        finally:
            if created:
                try:
                    if getattr(transport, "desynchronized", False):
                        with self.session() as recovery:
                            self.checked_write(recovery, f"MMEM:DEL '{filename}'")
                    else:
                        self.checked_write(transport, f"MMEM:DEL '{filename}'")
                except Exception as error:
                    if primary is not None:
                        primary.add_note(f"Временный файл прибора {filename}: {error}")
                    else:
                        raise InstrumentError(f"Не удалён временный файл прибора {filename}: {error}") from error

    @staticmethod
    def settings_from(raw):
        detector = raw["detector"].upper()
        detector = {"SAMP": "SAMPLE", "POSITIVE": "POS", "NEGATIVE": "NEG",
                    "APEAK": "APE", "AVERAGE": "AVER", "QPEAK": "QPE"}.get(detector, detector)
        points = float(raw["points"])
        if not points.is_integer():
            raise ValueError("Прибор вернул нецелое количество точек.")
        settings = AnalyzerSettings(*(float(raw[k]) for k in ("center_hz", "span_hz", "rbw_hz", "vbw_hz")),
                                    int(points), detector)
        settings.validate()
        return settings

    def probe(self):
        with self.session() as transport:
            self.identity = self.query(transport, "*IDN?")
            if not re.search(r"(?:^|,)FSL-?3(?:,|$)", self.identity, re.I):
                raise ValueError(f"Ожидался R&S FSL-3, получено: {self.identity}")
            raw = self.read_basic(transport)
        self.acquisition_blocked = False
        self.last_metadata = raw
        return self.identity, self.settings_from(raw), raw

    def read_settings(self):
        with self.session() as transport:
            return self.settings_from(self.read_basic(transport))

    def snapshot_state(self):
        with self.session() as transport:
            return self.metadata(transport)

    def apply_settings(self, settings):
        if self.diagnostics_only:
            raise ValueError("Прямой TCP доступен только для чтения настроек; для записи выберите PyVISA.")
        settings.validate()
        lower, upper = self.frequency_limits_hz
        if settings.center_hz - settings.span_hz / 2 < lower or settings.center_hz + settings.span_hz / 2 > upper:
            raise ValueError("Окно должно лежать в диапазоне FSL-3: 9 кГц–3 ГГц.")
        with self.session() as transport:
            self.validate_capture(self.metadata(transport))
            self.preexisting_errors.extend(self.read_error_queue(transport))
            self._settings_modified = True
            self.checked_write(transport, "FREQ:SPAN 1000")
            for command, value in [("FREQ:CENT", settings.center_hz), ("FREQ:SPAN", settings.span_hz),
                                   ("BAND:RES", settings.rbw_hz), ("BAND:VID", settings.vbw_hz),
                                   ("SWE:POIN", settings.points), ("DET", "SAMP" if settings.detector == "SAMPLE" else settings.detector)]:
                self.checked_write(transport, f"{command} {value}")
            return self.settings_from(self.metadata(transport))

    def restore_state(self, state):
        if self.diagnostics_only or not self._settings_modified:
            return
        with self.session() as transport:
            errors = []
            commands = [("FREQ:SPAN", "1000"), ("FREQ:CENT", state["center_hz"]),
                        ("FREQ:SPAN", state["span_hz"])]
            for field, command in [("rbw_hz", "BAND:RES"), ("vbw_hz", "BAND:VID"),
                                   ("points", "SWE:POIN"), ("detector", "DET"),
                                   ("sweep_time_s", "SWE:TIME"), ("reference_level", "DISP:TRAC1:Y:RLEV"),
                                   ("attenuation_db", "INP:ATT"), ("rbw_auto", "BAND:RES:AUTO"),
                                   ("vbw_auto", "BAND:VID:AUTO"), ("detector_auto", "DET:AUTO"),
                                   ("sweep_time_auto", "SWE:TIME:AUTO"), ("attenuation_auto", "INP:ATT:AUTO")]:
                commands.append((command, state[field]))
            for command, value in commands:
                try:
                    self.checked_write(transport, f"{command} {value}")
                except Exception as error:
                    errors.append(str(error))
            restored = self.metadata(transport)
            for field in ("center_hz", "span_hz", "points", "rbw_auto", "vbw_auto", "detector_auto",
                          "sweep_time_auto", "attenuation_auto", "reference_level"):
                if not math.isclose(float(state[field]), float(restored[field]), rel_tol=1e-8, abs_tol=1e-6):
                    errors.append(f"Не восстановлено {field}: {state[field]} → {restored[field]}")
            for field, auto in [("rbw_hz", "rbw_auto"), ("vbw_hz", "vbw_auto"),
                                ("sweep_time_s", "sweep_time_auto"), ("attenuation_db", "attenuation_auto")]:
                if float(state[auto]) == 0 and not math.isclose(float(state[field]), float(restored[field]), rel_tol=1e-6):
                    errors.append(f"Не восстановлено {field}")
            if float(state["detector_auto"]) == 0 and state["detector"] != restored["detector"]:
                errors.append("Не восстановлен detector")
            if errors:
                raise InstrumentError("; ".join(errors))
            self._settings_modified = False

    def series_snapshot(self):
        with self.session() as transport:
            raw = self.metadata(transport)
            raw["filter_type"] = self.query(transport, "BAND:TYPE?")
            raw["averaging_mode"] = self.query(transport, "CALC:MATH:MODE?")
            self.validate_capture(raw)
            return raw

    def apply_variant(self, variant, initial):
        settings = AnalyzerSettings(**variant["settings"])
        settings.validate()
        if any(getattr(settings, key) != getattr(initial, key) for key in ("center_hz", "span_hz", "points")):
            raise ValueError("Случайная серия не меняет центр, span и число точек.")
        if variant["trace_mode"] not in ("WRIT", "AVER", "MAXH"):
            raise ValueError("Недопустимый режим Trace 1 для серии.")
        if not 0 <= variant["sweep_count"] <= 32767:
            raise ValueError("Некорректное число накоплений.")
        with self.session() as transport:
            raw = self.metadata(transport)
            self.validate_capture(raw)
            if self.query(transport, "BAND:TYPE?").upper() != "NORM":
                raise ValueError("Для случайного перебора RBW/VBW установите Filter Type Normal. FFT/канальные фильтры автоматически не меняются.")
            self.preexisting_errors.extend(self.read_error_queue(transport))
            # No FREQ, SPAN or SWE:POIN writes, including during restoration.
            self._series_settings_modified = True
            for command, value in [("DISP:TRAC1:MODE", variant["trace_mode"]),
                                   ("DET:AUTO", 0), ("DET", "SAMP" if settings.detector == "SAMPLE" else settings.detector),
                                   ("SWE:TIME:AUTO", 1),
                                   ("BAND:RES:AUTO", 0), ("BAND:VID:AUTO", 0),
                                   ("BAND:RES", settings.rbw_hz), ("BAND:VID", settings.vbw_hz),
                                   ("SWE:COUN", variant["sweep_count"])]:
                self.checked_write(transport, f"{command} {value}")
            actual = self.metadata(transport)
            for key, wanted in [("rbw_hz", settings.rbw_hz), ("vbw_hz", settings.vbw_hz),
                                ("center_hz", initial.center_hz), ("span_hz", initial.span_hz),
                                ("points", initial.points), ("sweep_count", variant["sweep_count"])]:
                if not math.isclose(float(actual[key]), wanted, rel_tol=1e-8, abs_tol=1e-6):
                    raise InstrumentError(f"FSL не установил {key}: запрошено {wanted}, получено {actual[key]}")
            if self.settings_from(actual).detector != settings.detector or actual["trace_mode"] != variant["trace_mode"]:
                raise InstrumentError("FSL не подтвердил режим трассы или детектор")
            self.validate_capture(actual)
            return self.settings_from(actual)

    def restore_series_state(self, state, *, spectral=True):
        with self.session() as transport:
            self.preexisting_errors.extend(self.read_error_queue(transport))
            errors = []
            # QPE forbids programming VBW. Temporarily select POS before bands,
            # then restore the original detector after restoring the trace mode.
            for command in (("DET:AUTO 0", "DET POS") if spectral and self._series_settings_modified else ()):
                try:
                    self.checked_write(transport, command)
                except Exception as error:
                    errors.append(str(error))
            restore_fields = [("rbw_hz", "BAND:RES"), ("vbw_hz", "BAND:VID"),
                                   ("trace_mode", "DISP:TRAC1:MODE"),
                                   ("detector", "DET"), ("sweep_count", "SWE:COUN"),
                                   ("sweep_time_s", "SWE:TIME"),
                                   ("rbw_auto", "BAND:RES:AUTO"), ("vbw_auto", "BAND:VID:AUTO"),
                                   ("detector_auto", "DET:AUTO"), ("sweep_time_auto", "SWE:TIME:AUTO")]
            for field, command in (restore_fields if spectral and self._series_settings_modified else []):
                try:
                    self.checked_write(transport, f"{command} {state[field]}")
                except Exception as error:
                    errors.append(str(error))
            for field, command in [("data_format", "FORM"), ("continuous", "INIT:CONT")]:
                try:
                    if self.query(transport, command + "?") != state[field]:
                        self.checked_write(transport, f"{command} {state[field]}")
                except Exception as error:
                    errors.append(str(error))
            actual = self.metadata(transport)
            for key in ("center_hz", "span_hz", "points", "rbw_auto", "vbw_auto", "detector_auto", "sweep_time_auto", "sweep_count",
                        "reference_level", "attenuation_db", "attenuation_auto", "reference_offset_db"):
                if not math.isclose(float(actual[key]), float(state[key]), rel_tol=1e-8, abs_tol=1e-6):
                    errors.append(f"Не восстановлено {key}: {actual[key]} вместо {state[key]}")
            for key, auto in [("rbw_hz", "rbw_auto"), ("vbw_hz", "vbw_auto"), ("sweep_time_s", "sweep_time_auto")]:
                if float(state[auto]) == 0 and not math.isclose(float(actual[key]), float(state[key]), rel_tol=1e-6):
                    errors.append(f"Не восстановлено {key}")
            if actual["trace_mode"] != state["trace_mode"]:
                errors.append("Не восстановлена Trace 1")
            for key in ("data_format", "continuous"):
                if actual[key] != state[key]:
                    errors.append(f"Не восстановлено {key}")
            if float(state["detector_auto"]) == 0 and actual["detector"] != state["detector"]:
                errors.append("Не восстановлен детектор")
            if self.query(transport, "BAND:TYPE?") != state["filter_type"] or self.query(transport, "CALC:MATH:MODE?") != state["averaging_mode"]:
                errors.append("Изменились тип фильтра или способ усреднения")
            if errors:
                raise InstrumentError("; ".join(errors))
            self._series_settings_modified = False

    @staticmethod
    def validate_capture(raw):
        requirements = [(raw["instrument_mode"].upper().startswith("SAN"), "Нужен режим Spectrum Analyzer"),
                        (str(raw["frequency_axis"] or "").upper() in ("LIN", "LINEAR", "PENDING_EXPORT"), "Тип частотной оси не подтверждён"),
                        (raw["amplitude_unit"].upper() in ("DBM", "DBUV", "DBMV", "DBUA", "DBPW"), "Нужны поддерживаемые логарифмические единицы уровня"),
                        (raw["trace_mode"].upper() in ("WRIT", "WRITE", "MAXH", "MAXHOLD", "MINH", "MINHOLD", "AVER", "AVERAGE", "RMS"), "Trace 1 должна обновляться: Clear/Write, Max/Min Hold, Average или RMS"),
                        (raw["trigger"].upper().startswith("IMM"), "Нужен Free Run trigger"),
                        (float(raw["gated"]) == 0, "Gated sweep не поддерживается"),
                        (float(raw["signal_track"]) == 0, "Отключите Signal Track перед записью"),
                        (float(raw["trace_math"]) == 0, "Отключите Trace Math перед записью"),
                        (float(raw["statistics_apd"]) == 0 and float(raw["statistics_ccdf"]) == 0, "Статистические измерения APD/CCDF не поддерживаются"),
                        (0 <= float(raw["sweep_count"]) <= 32767 and float(raw["sweep_count"]).is_integer(), "Некорректный Sweep Count"),
                        (raw["trace_mode"].upper().startswith("WRIT") or float(raw["sweep_count"]) > 0, "Для нового накопления Max/Min Hold или Average установите Sweep Count не меньше 1"),
                        (raw["sweep_mode"].upper().startswith("AUTO"), "Нужен Sweep Mode Auto")]
        for valid, message in requirements:
            if not valid:
                raise ValueError(message + ". Настройки прибора не изменены.")

    def capture(self, sweeps, cancel):
        if self.diagnostics_only:
            raise ValueError("Для сбора и возврата ручного управления используйте PyVISA/VXI-11.")
        if not 1 <= sweeps <= 50:
            raise ValueError("Количество срезов: от 1 до 50.")
        rows, times, durations, exports, autopeak_min = [], [], [], [], []
        axis = None
        with self.session() as transport:
            raw = self.metadata(transport)
            self.validate_capture(raw)
            settings = self.settings_from(raw)
            count = max(1, int(float(raw["sweep_count"])))
            timeout = max(10.0, float(raw["sweep_time_s"]) * count * 2 + 10)
            if timeout > 300:
                raise ValueError("Одна трасса требует более 300 с; уменьшите Sweep Count или Sweep Time на приборе.")
            if hasattr(transport, "set_timeout"):
                transport.set_timeout(timeout)
            start, stop = float(raw["start_hz"]), float(raw["stop_hz"])
            if not (0 < start < stop and math.isclose(stop-start, settings.span_hz, rel_tol=1e-8)
                    and math.isclose((start+stop)/2, settings.center_hz, rel_tol=1e-8)):
                raise ValueError("Несогласованные границы/центр/span прибора.")
            if cancel.is_set():
                raise Cancelled()
            self.preexisting_errors.extend(self.read_error_queue(transport))
            primary, cleanup_errors = None, []
            try:
                self.checked_write(transport, "INIT:CONT OFF")
                self.checked_write(transport, "FORM ASC")
                for _ in range(sweeps):
                    if cancel.is_set():
                        raise Cancelled()
                    started = time.perf_counter()
                    if self.query(transport, "INIT:IMM;*OPC?") != "1":
                        raise InstrumentError("Прибор не подтвердил новый sweep", command=self.last_command)
                    completed = utc_now()
                    sweep_s = time.perf_counter() - started
                    transferred = time.perf_counter()
                    exported = self.export_trace(transport)
                    export_axis, values = exported["frequency_hz"], exported["amplitudes"]
                    if exported["amplitude_unit"].upper() != raw["amplitude_unit"].upper():
                        raise InstrumentError("Единицы в файле трассы не совпали с настройками")
                    if exported["frequency_axis"] != "LIN":
                        raise InstrumentError("Для сбора нужна линейная частотная шкала; файл прибора сообщает " + exported["frequency_axis"])
                    expected = tuple(start + (stop-start)*i/(settings.points-1) for i in range(settings.points))
                    if len(export_axis) != settings.points or any(not math.isclose(x,y,rel_tol=1e-9,abs_tol=.01) for x,y in zip(export_axis,expected)):
                        raise InstrumentError("Частоты в файле не соответствуют start/stop/points прибора")
                    if axis is not None and export_axis != axis:
                        raise InstrumentError("Частотная ось изменилась между трассами")
                    axis = export_axis
                    exports.append(exported["header"])
                    if exported["autopeak_min"]:
                        autopeak_min.append(exported["autopeak_min"])
                    transfer_s = time.perf_counter() - transferred
                    if len(values) != settings.points or not all(math.isfinite(v) and abs(v) < 1e6 for v in values):
                        raise InstrumentError(f"Некорректная трасса: {len(values)} точек, ожидалось {settings.points}")
                    rows.append(values); times.append(completed)
                    durations.append({"sweep_wait_s": sweep_s, "transfer_s": transfer_s})
                after = self.metadata(transport)
                for field in self.QUERIES.keys() - {"continuous", "data_format"}:
                    if after[field] != raw[field]:
                        raise InstrumentError(f"Настройка {field} изменилась во время записи")
            except BaseException as error:
                primary = error
                context = getattr(error, "context", {})
                context.update(completed_sweeps=len(rows), settings=raw, identity=self.identity,
                               last_command=self.last_command)
                error.context = context
                raise
            finally:
                def restore_capture(target):
                    for command, value in [("FORM", raw["data_format"]), ("INIT:CONT", raw["continuous"])]:
                        try:
                            self.checked_write(target, f"{command} {value}")
                            if self.query(target, command + "?") != value:
                                raise InstrumentError(f"Не восстановлено {command}")
                        except Exception as error:
                            cleanup_errors.append(str(error))
                if getattr(transport, "desynchronized", False):
                    try:
                        with self.session() as recovery:
                            restore_capture(recovery)
                    except Exception as error:
                        cleanup_errors.append(str(error))
                else:
                    restore_capture(transport)
                if cleanup_errors:
                    if primary is not None:
                        primary.add_note("Восстановление режима записи: " + "; ".join(cleanup_errors))
                    else:
                        raise InstrumentError("Не восстановлен режим записи: " + "; ".join(cleanup_errors))
            raw["frequency_axis"] = "LIN"
            raw["frequency_axis_source"] = "instrument ASCII trace file: x-Axis and explicit Hz column"
            metadata = {"identity": self.identity, "address": self.address, "backend": self.backend,
                        "hardware_validated": self.real_transport, "actual_settings": raw,
                        "validation_scope": "successful protocol acquisition, explicit frequency/unit/dimension checks and temporary mode restoration; not RF calibration",
                        "sweep_completed_at": times, "timings": durations,
                        "freshness_method": "INIT:IMM;*OPC? before each trace ASCII export",
                        "trace_history_reset": True, "underlying_sweeps_per_trace": count,
                        "trace_semantics": "new sequence; hold/average restarted by INIT (FSL manual 6.143)",
                        "trace_file_headers": exports, "settings_restored": True}
            metadata["preexisting_instrument_errors"] = list(self.preexisting_errors)
            metadata["underlying_sweep_count_source"] = "configured SWE:COUN?; sequence completion verified with OPC; SWE:COUN:CURR? not queried"
            metadata["export_header_discrepancies"] = [
                {"trace_index": index, "field": "sweep_count", "scpi_configured": raw["sweep_count"],
                 "file_header": header["sweepcount"]}
                for index, header in enumerate(exports)
                if "sweepcount" in header and float(header["sweepcount"]) != float(raw["sweep_count"])]
            if autopeak_min:
                if len(autopeak_min) != len(rows):
                    raise InstrumentError("Неполные данные AUTOPEAK между трассами")
                metadata["autopeak_min_sweeps"] = autopeak_min
            return Spectrum(settings, axis, tuple(rows), utc_now(), False, metadata, raw["amplitude_unit"].upper())

    def release_local(self):
        # Each public operation already returns local and closes its VISA session.
        pass
