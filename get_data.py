from __future__ import annotations

import argparse
import csv
import json
import math
import socket
import statistics
import time
from itertools import product
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple



@dataclass(frozen=True)
class AnalyzerProfile:
    name: str
    description: str
    span_khz: float
    rbw_hz: float
    vbw_hz: float
    sweep_points: int = 501
    detector: str = "POS"
    trace_mode: str = "WRIT"


@dataclass(frozen=True)
class FMStation:
    frequency_mhz: float
    name: str


@dataclass(frozen=True)
class AMGeneratorPreset:
    carrier_freq_mhz: float
    level_dbm: float
    modulation_freq_khz: float
    depth_percent: float
    waveform: str = "Sine"


@dataclass(frozen=True)
class PMGeneratorPreset:
    carrier_freq_mhz: float
    level_dbm: float
    modulation_freq_khz: float
    phase_deviation_rad: float
    waveform: str = "Sine"


ANALYZER_PROFILES: Dict[str, AnalyzerProfile] = {
    "baseline": AnalyzerProfile(
        "baseline", "span 250 kHz, RBW/VBW 10 kHz", 250.0, 10_000.0, 10_000.0
    ),
    "high_resolution": AnalyzerProfile(
        "high_resolution", "span 250 kHz, RBW/VBW 3 kHz", 250.0, 3_000.0, 3_000.0
    ),
    "smoothed": AnalyzerProfile(
        "smoothed", "span 250 kHz, RBW 10 kHz, VBW 3 kHz", 250.0, 10_000.0, 3_000.0
    ),
}
DEFAULT_PROFILE_NAMES = ("baseline", "high_resolution", "smoothed")
DEFAULT_NOISE_BANDS = ((80.0, 87.5), (87.5, 108.0), (108.0, 118.0))

FM_STATIONS: Tuple[FMStation, ...] = (
    FMStation(87.5, "Бизнес FM"), FMStation(87.9, "Like FM"),
    FMStation(88.3, "Ретро FM"), FMStation(88.7, "Юмор FM"),
    FMStation(89.1, "Радио Jazz"), FMStation(89.5, "Калина Красная"),
    FMStation(89.9, "Радио Record"), FMStation(90.3, "Авторадио"),
    FMStation(90.8, "Relax FM"), FMStation(91.2, "Радио Спутник"),
    FMStation(91.6, "Радио Культура"), FMStation(92.0, "Москва FM"),
    FMStation(92.4, "Радио Дача"), FMStation(92.8, "Радио РБК"),
    FMStation(93.2, "Studio 21"), FMStation(93.6, "Коммерсантъ FM"),
    FMStation(94.0, "Восток FM"), FMStation(94.4, "Первое Спортивное"),
    FMStation(94.8, "Говорит Москва"), FMStation(95.2, "Rock FM"),
    FMStation(95.6, "Радио Звезда"), FMStation(96.8, "Детское Радио"),
    FMStation(98.8, "Радио Рекорд / Радио Романтика"),
    FMStation(100.1, "Серебряный Дождь"), FMStation(101.2, "DFM"),
    FMStation(101.7, "Наше Радио"), FMStation(102.5, "Comedy Radio"),
    FMStation(103.0, "Радио Шансон"), FMStation(103.7, "Радио Maximum"),
    FMStation(104.2, "Радио Энергия (NRJ)"),
    FMStation(104.7, "Радио 7 на семи холмах"),
    FMStation(105.7, "Русское Радио"), FMStation(106.2, "Европа Плюс"),
    FMStation(107.0, "Радио Маяк"), FMStation(107.4, "Хит FM"),
    FMStation(107.8, "Новое Радио"),
)


FM_DOP_VARIANTS: Tuple[Tuple[float, float, str], ...] = (
    # center offset from the real station frequency [kHz], span [kHz], description
    (-80.0, 250.0, "signal shifted to the right side of the window"),
    (+80.0, 250.0, "signal shifted to the left side of the window"),
    (-140.0, 180.0, "mostly right part of the signal"),
    (+140.0, 180.0, "mostly left part of the signal"),
    (-180.0, 400.0, "signal near right side plus extra spectrum"),
    (+180.0, 400.0, "signal near left side plus extra spectrum"),
)


def build_am_generator_presets() -> Tuple[AMGeneratorPreset, ...]:
    """Balanced AM plan: two generator settings for each of nine carriers."""
    carriers_mhz = (88.1, 89.7, 91.8, 94.6, 97.8, 100.65, 102.3, 104.45, 106.6)
    combinations = (
        (-40.0, 20.0, 20.0), (-40.0, 50.0, 50.0), (-40.0, 80.0, 80.0),
        (-30.0, 20.0, 50.0), (-30.0, 50.0, 80.0), (-30.0, 80.0, 20.0),
        (-20.0, 20.0, 80.0), (-20.0, 50.0, 20.0), (-20.0, 80.0, 50.0),
    )
    presets = []
    for index, carrier in enumerate(carriers_mhz):
        # The cyclic offset repeats every parameter combination on another carrier.
        for combination_index in (index, (index + 4) % len(combinations)):
            level, depth, mod_freq = combinations[combination_index]
            presets.append(AMGeneratorPreset(carrier, level, mod_freq, depth))
    return tuple(presets)


AM_GENERATOR_PRESETS = build_am_generator_presets()


def build_pm_generator_presets(
    deviations_rad: Sequence[float],
) -> Tuple[PMGeneratorPreset, ...]:
    """Use the AM carrier, level and LF plan with explicit PM deviations."""
    if len(deviations_rad) != 3 or any(not math.isfinite(value) or value <= 0
                                        for value in deviations_rad):
        raise ValueError("Specify three positive PM deviations in radians")
    deviation_for_depth = dict(zip((20.0, 50.0, 80.0), deviations_rad))
    return tuple(
        PMGeneratorPreset(
            preset.carrier_freq_mhz,
            preset.level_dbm,
            preset.modulation_freq_khz,
            deviation_for_depth[preset.depth_percent],
            preset.waveform,
        )
        for preset in AM_GENERATOR_PRESETS
    )


@dataclass
class InstrumentConfig:
    resource: str = "TCPIP0::192.168.10.6::inst0::INSTR"
    timeout_ms: int = 30_000
    reset_on_connect: bool = True
    center_freq_mhz: float = 89.7
    span_khz: float = 250.0
    rbw_hz: float = 10_000.0
    vbw_hz: float = 10_000.0
    sweep_points: int = 501
    detector: str = "POS"
    trace_mode: str = "WRIT"
    reference_level_dbm: Optional[float] = None
    attenuation_db: Optional[float] = None
    sweeps_per_sample: int = 50
    samples_per_run: int = 1
    inter_sweep_delay_s: float = 0.05
    settle_after_config_s: float = 1.0
    dataset_root: str = "data"
    class_name: str = "FM"
    source_type: str = "air"
    source_name: str = ""
    analyzer_profile: str = "baseline"
    note: str = ""
    generator_level_dbm: Optional[float] = None
    generator_modulation_freq_khz: Optional[float] = None
    generator_depth_percent: Optional[float] = None
    generator_phase_deviation_rad: Optional[float] = None
    generator_waveform: Optional[str] = None
    prefer_binary: bool = False
    disable_display_updates: bool = False


@dataclass
class SampleMetadata:
    sample_id: str
    class_name: str
    sample_group_name: str
    source_type: str
    source_name: str
    analyzer_profile: str
    note: str
    instrument_model: str
    instrument_resource: str
    started_at: str
    finished_at: str
    center_freq_hz: float
    span_hz: float
    start_freq_hz: float
    stop_freq_hz: float
    rbw_hz: float
    vbw_hz: float
    sweep_points: int
    sweeps_per_sample: int
    detector: str
    trace_mode: str
    reference_level_dbm: Optional[float]
    attenuation_db: Optional[float]
    generator_level_dbm: Optional[float]
    generator_modulation_freq_khz: Optional[float]
    generator_depth_percent: Optional[float]
    generator_phase_deviation_rad: Optional[float]
    generator_waveform: Optional[str]
    csv_path: str


@dataclass
class CollectionSummary:
    collected: int = 0
    skipped_existing: int = 0
    skipped_by_user: int = 0
    failed: int = 0


@dataclass(frozen=True)
class NoiseSurvey:
    center_mhz: float
    band_index: int
    floor_dbm: float
    peak_excess_db: float


def utc_now_str() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_frequency_axis(center_freq_hz: float, span_hz: float, points: int) -> List[float]:
    if points < 2:
        return [center_freq_hz]
    start = center_freq_hz - span_hz / 2.0
    step = span_hz / (points - 1)
    return [start + i * step for i in range(points)]


def parse_ascii_trace(raw: str) -> List[float]:
    normalized = raw.strip().replace(";", ",").replace("\n", ",").replace("\r", ",")
    if not normalized:
        raise ValueError("Empty trace response")
    values = [float(part.strip().replace(",", ".")) for part in normalized.split(",") if part.strip()]
    if not values:
        raise ValueError("No numeric values parsed from ASCII trace")
    return values


def append_metadata_row(metadata_csv: Path, meta: SampleMetadata) -> None:
    ensure_dir(metadata_csv.parent)
    file_exists = metadata_csv.exists()
    with metadata_csv.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(meta).keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(asdict(meta))


def frequency_token(frequency_mhz: float) -> str:
    return f"{frequency_mhz:g}".replace(".", "_")


def make_sample_group_name(cfg: InstrumentConfig) -> str:
    return f"{cfg.class_name}_{frequency_token(cfg.center_freq_mhz)}"


def build_note(cfg: InstrumentConfig) -> str:
    analyzer = (
        f"profile={cfg.analyzer_profile}; center={cfg.center_freq_mhz:g} MHz; "
        f"span={cfg.span_khz:g} kHz; RBW={cfg.rbw_hz:g} Hz; VBW={cfg.vbw_hz:g} Hz; "
        f"points={cfg.sweep_points}; temporal_slices={cfg.sweeps_per_sample}"
    )
    if cfg.class_name == "FM":
        return f"FM эфир; station={cfg.source_name}; {analyzer}"
    if cfg.class_name == "AM":
        return (
            f"AM генератор; carrier={cfg.center_freq_mhz:g} MHz; "
            f"level={cfg.generator_level_dbm:g} dBm; depth={cfg.generator_depth_percent:g}%; "
            f"modulation={cfg.generator_modulation_freq_khz:g} kHz; "
            f"waveform={cfg.generator_waveform}; {analyzer}"
        )
    if cfg.class_name == "PM":
        return (
            f"PM генератор; carrier={cfg.center_freq_mhz:g} MHz; "
            f"level={cfg.generator_level_dbm:g} dBm; "
            f"phase_deviation={cfg.generator_phase_deviation_rad:g} rad; "
            f"modulation={cfg.generator_modulation_freq_khz:g} kHz; "
            f"waveform={cfg.generator_waveform}; {analyzer}"
        )
    return analyzer


def apply_profile(cfg: InstrumentConfig, profile: AnalyzerProfile) -> InstrumentConfig:
    updated = replace(
        cfg, analyzer_profile=profile.name, span_khz=profile.span_khz,
        rbw_hz=profile.rbw_hz, vbw_hz=profile.vbw_hz,
        sweep_points=profile.sweep_points, detector=profile.detector,
        trace_mode=profile.trace_mode,
    )
    return replace(updated, note=build_note(updated))


def select_profiles(names: Sequence[str]) -> Tuple[AnalyzerProfile, ...]:
    unknown = sorted(set(names) - set(ANALYZER_PROFILES))
    if unknown:
        raise ValueError(f"Unknown analyzer profiles: {unknown}")
    return tuple(ANALYZER_PROFILES[name] for name in names)


def capture_signature(cfg: InstrumentConfig) -> Dict[str, object]:
    return {
        "class_name": cfg.class_name,
        "center_freq_mhz": cfg.center_freq_mhz,
        "span_khz": cfg.span_khz,
        "rbw_hz": cfg.rbw_hz,
        "vbw_hz": cfg.vbw_hz,
        "analyzer_profile": cfg.analyzer_profile,
        "source_name": cfg.source_name,
        "generator_level_dbm": cfg.generator_level_dbm,
        "generator_modulation_freq_khz": cfg.generator_modulation_freq_khz,
        "generator_depth_percent": cfg.generator_depth_percent,
        "generator_phase_deviation_rad": cfg.generator_phase_deviation_rad,
        "generator_waveform": cfg.generator_waveform,
    }


def values_match(left: object, right: object) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) < 1e-9
    return left == right


def capture_already_exists(cfg: InstrumentConfig) -> bool:
    group_dir = Path(cfg.dataset_root) / cfg.class_name / make_sample_group_name(cfg)
    if not group_dir.exists():
        return False
    expected = capture_signature(cfg)
    for config_path in group_dir.glob("*/run_config.json"):
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if all(values_match(payload.get(key), value) for key, value in expected.items()):
            if any((config_path.parent / "samples").glob("*.csv")):
                return True
    return False


def append_failure(dataset_root: str, cfg: InstrumentConfig, error: Exception) -> None:
    root = Path(dataset_root)
    ensure_dir(root)
    payload = {"failed_at": utc_now_str(), "error": repr(error), "config": asdict(cfg)}
    with (root / "collection_failures.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


class RSFSLCollector:
    def __init__(self, cfg: InstrumentConfig):
        self.cfg = cfg
        self.sock: Optional[socket.socket] = None
        self.idn = "UNKNOWN"

    def _get_host_port(self) -> Tuple[str, int]:
        resource = self.cfg.resource.strip()

        # Поддерживаем старую строку вида:
        # TCPIP0::192.168.10.24::inst0::INSTR
        if "::" in resource:
            parts = resource.split("::")
            if len(parts) >= 2:
                return parts[1], 5025

        # И более простой вариант:
        # 192.168.10.24:5025
        if ":" in resource:
            host, port = resource.rsplit(":", 1)
            return host, int(port)

        return resource, 5025

    def connect(self) -> None:
        host, port = self._get_host_port()

        self.sock = socket.create_connection(
            (host, port),
            timeout=self.cfg.timeout_ms / 1000.0,
        )
        self.sock.settimeout(self.cfg.timeout_ms / 1000.0)

        if self.cfg.reset_on_connect:
            self.write("*RST")
            time.sleep(2.0)

        self.write("*CLS")
        self.try_write(
            "SYST:DISP:UPD OFF"
            if self.cfg.disable_display_updates
            else "SYST:DISP:UPD ON"
        )
        self.write("INIT:CONT OFF")
        self.idn = self.query("*IDN?").strip()

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
            finally:
                self.sock = None

    def write(self, cmd: str) -> None:
        if self.sock is None:
            raise RuntimeError("Instrument is not connected")

        self.sock.sendall((cmd + "\n").encode("ascii"))

    def try_write(self, cmd: str) -> bool:
        try:
            self.write(cmd)
            return True
        except Exception:
            return False

    def query(self, cmd: str) -> str:
        if self.sock is None:
            raise RuntimeError("Instrument is not connected")

        self.write(cmd)

        chunks = bytearray()
        while True:
            chunk = self.sock.recv(65536)
            if not chunk:
                break

            chunks.extend(chunk)

            if b"\n" in chunk:
                break

        if not chunks:
            raise RuntimeError(f"Empty response for SCPI query: {cmd}")

        return chunks.decode("ascii", errors="replace").strip()

    def configure(self, cfg: InstrumentConfig) -> Dict[str, float]:
        self.cfg = cfg
        center_hz, span_hz = cfg.center_freq_mhz * 1e6, cfg.span_khz * 1e3
        self.write(f"FREQ:CENT {center_hz}")
        self.write(f"FREQ:SPAN {span_hz}")
        self.write(f"BAND:RES {cfg.rbw_hz}")
        self.write(f"BAND:VID {cfg.vbw_hz}")

        if not self.try_write(f"SWE:POIN {cfg.sweep_points}"):
            if not self.try_write(f"DISP:TRAC:POIN {cfg.sweep_points}"):
                raise RuntimeError("Could not set sweep points on instrument")

        self.try_write(f"DET {cfg.detector}")
        self.try_write(f"DET:TRAC {cfg.detector}")
        self.try_write(f"DISP:WIND:TRAC1:MODE {cfg.trace_mode}")
        self.try_write(f"TRAC1:MODE {cfg.trace_mode}")
        self.try_write(f"TRAC:MODE {cfg.trace_mode}")

        if cfg.reference_level_dbm is not None:
            self.try_write(f"DISP:WIND:TRAC:Y:RLEV {cfg.reference_level_dbm}DBM")
            self.try_write(f"DISP:TRAC:Y:RLEV {cfg.reference_level_dbm}DBM")

        if cfg.attenuation_db is not None:
            self.try_write("INP:ATT:AUTO OFF")
            self.try_write(f"INP:ATT {cfg.attenuation_db}")
        else:
            self.try_write("INP:ATT:AUTO ON")

        self.query("*OPC?")
        time.sleep(cfg.settle_after_config_s)

        return {
            "center_hz": center_hz,
            "span_hz": span_hz,
            "start_hz": center_hz - span_hz / 2.0,
            "stop_hz": center_hz + span_hz / 2.0,
        }

    def acquire_single_sweep(self) -> List[float]:
        self.write("INIT;*WAI")

        trace_commands = (
            "FORM ASC; TRAC? TRACE1",
            "FORM ASC; TRAC:DATA? TRACE1",
            "FORM ASC; TRACE:DATA? TRACE1",
            "TRAC? TRACE1",
            "TRAC:DATA? TRACE1",
            "TRACE:DATA? TRACE1",
        )

        last_error: Optional[Exception] = None

        for cmd in trace_commands:
            try:
                values = parse_ascii_trace(self.query(cmd))
                if values:
                    return values
            except Exception as error:
                last_error = error

        raise RuntimeError(
            f"Failed to read trace data over TCP socket. Last error: {last_error}"
        )

    def acquire_sample(self) -> Tuple[List[float], List[List[float]]]:
        axis = build_frequency_axis(
            self.cfg.center_freq_mhz * 1e6,
            self.cfg.span_khz * 1e3,
            self.cfg.sweep_points,
        )

        sweeps: List[List[float]] = []

        for sweep_idx in range(self.cfg.sweeps_per_sample):
            values = self.acquire_single_sweep()

            if len(values) < self.cfg.sweep_points:
                raise RuntimeError(
                    f"Trace has {len(values)} points, "
                    f"expected {self.cfg.sweep_points}"
                )

            sweeps.append(values[: self.cfg.sweep_points])

            if (
                sweep_idx < self.cfg.sweeps_per_sample - 1
                and self.cfg.inter_sweep_delay_s > 0
            ):
                time.sleep(self.cfg.inter_sweep_delay_s)

        return axis, sweeps


def save_sample_csv(
    csv_path: Path, frequency_axis_hz: List[float], sweep_matrix_dbm: List[List[float]],
    sample_id: str, class_name: str, sample_group_name: str,
) -> None:
    ensure_dir(csv_path.parent)
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "sample_id", "class_name", "sample_group_name", "sweep_idx",
            "point_idx", "frequency_hz", "amplitude_dbm",
        ])
        for sweep_idx, row in enumerate(sweep_matrix_dbm):
            for point_idx, (freq_hz, amp_dbm) in enumerate(zip(frequency_axis_hz, row)):
                writer.writerow([
                    sample_id, class_name, sample_group_name, sweep_idx, point_idx,
                    f"{freq_hz:.3f}", f"{amp_dbm:.6f}",
                ])


def save_run_config(run_dir: Path, cfg: InstrumentConfig, instrument_idn: str, group: str) -> None:
    payload = asdict(cfg).copy()
    payload.update(sample_group_name=group, instrument_idn=instrument_idn, saved_at=utc_now_str())
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def collect_configured_run(
    collector: RSFSLCollector, cfg: InstrumentConfig,
    validate_sample: Optional[Callable[[List[List[float]]], None]] = None,
) -> None:
    group = make_sample_group_name(cfg)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path(cfg.dataset_root) / cfg.class_name / group / run_ts
    samples_dir, metadata_csv = run_dir / "samples", run_dir / "metadata.csv"
    applied = collector.configure(cfg)
    print(
        f"  Analyzer: center={cfg.center_freq_mhz:g} MHz, span={cfg.span_khz:g} kHz, "
        f"RBW={cfg.rbw_hz:g} Hz, VBW={cfg.vbw_hz:g} Hz, "
        f"points={cfg.sweep_points}, slices={cfg.sweeps_per_sample}"
    )
    for sample_idx in range(cfg.samples_per_run):
        sample_id = f"{group}_{run_ts}_{sample_idx:04d}"
        sample_csv = samples_dir / f"{sample_id}.csv"
        print(f"  [{sample_idx + 1}/{cfg.samples_per_run}] Collecting {sample_id} ...")
        started_at = utc_now_str()
        axis, matrix = collector.acquire_sample()
        if validate_sample is not None:
            validate_sample(matrix)
        ensure_dir(samples_dir)
        if not (run_dir / "run_config.json").exists():
            save_run_config(run_dir, cfg, collector.idn, group)
        save_sample_csv(sample_csv, axis, matrix, sample_id, cfg.class_name, group)
        meta = SampleMetadata(
            sample_id=sample_id,
            class_name=cfg.class_name,
            sample_group_name=group,
            source_type=cfg.source_type,
            source_name=cfg.source_name,
            analyzer_profile=cfg.analyzer_profile,
            note=cfg.note,
            instrument_model=collector.idn,
            instrument_resource=cfg.resource,
            started_at=started_at,
            finished_at=utc_now_str(),
            center_freq_hz=applied["center_hz"],
            span_hz=applied["span_hz"],
            start_freq_hz=applied["start_hz"],
            stop_freq_hz=applied["stop_hz"],
            rbw_hz=cfg.rbw_hz,
            vbw_hz=cfg.vbw_hz,
            sweep_points=cfg.sweep_points,
            sweeps_per_sample=cfg.sweeps_per_sample,
            detector=cfg.detector,
            trace_mode=cfg.trace_mode,
            reference_level_dbm=cfg.reference_level_dbm,
            attenuation_db=cfg.attenuation_db,
            generator_level_dbm=cfg.generator_level_dbm,
            generator_modulation_freq_khz=cfg.generator_modulation_freq_khz,
            generator_depth_percent=cfg.generator_depth_percent,
            generator_phase_deviation_rad=cfg.generator_phase_deviation_rad,
            generator_waveform=cfg.generator_waveform,
            csv_path=str(sample_csv.resolve()),
        )
        append_metadata_row(metadata_csv, meta)
        print(f"  Saved: {sample_csv}")


def build_fm_configs(base: InstrumentConfig, profiles: Sequence[AnalyzerProfile]) -> List[InstrumentConfig]:
    configs: List[InstrumentConfig] = []
    for station in FM_STATIONS:
        station_cfg = replace(
            base, center_freq_mhz=station.frequency_mhz, class_name="FM",
            source_type="air", source_name=station.name,
            generator_level_dbm=None, generator_modulation_freq_khz=None,
            generator_depth_percent=None, generator_waveform=None,
        )
        configs.extend(apply_profile(station_cfg, profile) for profile in profiles)
    return configs


def build_fm_dop_configs(base: InstrumentConfig) -> List[InstrumentConfig]:
    configs: List[InstrumentConfig] = []

    for station_index, station in enumerate(FM_STATIONS):
        # Three captures per station. The starting variant rotates so that all
        # offset/span combinations are represented across the dataset.
        start_variant = (station_index * 3) % len(FM_DOP_VARIANTS)

        for local_index in range(3):
            variant_index = (start_variant + local_index) % len(FM_DOP_VARIANTS)
            offset_khz, span_khz, description = FM_DOP_VARIANTS[variant_index]
            center_mhz = station.frequency_mhz + offset_khz / 1000.0

            cfg = replace(
                base,
                dataset_root=str(Path(base.dataset_root) / "FM_dop"),
                center_freq_mhz=center_mhz,
                span_khz=span_khz,
                rbw_hz=10_000.0,
                vbw_hz=10_000.0,
                sweep_points=501,
                detector="POS",
                trace_mode="WRIT",
                class_name="FM",
                source_type="air",
                source_name=station.name,
                analyzer_profile=f"fm_dop_v{variant_index + 1}",
                generator_level_dbm=None,
                generator_modulation_freq_khz=None,
                generator_depth_percent=None,
                generator_waveform=None,
            )

            note = (
                f"FM_dop; station={station.name}; "
                f"station_frequency={station.frequency_mhz:g} MHz; "
                f"center={center_mhz:g} MHz; "
                f"center_offset={offset_khz:+g} kHz; "
                f"span={span_khz:g} kHz; {description}"
            )
            configs.append(replace(cfg, note=note))

    return configs


def build_am_configs(
    base: InstrumentConfig, preset: AMGeneratorPreset, profiles: Sequence[AnalyzerProfile]
) -> List[InstrumentConfig]:
    preset_cfg = replace(
        base, center_freq_mhz=preset.carrier_freq_mhz, class_name="AM",
        source_type="generator", source_name="Rohde & Schwarz SMB100A",
        generator_level_dbm=preset.level_dbm,
        generator_modulation_freq_khz=preset.modulation_freq_khz,
        generator_depth_percent=preset.depth_percent,
        generator_waveform=preset.waveform,
    )
    return [apply_profile(preset_cfg, profile) for profile in profiles]


def build_pm_configs(
    base: InstrumentConfig, preset: PMGeneratorPreset,
    profiles: Sequence[AnalyzerProfile], spans_khz: Sequence[float],
) -> List[InstrumentConfig]:
    preset_cfg = replace(
        base, center_freq_mhz=preset.carrier_freq_mhz, class_name="PM",
        source_type="generator", source_name="Rohde & Schwarz SMB100A",
        generator_level_dbm=preset.level_dbm,
        generator_modulation_freq_khz=preset.modulation_freq_khz,
        generator_depth_percent=None,
        generator_phase_deviation_rad=preset.phase_deviation_rad,
        generator_waveform=preset.waveform,
    )
    configs = []
    for span_khz, profile in product(spans_khz, profiles):
        span_profile = replace(
            profile, name=f"{profile.name}_span{span_khz:g}k",
            span_khz=span_khz,
        )
        configs.append(apply_profile(preset_cfg, span_profile))
    return configs


def execute_configs(
    collector: RSFSLCollector, configs: Sequence[InstrumentConfig],
    summary: CollectionSummary, resume: bool,
) -> None:
    for index, cfg in enumerate(configs, start=1):
        print(
            f"[{index}/{len(configs)}] {cfg.class_name} {cfg.center_freq_mhz:g} MHz | "
            f"{cfg.source_name} | profile={cfg.analyzer_profile}"
        )
        if resume and capture_already_exists(cfg):
            print("  Already collected, skipping.")
            summary.skipped_existing += 1
            continue
        try:
            collect_configured_run(collector, cfg)
            summary.collected += 1
        except KeyboardInterrupt:
            raise
        except Exception as error:
            summary.failed += 1
            append_failure(cfg.dataset_root, cfg, error)
            print(f"  ERROR: {error}")
            print("  Failure logged; continuing with the next configuration.")


def run_fm_mode(
    base: InstrumentConfig, profiles: Sequence[AnalyzerProfile], dry_run: bool, resume: bool
) -> CollectionSummary:
    configs = build_fm_configs(base, profiles)
    print(f"FM plan: {len(FM_STATIONS)} stations x {len(profiles)} profiles = {len(configs)} captures.")
    if dry_run:
        for index, cfg in enumerate(configs, start=1):
            print(f"[{index}/{len(configs)}] {cfg.center_freq_mhz:g} MHz | {cfg.note}")
        return CollectionSummary()
    collector, summary = RSFSLCollector(base), CollectionSummary()
    try:
        print("Connecting to instrument...")
        collector.connect()
        print(f"Connected: {collector.idn}")
        execute_configs(collector, configs, summary, resume)
    finally:
        collector.close()
    return summary


def run_fm_dop_mode(
    base: InstrumentConfig, dry_run: bool, resume: bool
) -> CollectionSummary:
    configs = build_fm_dop_configs(base)
    print(f"FM_dop plan: {len(FM_STATIONS)} stations x 3 variants = {len(configs)} captures.")

    if dry_run:
        for index, cfg in enumerate(configs, start=1):
            print(
                f"[{index}/{len(configs)}] center={cfg.center_freq_mhz:g} MHz | "
                f"span={cfg.span_khz:g} kHz | {cfg.source_name} | "
                f"profile={cfg.analyzer_profile}"
            )
        return CollectionSummary()

    collector, summary = RSFSLCollector(base), CollectionSummary()
    try:
        print("Connecting to instrument...")
        collector.connect()
        print(f"Connected: {collector.idn}")
        execute_configs(collector, configs, summary, resume)
    finally:
        collector.close()
    return summary


def noise_candidate_centers(
    band_start_mhz: float, band_stop_mhz: float, span_khz: float,
    station_guard_khz: float,
) -> List[float]:
    """Find windows that do not overlap a listed FM station or band edge."""
    if band_start_mhz >= band_stop_mhz:
        raise ValueError("Noise band start must be below band stop")
    half_span_mhz = span_khz / 2000.0
    exclusion_mhz = half_span_mhz + 0.100 + station_guard_khz / 1000.0
    centers = []
    first_index = math.ceil((band_start_mhz + half_span_mhz) * 4)
    last_index = math.floor((band_stop_mhz - half_span_mhz) * 4)
    for index in range(first_index, last_index + 1):
        center = index / 4.0
        if all(abs(center - station.frequency_mhz) > exclusion_mhz for station in FM_STATIONS):
            centers.append(center)
    return centers


def noise_trace_stats(sweeps: Sequence[Sequence[float]]) -> Tuple[float, float]:
    """Return median floor and strongest local peak seen in any sweep, in dB."""
    if not sweeps or not sweeps[0]:
        raise ValueError("Empty noise trace")
    points = len(sweeps[0])
    if any(len(row) != points or any(not math.isfinite(value) for value in row) for row in sweeps):
        raise ValueError("Invalid noise trace values or length")
    floor_dbm = statistics.median(statistics.median(row) for row in sweeps)
    width = min(9, points)
    peak_excess_db = max(
        statistics.mean(row[start:start + width]) - statistics.median(row)
        for row in sweeps for start in range(points - width + 1)
    )
    return floor_dbm, peak_excess_db


def survey_noise_center(
    collector: RSFSLCollector, cfg: InstrumentConfig, survey_sweeps: int,
    band_index: int,
) -> NoiseSurvey:
    collector.configure(cfg)
    sweeps = [collector.acquire_single_sweep() for _ in range(survey_sweeps)]
    floor_dbm, peak_excess_db = noise_trace_stats(sweeps)
    return NoiseSurvey(cfg.center_freq_mhz, band_index, floor_dbm, peak_excess_db)


def run_noise_mode(
    base: InstrumentConfig, profiles: Sequence[AnalyzerProfile], dry_run: bool,
    resume: bool, count: int, bands: Sequence[Tuple[float, float]],
    spans_khz: Sequence[float],
    station_guard_khz: float, peak_threshold_db: float,
    floor_margin_db: float, survey_sweeps: int,
    requested_centers: Optional[Sequence[float]] = None,
) -> CollectionSummary:
    if count < 1 or survey_sweeps < 2:
        raise ValueError("Noise count must be positive and survey sweeps at least 2")
    if station_guard_khz < 0 or peak_threshold_db <= 0 or floor_margin_db <= 0:
        raise ValueError("Noise guard and thresholds must be positive")
    if not spans_khz or any(not math.isfinite(span) or span <= 0 for span in spans_khz):
        raise ValueError("Noise spans must be positive finite numbers")
    if not bands:
        raise ValueError("At least one Noise band is required")
    baseline = ANALYZER_PROFILES["baseline"]
    spans_khz = tuple(sorted(set(spans_khz)))
    widest_span_khz = max(spans_khz)
    survey_profile = replace(baseline, span_khz=widest_span_khz)
    candidates = [
        (band_index, center)
        for band_index, (start, stop) in enumerate(bands)
        for center in noise_candidate_centers(
            start, stop, widest_span_khz, station_guard_khz,
        )
    ]
    if requested_centers:
        candidates = []
        half_span_mhz = widest_span_khz / 2000.0
        exclusion_mhz = half_span_mhz + 0.100 + station_guard_khz / 1000.0
        for center in requested_centers:
            band_index = next((
                index for index, (start, stop) in enumerate(bands)
                if center - half_span_mhz >= start and center + half_span_mhz <= stop
            ), None)
            if (not math.isfinite(center) or band_index is None
                    or any(abs(center - station.frequency_mhz) <= exclusion_mhz
                           for station in FM_STATIONS)):
                raise ValueError("A requested Noise center overlaps a station guard or band edge")
            candidates.append((band_index, float(center)))
    if not candidates:
        raise ValueError("No candidate Noise windows remain in the selected bands")

    print(
        f"Noise survey: {len(candidates)} candidate windows in {len(bands)} bands, "
        f"{survey_sweeps} sweeps each; target {count} centers x "
        f"{len(spans_khz)} spans x {len(profiles)} profiles."
    )
    print("  Spans (kHz): " + ", ".join(f"{span:g}" for span in spans_khz))
    for band_index, (start, stop) in enumerate(bands, start=1):
        total = sum(index == band_index - 1 for index, _ in candidates)
        print(f"  Band {band_index}: {start:g}–{stop:g} MHz, {total} candidates")
    print("Выключите RF-выход AM-генератора; оставьте приемную антенну как при FM-сборе.")
    if dry_run:
        print("Candidate centers (MHz): " + ", ".join(f"{center:g}" for _, center in candidates))
        print("Actual quiet centers will be selected from analyzer traces during collection.")
        return CollectionSummary()

    input("После проверки установки нажмите Enter для сканирования частот: ")
    collector, summary = RSFSLCollector(base), CollectionSummary()
    try:
        print("Connecting to instrument...")
        collector.connect()
        print(f"Connected: {collector.idn}")
        surveys = []
        for index, (band_index, center) in enumerate(candidates, start=1):
            cfg = apply_profile(replace(
                base, center_freq_mhz=center, class_name="Noise",
                source_type="air", source_name="quiet spectrum window",
            ), survey_profile)
            result = survey_noise_center(collector, cfg, survey_sweeps, band_index)
            surveys.append(result)
            print(
                f"  Survey [{index}/{len(candidates)}] {center:g} MHz: "
                f"floor={result.floor_dbm:.1f} dBm, "
                f"peak={result.peak_excess_db:.1f} dB"
            )

        reference_floors = {}
        quiet_by_band: Dict[int, List[NoiseSurvey]] = {}
        for band_index in range(len(bands)):
            band_surveys = [item for item in surveys if item.band_index == band_index]
            if not band_surveys:
                quiet_by_band[band_index] = []
                continue
            sorted_floors = sorted(item.floor_dbm for item in band_surveys)
            reference_floor = sorted_floors[len(sorted_floors) // 5]
            reference_floors[band_index] = reference_floor
            quiet_by_band[band_index] = sorted(
                (
                    item for item in band_surveys
                    if item.peak_excess_db <= peak_threshold_db
                    and item.floor_dbm <= reference_floor + floor_margin_db
                ),
                key=lambda item: (item.peak_excess_db, item.floor_dbm),
            )
            print(
                f"  Band {band_index + 1} reference floor: {reference_floor:.1f} dBm; "
                f"quiet candidates: {len(quiet_by_band[band_index])}"
            )

        selected: List[NoiseSurvey] = []
        while len(selected) < count:
            progress = False
            for band_index in range(len(bands)):
                pool = quiet_by_band[band_index]
                while pool:
                    item = pool.pop(0)
                    if all(abs(item.center_mhz - earlier.center_mhz) >= 0.5
                           for earlier in selected):
                        selected.append(item)
                        progress = True
                        break
                if len(selected) == count:
                    break
            if not progress:
                break
        if not selected:
            raise RuntimeError("No quiet windows passed the spectrum check; no Noise data saved")
        print(f"Selected {len(selected)} quiet windows across the bands.")
        if len(selected) < count:
            print(f"Only {len(selected)} of {count} requested centers passed. No marginal centers added.")

        survey_path = Path(base.dataset_root) / "Noise" / (
            "survey_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
        )
        ensure_dir(survey_path.parent)
        chosen = {item.center_mhz for item in selected}
        with survey_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                "band_start_mhz", "band_stop_mhz", "center_mhz",
                "survey_span_khz", "floor_dbm", "peak_excess_db",
                "reference_floor_dbm", "selected",
            ])
            for item in surveys:
                start, stop = bands[item.band_index]
                writer.writerow([
                    f"{start:.3f}", f"{stop:.3f}",
                    f"{item.center_mhz:.3f}", f"{widest_span_khz:g}",
                    f"{item.floor_dbm:.3f}",
                    f"{item.peak_excess_db:.3f}",
                    f"{reference_floors[item.band_index]:.3f}",
                    int(item.center_mhz in chosen),
                ])
        print(f"Survey saved: {survey_path}")

        for item in selected:
            reference_floor = reference_floors[item.band_index]
            band_start, band_stop = bands[item.band_index]
            for span_khz, profile in product(spans_khz, profiles):
                span_profile = replace(
                    profile, name=f"{profile.name}_span{span_khz:g}k",
                    span_khz=span_khz,
                )
                cfg = apply_profile(replace(
                    base, center_freq_mhz=item.center_mhz, class_name="Noise",
                    source_type="air", source_name="quiet spectrum window",
                ), span_profile)
                cfg = replace(cfg, note=(
                    f"Noise эфир; band={band_start:g}-{band_stop:g} MHz; "
                    f"survey_floor={item.floor_dbm:.2f} dBm; "
                    f"survey_peak={item.peak_excess_db:.2f} dB; "
                    f"reference_floor={reference_floor:.2f} dBm; "
                    f"peak_limit={peak_threshold_db:g} dB; "
                    f"floor_margin={floor_margin_db:g} dB; {cfg.note}"
                ))
                if resume and capture_already_exists(cfg):
                    summary.skipped_existing += 1
                    print(f"  Already collected: {item.center_mhz:g} MHz / {span_profile.name}")
                    continue

                def validate_sample(matrix: List[List[float]]) -> None:
                    floor_dbm, peak_excess_db = noise_trace_stats(matrix)
                    if peak_excess_db > peak_threshold_db:
                        raise ValueError(
                            f"Noise capture rejected: persistent peak {peak_excess_db:.1f} dB"
                        )
                    if floor_dbm > reference_floor + floor_margin_db:
                        raise ValueError(
                            f"Noise capture rejected: floor {floor_dbm:.1f} dBm exceeds "
                            f"{reference_floor + floor_margin_db:.1f} dBm"
                        )

                try:
                    print(f"Noise {item.center_mhz:g} MHz / {span_profile.name}")
                    collect_configured_run(collector, cfg, validate_sample=validate_sample)
                    summary.collected += 1
                except KeyboardInterrupt:
                    raise
                except Exception as error:
                    summary.failed += 1
                    append_failure(cfg.dataset_root, cfg, error)
                    print(f"  ERROR: {error}; no Noise CSV saved for this run.")
    finally:
        collector.close()
    return summary


def print_generator_instructions(preset: AMGeneratorPreset, index: int, total: int) -> None:
    print("\n" + "=" * 72)
    print(f"AM generator setup [{index}/{total}]")
    print(f"  Carrier frequency : {preset.carrier_freq_mhz:g} MHz")
    print(f"  RF level          : {preset.level_dbm:g} dBm")
    print("  AM state          : ON")
    print("  AM source         : Internal")
    print(f"  AM depth          : {preset.depth_percent:g} %")
    print(f"  LF generator freq : {preset.modulation_freq_khz:g} kHz")
    print(f"  LF waveform       : {preset.waveform}")
    print("  RF output         : ON")
    print("=" * 72)


def wait_for_generator_ready() -> str:
    while True:
        answer = input(
            "Выставьте настройки. Enter/готов — собирать, s — пропустить, q — завершить: "
        ).strip().lower()
        if answer in {"", "готов", "ready", "r"}:
            return "ready"
        if answer in {"s", "skip", "пропустить"}:
            return "skip"
        if answer in {"q", "quit", "exit", "выход"}:
            return "quit"
        print("Неизвестная команда.")


def run_am_mode(
    base: InstrumentConfig, profiles: Sequence[AnalyzerProfile], dry_run: bool, resume: bool
) -> CollectionSummary:
    total = len(AM_GENERATOR_PRESETS) * len(profiles)
    print(f"AM plan: {len(AM_GENERATOR_PRESETS)} generator setups x {len(profiles)} profiles = {total} captures.")
    summary = CollectionSummary()
    if dry_run:
        for index, preset in enumerate(AM_GENERATOR_PRESETS, start=1):
            print_generator_instructions(preset, index, len(AM_GENERATOR_PRESETS))
            for cfg in build_am_configs(base, preset, profiles):
                print(f"  analyzer profile={cfg.analyzer_profile}; note: {cfg.note}")
        return summary
    collector = RSFSLCollector(base)
    try:
        print("Connecting to instrument...")
        collector.connect()
        print(f"Connected: {collector.idn}")
        for index, preset in enumerate(AM_GENERATOR_PRESETS, start=1):
            configs = build_am_configs(base, preset, profiles)
            pending = [cfg for cfg in configs if not (resume and capture_already_exists(cfg))]
            summary.skipped_existing += len(configs) - len(pending)
            if not pending:
                print(f"AM setup [{index}/{len(AM_GENERATOR_PRESETS)}] already collected, skipping.")
                continue
            print_generator_instructions(preset, index, len(AM_GENERATOR_PRESETS))
            action = wait_for_generator_ready()
            if action == "quit":
                print("Collection stopped. The next run will resume missing captures.")
                break
            if action == "skip":
                summary.skipped_by_user += len(pending)
                continue
            execute_configs(collector, pending, summary, resume=False)
    finally:
        collector.close()
    return summary


def print_pm_generator_instructions(preset: PMGeneratorPreset, index: int, total: int) -> None:
    print("\n" + "=" * 72)
    print(f"PM generator setup [{index}/{total}]")
    print(f"  Carrier frequency : {preset.carrier_freq_mhz:g} MHz")
    print(f"  RF level          : {preset.level_dbm:g} dBm")
    print("  AM state          : OFF")
    print("  FM state          : OFF")
    print("  PhiM / PM state   : ON")
    print("  PhiM mode         : Normal")
    print("  PhiM source       : Internal")
    print(f"  PhiM deviation   : {preset.phase_deviation_rad:g} rad")
    print(f"  LF generator freq : {preset.modulation_freq_khz:g} kHz")
    print(f"  LF waveform       : {preset.waveform}")
    print("  RF output         : ON")
    print("=" * 72)


def run_pm_mode(
    base: InstrumentConfig, profiles: Sequence[AnalyzerProfile], dry_run: bool,
    resume: bool, deviations_rad: Sequence[float], spans_khz: Sequence[float],
) -> CollectionSummary:
    if not spans_khz or any(not math.isfinite(span) or span <= 0 for span in spans_khz):
        raise ValueError("PM spans must be positive finite numbers")
    spans_khz = tuple(sorted(set(spans_khz)))
    presets = build_pm_generator_presets(deviations_rad)
    total = len(presets) * len(spans_khz) * len(profiles)
    print(
        f"PM plan: {len(presets)} generator setups x {len(spans_khz)} spans x "
        f"{len(profiles)} profiles = {total} captures."
    )
    summary = CollectionSummary()
    if dry_run:
        for index, preset in enumerate(presets, start=1):
            print_pm_generator_instructions(preset, index, len(presets))
            for cfg in build_pm_configs(base, preset, profiles, spans_khz):
                print(
                    f"  analyzer profile={cfg.analyzer_profile}; "
                    f"span={cfg.span_khz:g} kHz"
                )
        return summary

    collector = RSFSLCollector(base)
    try:
        print("Connecting to instrument...")
        collector.connect()
        print(f"Connected: {collector.idn}")
        for index, preset in enumerate(presets, start=1):
            configs = build_pm_configs(base, preset, profiles, spans_khz)
            pending = [cfg for cfg in configs if not (resume and capture_already_exists(cfg))]
            summary.skipped_existing += len(configs) - len(pending)
            if not pending:
                print(f"PM setup [{index}/{len(presets)}] already collected, skipping.")
                continue
            print_pm_generator_instructions(preset, index, len(presets))
            action = wait_for_generator_ready()
            if action == "quit":
                print("Collection stopped. The next run will resume missing captures.")
                break
            if action == "skip":
                summary.skipped_by_user += len(pending)
                continue
            execute_configs(collector, pending, summary, resume=False)
    finally:
        collector.close()
    return summary


def collect_dataset(cfg: InstrumentConfig) -> None:
    """Backward-compatible single-configuration collection."""
    collector = RSFSLCollector(cfg)
    try:
        print("Connecting to instrument...")
        collector.connect()
        print(f"Connected: {collector.idn}")
        collect_configured_run(collector, replace(cfg, note=cfg.note or build_note(cfg)))
    finally:
        collector.close()


def choose_mode() -> str:
    print("Выберите режим сбора:")
    print("  1 — FM: автоматический проход по радиостанциям")
    print("  2 — AM: пошаговые настройки генератора")
    print("  3 — FM_dop: смещенный центр и измененный span")
    print("  4 — Noise: поиск свободных участков в заданных диапазонах")
    print("  5 — PM: пошаговые настройки фазовой модуляции")
    while True:
        answer = input("Режим [1/2/3/4/5]: ").strip().lower()
        if answer in {"1", "fm"}:
            return "fm"
        if answer in {"2", "am"}:
            return "am"
        if answer in {"3", "fm_dop"}:
            return "fm_dop"
        if answer in {"4", "noise"}:
            return "noise"
        if answer in {"5", "pm"}:
            return "pm"
        print("Введите 1, 2, 3, 4 или 5.")


def choose_profile_names() -> Tuple[str, ...]:
    print("Профили анализатора:")
    print("  1 — baseline + high_resolution + smoothed (рекомендуется)")
    print("  2 — только baseline (быстрый сбор)")
    while True:
        answer = input("Профили [1/2, Enter=1]: ").strip().lower()
        if answer in {"", "1", "all"}:
            return DEFAULT_PROFILE_NAMES
        if answer in {"2", "baseline"}:
            return ("baseline",)
        print("Введите 1 или 2.")


def print_summary(summary: CollectionSummary) -> None:
    print("\nCollection summary:")
    print(f"  collected        : {summary.collected}")
    print(f"  skipped existing : {summary.skipped_existing}")
    print(f"  skipped by user  : {summary.skipped_by_user}")
    print(f"  failed           : {summary.failed}")


def parse_noise_bands(values: Optional[Sequence[str]]) -> Tuple[Tuple[float, float], ...]:
    if not values:
        return DEFAULT_NOISE_BANDS
    bands = []
    for value in values:
        try:
            start_text, stop_text = value.split(":", 1)
            start, stop = float(start_text), float(stop_text)
        except ValueError as error:
            raise ValueError(f"Invalid Noise band {value!r}; use START:STOP in MHz") from error
        if not math.isfinite(start) or not math.isfinite(stop) or start >= stop:
            raise ValueError(f"Invalid Noise band {value!r}")
        bands.append((start, stop))
    bands.sort()
    if any(left[1] > right[0] for left, right in zip(bands, bands[1:])):
        raise ValueError("Noise bands must not overlap")
    return tuple(bands)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect FM, AM, PM or Noise datasets with an R&S FSL analyzer.")
    parser.add_argument("--mode", choices=("fm", "am", "fm_dop", "noise", "pm"))
    parser.add_argument("--profiles", nargs="+", choices=tuple(ANALYZER_PROFILES))
    parser.add_argument("--resource", default=InstrumentConfig.resource)
    parser.add_argument("--dataset-root", default="data")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without instrument connection.")
    parser.add_argument("--force", action="store_true", help="Do not skip existing captures.")
    parser.add_argument("--no-reset", action="store_true", help="Do not reset analyzer on connect.")
    parser.add_argument("--noise-count", type=int, default=12,
                        help="Target number of independent quiet frequency windows (default: 12).")
    parser.add_argument("--noise-spans-khz", nargs="+", type=float,
                        default=(180.0, 250.0, 400.0),
                        help="Span values for every profile; default: 180 250 400 kHz.")
    parser.add_argument("--noise-bands", nargs="+", metavar="START:STOP",
                        help="Bands in MHz; default: 80:87.5 87.5:108 108:118.")
    parser.add_argument("--noise-centers", nargs="+", type=float,
                        help="Optional candidate center frequencies in MHz; each is still checked.")
    parser.add_argument("--noise-station-guard-khz", type=float, default=100.0,
                        help="Additional clearance beyond nominal 200 kHz FM channel (default: 100).")
    parser.add_argument("--noise-peak-threshold-db", type=float, default=10.0)
    parser.add_argument("--noise-floor-margin-db", type=float, default=6.0)
    parser.add_argument("--noise-survey-sweeps", type=int, default=3)
    parser.add_argument("--pm-phase-deviations-rad", nargs=3, type=float,
                        default=(0.2, 0.5, 1.0),
                        metavar=("LOW", "MID", "HIGH"),
                        help="PM deviations matching the AM 20/50/80%% plan; default: 0.2 0.5 1.0 rad.")
    parser.add_argument("--pm-spans-khz", nargs="+", type=float,
                        default=(180.0, 250.0, 400.0),
                        help="Span values for each PM analyzer profile; default: 180 250 400 kHz.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = args.mode or choose_mode()

    base = InstrumentConfig(
        resource=args.resource, reset_on_connect=not args.no_reset,
        dataset_root=args.dataset_root,
    )
    print(f"Data directory: {Path(base.dataset_root).resolve()}")

    if mode == "fm_dop":
        summary = run_fm_dop_mode(base, args.dry_run, resume=not args.force)
    else:
        profile_names = tuple(args.profiles) if args.profiles else choose_profile_names()
        profiles = select_profiles(profile_names)
        print("Profiles: " + ", ".join(profile.name for profile in profiles))

        if mode == "fm":
            summary = run_fm_mode(base, profiles, args.dry_run, resume=not args.force)
        elif mode == "noise":
            summary = run_noise_mode(
                base, profiles, args.dry_run, resume=not args.force,
                count=args.noise_count,
                bands=parse_noise_bands(args.noise_bands),
                spans_khz=args.noise_spans_khz,
                station_guard_khz=args.noise_station_guard_khz,
                peak_threshold_db=args.noise_peak_threshold_db,
                floor_margin_db=args.noise_floor_margin_db,
                survey_sweeps=args.noise_survey_sweeps,
                requested_centers=args.noise_centers,
            )
        elif mode == "pm":
            summary = run_pm_mode(
                base, profiles, args.dry_run, resume=not args.force,
                deviations_rad=args.pm_phase_deviations_rad,
                spans_khz=args.pm_spans_khz,
            )
        else:
            summary = run_am_mode(base, profiles, args.dry_run, resume=not args.force)

    print_summary(summary)


if __name__ == "__main__":
    main()
