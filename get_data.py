from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import pyvisa
except ImportError:  # Plan inspection should work without VISA.
    pyvisa = None


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


@dataclass
class InstrumentConfig:
    resource: str = "TCPIP0::192.168.10.24::inst0::INSTR"
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
    generator_waveform: Optional[str]
    csv_path: str


@dataclass
class CollectionSummary:
    collected: int = 0
    skipped_existing: int = 0
    skipped_by_user: int = 0
    failed: int = 0


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
        "analyzer_profile": cfg.analyzer_profile,
        "source_name": cfg.source_name,
        "generator_level_dbm": cfg.generator_level_dbm,
        "generator_modulation_freq_khz": cfg.generator_modulation_freq_khz,
        "generator_depth_percent": cfg.generator_depth_percent,
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
        self.rm = None
        self.inst = None
        self.idn = "UNKNOWN"

    def connect(self) -> None:
        if pyvisa is None:
            raise RuntimeError("PyVISA is not installed. Run: python -m pip install pyvisa")
        self.rm = pyvisa.ResourceManager()
        self.inst = self.rm.open_resource(self.cfg.resource)
        self.inst.timeout = self.cfg.timeout_ms
        self.inst.read_termination = "\n"
        self.inst.write_termination = "\n"
        if self.cfg.reset_on_connect:
            self.write("*RST")
            time.sleep(2.0)
        self.write("*CLS")
        self.try_write("SYST:DISP:UPD OFF" if self.cfg.disable_display_updates else "SYST:DISP:UPD ON")
        self.write("INIT:CONT OFF")
        self.idn = self.query("*IDN?").strip()

    def close(self) -> None:
        if self.inst is not None:
            try:
                self.inst.close()
            except Exception:
                pass
        if self.rm is not None:
            try:
                self.rm.close()
            except Exception:
                pass

    def write(self, cmd: str) -> None:
        self.inst.write(cmd)

    def try_write(self, cmd: str) -> bool:
        try:
            self.inst.write(cmd)
            return True
        except Exception:
            return False

    def query(self, cmd: str) -> str:
        return self.inst.query(cmd)

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
            "center_hz": center_hz, "span_hz": span_hz,
            "start_hz": center_hz - span_hz / 2.0,
            "stop_hz": center_hz + span_hz / 2.0,
        }

    def acquire_single_sweep(self) -> List[float]:
        self.write("INIT;*WAI")
        ascii_cmds = (
            "FORM ASC; TRAC? TRACE1", "FORM ASC; TRAC:DATA? TRACE1",
            "FORM ASC; TRACE:DATA? TRACE1", "FORM ASC; TRAC1? TRACE1",
        )
        if not self.cfg.prefer_binary:
            for cmd in ascii_cmds:
                try:
                    values = parse_ascii_trace(self.query(cmd))
                    if values:
                        return values
                except Exception:
                    continue
        for cmd in (
            "FORM REAL,32; TRAC? TRACE1", "FORM REAL,32; TRAC:DATA? TRACE1",
            "FORM REAL,32; TRACE:DATA? TRACE1",
        ):
            try:
                self.write("INIT;*WAI")
                self.write(cmd)
                values = self.inst.query_binary_values("", datatype="f", is_big_endian=False, container=list)
                if values:
                    return [float(value) for value in values]
            except Exception:
                continue
        for cmd in ("TRAC? TRACE1", "TRAC:DATA? TRACE1", "TRACE:DATA? TRACE1"):
            try:
                values = parse_ascii_trace(self.query(cmd))
                if values:
                    return values
            except Exception:
                continue
        raise RuntimeError("Failed to read trace data; SCPI trace command may need adjustment.")

    def acquire_sample(self) -> Tuple[List[float], List[List[float]]]:
        axis = build_frequency_axis(
            self.cfg.center_freq_mhz * 1e6, self.cfg.span_khz * 1e3, self.cfg.sweep_points
        )
        sweeps: List[List[float]] = []
        for sweep_idx in range(self.cfg.sweeps_per_sample):
            values = self.acquire_single_sweep()
            if len(values) < self.cfg.sweep_points:
                raise RuntimeError(f"Trace has {len(values)} points, expected {self.cfg.sweep_points}")
            sweeps.append(values[: self.cfg.sweep_points])
            if sweep_idx < self.cfg.sweeps_per_sample - 1 and self.cfg.inter_sweep_delay_s > 0:
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


def collect_configured_run(collector: RSFSLCollector, cfg: InstrumentConfig) -> None:
    group = make_sample_group_name(cfg)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path(cfg.dataset_root) / cfg.class_name / group / run_ts
    samples_dir, metadata_csv = run_dir / "samples", run_dir / "metadata.csv"
    ensure_dir(samples_dir)
    applied = collector.configure(cfg)
    print(
        f"  Analyzer: center={cfg.center_freq_mhz:g} MHz, span={cfg.span_khz:g} kHz, "
        f"RBW={cfg.rbw_hz:g} Hz, VBW={cfg.vbw_hz:g} Hz, "
        f"points={cfg.sweep_points}, slices={cfg.sweeps_per_sample}"
    )
    save_run_config(run_dir, cfg, collector.idn, group)
    for sample_idx in range(cfg.samples_per_run):
        sample_id = f"{group}_{run_ts}_{sample_idx:04d}"
        sample_csv = samples_dir / f"{sample_id}.csv"
        print(f"  [{sample_idx + 1}/{cfg.samples_per_run}] Collecting {sample_id} ...")
        started_at = utc_now_str()
        axis, matrix = collector.acquire_sample()
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
    while True:
        answer = input("Режим [1/2]: ").strip().lower()
        if answer in {"1", "fm"}:
            return "fm"
        if answer in {"2", "am"}:
            return "am"
        print("Введите 1 или 2.")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect FM or AM datasets with an R&S FSL analyzer.")
    parser.add_argument("--mode", choices=("fm", "am"))
    parser.add_argument("--profiles", nargs="+", choices=tuple(ANALYZER_PROFILES))
    parser.add_argument("--resource", default=InstrumentConfig.resource)
    parser.add_argument("--dataset-root", default="data")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without VISA connection.")
    parser.add_argument("--force", action="store_true", help="Do not skip existing captures.")
    parser.add_argument("--no-reset", action="store_true", help="Do not reset analyzer on connect.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = args.mode or choose_mode()
    profile_names = tuple(args.profiles) if args.profiles else choose_profile_names()
    profiles = select_profiles(profile_names)
    base = InstrumentConfig(
        resource=args.resource, reset_on_connect=not args.no_reset,
        dataset_root=args.dataset_root,
    )
    print(f"Data directory: {Path(base.dataset_root).resolve()}")
    print("Profiles: " + ", ".join(profile.name for profile in profiles))
    if mode == "fm":
        summary = run_fm_mode(base, profiles, args.dry_run, resume=not args.force)
    else:
        summary = run_am_mode(base, profiles, args.dry_run, resume=not args.force)
    print_summary(summary)


if __name__ == "__main__":
    main()
