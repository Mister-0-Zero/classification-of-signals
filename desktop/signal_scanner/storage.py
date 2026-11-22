from __future__ import annotations

from dataclasses import asdict
import csv
import json
import os
from pathlib import Path
import sys
import uuid

from .domain import CaptureLabels, Spectrum, utc_now


def application_home() -> Path:
    override = os.environ.get("SIGNAL_SCANNER_HOME")
    if override:
        return Path(override).expanduser().resolve()
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "workspace"
    return Path(__file__).resolve().parents[2] / ".local" / "scanner_workspace"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class Repository:
    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()

    def new_session(self, category: str, *, simulated: bool = True) -> Path:
        if category not in {"scans", "captures"}:
            raise ValueError("Unknown session category")
        stamp = utc_now().replace(":", "").replace("+", "_")
        path = self.root / ("simulation" if simulated else "measurements") / category / f"{stamp}_{uuid.uuid4().hex[:8]}"
        path.mkdir(parents=True, exist_ok=False)
        return path

    def save_spectrum(self, run: Path, name: str, spectrum: Spectrum) -> None:
        spectrum.validate()
        expected = "simulation" if spectrum.simulated else "measurements"
        if run.parent.parent != self.root / expected:
            raise ValueError("Происхождение спектра не совпало с веткой хранения.")
        csv_path = run / "samples" / f"{name}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            amplitude_column = "amplitude_" + spectrum.amplitude_unit.lower()
            minima = spectrum.measurement_metadata.get("autopeak_min_sweeps")
            extra_header = ["amplitude_min_" + spectrum.amplitude_unit.lower()] if minima is not None else []
            writer.writerow(["sample_id", "class_name", "sample_group_name", "sweep_idx",
                             "point_idx", "frequency_hz", amplitude_column, "simulated", "amplitude_unit"] + extra_header)
            for sweep_index, row in enumerate(spectrum.sweeps_dbm):
                for point_index, (freq, level) in enumerate(zip(spectrum.frequency_hz, row)):
                    extra_values = [f"{minima[sweep_index][point_index]:.9g}"] if minima is not None else []
                    writer.writerow([name, "UNLABELED", run.name, sweep_index,
                                     point_index, f"{freq:.6f}", f"{level:.9g}", spectrum.simulated, spectrum.amplitude_unit] + extra_values)
        write_json(run / "settings" / f"{name}.json", {
            "schema_version": 3, "simulated": spectrum.simulated, "captured_at": spectrum.captured_at,
            "settings": asdict(spectrum.settings), "sweeps": len(spectrum.sweeps_dbm),
            "frequency_axis": "linear", "amplitude_unit": spectrum.amplitude_unit,
            "amplitude_column": amplitude_column,
            "measurement_metadata": spectrum.measurement_metadata,
        })

    def save_capture(self, run: Path, index: int, spectrum: Spectrum,
                     labels: CaptureLabels, *, variant: bool, scene: str) -> None:
        labels.validate()
        name = f"capture_{index:03}"
        self.save_spectrum(run, name, spectrum)
        write_json(run / "labels" / f"{name}.json", {
            "schema_version": 2, "simulated": spectrum.simulated, "group_id": labels.observation_id or run.name,
            "labels": asdict(labels), "labels_need_review": variant or labels.label_certainty == "uncertain",
            "signal_present": labels.signal_present, "training_eligible": labels.training_eligible,
            "variant": variant, "simulation_scene": scene if spectrum.simulated else None,
            "session_id": labels.session_id or run.name, "observation_id": labels.observation_id or run.name,
            "run_id": f"{run.name}_{name}", "variant_id": name,
            "annotation_source": "operator", "signals": [],
            "localization_status": "unannotated",
        })

    def append_event(self, run: Path, event: dict) -> None:
        with (run / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def load_preferences(self) -> dict:
        try:
            value = json.loads((self.root / "preferences.json").read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            return {}

    def save_preferences(self, preferences: dict) -> None:
        write_json(self.root / "preferences.json", preferences)
