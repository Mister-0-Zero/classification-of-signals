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

    def new_session(self, category: str) -> Path:
        if category not in {"scans", "captures"}:
            raise ValueError("Unknown session category")
        stamp = utc_now().replace(":", "").replace("+", "_")
        path = self.root / "simulation" / category / f"{stamp}_{uuid.uuid4().hex[:8]}"
        path.mkdir(parents=True, exist_ok=False)
        return path

    def save_spectrum(self, run: Path, name: str, spectrum: Spectrum) -> None:
        csv_path = run / "samples" / f"{name}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_id", "class_name", "sample_group_name", "sweep_idx",
                             "point_idx", "frequency_hz", "amplitude_dbm", "simulated"])
            for sweep_index, row in enumerate(spectrum.sweeps_dbm):
                for point_index, (freq, level) in enumerate(zip(spectrum.frequency_hz, row)):
                    writer.writerow([name, "UNLABELED", run.name, sweep_index,
                                     point_index, f"{freq:.3f}", f"{level:.6f}", True])
        write_json(run / "settings" / f"{name}.json", {
            "schema_version": 1, "simulated": True, "captured_at": spectrum.captured_at,
            "settings": asdict(spectrum.settings), "sweeps": len(spectrum.sweeps_dbm),
            "frequency_axis": "linear", "amplitude_unit": "dBm",
        })

    def save_capture(self, run: Path, index: int, spectrum: Spectrum,
                     labels: CaptureLabels, *, variant: bool, scene: str) -> None:
        labels.validate()
        name = f"capture_{index:03}"
        self.save_spectrum(run, name, spectrum)
        write_json(run / "labels" / f"{name}.json", {
            "schema_version": 1, "simulated": True, "group_id": run.name,
            "labels": asdict(labels), "labels_need_review": variant,
            "variant": variant, "simulation_scene": scene,
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
