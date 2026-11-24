from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import threading
import time
from typing import Callable

from .detection import ThresholdDetector
from .domain import AnalyzerSettings, CaptureLabels, Detection, ScanPlan, utc_now
from .instruments import Analyzer, Cancelled
from .storage import Repository, write_json


def finish_device(analyzer, state, changed, summary):
    errors = []
    if changed:
        try:
            analyzer.restore_state(state)
        except Exception as error:
            errors.append(error)
    try:
        analyzer.release_local()
    except Exception as error:
        errors.append(error)
    if errors:
        summary.update(status="error", cleanup_errors=[str(e) for e in errors])
    return errors[0] if errors else None


class Control:
    def __init__(self):
        self.cancel = threading.Event()
        self.paused = threading.Event()

    def checkpoint(self) -> None:
        while self.paused.is_set():
            if self.cancel.wait(.05):
                raise Cancelled()
        if self.cancel.is_set():
            raise Cancelled()


class FindingRegistry:
    def __init__(self):
        self.rows: list[dict] = []

    def begin_pass(self) -> None:
        for row in self.rows:
            row["status"] = "Ожидает проверки"

    def observe(self, finding: Detection, pass_number: int, refined: bool, *, simulated: bool = True) -> int:
        match = None
        for row in self.rows:
            distance = abs(row["center_hz"] - finding.center_hz)
            tolerance = max(50_000, (row["stop_hz"] - row["start_hz"]) * .5,
                            (finding.stop_hz - finding.start_hz) * .5)
            if distance <= tolerance:
                match = row
                break
        if match is None:
            match = {"id": len(self.rows) + 1, "first_seen": utc_now(), "observations": 0}
            self.rows.append(match)
        if not refined and match.get("last_pass") == pass_number and match.get("refined"):
            return match["id"]
        match.update(asdict(finding), last_seen=utc_now(), last_pass=pass_number,
                     status="Обнаружен", refined=refined,
                     observations=match["observations"] + 1, simulated=simulated)

        return match["id"]

    def finish_pass(self, pass_number: int) -> None:
        for row in self.rows:
            if row["last_pass"] != pass_number:
                row["status"] = "Ранее обнаружен"

    def snapshot(self) -> list[dict]:
        return [dict(row) for row in sorted(self.rows, key=lambda row: row["center_hz"])]


def scan(analyzer: Analyzer, plan: ScanPlan, repository: Repository,
         control: Control, emit: Callable[[dict], None], *, save_log: bool = True,
         save_raw: bool = False) -> None:
    windows = plan.windows()
    classifier = None
    if plan.refine:
        from models.modulation.scanner import ScannerClassifier
        classifier = ScannerClassifier(plan.classification_mode)
    analyzer.read_settings()
    initial_state = analyzer.snapshot_state()
    detector = ThresholdDetector()
    registry = FindingRegistry()
    started = time.perf_counter()
    run = repository.new_session("scans", simulated=analyzer.simulated) if save_log or save_raw else None
    summary = {"schema_version": 2, "simulated": analyzer.simulated, "status": "running",
               "started_at": utc_now(), "plan": asdict(plan),
               "algorithm": "ThresholdDetector", "model": plan.classification_mode if plan.refine else None,
               "save_log": save_log, "save_raw": save_raw}
    operation_error = None
    captures = 0
    pass_number = 0
    complete_passes = 0
    coverage: list[tuple[float, float]] = []
    try:
        if run:
            write_json(run / "session.json", summary)
        while True:
            control.checkpoint()
            pass_number += 1
            registry.begin_pass()
            coverage = []
            if hasattr(analyzer, "pass_number"):
                analyzer.pass_number = pass_number
            emit({"type": "pass_start", "pass": pass_number, "rows": registry.snapshot(),
                  "total": len(windows), "run": str(run) if run else ""})
            refined_ids = set()
            for index, (center, span) in enumerate(windows):
                control.checkpoint()
                requested = replace(plan.settings, center_hz=center, span_hz=span)
                analyzer.apply_settings(requested)
                spectrum = analyzer.capture(plan.sweeps, control.cancel)
                captures += 1
                found = detector.detect(spectrum, plan.threshold_db)
                coverage.append((spectrum.frequency_hz[0], spectrum.frequency_hz[-1]))
                if run and save_raw:
                    repository.save_spectrum(run, f"p{pass_number:04}_w{index:05}_overview", spectrum)
                ids = [registry.observe(finding, pass_number, False, simulated=analyzer.simulated)
                       for finding in found]
                emit({"type": "overview", "spectrum": spectrum, "detections": found,
                      "rows": registry.snapshot(), "pass": pass_number,
                      "index": index + 1, "total": len(windows),
                      "elapsed": time.perf_counter() - started})
                if run and save_log:
                    repository.append_event(run, {"type": "overview", "time": spectrum.captured_at,
                        "pass": pass_number, "window": index, "settings": asdict(spectrum.settings),
                        "detections": [asdict(v) for v in found], "simulated": analyzer.simulated})
                if not plan.refine:
                    continue
                for finding, row_id in zip(found, ids):
                    if row_id in refined_ids:
                        continue
                    control.checkpoint()
                    refined_ids.add(row_id)
                    detail_span = 250_000
                    lower, upper = analyzer.frequency_limits_hz
                    detail_center = min(max(finding.center_hz, max(1, lower) + detail_span / 2),
                                        upper - detail_span / 2)
                    detail = replace(requested, center_hz=detail_center, span_hz=detail_span, points=501,
                                     rbw_hz=min(requested.rbw_hz, 3000),
                                     vbw_hz=min(requested.vbw_hz, 3000))
                    analyzer.apply_settings(detail)
                    detailed_spectrum = analyzer.capture(classifier.sweeps, control.cancel)
                    captures += 1
                    label, origin = classifier.predict(detailed_spectrum)
                    classified = replace(finding, modulation=label, origin=origin)
                    candidates = [classified]
                    registry.observe(classified, pass_number, True, simulated=analyzer.simulated)
                    emit({"type": "detail", "spectrum": detailed_spectrum,
                          "detections": candidates, "rows": registry.snapshot(),
                          "pass": pass_number, "elapsed": time.perf_counter() - started})
                    if run and save_raw:
                        repository.save_spectrum(run, f"p{pass_number:04}_w{index:05}_detail{captures}", detailed_spectrum)
                    if run and save_log:
                        repository.append_event(run, {"type": "detail", "time": detailed_spectrum.captured_at,
                            "pass": pass_number, "settings": asdict(detailed_spectrum.settings),
                            "detections": [asdict(v) for v in candidates], "simulated": analyzer.simulated})
            registry.finish_pass(pass_number)
            complete_passes += 1
            emit({"type": "pass_complete", "pass": pass_number, "rows": registry.snapshot(),
                  "elapsed": time.perf_counter() - started})
            if run and save_log:
                write_json(run / "findings.json", registry.snapshot())
            if not plan.continuous:
                break
            if control.cancel.wait(.2):
                raise Cancelled()
        summary["status"] = "complete"
    except Cancelled:
        summary["status"] = "stopped"
    except Exception as error:
        operation_error = error
        summary.update(status="error", error=str(error))
        raise
    finally:
        cleanup_error = finish_device(analyzer, initial_state, True, summary)
        summary.update(finished_at=utc_now(), elapsed_s=time.perf_counter() - started,
                       passes_complete=complete_passes, captures=captures,
                       coverage_last_pass_hz=coverage)
        if run:
            write_json(run / "session.json", summary)
            if save_log:
                write_json(run / "findings.json", registry.snapshot())
        emit({"type": "scan_finished", "status": summary["status"],
              "rows": registry.snapshot(), "run": str(run) if run else "",
              "elapsed": summary["elapsed_s"], "captures": captures,
              "passes": complete_passes})
        if cleanup_error and operation_error is None:
            raise RuntimeError("Не удалось восстановить прибор: " + str(cleanup_error)) from cleanup_error


def collect(analyzer: Analyzer, repository: Repository, control: Control,
            emit: Callable[[dict], None], labels: CaptureLabels, settings: AnalyzerSettings,
            sweeps: int, *, read_manual: bool = False, variants: bool = False,
            scene: str = "Эфир") -> None:
    labels.validate()
    if not 1 <= sweeps <= 50:
        raise ValueError("Количество срезов: от 1 до 50.")
    initial = analyzer.read_settings()
    initial_state = analyzer.snapshot_state()
    baseline = initial if read_manual else settings
    baseline.validate()
    configurations = [baseline]
    if variants:
        for rbw, vbw in ((3000, 3000), (10000, 3000)):
            candidate = replace(baseline, rbw_hz=min(rbw, baseline.span_hz),
                                vbw_hz=min(vbw, baseline.span_hz))
            if candidate not in configurations:
                configurations.append(candidate)
    run = repository.new_session("captures", simulated=analyzer.simulated)
    session = {"schema_version": 2, "simulated": analyzer.simulated, "status": "running",
               "started_at": utc_now(), "labels": asdict(labels), "scene": scene if analyzer.simulated else None,
               "settings_source": ("simulated_front_panel" if analyzer.simulated else "instrument_front_panel") if read_manual else "application",
               "initial_settings": asdict(initial), "initial_state": initial_state if isinstance(initial_state, dict) else asdict(initial_state), "captured": 0,
               "signal_present": labels.signal_present, "training_eligible": labels.training_eligible}
    changed = False
    operation_error = None
    try:
        write_json(run / "session.json", session)
        for index, cfg in enumerate(configurations):
            control.checkpoint()
            if not (read_manual and index == 0):
                changed = True  # a partial or rejected write can still alter state
                analyzer.apply_settings(cfg)
            spectrum = analyzer.capture(sweeps, control.cancel)
            repository.save_capture(run, index, spectrum, labels, variant=index > 0, scene=scene)
            session["captured"] += 1
            emit({"type": "capture", "spectrum": spectrum, "index": index + 1,
                  "total": len(configurations), "run": str(run), "variant": index > 0})
        session["status"] = "complete"
    except Cancelled:
        session["status"] = "stopped"
    except Exception as error:
        operation_error = error
        session.update(status="error", error=str(error), diagnostic_context=getattr(error, "context", None),
                       error_notes=getattr(error, "__notes__", []))
        raise
    finally:
        cleanup_error = finish_device(analyzer, initial_state, changed, session)
        session["finished_at"] = utc_now()
        write_json(run / "session.json", session)
        emit({"type": "capture_finished", "status": session["status"],
              "run": str(run), "captured": session["captured"]})
        if cleanup_error and operation_error is None:
            raise RuntimeError("Не удалось восстановить прибор: " + str(cleanup_error)) from cleanup_error
