"""Bounded laboratory check using the application's real adapter and storage.

python -m signal_scanner.fsl_check --address 192.168.10.24 --capture 3 --wait 180
No reset, VPN changes, frequency changes or invented operator labels.
"""
import argparse
from dataclasses import asdict
from pathlib import Path
import socket
import threading
import time

from .domain import utc_now
from .fsl import FslAnalyzer
from .storage import Repository, application_home, write_json


def check(address, output, capture=0, wait=0):
    output = Path(output)
    report = {"started_at": utc_now(), "address": address, "status": "waiting",
              "hardware_acquisition_verified": False, "queries": {}}
    write_json(output / "fsl_check_latest.json", report)
    analyzer = FslAnalyzer(address)
    deadline = time.monotonic() + wait
    while True:
        try:
            # A single known host/port; no network scan or route changes.
            with socket.create_connection((address, 111), timeout=2):
                break
        except OSError as error:
            if time.monotonic() >= deadline:
                report.update(status="unreachable", error=str(error), finished_at=utc_now())
                write_json(output / "fsl_check_latest.json", report)
                return report
            time.sleep(1)
    try:
        identity, settings, raw = analyzer.probe()
        report.update(status="diagnosing", identity=identity, basic_settings=asdict(settings))
        write_json(output / "fsl_check_latest.json", report)
        for name, command in analyzer.QUERIES.items():
            if name == "frequency_axis":
                continue  # verified via explicit x values in exported trace, never guessed
            try:
                with analyzer.session() as transport:
                    value = analyzer.query(transport, command)
                report["queries"][name] = {"command": command, "value": value}
            except Exception as error:
                report["queries"][name] = {"command": command, "error": str(error),
                                           "notes": getattr(error, "__notes__", [])}
            write_json(output / "fsl_check_latest.json", report)
        failures = [name for name, value in report["queries"].items() if "error" in value]
        if failures:
            report.update(status="unsupported_queries", failed_queries=failures)
        elif capture:
            before = analyzer.snapshot_state()
            spectrum = analyzer.capture(capture, threading.Event())
            after = analyzer.snapshot_state()
            changed = {key: [before[key], after[key]] for key in before if before[key] != after[key]}
            if changed:
                raise RuntimeError("Не совпало состояние после сбора: " + str(changed))
            spectrum.validate()
            repository = Repository(output)
            run = repository.new_session("captures", simulated=False)
            repository.save_spectrum(run, "hardware_check", spectrum)
            session = {"purpose": "hardware_validation", "training_eligible": False,
                       "status": "complete", "simulated": False, "identity": identity,
                       "started_at": report["started_at"], "finished_at": utc_now(),
                       "before": before, "after": after, "restored": True,
                       "operator_labels": None, "sweeps": len(spectrum.sweeps_dbm)}
            write_json(run / "session.json", session)
            report.update(status="complete", hardware_acquisition_verified=True,
                          data_run=str(run), sweeps=len(spectrum.sweeps_dbm),
                          points=len(spectrum.frequency_hz), amplitude_unit=spectrum.amplitude_unit,
                          min_level=min(min(row) for row in spectrum.sweeps_dbm),
                          max_level=max(max(row) for row in spectrum.sweeps_dbm),
                          start_hz=spectrum.frequency_hz[0], stop_hz=spectrum.frequency_hz[-1],
                          settings_restored=True)
        else:
            report["status"] = "settings_verified"
    except Exception as error:
        report.update(status="error", error=str(error), command=analyzer.last_command,
                      diagnostic_context=getattr(error, "context", None),
                      notes=getattr(error, "__notes__", []))
    report["finished_at"] = utc_now()
    write_json(output / "fsl_check_latest.json", report)
    return report


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", default="192.168.10.24")
    parser.add_argument("--output", type=Path, default=application_home() / "data")
    parser.add_argument("--capture", type=int, choices=range(1, 6), default=0)
    parser.add_argument("--wait", type=int, default=0)
    args = parser.parse_args()
    print("FSL check: waiting for LAN; no network settings are modified.", flush=True)
    result = check(args.address, args.output, args.capture, min(300, max(0, args.wait)))
    print(json.dumps(result, ensure_ascii=True, indent=2), flush=True)
    return 0 if result["status"] in ("complete", "settings_verified") else 1


if __name__ == "__main__":
    raise SystemExit(main())
