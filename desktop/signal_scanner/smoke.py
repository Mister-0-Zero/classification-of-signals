"""Exercise the actual Qt widgets and worker threads without instrument access."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

from PySide6.QtCore import QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from .gui import MainWindow, configure_application


def main() -> int:
    output = Path(os.environ.get('SIGNAL_SCANNER_HOME', 'smoke_output')).resolve()
    output.mkdir(parents=True, exist_ok=True)
    app = QApplication(sys.argv)
    configure_application(app)
    window = MainWindow(output)
    window._error = lambda message: errors.append(message)
    errors = []
    window.show()
    state = {'step': 0, 'deadline': time.monotonic() + 45, 'passed': []}

    def done(failed: str | None = None):
        timer.stop()
        report = {'passed': state['passed'], 'error': failed, 'ui_errors': errors}
        (output / 'smoke_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        if window.job:
            window.job.control.cancel.set()
            window.job.wait(3000)
            window.job = None
        window.close()
        app.exit(1 if failed else 0)

    def advance():
        try:
            if time.monotonic() > state['deadline']:
                raise AssertionError('Qt scenario timed out')
            step = state['step']
            if step == 0:
                assert not window.scan_start_button.isEnabled()
                window.nav.setCurrentRow(2)
                window.connect_button.click()
                assert window.analyzer is not None
                window.analyzer.delay_scale = .05
                window.nav.setCurrentRow(0)
                window.scan_start.setValue(87)
                window.scan_stop.setValue(90)
                window.scan_sweeps.setValue(4)
                window.scan_start_button.click()
                assert window.job is not None
                assert not window.record_button.isEnabled()
                state['step'] = 1
            elif step == 1 and window.job is None:
                assert window.rows and window.table.rowCount() == len(window.rows)
                assert all(row['modulation'] == 'Неизвестно' for row in window.rows)
                assert window.progress.value() == 100
                assert window.scan_sweeps.value() == 4
                assert Path(window.last_run, 'session.json').exists()
                window.grab().save(str(output / 'scanner.png'))
                state['passed'].append('single scan, refinement, table, mutual exclusion, journal')
                window.nav.setCurrentRow(1)
                window.scene.setCurrentText('Один FM')
                window.variants_checkbox.setChecked(True)
                window.capture_sweeps.setValue(2)
                window.record_button.click()
                assert window.job is not None
                state['step'] = 2
            elif step == 2 and window.job is None:
                assert len(list(Path(window.last_run, 'samples').glob('*.csv'))) == 3
                window.grab().save(str(output / 'capture.png'))
                state['passed'].append('capture, three variants, persistence')
                window.nav.setCurrentRow(2)
                window.panel.center.setValue(89.5)
                window.apply_front_panel()
                window.nav.setCurrentRow(1)
                window.settings_source.setCurrentIndex(1)
                window.capture_settings.center.setValue(.001)
                window.variants_checkbox.setChecked(False)
                window.record_button.click()
                state['step'] = 3
            elif step == 3 and window.job is None:
                config = json.loads(next(Path(window.last_run, 'settings').glob('*.json')).read_text())
                assert config['settings']['center_hz'] == 89.5e6
                assert window.analyzer.settings.center_hz == 89.5e6
                state['passed'].append('manual panel readback and restoration')
                window.nav.setCurrentRow(0)
                window.scan_mode.setCurrentText('Непрерывный')
                window.scan_start_button.click()
                window.pause_button.click()
                assert window.job.control.paused.is_set()
                window.stop_button.click()
                state['step'] = 4
            elif step == 4 and window.job is None:
                state['passed'].append('continuous start, pause, stop, unlocked controls')
                window.nav.setCurrentRow(2)
                window.connect_button.click()
                window.transport.setCurrentText('TCP-сокет')
                window.connect_button.click()
                assert window.analyzer is None and errors
                assert 'имитация' in errors[-1]
                state['passed'].append('real transport blocked without network')
                done()
        except Exception as error:
            done(f'{type(error).__name__}: {error}')

    timer = QTimer()
    timer.timeout.connect(advance)
    timer.start(100)
    return app.exec()
