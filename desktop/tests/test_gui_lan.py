"""Exercise the existing Qt forms with a fake FSL, never an instrument."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from signal_scanner.fsl import FslAnalyzer
from signal_scanner.gui import MainWindow
from test_fsl import FakeFsl


class GuiLanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.window=MainWindow(Path(self.tmp.name))
        self.errors=[]
        self.window._error=self.errors.append
        self.fake=FakeFsl()
        self.analyzer=FslAnalyzer('offline-fixture',transport_factory=self.fake.factory)

    def tearDown(self):
        if self.window.job:
            self.window.job.control.cancel.set()
            self.window.job.wait(5000)
            self.app.processEvents()
        self.window.close()
        self.app.processEvents()
        self.tmp.cleanup()

    def test_form_connect_read_manual_record_unknown_and_group(self):
        self.window.transport.setCurrentText('PyVISA')
        self.window.address.setText('offline-fixture')
        with patch('signal_scanner.gui.create_analyzer',return_value=self.analyzer):
            self.window.toggle_connection()
        self.assertIs(self.window.analyzer,self.analyzer)
        self.assertFalse(self.window.panel.isEnabled())
        self.assertFalse(self.window.scene.isEnabled())
        self.assertIn('FSL-3',self.window.connection_info.text())
        self.assertIn('100.000000',self.window.connection_info.text())
        self.assertIn('Trace 1: Clear/Write',self.window.connection_info.text())
        self.assertIn('Sweep Count: 0',self.window.connection_info.text())
        self.window.settings_source.setCurrentIndex(1)
        self.window.capture_sweeps.setValue(2)
        self.window.mod_combo.setCurrentText('Неизвестно')
        self.window.observation_id.setText('station_A')
        self.window.variant_count.setValue(10)
        self.window.start_capture()
        # Twelve captures write CSV/JSON and atomically update the session.
        # Allow slow laboratory disks while retaining every completion assertion.
        deadline=time.monotonic()+60
        while self.window.job is not None and time.monotonic()<deadline:
            self.app.processEvents()
            QTest.qWait(10)
        self.assertIsNone(self.window.job)
        self.assertEqual(self.errors,[])
        self.assertIn('прибор FSL',self.window.capture_plot.title)
        run=Path(self.window.last_run)
        self.assertEqual(run.parent.parent.name,'measurements')
        labels=json.loads(next((run/'labels').glob('*.json')).read_text(encoding='utf-8'))
        self.assertEqual(labels['labels']['modulation'],'unknown')
        self.assertEqual(labels['group_id'],'station_A')
        self.assertEqual(labels['session_id'],self.window.capture_session_id)
        self.assertEqual(len(list((run/'samples').glob('*.csv'))),12)
        self.assertTrue(all(not cmd.startswith(('FREQ:', 'SWE:POIN ')) or '?' in cmd for _,cmd in self.fake.log))
        self.assertEqual(self.fake.raw['trace_mode'],'WRIT')
        self.assertEqual(self.fake.raw['sweep_count'],'0')

    def test_manual_panel_read_updates_form_and_trace_in_background(self):
        with patch('signal_scanner.gui.create_analyzer',return_value=self.analyzer):
            self.window.toggle_connection()
        self.fake.raw.update(trace_mode='MAXH',sweep_count='5',amplitude_unit='DBUV')
        self.fake.raw['center_hz']='36200000'
        self.window.read_front_panel()
        deadline=time.monotonic()+5
        while self.window.job is not None and time.monotonic()<deadline:
            self.app.processEvents()
            QTest.qWait(10)
        self.assertEqual(self.errors,[])
        self.assertEqual(self.window.capture_settings.settings().center_hz,36200000)
        self.assertIn('MAXH',self.window.capture_trace_info.text())
        self.assertIn('DBUV',self.window.capture_trace_info.text())

    def test_tcp_diagnostics_disables_acquisition_buttons(self):
        self.analyzer.diagnostics_only=True
        with patch('signal_scanner.gui.create_analyzer',return_value=self.analyzer):
            self.window.toggle_connection()
        self.assertIsNotNone(self.window.analyzer)
        self.assertFalse(self.window.record_button.isEnabled())
        self.assertFalse(self.window.preview_button.isEnabled())
        self.assertFalse(self.window.scan_start_button.isEnabled())
        self.assertIn('Только чтение TCP',self.window.connection_info.text())
