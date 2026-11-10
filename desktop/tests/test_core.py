from dataclasses import replace
import json
from pathlib import Path
import tempfile
import threading
import unittest

from signal_scanner.detection import ThresholdDetector
from signal_scanner.domain import AnalyzerSettings, CaptureLabels, ScanPlan
from signal_scanner.instruments import Cancelled, Simulator, create_analyzer
from signal_scanner.services import Control, FindingRegistry, collect, scan
from signal_scanner.storage import Repository


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.repository = Repository(Path(self.temporary.name))
        self.analyzer = Simulator(delay_scale=0)

    def tearDown(self):
        self.temporary.cleanup()

    def plan(self, **kwargs):
        args = dict(start_hz=87e6, stop_hz=90e6, settings=AnalyzerSettings(span_hz=1e6), refine=False)
        args.update(kwargs)
        return ScanPlan(**args)

    def test_windows_cover_range_with_overlap_and_last_edge(self):
        plan = self.plan()
        windows = plan.windows()
        self.assertEqual(windows[0][0] - windows[0][1] / 2, plan.start_hz)
        self.assertEqual(windows[-1][0] + windows[-1][1] / 2, plan.stop_hz)
        for (center, span), (next_center, next_span) in zip(windows, windows[1:]):
            self.assertLessEqual(next_center - next_span / 2, center + span / 2)
        self.assertEqual(len(windows), len(set(windows)))

    def test_window_narrower_than_requested_span(self):
        self.assertEqual(self.plan(stop_hz=87.1e6).windows(), ((87.05e6, 100_000),))

    def test_large_span_is_validated_against_actual_scan_center(self):
        plan = self.plan(start_hz=1e9, stop_hz=1.6e9,
                         settings=AnalyzerSettings(center_hz=100.1e6, span_hz=300e6))
        self.assertEqual(plan.windows()[0], (1.15e9, 300e6))

    def test_invalid_range_and_excessive_windows(self):
        with self.assertRaises(ValueError): self.plan(stop_hz=1).windows()
        with self.assertRaises(ValueError): self.plan(overlap=.9).windows()
        with self.assertRaises(ValueError):
            self.plan(start_hz=1e6, stop_hz=1e12, settings=AnalyzerSettings(span_hz=1000, rbw_hz=100, vbw_hz=100)).windows()

    def test_no_hardware_transport_can_connect(self):
        for name in ('TCP-сокет', 'PyVISA'):
            with self.assertRaisesRegex(ValueError, 'только имитация'):
                create_analyzer(name, '127.0.0.1')

    def test_empty_scene_has_no_detection(self):
        self.analyzer.set_scene('Нет сигнала')
        spectrum = self.analyzer.capture(3, threading.Event())
        self.assertEqual(ThresholdDetector().detect(spectrum), [])

    def test_signal_detected_without_exposing_simulation_truth(self):
        self.analyzer.set_scene('Один FM')
        spectrum = self.analyzer.capture(3, threading.Event())
        findings = ThresholdDetector().detect(spectrum)
        self.assertTrue(findings)
        self.assertTrue(all(x.modulation == 'Неизвестно' for x in findings))
        self.assertLess(abs(findings[0].center_hz - self.analyzer.settings.center_hz), 100_000)

    def test_distinct_signals_are_localized(self):
        self.analyzer.set_scene('Несколько')
        findings = ThresholdDetector().detect(self.analyzer.capture(3, threading.Event()))
        self.assertGreaterEqual(len(findings), 2)

    def test_cancelled_capture(self):
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(Cancelled): self.analyzer.capture(3, cancel)

    def test_scan_saves_journal_but_not_raw_by_default(self):
        initial = self.analyzer.settings
        events = []
        scan(self.analyzer, self.plan(), self.repository, Control(), events.append)
        finished = events[-1]
        run = Path(finished['run'])
        self.assertEqual(finished['status'], 'complete')
        self.assertTrue((run / 'events.jsonl').exists())
        self.assertFalse((run / 'samples').exists())
        self.assertEqual(self.analyzer.settings, initial)
        self.assertEqual(json.loads((run / 'session.json').read_text())['passes_complete'], 1)

    def test_raw_saving_and_no_log(self):
        events = []
        scan(self.analyzer, self.plan(stop_hz=87.5e6), self.repository, Control(), events.append,
             save_log=False, save_raw=True)
        run = Path(events[-1]['run'])
        self.assertTrue(list((run / 'samples').glob('*.csv')))
        self.assertFalse((run / 'events.jsonl').exists())

    def test_disabled_saving_makes_no_session(self):
        events = []
        scan(self.analyzer, self.plan(), self.repository, Control(), events.append, save_log=False, save_raw=False)
        self.assertEqual(events[-1]['run'], '')
        self.assertEqual(list(Path(self.temporary.name).iterdir()), [])

    def test_refinement_performs_additional_measurements(self):
        events = []
        plan = self.plan(refine=True)
        scan(self.analyzer, plan, self.repository, Control(), events.append)
        self.assertTrue(any(x['type'] == 'detail' for x in events))
        self.assertGreater(events[-1]['captures'], len(plan.windows()))

    def test_stop_preserves_partial_coverage_and_restores_settings(self):
        initial = self.analyzer.settings
        control, events = Control(), []
        def emit(event):
            events.append(event)
            if event['type'] == 'overview': control.cancel.set()
        scan(self.analyzer, self.plan(), self.repository, control, emit)
        self.assertEqual(events[-1]['status'], 'stopped')
        summary = json.loads((Path(events[-1]['run']) / 'session.json').read_text())
        self.assertEqual(len(summary['coverage_last_pass_hz']), 1)
        self.assertEqual(summary['passes_complete'], 0)
        self.assertEqual(self.analyzer.settings, initial)

    def test_continuous_scan_stops_after_two_passes(self):
        control, events = Control(), []
        def emit(event):
            events.append(event)
            if event['type'] == 'pass_complete' and event['pass'] == 2: control.cancel.set()
        scan(self.analyzer, self.plan(continuous=True), self.repository, control, emit)
        self.assertEqual(events[-1]['passes'], 2)

    def test_registry_does_not_invent_absence_after_incomplete_pass(self):
        self.analyzer.set_scene('Один FM')
        found = ThresholdDetector().detect(self.analyzer.capture(2, threading.Event()))[0]
        registry = FindingRegistry()
        registry.observe(found, 1, False)
        registry.begin_pass()
        self.assertEqual(registry.rows[0]['status'], 'Ожидает проверки')
        registry.finish_pass(2)
        self.assertEqual(registry.rows[0]['status'], 'Ранее обнаружен')
        registry.observe(found, 3, True)
        self.assertEqual(len(registry.rows), 1)
        self.assertTrue(registry.rows[0]['refined'])

    def test_variants_restore_settings_and_share_group(self):
        initial = self.analyzer.settings
        labels = CaptureLabels('one', 'FM')
        events = []
        collect(self.analyzer, self.repository, Control(), events.append, labels,
                replace(initial, center_hz=89e6), 2, variants=True)
        run = Path(events[-1]['run'])
        records = [json.loads(p.read_text()) for p in sorted((run / 'labels').glob('*.json'))]
        self.assertEqual(len(records), 3)
        self.assertFalse(records[0]['labels_need_review'])
        self.assertTrue(all(r['labels_need_review'] for r in records[1:]))
        self.assertEqual(len({r['group_id'] for r in records}), 1)
        self.assertEqual(self.analyzer.settings, initial)
        self.assertTrue(all(r['simulated'] for r in records))

    def test_manual_capture_uses_current_panel_not_form(self):
        manual = replace(self.analyzer.settings, center_hz=89.5e6, rbw_hz=3000)
        self.analyzer.apply_settings(manual)
        events = []
        collect(self.analyzer, self.repository, Control(), events.append,
                CaptureLabels('one', 'AM', settings_source='manual'), AnalyzerSettings(), 1, read_manual=True)
        spectrum = next(e['spectrum'] for e in events if e['type'] == 'capture')
        self.assertEqual(spectrum.settings, manual)
        self.assertEqual(self.analyzer.settings, manual)

    def test_manual_capture_ignores_invalid_unused_form(self):
        events = []
        invalid = AnalyzerSettings(center_hz=1000, span_hz=1e6)
        collect(self.analyzer, self.repository, Control(), events.append,
                CaptureLabels('one', 'FM', settings_source='manual'), invalid, 1, read_manual=True)
        self.assertEqual(events[-1]['status'], 'complete')

    def test_invalid_labels_create_no_files(self):
        with self.assertRaises(ValueError):
            collect(self.analyzer, self.repository, Control(), lambda _: None,
                    CaptureLabels('none', 'AM'), AnalyzerSettings(), 2)
        self.assertEqual(list(Path(self.temporary.name).iterdir()), [])

    def test_capture_error_records_status_and_restores(self):
        initial = self.analyzer.settings
        class Broken(Simulator):
            def capture(self, sweeps, cancel):
                raise OSError('synthetic acquisition failure')
        analyzer = Broken(delay_scale=0)
        events = []
        with self.assertRaises(OSError):
            collect(analyzer, self.repository, Control(), events.append, CaptureLabels('one', 'FM'),
                    replace(initial, center_hz=90e6), 1)
        self.assertEqual(analyzer.settings, initial)
        self.assertEqual(events[-1]['status'], 'error')

    def test_cancel_while_paused(self):
        control = Control()
        control.paused.set()
        control.cancel.set()
        with self.assertRaises(Cancelled): control.checkpoint()


if __name__ == '__main__':
    unittest.main()
