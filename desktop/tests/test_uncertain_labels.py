import json
from pathlib import Path
import tempfile
import unittest

from PySide6.QtWidgets import QApplication
from signal_scanner.domain import AnalyzerSettings, CaptureLabels
from signal_scanner.gui import MainWindow
from signal_scanner.instruments import Simulator
from signal_scanner.services import collect, Control
from signal_scanner.capture_series import collect_series, SeriesPlan
from signal_scanner.storage import Repository


class UncertainLabelsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])

    def test_both_choices_keep_count_and_uncertainty(self):
        with tempfile.TemporaryDirectory() as folder:
            window=MainWindow(Path(folder))
            try:
                for choice,count in [('Сомнительно один','one'),('Сомнительно несколько','multiple')]:
                    window.count_combo.setCurrentText(choice)
                    labels=window._labels()
                    labels.validate()
                    self.assertEqual(labels.signal_count,count)
                    self.assertEqual(labels.label_certainty,'uncertain')
                    self.assertIsNone(labels.signal_present)
                    self.assertFalse(labels.training_eligible)
            finally:
                window.close()
                self.app.processEvents()

    def test_uncertain_labels_persist_for_single_capture_and_series(self):
        for series in (False,True):
            with self.subTest(series=series),tempfile.TemporaryDirectory() as folder:
                repository=Repository(Path(folder))
                events=[]
                labels=CaptureLabels('multiple','unknown',label_certainty='uncertain')
                analyzer=Simulator(delay_scale=0)
                if series:
                    collect_series(analyzer,repository,Control(),events.append,labels,1,SeriesPlan(variants=10,seed=1))
                else:
                    collect(analyzer,repository,Control(),events.append,labels,AnalyzerSettings(),1,read_manual=True)
                run=Path(events[-1]['run'])
                session=json.loads((run/'session.json').read_text(encoding='utf-8'))
                self.assertIsNone(session['signal_present'])
                self.assertFalse(session['training_eligible'])
                for file in (run/'labels').glob('*.json'):
                    data=json.loads(file.read_text(encoding='utf-8'))
                    self.assertEqual(data['labels']['signal_count'],'multiple')
                    self.assertEqual(data['labels']['label_certainty'],'uncertain')
                    self.assertTrue(data['labels_need_review'])
                    self.assertIsNone(data['signal_present'])
                    self.assertFalse(data['training_eligible'])
