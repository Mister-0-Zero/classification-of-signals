from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from signal_scanner.domain import AnalyzerSettings, ScanPlan
from signal_scanner.instruments import Simulator
from signal_scanner.services import Control, scan
from signal_scanner.storage import Repository


class ClassificationTests(unittest.TestCase):
    def test_both_modes_classify_and_persist_the_acquired_candidate(self):
        for mode, count in [("fast", 1), ("quality", 7)]:
            with self.subTest(mode=mode), TemporaryDirectory() as tmp:
                events = []
                analyzer = Simulator(delay_scale=0)
                before = analyzer.read_settings()
                plan = ScanPlan(87e6, 90e6, AnalyzerSettings(), classification_mode=mode)
                scan(analyzer, plan, Repository(Path(tmp)), Control(), events.append)
                details = [e for e in events if e["type"] == "detail"]
                self.assertTrue(details)
                for event in details:
                    self.assertEqual(len(event["spectrum"].sweeps_dbm), count)
                    self.assertEqual(event["spectrum"].settings.points, 501)
                    self.assertIn(event["detections"][0].modulation, ("АМ", "ЧМ", "Шум"))
                run = Path(events[-1]["run"])
                saved = json.loads((run / "findings.json").read_text(encoding="utf-8"))
                self.assertEqual(saved, events[-1]["rows"])
                self.assertTrue(any(row["refined"] and row["modulation"] != "Неизвестно" for row in saved))
                self.assertEqual(analyzer.read_settings(), before)
