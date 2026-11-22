import math
import unittest

from signal_scanner.fsl_export import parse_trace_export


class TraceFileTests(unittest.TestCase):
    def sample(self, rows="100;10;2\n200;11;3", count=2, unit="dBµV", spacing="LIN"):
        return (f"Type;FSL;\nx-Axis;{spacing};\nTrace 1:;;\nx-Unit;Hz;\ny-Unit;{unit};\nValues;{count};\n" + rows).encode("latin-1")

    def test_native_units_and_both_autopeak_columns(self):
        result=parse_trace_export(self.sample())
        self.assertEqual(result['amplitude_unit'],'DBUV')
        self.assertEqual(result['frequency_hz'],(100.,200.))
        self.assertEqual(result['amplitudes'],(10.,11.))
        self.assertEqual(result['autopeak_min'],(2.,3.))

    def test_decimal_comma_and_trailing_semicolon(self):
        result=parse_trace_export(self.sample(rows='100;10,5;\n200;11,25;'))
        self.assertEqual(result['amplitudes'],(10.5,11.25))
        self.assertEqual(result['autopeak_min'],())

    def test_corrupt_or_truncated_exports_rejected(self):
        for rows,count in [('100;10;2\n200;11;',2), ('100;nan\n200;11',2),
                           ('100;10\n200;11',3), ('100;10\n100;11',2),
                           ('100;10;2;4\n200;11;3;5',2)]:
            with self.subTest(rows=rows):
                with self.assertRaises(ValueError):
                    parse_trace_export(self.sample(rows,count))

    def test_unknown_units_and_missing_axis_rejected(self):
        for unit,axis in [('V','LIN'),('dBm','')]:
            with self.assertRaises(ValueError):
                parse_trace_export(self.sample(unit=unit,spacing=axis))
