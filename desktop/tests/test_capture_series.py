from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest

from signal_scanner.capture_series import SeriesPlan, variants_for, collect_series
from signal_scanner.domain import AnalyzerSettings, CaptureLabels
from signal_scanner.fsl import FslAnalyzer
from signal_scanner.instruments import Simulator
from signal_scanner.services import Control
from signal_scanner.storage import Repository
from test_fsl import FakeFsl


class SeriesTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.repository=Repository(Path(self.tmp.name))
        self.fake=FakeFsl()
        self.analyzer=FslAnalyzer('fixture',transport_factory=self.fake.factory)
        self.labels=CaptureLabels('multiple','unknown',observation_id='same_scene')
    def tearDown(self):
        self.tmp.cleanup()

    def run_series(self, analyzer=None, control=None, emit=None, labels=None):
        events=[]
        def record(event):
            events.append(event)
            if emit: emit(event)
        collect_series(analyzer or self.analyzer,self.repository,control or Control(),record,
                       labels or self.labels,1,SeriesPlan(variants=10,seed=112))
        return events,Path(events[-1]['run'])

    def test_generator_reproducible_unique_bounded_and_frozen(self):
        settings=AnalyzerSettings(center_hz=36.2e6,span_hz=2e6,points=10001)
        plan=SeriesPlan(seed=123)
        choices=variants_for(settings,{},plan)
        self.assertEqual(choices,variants_for(settings,{},plan))
        self.assertNotEqual(choices,variants_for(settings,{},replace(plan,seed=124)))
        self.assertEqual(len(choices),15)
        self.assertEqual(len({json.dumps(v,sort_keys=True) for v in choices}),15)
        for variant in choices:
            cfg=AnalyzerSettings(**variant['settings'])
            self.assertEqual((cfg.center_hz,cfg.span_hz,cfg.points),(settings.center_hz,settings.span_hz,settings.points))
            self.assertTrue(.002 <= cfg.rbw_hz/settings.span_hz <= .2)
            self.assertTrue(.1 <= cfg.vbw_hz/cfg.rbw_hz <= 1)
            self.assertIn(cfg.detector,('POS','RMS','SAMPLE'))

    def test_baselines_first_last_groups_native_units_and_full_restore(self):
        self.fake.raw.update(trace_mode='MAXH',sweep_count='5',amplitude_unit='DBUV')
        before=dict(self.fake.raw)
        events,run=self.run_series()
        captures=[event for event in events if event['type']=='capture']
        self.assertEqual(len(captures),12)
        self.assertEqual([captures[0]['role'],captures[-1]['role']],['baseline_start','baseline_end'])
        for key,value in before.items():
            self.assertEqual(self.fake.raw[key],value,key)
        self.assertEqual(captures[0]['spectrum'].settings,captures[-1]['spectrum'].settings)
        configs=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((run/'settings').glob('*.json'))]
        self.assertTrue(all(c['amplitude_unit']=='DBUV' for c in configs))
        labels=[json.loads(p.read_text(encoding='utf-8')) for p in (run/'labels').glob('*.json')]
        self.assertEqual({v['group_id'] for v in labels},{'same_scene'})
        self.assertEqual(sum(v['labels_need_review'] for v in labels),10)
        self.assertTrue(all(c['measurement_metadata']['series']['quality_score'] is None for c in configs))
        self.assertTrue(all(not cmd.startswith(('FREQ:', 'SWE:POIN ')) or '?' in cmd for _,cmd in self.fake.log))

    def test_zero_count_temporary_accumulations_restored(self):
        _,run=self.run_series()
        self.assertEqual(self.fake.raw['sweep_count'],'0')
        configs=[json.loads(p.read_text(encoding='utf-8')) for p in (run/'settings').glob('*.json')]
        modes={c['measurement_metadata']['actual_settings']['trace_mode'] for c in configs}
        self.assertTrue({'WRIT','AVER','MAXH'} <= modes)
        for c in configs:
            state=c['measurement_metadata']['actual_settings']
            if state['trace_mode']!='WRIT': self.assertEqual(state['sweep_count'],'5')

    def test_stop_preserves_completed_records_and_restores(self):
        control=Control()
        def stop(event):
            if event['type']=='capture' and event['index']==3: control.cancel.set()
        events,run=self.run_series(control=control,emit=stop)
        session=json.loads((run/'session.json').read_text(encoding='utf-8'))
        self.assertEqual(session['status'],'stopped')
        self.assertEqual(session['captured'],3)
        self.assertTrue(session['settings_restored'])
        self.assertEqual(len(list((run/'samples').glob('*.csv'))),3)
        self.assertEqual(self.fake.raw['trace_mode'],'WRIT')

    def test_partially_rejected_variant_is_restored(self):
        original=self.analyzer.apply_variant
        def fail(variant,initial):
            original(variant,initial)
            raise RuntimeError('synthetic partial write failure')
        self.analyzer.apply_variant=fail
        with self.assertRaisesRegex(RuntimeError,'partial write'):
            self.run_series()
        run=next((self.repository.root/'measurements'/'captures').iterdir())
        session=json.loads((run/'session.json').read_text(encoding='utf-8'))
        self.assertEqual(session['captured'],1)
        self.assertEqual(session['status'],'error')
        self.assertTrue(session['settings_restored'])
        self.assertEqual(self.fake.raw['rbw_hz'],'10000')
        self.assertEqual(self.fake.raw['rbw_auto'],'1')

    def test_empty_window_remains_negative_and_no_modulation_task(self):
        sim=Simulator(delay_scale=0)
        sim.set_scene('Нет сигнала')
        _,run=self.run_series(analyzer=sim,labels=CaptureLabels('none','none'))
        session=json.loads((run/'session.json').read_text(encoding='utf-8'))
        self.assertFalse(session['signal_present'])
        self.assertEqual(session['purpose'],'signal_presence_and_settings_comparison')
        self.assertEqual(session['captured'],12)

    def test_technical_series_does_not_invent_signal_presence(self):
        sim=Simulator(delay_scale=0)
        _,run=self.run_series(analyzer=sim,labels=CaptureLabels('unknown','unknown'))
        session=json.loads((run/'session.json').read_text(encoding='utf-8'))
        self.assertIsNone(session['signal_present'])
        self.assertFalse(session['training_eligible'])

    def test_foreign_window_rejected_before_writes(self):
        initial=self.analyzer.read_settings()
        variant=variants_for(initial,{},SeriesPlan(seed=1))[0]
        variant['settings']['points']+=2
        self.fake.log.clear()
        with self.assertRaisesRegex(ValueError,'центр'):
            self.analyzer.apply_variant(variant,initial)
        self.assertEqual(self.fake.log,[])

    def test_qpe_video_bandwidth_command_order_and_restore(self):
        self.fake.raw.update(detector='QPE',detector_auto='0')
        factory=self.fake.factory
        def guarded():
            transport=factory()
            write=transport.write
            def guard(command):
                if command.startswith('BAND:VID ') and self.fake.raw['detector']=='QPE':
                    raise RuntimeError('VBW forbidden with QPE')
                return write(command)
            transport.write=guard
            return transport
        self.analyzer.factory=guarded
        _,run=self.run_series()
        self.assertEqual(self.fake.raw['detector'],'QPE')
        self.assertEqual(json.loads((run/'session.json').read_text(encoding='utf-8'))['status'],'complete')

    def test_failed_baseline_format_restore_is_not_reported_restored(self):
        self.fake.raw['data_format']='REAL,32'
        self.fake.fail_sweep=1
        factory=self.fake.factory
        def fail_restore():
            transport=factory()
            write=transport.write
            def fail(command):
                if command=='FORM REAL,32':
                    raise RuntimeError('synthetic FORM restore rejection')
                return write(command)
            transport.write=fail
            return transport
        self.analyzer.factory=fail_restore
        with self.assertRaisesRegex(Exception,'sweep timeout'):
            self.run_series()
        run=next((self.repository.root/'measurements'/'captures').iterdir())
        session=json.loads((run/'session.json').read_text(encoding='utf-8'))
        self.assertFalse(session['settings_restored'])
        self.assertTrue(session['cleanup_errors'])
        self.assertEqual(self.fake.raw['data_format'],'ASC')
        self.assertFalse(any(cmd.startswith(('DET ', 'BAND:RES ', 'BAND:VID ')) for _,cmd in self.fake.log))

    def test_filter_preflight_failure_does_not_write_spectral_settings(self):
        factory=self.fake.factory
        def fft():
            transport=factory()
            query=transport.query
            transport.query=lambda command: 'FFT' if command=='BAND:TYPE?' else query(command)
            return transport
        self.analyzer.factory=fft
        with self.assertRaisesRegex(ValueError,'Normal'):
            self.run_series()
        self.assertFalse(any(cmd.startswith(('DET ', 'BAND:RES ', 'BAND:VID ')) for _,cmd in self.fake.log))
