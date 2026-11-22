"""Offline protocol tests. Fake replies are not hardware validation."""
from dataclasses import replace
import csv
import json
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from signal_scanner.domain import AnalyzerSettings, CaptureLabels, utc_now
from signal_scanner.fsl import FslAnalyzer, InstrumentError, SocketTransport, VisaTransport
from signal_scanner.instruments import Simulator
from signal_scanner.services import Control, collect, scan
from signal_scanner.storage import Repository


class FakeFsl:
    def __init__(self):
        self.raw = dict(
            instrument_mode='SAN', start_hz='99500000', stop_hz='100500000',
            center_hz='100000000', span_hz='1000000', rbw_hz='10000', vbw_hz='10000',
            points='101', detector='POS', rbw_auto='1', vbw_auto='1', detector_auto='1',
            trace_mode='WRIT', frequency_axis='LIN', amplitude_unit='DBM',
            sweep_time_s='0.01', sweep_time_auto='1', sweep_count='0', continuous='1',
            trigger='IMM', reference_level='0', attenuation_db='10',
            attenuation_auto='1', data_format='ASC', gated='0', signal_track='0',
            sweep_mode='AUTO', trace_math='0', statistics_apd='0', statistics_ccdf='0',
            reference_offset_db='0')
        self.log = []
        self.completed = 0
        self.fail_sweep = None
        self.trace_points = None
        self.error_response = '0,"No error"'
        self.fail_local = False
        self.connections = 0

    def factory(self):
        device = self
        device.connections += 1
        channel = device.connections
        class Transport:
            def query_file(self, command):
                device.log.append((channel, command))
                n = device.trace_points or int(device.raw['points'])
                start, stop = float(device.raw['start_hz']), float(device.raw['stop_hz'])
                lines = ['Type;FSL;', 'x-Axis;'+device.raw['frequency_axis']+';',
                         'Trace 1:;;', 'x-Unit;Hz;', 'y-Unit;'+device.raw['amplitude_unit']+';',
                         f'Values;{n};']
                lines.extend(f'{start+(stop-start)*i/(n-1):.12g};{-90+device.completed/10};' for i in range(n))
                return '\n'.join(lines).encode('ascii')
            def query(self, command):
                device.log.append((channel, command))
                if command == '*OPC?': return '1'
                if command == 'BAND:TYPE?': return 'NORM'
                if command == 'CALC:MATH:MODE?': return 'LOG'
                if command == '*IDN?': return 'Rohde&Schwarz,FSL-3,123456,2.00'
                if command == 'SYST:ERR?': return device.error_response
                if command == 'INIT:IMM;*OPC?':
                    if device.fail_sweep == device.completed + 1:
                        raise TimeoutError('synthetic sweep timeout')
                    device.completed += 1
                    return '1'
                if command == 'TRAC? TRACE1':
                    return ','.join(str(-90+device.completed/10) for _ in range(device.trace_points or int(device.raw['points'])))
                key = next((key for key,value in FslAnalyzer.QUERIES.items() if command == value), None)
                if key is None: raise AssertionError(command)
                return device.raw[key]
            def write(self, command):
                device.log.append((channel, command))
                if command.startswith(('MMEM:STOR:TRAC ', 'MMEM:DEL ')): return
                head, value = command.split(' ',1)
                key = next((key for key,query in FslAnalyzer.QUERIES.items() if query == head+'?'), None)
                if key is None: raise AssertionError(command)
                device.raw[key] = value
                if key in ('center_hz','span_hz'):
                    center,span=float(device.raw['center_hz']),float(device.raw['span_hz'])
                    device.raw['start_hz']=str(center-span/2)
                    device.raw['stop_hz']=str(center+span/2)
                if key in ('rbw_hz','vbw_hz','detector','sweep_time_s','attenuation_db'):
                    auto={'rbw_hz':'rbw_auto','vbw_hz':'vbw_auto','detector':'detector_auto',
                          'sweep_time_s':'sweep_time_auto','attenuation_db':'attenuation_auto'}[key]
                    device.raw[auto]='0'
            def local(self):
                device.log.append((channel,'LOCAL_RPC'))
                if device.fail_local: raise OSError('synthetic Local failure')
            def close(self):
                device.log.append((channel,'CLOSE'))
        return Transport()


class FslTests(unittest.TestCase):
    def setUp(self):
        self.fake=FakeFsl()
        self.analyzer=FslAnalyzer('offline-fixture',transport_factory=self.fake.factory)
        self.tmp=tempfile.TemporaryDirectory()
        self.repository=Repository(Path(self.tmp.name))
    def tearDown(self):
        self.tmp.cleanup()
    def collect(self, **kw):
        events=[]
        collect(self.analyzer,self.repository,Control(),events.append,CaptureLabels('one','unknown'),
                AnalyzerSettings(center_hz=100e6,points=101),2,**kw)
        return events,Path(events[-1]['run'])

    def test_probe_only_queries_and_protocol_local(self):
        identity,settings,raw=self.analyzer.probe()
        self.assertIn('FSL-3',identity)
        self.assertEqual(settings.points,101)
        self.assertTrue(all('?' in cmd or cmd in ('LOCAL_RPC','CLOSE') for _,cmd in self.fake.log))
        self.assertEqual([cmd for _,cmd in self.fake.log[-2:]],['LOCAL_RPC','CLOSE'])

    def test_probe_and_manual_read_do_not_query_unsupported_axis(self):
        original=self.fake.factory
        def factory():
            transport=original()
            query=transport.query
            def reply(command):
                if command=='DISP:TRAC1:X:SPAC?':
                    self.fake.log.append((self.fake.connections,command))
                    raise TimeoutError('unsupported axis query')
                return query(command)
            transport.query=reply
            return transport
        self.analyzer.factory=factory
        identity,settings,raw=self.analyzer.probe()
        self.assertIn('FSL-3',identity)
        self.assertEqual(settings.center_hz,100e6)
        self.assertEqual(raw['frequency_axis'],'PENDING_EXPORT')
        self.assertFalse(self.analyzer.acquisition_blocked)
        self.assertEqual(self.analyzer.read_settings().center_hz,100e6)
        spectrum=self.analyzer.capture(1,threading.Event())
        self.assertEqual(spectrum.measurement_metadata['actual_settings']['frequency_axis'],'LIN')
        self.assertFalse(any(cmd=='DISP:TRAC1:X:SPAC?' for _,cmd in self.fake.log))

    def test_manual_collect_never_writes_spectral_settings(self):
        events,run=self.collect(read_manual=True)
        commands=[cmd for _,cmd in self.fake.log]
        for prefix in ('FREQ:', 'BAND:', 'SWE:', 'DET ', 'INP:', 'DISP:'):
            self.assertFalse(any(cmd.startswith(prefix) and '?' not in cmd for cmd in commands),prefix)
        self.assertEqual(self.fake.completed,2)
        self.assertEqual(self.fake.raw['continuous'],'1')
        self.assertEqual(self.fake.raw['data_format'],'ASC')
        self.assertFalse(events[0]['spectrum'].simulated)

    def test_real_branch_provenance_dimensions_and_unknown_label(self):
        _,run=self.collect(read_manual=True)
        self.assertEqual(run.parent.parent.name,'measurements')
        with next((run/'samples').glob('*.csv')).open(newline='',encoding='utf-8') as f:
            rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),202)
        self.assertTrue(all(r['simulated']=='False' for r in rows))
        cfg=json.loads(next((run/'settings').glob('*.json')).read_text(encoding='utf-8'))
        labels=json.loads(next((run/'labels').glob('*.json')).read_text(encoding='utf-8'))
        self.assertFalse(cfg['simulated'])
        self.assertFalse(labels['simulated'])
        self.assertEqual(labels['labels']['modulation'],'unknown')
        self.assertEqual(labels['localization_status'],'unannotated')
        self.assertEqual(len(cfg['measurement_metadata']['sweep_completed_at']),2)
        self.assertEqual(cfg['measurement_metadata']['actual_settings']['rbw_auto'],'1')

    def test_sweep_completion_precedes_each_trace(self):
        self.analyzer.capture(3,threading.Event())
        commands=[cmd for _,cmd in self.fake.log if cmd=='INIT:IMM;*OPC?' or cmd.startswith('MMEM:STOR:TRAC ')]
        self.assertEqual([cmd if cmd=='INIT:IMM;*OPC?' else 'EXPORT' for cmd in commands],['INIT:IMM;*OPC?','EXPORT']*3)

    def test_unsafe_modes_rejected_without_writes(self):
        for field,value in [('trace_mode','VIEW'),('trigger','EXT'),
                            ('instrument_mode','ADEM'),
                            ('amplitude_unit','V'),('gated','1'),('signal_track','1'),
                            ('sweep_count','-1'),('sweep_mode','ESP'),('trace_math','1'),
                            ('statistics_apd','1'),('statistics_ccdf','1')]:
            with self.subTest(field=field,value=value):
                before=self.fake.raw[field]
                self.fake.raw[field]=value
                self.fake.log.clear()
                with self.assertRaises(ValueError): self.analyzer.capture(2,threading.Event())
                self.assertTrue(all('?' in cmd or cmd in ('LOCAL_RPC','CLOSE') for _,cmd in self.fake.log))
                self.fake.raw[field]=before

    def test_unsupported_mode_in_service_does_not_trigger_restore_writes(self):
        self.fake.raw['trace_mode']='VIEW'
        with self.assertRaises(ValueError):
            self.collect(read_manual=False)
        self.assertTrue(all('?' in cmd or cmd in ('LOCAL_RPC','CLOSE') for _,cmd in self.fake.log))

    def test_mismatching_trace_rejected_without_truncation(self):
        self.fake.trace_points=102
        with self.assertRaises(InstrumentError): self.analyzer.capture(1,threading.Event())
        self.assertEqual(self.fake.raw['continuous'],'1')

    def test_max_hold_five_sweeps_native_dbuv_roundtrip(self):
        self.fake.raw.update(trace_mode='MAXH',sweep_count='5',amplitude_unit='DBUV')
        _,run=self.collect(read_manual=True)
        with next((run/'samples').glob('*.csv')).open(newline='',encoding='utf-8') as f:
            rows=list(csv.DictReader(f))
        self.assertIn('amplitude_dbuv',rows[0])
        self.assertNotIn('amplitude_dbm',rows[0])
        self.assertEqual(rows[0]['amplitude_unit'],'DBUV')
        self.assertAlmostEqual(float(rows[0]['amplitude_dbuv']),-89.9)
        cfg=json.loads(next((run/'settings').glob('*.json')).read_text(encoding='utf-8'))
        self.assertEqual(cfg['amplitude_unit'],'DBUV')
        self.assertEqual(cfg['measurement_metadata']['underlying_sweeps_per_trace'],5)
        self.assertTrue(cfg['measurement_metadata']['trace_history_reset'])
        self.assertEqual(self.fake.raw['trace_mode'],'MAXH')
        self.assertEqual(self.fake.raw['sweep_count'],'5')

    def test_log_axis_export_is_rejected_and_settings_restored(self):
        self.fake.raw['frequency_axis']='LOG'
        with self.assertRaisesRegex(InstrumentError,'линейная'):
            self.analyzer.capture(1,threading.Event())
        self.assertEqual(self.fake.raw['continuous'],'1')
        self.assertTrue(any(cmd.startswith('MMEM:DEL ') for _,cmd in self.fake.log))

    def test_read_settings_does_not_require_extended_flags(self):
        self.fake.raw.pop('gated')
        self.fake.raw.pop('detector_auto')
        self.assertEqual(self.analyzer.read_settings().points,101)

    def test_timeout_diagnostic_and_recovery_uses_new_channel(self):
        self.fake.fail_sweep=2
        events=[]
        with self.assertRaises(InstrumentError) as result:
            collect(self.analyzer,self.repository,Control(),events.append,CaptureLabels('one','unknown'),
                    AnalyzerSettings(points=101),2,read_manual=True)
        summary=json.loads((Path(events[-1]['run'])/'session.json').read_text(encoding='utf-8'))
        self.assertEqual(summary['status'],'error')
        self.assertEqual(summary['diagnostic_context']['completed_sweeps'],1)
        self.assertEqual(summary['captured'],0)
        self.assertEqual(self.fake.raw['continuous'],'1')
        failed_channel=next(ch for ch,cmd in self.fake.log if cmd=='INIT:IMM;*OPC?')
        # read_settings, snapshot, capture each own one channel.
        self.assertTrue(any(ch>failed_channel and cmd=='INIT:CONT 1' for ch,cmd in self.fake.log))

    def test_variants_share_group_and_restore_auto_flags(self):
        before=dict(self.fake.raw)
        _,run=self.collect(read_manual=True,variants=True)
        for key in ('center_hz','span_hz','rbw_hz','vbw_hz','rbw_auto','vbw_auto','detector_auto','sweep_time_auto'):
            self.assertEqual(float(self.fake.raw[key]),float(before[key]))
        labels=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((run/'labels').glob('*.json'))]
        self.assertEqual(len(labels),3)
        self.assertEqual(len({r['group_id'] for r in labels}),1)
        self.assertEqual(len({r['run_id'] for r in labels}),3)
        self.assertEqual([r['labels_need_review'] for r in labels],[False,True,True])

    def test_command_rejection_is_not_success(self):
        self.fake.error_response='-222,"Data out of range"'
        with patch.object(self.analyzer,'read_error_queue',return_value=[]):
            with self.assertRaisesRegex(InstrumentError,'отклонил'):
                self.analyzer.apply_settings(AnalyzerSettings(center_hz=100e6,points=101))

    def test_local_failure_is_visible_and_connection_closed(self):
        self.fake.fail_local=True
        with self.assertRaisesRegex(InstrumentError,'ручного управления'):
            self.analyzer.read_settings()
        self.assertEqual(self.fake.log[-1][1],'CLOSE')

    def test_tcp_capture_is_blocked(self):
        analyzer=FslAnalyzer('unused',diagnostics_only=True,transport_factory=self.fake.factory)
        with self.assertRaisesRegex(ValueError,'PyVISA'): analyzer.capture(1,threading.Event())
        self.assertEqual(self.fake.connections,0)

    def test_storage_rejects_bad_dimensions_before_creating_csv(self):
        spectrum=Simulator(delay_scale=0).capture(1,threading.Event())
        run=self.repository.new_session('captures')
        bad=replace(spectrum,sweeps_dbm=(spectrum.sweeps_dbm[0][:-1],))
        with self.assertRaises(ValueError): self.repository.save_spectrum(run,'bad',bad)
        self.assertFalse((run/'samples').exists())

    def test_storage_rejects_wrong_origin(self):
        spectrum=Simulator(delay_scale=0).capture(1,threading.Event())
        run=self.repository.new_session('captures',simulated=False)
        with self.assertRaises(ValueError): self.repository.save_spectrum(run,'bad',spectrum)

    def test_cleanup_error_marks_session_error_and_releases(self):
        class RestoreFails(Simulator):
            released=False
            def restore_state(self,state): raise OSError('synthetic restore failure')
            def release_local(self): self.released=True
        analyzer=RestoreFails(delay_scale=0)
        events=[]
        with self.assertRaisesRegex(RuntimeError,'восстановить'):
            collect(analyzer,self.repository,Control(),events.append,CaptureLabels('one','unknown'),
                    AnalyzerSettings(),1)
        summary=json.loads((Path(events[-1]['run'])/'session.json').read_text(encoding='utf-8'))
        self.assertEqual(summary['status'],'error')
        self.assertTrue(summary['cleanup_errors'])
        self.assertTrue(analyzer.released)

    def test_original_error_preserved_when_cleanup_fails(self):
        class BothFail(Simulator):
            def capture(self,*args): raise TimeoutError('original timeout')
            def restore_state(self,*args): raise OSError('restore failure')
        with self.assertRaisesRegex(TimeoutError,'original timeout'):
            collect(BothFail(),self.repository,Control(),lambda _:None,CaptureLabels('one','unknown'),
                    AnalyzerSettings(),1)

    def test_completed_baseline_survives_variant_failure(self):
        class VariantFails(Simulator):
            count=0
            def capture(self,*args):
                self.count+=1
                if self.count==2: raise TimeoutError('variant timeout')
                return super().capture(*args)
        events=[]
        with self.assertRaises(TimeoutError):
            collect(VariantFails(delay_scale=0),self.repository,Control(),events.append,
                    CaptureLabels('one','unknown'),AnalyzerSettings(),1,variants=True)
        run=Path(events[-1]['run'])
        self.assertEqual(events[-1]['captured'],1)
        self.assertEqual(len(list((run/'samples').glob('*.csv'))),1)
        self.assertEqual(json.loads((run/'session.json').read_text(encoding='utf-8'))['status'],'error')


class TransportTests(unittest.TestCase):
    def test_tcp_fragmentation_and_preserved_tail(self):
        sock=Mock()
        sock.recv.side_effect=[b'abc',b'd\nnext\n']
        with patch('socket.create_connection',return_value=sock):
            transport=SocketTransport('127.0.0.1:5025')
            self.assertEqual(transport.query('*IDN?'),'abcd')
            self.assertEqual(transport.query('INST?'),'next')
            self.assertEqual(sock.recv.call_count,2)

    def test_tcp_eof_before_terminator_is_error(self):
        sock=Mock()
        sock.recv.side_effect=[b'partial',b'']
        with patch('socket.create_connection',return_value=sock):
            transport=SocketTransport('127.0.0.1')
            with self.assertRaises(ConnectionError): transport.query('*IDN?')

    def test_tcp_rejects_visa_resource_without_connecting(self):
        with patch('socket.create_connection',side_effect=AssertionError('network')):
            with self.assertRaises(ValueError): SocketTransport('TCPIP0::127.0.0.1::inst0::INSTR')

    @unittest.skipUnless(find_spec("pyvisa"), "optional PyVISA dependency not installed")
    def test_pyvisa_pure_backend_uses_device_local_rpc(self):
        import pyvisa
        transport=VisaTransport.__new__(VisaTransport)
        instrument=Mock()
        instrument.session=17
        instrument.timeout=5000
        instrument.control_ren.side_effect=pyvisa.errors.VisaIOError(pyvisa.constants.StatusCode.error_nonsupported_operation)
        protocol=Mock()
        protocol.device_local.return_value=0
        transport.instrument=instrument
        transport.manager=SimpleNamespace(visalib=SimpleNamespace(sessions={17:SimpleNamespace(interface=protocol,link=42,lock_timeout=5000)}))
        transport.local_operation=pyvisa.constants.RENLineOperation.deassert_gtl
        transport.backend='@py'
        with patch('signal_scanner.fsl.version',return_value='0.8.1'):
            transport.local()
        protocol.device_local.assert_called_once_with(42,0,5000,5000)

    @unittest.skipUnless(find_spec("pyvisa"), "optional PyVISA dependency not installed")
    def test_rpc_local_error_propagates(self):
        import pyvisa
        transport=VisaTransport.__new__(VisaTransport)
        instrument=Mock()
        instrument.session=17; instrument.timeout=5000
        instrument.control_ren.side_effect=pyvisa.errors.VisaIOError(pyvisa.constants.StatusCode.error_nonsupported_operation)
        protocol=Mock(); protocol.device_local.return_value=8
        transport.instrument=instrument
        transport.manager=SimpleNamespace(visalib=SimpleNamespace(sessions={17:SimpleNamespace(interface=protocol,link=42,lock_timeout=5000)}))
        transport.local_operation=pyvisa.constants.RENLineOperation.deassert_gtl
        transport.backend='@py'
        with patch('signal_scanner.fsl.version',return_value='0.8.1'):
            with self.assertRaisesRegex(InstrumentError,'device_local'): transport.local()
