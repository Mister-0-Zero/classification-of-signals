"""Technical full-series check without invented operator signal labels."""
import argparse
import json
from pathlib import Path
import socket
import time

from .capture_series import SeriesPlan, collect_series
from .domain import CaptureLabels, utc_now
from .fsl import FslAnalyzer
from .services import Control
from .storage import Repository, application_home, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--address',default='192.168.10.24')
    parser.add_argument('--wait',type=int,default=180)
    parser.add_argument('--output',type=Path,default=application_home()/'data')
    args=parser.parse_args()
    report={'started_at':utc_now(),'address':args.address,'status':'waiting','hardware_series_verified':False}
    path=args.output/'series_check_latest.json'
    write_json(path,report)
    print('Waiting for FSL LAN; network settings are unchanged.',flush=True)
    deadline=time.monotonic()+min(300,max(0,args.wait))
    while True:
        try:
            with socket.create_connection((args.address,111),timeout=2): break
        except OSError as error:
            if time.monotonic()>=deadline:
                report.update(status='unreachable',error=str(error),finished_at=utc_now())
                write_json(path,report)
                print(json.dumps(report,ensure_ascii=True),flush=True)
                return 1
            time.sleep(1)
    analyzer=FslAnalyzer(args.address)
    try:
        identity,_,_=analyzer.probe()
        report.update(identity=identity,status='collecting')
        write_json(path,report)
        def event(value):
            if value.get('run'): report['run']=value['run']
            if value['type']=='capture':
                report.update(captured=value['index'],total=value['total'],last_role=value['role'])
            if value['type']=='capture_finished':
                report.update(status=value['status'],captured=value['captured'])
            write_json(path,report)
        collect_series(analyzer,Repository(args.output),Control(),event,
                       CaptureLabels('unknown','unknown',comment='Technical series validation; signal count not annotated'),
                       1,SeriesPlan(variants=10,seed=20261114))
        report['hardware_series_verified']=report['status']=='complete'
    except Exception as error:
        report.update(status='error',error=str(error),last_command=analyzer.last_command,
                      error_notes=getattr(error,'__notes__',[]))
    report['finished_at']=utc_now()
    write_json(path,report)
    print(json.dumps(report,ensure_ascii=True,indent=2),flush=True)
    return 0 if report['hardware_series_verified'] else 1


if __name__=='__main__':
    raise SystemExit(main())
