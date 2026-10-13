"""Create an immutable development release. Deliberately does not enable main inference."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from datetime import datetime, timezone
from study import read_json, schedule, validate_cases

ROOT=Path(__file__).resolve().parent
SOURCES=['study.py','run_model.py','analyze.py','freeze.py','test_study.py','config.json',
         'development_cases.json','build_development_cases.py','protocol.txt','analysis_plan.txt',
         'human_extension.txt','case_authoring.txt','README.txt']

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    tmp.replace(path)

def verify(release):
    manifest=read_json(release/'manifest.json')
    for name,expected in manifest['sha256'].items():
        target=release/name
        if target.resolve().parent != release.resolve(): raise ValueError('Unsafe manifest path')
        if sha(target)!=expected: raise ValueError('Release hash mismatch: '+name)
    config=read_json(release/'config.json');cases=read_json(release/'development_cases.json')
    validate_cases(cases,config)
    assignments=read_json(release/'assignments.json')
    if assignments!=schedule(config,cases): raise ValueError('Schedule differs from fixed construction')
    return config,cases,assignments,manifest

def freeze(target):
    if target.exists(): raise ValueError('Use a new release directory; previous releases must remain unchanged')
    config=read_json(ROOT/'config.json');cases=read_json(ROOT/'development_cases.json')
    validate_cases(cases,config)
    target.mkdir(parents=True)
    for name in SOURCES: shutil.copyfile(ROOT/name,target/name)
    assignments=schedule(config,cases)
    save(target/'assignments.json',assignments)
    save(target/'manifest.json',{'status':'development_pilot_only','frozen_at_utc':datetime.now(timezone.utc).isoformat(),
         'external_registration':False,'n_initial':sum(r['stage']=='initial' for r in assignments),
         'n_final':sum(r['stage']=='final' for r in assignments),
         'sha256':{name:sha(target/name) for name in SOURCES+['assignments.json']}})
    verify(target)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    freeze(a.output)
    print(json.dumps({'release':str(a.output),'status':'development_pilot_only'}))
