#!/usr/bin/env python3
"""Recalculate the manuscript's numerical results from saved outputs, offline.

No model imports, inference, API calls, or remote downloads. Use a fresh output
directory so the supplied archives remain unchanged.
"""
from pathlib import Path
import argparse,hashlib,json,math,shutil,subprocess,sys,zipfile

ROOT=Path(__file__).resolve().parent

def same(a,b,path='root'):
    if isinstance(a,dict):
        assert isinstance(b,dict) and a.keys()==b.keys(),path
        for k in a:same(a[k],b[k],path+'.'+k)
    elif isinstance(a,list):
        assert isinstance(b,list) and len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):same(x,y,f'{path}[{i}]')
    elif isinstance(a,(int,float)) and not isinstance(a,bool):
        assert isinstance(b,(int,float)) and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),(path,a,b)
    else:assert a==b,(path,a,b)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'_reproduction')
    p.add_argument('--questionnaire',action='store_true',help='Also reproduce the earlier paired questionnaire results.')
    p.add_argument('--archived-api',action='store_true',help='Also analyze the archived API extension. This does not call the API.')
    p.add_argument('--verify-only',action='store_true',help='Verify and extract only.')
    a=p.parse_args();work=a.output.resolve()
    if work.exists():raise SystemExit('Choose a new --output directory; existing results are never overwritten.')
    work.mkdir(parents=True)
    names=['governance_evidence.zip']+(['questionnaire_evidence.zip'] if a.questionnaire else [])
    checksums={line.split('  ',1)[1]:line.split('  ',1)[0] for line in (ROOT/'archives/SHA256SUMS').read_text().splitlines()}
    verified=0
    for name in names:
        archive=ROOT/'archives'/name
        assert hashlib.sha256(archive.read_bytes()).hexdigest()==checksums[name],name
        with zipfile.ZipFile(archive) as z:
            manifest=json.loads(z.read('MANIFEST.json'))['files']
            assert set(z.namelist())==set(manifest)|{'MANIFEST.json'}
            for n,h in manifest.items():
                dest=work/n
                assert dest.resolve().is_relative_to(work),n
                b=z.read(n);assert hashlib.sha256(b).hexdigest()==h,n
                if n=='README.txt':dest=work/(name.removesuffix('.zip')+'_README.txt')
                if dest.exists():assert dest.read_bytes()==b,n
                else:dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
                verified+=1
    logs=work/'reproduction_logs';logs.mkdir()
    report={'verified_archive_members':verified,'comparisons':[],'no_inference':True,'numeric_tolerance':1e-10}
    def run(label,args,result=None):
        before=json.loads((work/result).read_text()) if result else None
        with (logs/(label+'.txt')).open('w') as stream:
            subprocess.run([sys.executable,*args],cwd=work,stdout=stream,stderr=subprocess.STDOUT,check=True)
        if result:same(before,json.loads((work/result).read_text()))
        report['comparisons'].append({'step':label,'saved_result_matches':True if result else None})
        print('PASS',label,flush=True)
    if not a.verify_only:
        for model,runname in [('qwen','qwen_main_v2'),('gemma','gemma_main_v3_merged')]:
            run_dir='results/incident_review_study/'+runname
            run('review_'+model,['scripts/analyze_incident_review_study.py',run_dir],run_dir+'/analysis.json')
            run('review_audit_'+model,['scripts/audit_incident_review_results.py',run_dir,'--output',str(logs/(model+'_audit.json'))])
        for model in ['qwen','gemma']:
            run_dir=f'results/incident_resolution_diagnostic/{model}_v1'
            release='results/incident_resolution_diagnostic/frozen_release_v1'
            run('diagnostic_'+model,[release+'/scripts/analyze_resolution_diagnostic.py',run_dir,'--release',release],run_dir+'/analysis.json')
            run('diagnostic_audit_'+model,['scripts/audit_resolution_diagnostic.py','--run',run_dir])
            run_dir=f'results/incident_repair_study/runs/{model}_main_v1'
            run('repair_'+model,['scripts/analyze_incident_repair_study.py',run_dir],run_dir+'/analysis.json')
        run('signal_detection',['scripts/analyze_review_signal_detection.py'],'results/incident_review_study/signal_detection.json')
        run('design_figures',['scripts/build_paper_design_figures.py'])
        same(json.loads((work/'presentation/reader_redraft/diagram_record.json').read_text()),json.loads((work/'tmp/pdfs/who_asks_the_model/reader_redraft/figures/diagram_record.json').read_text()))
        run('dot_chart',['scripts/build_gemma_form_dotplot.py'])
        same(json.loads((work/'presentation/thesis_redraft/dotplot_data.json').read_text()),json.loads((work/'tmp/pdfs/who_asks_the_model/thesis_redraft/figures/dotplot_data.json').read_text()))
        (work/'figures').mkdir(exist_ok=True)
        for n in ['experimental_design.pdf','evidence_example.pdf']:shutil.copy2(work/'tmp/pdfs/who_asks_the_model/reader_redraft/figures'/n,work/'figures'/n)
        shutil.copy2(work/'tmp/pdfs/who_asks_the_model/thesis_redraft/figures/gemma_form_misses.pdf',work/'figures/gemma_form_misses.pdf')
        if a.questionnaire:
            for model in ['qwen','gemma']:
                run('questionnaire_'+model,['scripts/analyze_essay_mechanism_questionnaire.py','--model',model],f'results/essay_mechanism/{model}_questionnaire_analysis.json')
        if a.archived_api:
            release='results/frontier_review_extension/release_v2';run_dir='results/frontier_review_extension/main_v1'
            run('archived_api',[release+'/scripts/analyze_frontier_review_extension.py','--release',release,'--run',run_dir],run_dir+'/analysis.json')
    (work/'REPRODUCTION_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Saved',work/'REPRODUCTION_REPORT.json')

if __name__=='__main__':main()
