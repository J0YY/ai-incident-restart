"""Independent raw-output count of core accuracy and constant-decision baselines."""
from collections import Counter,defaultdict
from pathlib import Path
import argparse,hashlib,json,zipfile

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();cells=defaultdict(lambda: {'n':0,'correct':0,'labels':Counter(),'choices':Counter()})
    with zipfile.ZipFile(a.archive) as z:
        manifest=json.loads(z.read('MANIFEST.json'))['files']
        for model,folder in [('qwen','qwen_main_v2'),('gemma','gemma_main_v3_merged')]:
            names=sorted(n for n in z.namelist() if n.startswith('results/incident_review_study/'+folder+'/batches/') and n.endswith('.json'))
            for name in names:
                raw=z.read(name);assert hashlib.sha256(raw).hexdigest()==manifest[name]
                for r in json.loads(raw):
                    assignment=r['assignment']
                    if assignment['kind']!='core':continue
                    text=r['output']['text'].strip()
                    if text.startswith('```'):
                        lines=text.splitlines();assert lines[0] in ['```json','```'] and lines[-1]=='```'
                        text='\n'.join(lines[1:-1])
                    v=json.loads(text);assert not r['output']['truncated']
                    gold=assignment['gold']['decision'];choice=v['decision'];assert choice in ['FULL','LIMITED','HOLD']
                    cell=cells[model,assignment['guide']];cell['n']+=1;cell['correct']+=choice==gold
                    cell['labels'][gold]+=1;cell['choices'][choice]+=1
    rows=[]
    for (model,guide),cell in sorted(cells.items()):
        assert cell['n']==576 and dict(cell['labels'])=={'FULL':288,'LIMITED':144,'HOLD':144}
        rows.append({'model':model,'guide':guide,**cell,'accuracy_percent':100*cell['correct']/576,
            'constant_full_accuracy_percent':50,'constant_full_wrong_approvals':288,'constant_full_unsupported_n':288})
    assert sorted(r['correct'] for r in rows if r['model']=='qwen')==[221,225,236]
    assert all(r['correct']==576 for r in rows if r['model']=='gemma')
    result={'source_archive_sha256':hashlib.sha256(a.archive.read_bytes()).hexdigest(),'source':'Independently parsed complete raw core responses, no new inference','cells':rows}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print('Verified core accuracies and class imbalance from all 3456 raw core responses.')
if __name__=='__main__':main()
