"""Verify the complete frozen rerun and reproduce saved analyses, without inference."""
import hashlib,json,subprocess,sys,tempfile,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    archive=ROOT/'completed.zip'
    expected=(ROOT/'SHA256SUMS').read_text().strip().split()[0]
    if sha(archive)!=expected: raise ValueError('Archive hash mismatch')
    with tempfile.TemporaryDirectory(prefix='gemma-reasoning-verify-') as tmp:
        root=Path(tmp).resolve()
        with zipfile.ZipFile(archive) as z:
            hashes=json.loads(z.read('MANIFEST.json'))['files']
            if set(z.namelist())!=set(hashes)|{'MANIFEST.json'}: raise ValueError('Unexpected archive members')
            for name,h in hashes.items():
                target=(root/name).resolve()
                if not target.is_relative_to(root): raise ValueError('Unsafe archive member')
                data=z.read(name)
                if hashlib.sha256(data).hexdigest()!=h: raise ValueError('Member hash mismatch')
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        release=root/'release_v3';run=root/'attempt_01'
        manifest=json.loads((release/'manifest.json').read_text())
        for name,h in manifest['sha256'].items():
            if sha(release/name)!=h:raise ValueError('Frozen release changed')
        assignments=json.loads((release/'assignments.json').read_text())
        count=0
        for arm in ['off_320','off_4096','on_4096']:
            for offset in range(0,288,8):
                rows=json.loads((run/arm/'batches'/f'{offset:05d}.json').read_text())
                if [r['assignment'] for r in rows]!=assignments[offset:offset+8]: raise ValueError('Assignment mismatch')
                count+=len(rows)
        if count!=864: raise ValueError('Wrong assignment count')
        preempt=json.loads((root/'preemption_430695.json').read_text())
        for name,h in preempt['batch_sha256'].items():
            if sha(run/name)!=h:raise ValueError('Preemption snapshot changed')
        before=json.loads((run/'analysis.json').read_text())
        subprocess.run([sys.executable,str(release/'scripts/analyze_gemma_reasoning_check.py'),'--run',str(run)],check=True,stdout=subprocess.PIPE,text=True)
        after=json.loads((run/'analysis.json').read_text())
        if before!=after: raise ValueError('Saved analysis differs from reparse')
        # Every condition must be present, including cells without errors.
        if len(after['cells'])!=36 or len(after['base_cells'])!=432:raise ValueError('Missing cells')
        if len(list((run/'environments').glob('*.json')))!=2:raise ValueError('Missing invocation environment')
        print(json.dumps({'verified_assignments':count,'preemption_batches_unchanged':len(preempt['batch_sha256']),
            'saved_analysis_reproduced':True,'summaries':after['summaries'],'no_inference':True},indent=2))
if __name__=='__main__':main()
