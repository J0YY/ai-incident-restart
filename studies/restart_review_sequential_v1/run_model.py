"""Pinned Gemma development runner. Offline cache only; one attempt, no answer retries."""
import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
import fcntl
import hashlib
import json
from pathlib import Path
import time
import traceback
from freeze import verify,save,sha
from study import messages,digest,read_json,final_answer


def batches(assignments,size):
    for stage in ['initial','final']:
        rows=[r for r in assignments if r['stage']==stage]
        for offset in range(0,len(rows),size):
            yield f'{stage}_{offset:05d}',rows[offset:offset+size]


@contextmanager
def locked(output):
    output.mkdir(parents=True,exist_ok=True)
    with (output/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try: yield
        finally: fcntl.flock(lock,fcntl.LOCK_UN)


class Gemma:
    def __init__(self,config):
        import torch,transformers
        from transformers import AutoProcessor,AutoModelForMultimodalLM
        self.torch=torch;self.config=config
        self.processor=AutoProcessor.from_pretrained(config['model'],revision=config['revision'],trust_remote_code=False,local_files_only=True)
        self.tok=self.processor.tokenizer;self.tok.padding_side='left'
        self.model=AutoModelForMultimodalLM.from_pretrained(config['model'],revision=config['revision'],dtype=torch.bfloat16,
           device_map={'':'cuda:0'},attn_implementation='sdpa',trust_remote_code=False,local_files_only=True).eval()
        if any(m._forward_hooks for m in self.model.modules()): raise ValueError('Unexpected activation hook')
        self.environment={'torch':torch.__version__,'transformers':transformers.__version__,'gpu':torch.cuda.get_device_name(),
            'model':config['model'],'revision':config['revision'],'generation_config':self.model.generation_config.to_dict(),
            'chat_template':self.processor.chat_template}

    def generate(self,prompts,seed):
        c=self.config;t=self.torch
        kwargs=dict(add_generation_prompt=True,enable_thinking=c['thinking'])
        prefixes=[self.processor.apply_chat_template(m,tokenize=False,**kwargs) for m in prompts]
        off=self.processor.apply_chat_template(prompts[0],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        if prefixes[0]==off or '<|think|>' not in prefixes[0]: raise ValueError('Thinking template did not enable thought')
        inputs=self.processor.apply_chat_template(prompts,tokenize=True,padding=True,return_dict=True,return_tensors='pt',**kwargs).to('cuda:0')
        width=inputs['input_ids'].shape[1]
        if width+c['max_new_tokens']>=16000: raise ValueError('Context exceeds frozen bound; do not truncate prompts')
        for i,prefix in enumerate(prefixes):
            ids=inputs['input_ids'][i][inputs['attention_mask'][i].bool()].tolist()
            if self.tok(prefix,add_special_tokens=False)['input_ids']!=ids: raise ValueError('Template/tokenization mismatch')
        t.manual_seed(seed);t.cuda.manual_seed_all(seed);start=time.monotonic()
        with t.inference_mode():
            out=self.model.generate(**inputs,do_sample=True,temperature=c['temperature'],top_p=c['top_p'],top_k=c['top_k'],
               max_new_tokens=c['max_new_tokens'],use_cache=True,pad_token_id=self.tok.pad_token_id)
        elapsed=time.monotonic()-start
        eos=self.model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else list(eos or [])
        values=[]
        for i in range(len(prompts)):
            ids=out[i,width:].tolist();stop=next((j for j,v in enumerate(ids) if v in eos),None)
            ids=ids[:stop+1] if stop is not None else ids
            raw=self.tok.decode(ids,skip_special_tokens=False)
            final,valid,thought=final_answer(raw)
            values.append({'raw_text':raw,'final_text':final,'channel_valid':valid,'thought_channel':thought,
                'new_token_ids':ids,'input_tokens':int(inputs['attention_mask'][i].sum()),
                'rendered_input_sha256':hashlib.sha256(prefixes[i].encode()).hexdigest(),
                'sampler_seed':seed,'truncated':stop is None and len(ids)>=c['max_new_tokens'],
                'stop_token':None if stop is None else ids[-1], 'batch_seconds':elapsed})
        return values


def run(release,output,resume=False,engine_factory=Gemma):
    config,cases,assignments,manifest=verify(release)
    bycase={c['family_id']:c for c in cases}
    with locked(output):
        runfile=output/'run.json'
        if runfile.exists():
            if not resume or read_json(runfile)['manifest']!=manifest: raise ValueError('Explicit resume of the same release required')
        else:
            if resume: raise ValueError('Cannot resume an uninitialized run')
            if any(output.glob('batches/*.json')): raise ValueError('Orphaned batches')
            save(runfile,{'manifest':manifest,'config':config,'status':'development_pilot_only','release_path':str(release.resolve())})
        permitted={name+'.json' for name,_ in batches(assignments,config['batch_size'])}
        saved_batches={p.name for p in (output/'batches').glob('*.json')}
        saved_receipts={p.name for p in (output/'receipts').glob('*.json')}
        extra=(saved_batches|saved_receipts)-permitted
        if extra: raise ValueError('Unexpected batches or receipts: '+str(extra))
        if saved_batches!=saved_receipts:
            raise ValueError('Orphaned batch or receipt; preserve both and investigate before resuming')
        for name in saved_batches:
            if read_json(output/'receipts'/name)['sha256']!=sha(output/'batches'/name):
                raise ValueError('Saved batch hash changed; investigate before loading the model')
        engine=None;prior={};done=0;start=time.monotonic()
        invocation=str(time.time_ns())
        try:
            for batch_id,chunk in batches(assignments,config['batch_size']):
                seed=int(digest([config['schedule_seed'],batch_id])[:8],16)%(2**31)
                prompts=[messages(bycase[a['family_id']],a,prior.get(a['initial_id'])) for a in chunk]
                path=output/'batches'/(batch_id+'.json')
                receipt=output/'receipts'/(batch_id+'.json')
                if path.exists():
                    if not receipt.exists() or read_json(receipt)['sha256']!=sha(path):
                        raise ValueError('Saved batch hash missing or changed; preserve and investigate')
                    batch=read_json(path)
                    if batch['assignments']!=chunk or batch['messages']!=prompts or batch['seed']!=seed:
                        raise ValueError('Saved batch does not match frozen assignment/prompts')
                else:
                    if engine is None:
                        engine=engine_factory(config)
                        save(output/'environments'/(invocation+'.json'),engine.environment)
                    values=engine.generate(prompts,seed)
                    if len(values)!=len(chunk): raise ValueError('Missing responses in generated batch')
                    batch={'assignments':chunk,'messages':prompts,'seed':seed,'outputs':values}
                    save(path,batch)
                    save(receipt,{'sha256':sha(path),'batch_id':batch_id})
                if len(batch['outputs'])!=len(chunk): raise ValueError('Saved response count mismatch')
                for a,value in zip(chunk,batch['outputs']):
                    if a['stage']=='initial': prior[a['id']]=value
                done+=len(chunk)
                save(output/'progress.json',{'completed':done,'total':len(assignments),'seconds_this_invocation':time.monotonic()-start})
                print(batch_id,done,'/',len(assignments),flush=True)
            save(output/'completion.json',{'complete':True,'assignments':done,'initial':len(prior),
                'final':done-len(prior),'status':'development_pilot_only','completed_at_utc':datetime.now(timezone.utc).isoformat()})
        except BaseException as e:
            save(output/'interruptions'/(invocation+'.json'),{'type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(),
                'completed_this_scan':done,'no_answer_retry_authorized':True})
            raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    if Path(__file__).resolve()!=a.release.resolve()/'run_model.py': raise SystemExit('Run the frozen release copy of this script')
    run(a.release,a.output,a.resume)
