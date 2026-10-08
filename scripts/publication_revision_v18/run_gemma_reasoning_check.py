"""Frozen three-arm Gemma diagnostic. No outcome-dependent retries or selection."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    temp.replace(path)


def final_answer(raw):
    """Extract only a completed final channel; never mine JSON from thought text."""
    text=raw.strip()
    for token in ['<pad>', '<eos>', '<turn|>']:
        while text.endswith(token): text=text[:-len(token)].rstrip()
    marker='<|channel>thought\n'
    thought=False
    if text.startswith(marker):
        thought=True
        if text.count('<channel|>')!=1:
            return '', {'valid_channel':False,'thought_channel':True,'error':'missing_or_repeated_boundary'}
        _,text=text.split('<channel|>',1)
    # Reject unknown, repeated, or unterminated channels, not arbitrary content before JSON.
    if any(x in text for x in ['<|channel>','<channel|>','<|turn>','<turn|>','<|think|>']):
        return '', {'valid_channel':False,'thought_channel':thought,'error':'unexpected_channel'}
    return text.strip(), {'valid_channel':True,'thought_channel':thought,'error':None}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--release',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true',help='Resume only missing preassigned batches after a retained interruption.')
    a=p.parse_args()
    manifest=json.loads((a.release/'manifest.json').read_text())
    for name,expected in manifest['sha256'].items(): assert sha(a.release/name)==expected,name
    assert Path(__file__).resolve()==(a.release/'scripts/run_gemma_reasoning_check.py').resolve()
    protocol=json.loads((a.release/'protocol.json').read_text())
    assignments=json.loads((a.release/'assignments.json').read_text())
    assert len(assignments)==288
    if a.output.exists():
        assert a.resume, 'Use a fresh attempt directory or explicit --resume'
        old=json.loads((a.output/'run.json').read_text())
        assert old['manifest']==manifest
    else:
        assert not a.resume
        save(a.output/'run.json',{'manifest':manifest,'protocol':protocol,'n':864,'status':'started'})
    import torch,transformers
    from gpu_essay import Runner,MODEL,REV
    from run_resolution_diagnostic import score
    assert [MODEL,REV]==protocol['model']
    runner=Runner()
    template_prefixes={}
    for mode in [False,True]:
        template_prefixes[str(mode)]=runner.processor.apply_chat_template(assignments[0]['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=mode)
    assert template_prefixes['False']!=template_prefixes['True']
    assert '<|think|>' in template_prefixes['True'] and '<|think|>' not in template_prefixes['False']
    save(a.output/'environments'/f'{time.time_ns()}.json',{'torch':torch.__version__,'transformers':transformers.__version__,
        'gpu':torch.cuda.get_device_name(),'generation_config':runner.model.generation_config.to_dict(),
        'template_prefixes':template_prefixes,'chat_template':runner.processor.chat_template,
        'revision':REV,'model':MODEL,'device_map':getattr(runner.model,'hf_device_map',None)})
    schedule=[]
    for offset in range(0,288,8):
        arm_order=list(protocol['arms'])
        random.Random(6100801+offset).shuffle(arm_order)
        schedule.extend((offset,arm) for arm in arm_order)
    save(a.output/'schedule.json',schedule)
    started=time.time();done=0
    for offset,arm in schedule:
        target=a.output/arm['name']/'batches'/f'{offset:05}.json'
        chunk=assignments[offset:offset+8]
        if target.exists():
            prior=json.loads(target.read_text())
            assert [r['assignment'] for r in prior]==chunk and all(r['arm']==arm for r in prior)
            done+=len(chunk);continue
        messages=[x['messages'] for x in chunk]
        seed=int(digest(['gemma','resolution_diagnostic',offset])[:8],16)%(2**31)
        kwargs={'add_generation_prompt':True,'enable_thinking':arm['thinking']}
        prefixes=[runner.processor.apply_chat_template(m,tokenize=False,**kwargs) for m in messages]
        inputs=runner.processor.apply_chat_template(messages,tokenize=True,padding=True,return_dict=True,return_tensors='pt',**kwargs).to('cuda:0')
        width=inputs['input_ids'].shape[1]
        assert width+arm['max_new_tokens']<16000
        for i,prefix in enumerate(prefixes):
            unpad=inputs['input_ids'][i][inputs['attention_mask'][i].bool()].tolist()
            assert runner.tok(prefix,add_special_tokens=False)['input_ids']==unpad
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        batch_start=time.time()
        with torch.inference_mode():
            output=runner.model.generate(**inputs,do_sample=True,temperature=.7,top_p=.9,top_k=0,
                max_new_tokens=arm['max_new_tokens'],use_cache=True,pad_token_id=runner.tok.pad_token_id)
        eos=runner.model.generation_config.eos_token_id
        eos=[eos] if isinstance(eos,int) else list(eos or [])
        rows=[]
        for i,assignment in enumerate(chunk):
            ids=output[i,width:].tolist()
            stop=next((j for j,t in enumerate(ids) if t in eos),None)
            effective=ids[:stop+1] if stop is not None else ids
            raw=runner.tok.decode(effective,skip_special_tokens=False)
            answer,channel=final_answer(raw)
            value={'text':answer,'raw_text':raw,'new_token_ids':effective,'channel':channel,
                'input_sha256':hashlib.sha256(prefixes[i].encode()).hexdigest(),
                'input_tokens':int(inputs['attention_mask'][i].sum()),'sampler_seed':seed,
                'max_new_tokens':arm['max_new_tokens'],
                'truncated':stop is None and len(effective)>=arm['max_new_tokens'],
                'stop_token':None if stop is None else ids[stop],'batch_seconds':time.time()-batch_start}
            parsed,metrics=score(assignment,value)
            rows.append({'assignment':assignment,'arm':arm,'output':value,'parsed':parsed,'metrics':metrics})
        save(target,rows);done+=len(rows)
        save(a.output/'progress.json',{'done':done,'total':864,'seconds_this_attempt':time.time()-started})
        print('REASONING_CHECK',done,'/',864,arm['name'],offset,flush=True)
    save(a.output/'completion.json',{'complete':True,'n':done,'seconds_this_attempt':time.time()-started,
        'max_gpu_bytes':torch.cuda.max_memory_allocated()})

if __name__=='__main__': main()
