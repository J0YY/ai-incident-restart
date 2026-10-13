"""Reconstruct complete assignments and descriptive DEVELOPMENT results, never significance."""
import argparse
from collections import defaultdict
from itertools import product
import json
from pathlib import Path
from freeze import verify,save,sha
from run_model import batches
from study import read_json,messages,final_answer,digest,score,parse


def summarize(rows):
    n=len(rows);support=[r for r in rows if r['assignment']['branch']=='resolved'];other=[r for r in rows if r['assignment']['branch']!='resolved']
    result={'n':n,'supported_n':len(support),'unsupported_n':len(other)}
    for field in ['scope_choice_matches_key','wrong_approval','unsafe_full_signal','wrong_approval_upper','needless_withholding','valid_refusal','invalid','truncated','original_status_correct','active_objections_correct','schema_valid','evidence_ids_known']:
        result[field]=sum(r['metrics'][field] for r in rows)
    result['generated_tokens']=sum(len(r['output']['new_token_ids']) for r in rows)
    result['semantic_review_complete']=False
    return result


def analyze(release,output):
    config,cases,assignments,manifest=verify(release)
    if read_json(output/'run.json')['manifest']!=manifest: raise ValueError('Wrong release')
    complete=read_json(output/'completion.json')
    if not complete['complete'] or complete['assignments']!=len(assignments): raise ValueError('Run incomplete')
    case_map={c['family_id']:c for c in cases};rows=[];prior={};initial=[];batch_records=[]
    expected={b+'.json' for b,_ in batches(assignments,config['batch_size'])}
    if {p.name for p in (output/'batches').glob('*.json')}!=expected: raise ValueError('Missing or extra batch files')
    for batch_id,chunk in batches(assignments,config['batch_size']):
        batch_path=output/'batches'/(batch_id+'.json')
        if read_json(output/'receipts'/(batch_id+'.json'))['sha256']!=sha(batch_path):
            raise ValueError('Saved batch hash mismatch')
        batch=read_json(batch_path)
        prompts=[messages(case_map[a['family_id']],a,prior.get(a['initial_id'])) for a in chunk]
        seed=int(digest([config['schedule_seed'],batch_id])[:8],16)%(2**31)
        if batch['assignments']!=chunk or batch['messages']!=prompts or batch['seed']!=seed: raise ValueError('Assignment or message mismatch')
        if len(batch['outputs'])!=len(chunk): raise ValueError('Missing response')
        for a,value in zip(chunk,batch['outputs']):
            final,valid,thought=final_answer(value['raw_text'])
            if (final,valid,thought)!=(value['final_text'],value['channel_valid'],value['thought_channel']): raise ValueError('Final-channel mismatch')
            if value['sampler_seed']!=seed: raise ValueError('Output seed mismatch')
            if a['stage']=='initial':
                prior[a['id']]=value
                initial.append({'assignment':a,'decision_valid':parse(final)['decision_valid'] if valid else False,'truncated':value['truncated']})
            else: rows.append({'assignment':a,'output':value,'metrics':score(case_map[a['family_id']],a,value),'initial_decision_valid':parse(prior[a['initial_id']]['final_text'])['decision_valid'],'missing_prior_final':not bool(prior[a['initial_id']]['final_text'])})
        # Batch latency is shared; summing per-row copies would multiply it by batch size.
        batch_records.append({'batch_id':batch_id,'n':len(chunk),'seconds':batch['outputs'][0]['batch_seconds']})
    cells=[]
    for form,history,branch,order in product(config['forms'],config['histories'],config['branches'],config['orders']):
        subset=[r for r in rows if all(r['assignment'][k]==v for k,v in [('form',form),('history',history),('branch',branch),('order',order)])]
        cells.append(dict(form=form,history=history,branch=branch,order=order,**summarize(subset)))
    contrasts=[]
    for history in config['histories']:
        for metric,branches in [('needless_withholding',['resolved']),('wrong_approval',[b for b in config['branches'] if b!='resolved']),('wrong_approval_upper',[b for b in config['branches'] if b!='resolved']),('unsafe_full_signal',[b for b in config['branches'] if b!='resolved'])]:
            family_differences=[]
            for case in cases:
                rates={}
                for form in config['forms']:
                    rates[form]=sum(sum(r['metrics'][metric] for r in rows if r['assignment']['family_id']==case['family_id'] and r['assignment']['history']==history and r['assignment']['form']==form and r['assignment']['branch']==branch)/(len(config['orders'])*config['repetitions']) for branch in branches)/len(branches)
                family_differences.append({'family_id':case['family_id'],'difference':rates['focused']-rates['ordinary']})
            contrasts.append({'history':history,'metric':metric,'focused_minus_ordinary':sum(x['difference'] for x in family_differences)/len(cases),'family_differences':family_differences,'n_families':len(cases)})
    history_contrasts=[]
    for form,branch,comparison in product(config['forms'],config['branches'],['fresh','other']):
        for metric in ['needless_withholding','wrong_approval','wrong_approval_upper','original_status_correct','active_objections_correct','invalid']:
            diffs=[]
            for case in cases:
                rates={}
                for history in ['own',comparison]:
                    subset=[r for r in rows if all(r['assignment'][k]==v for k,v in [('family_id',case['family_id']),('form',form),('branch',branch),('history',history)])]
                    rates[history]=sum(r['metrics'][metric] for r in subset)/len(subset)
                diffs.append({'family_id':case['family_id'],'difference':rates['own']-rates[comparison]})
            history_contrasts.append({'form':form,'branch':branch,'metric':metric,'comparison':'own_minus_'+comparison,
                'difference':sum(d['difference'] for d in diffs)/len(diffs),'family_differences':diffs})
    return {'status':'development_pilot_descriptive_only','n_initial':len(initial),'n_final':len(rows),
        'n_independent_human_validated_cases':0,'overall':summarize(rows),'initial_outputs':initial,'cells':cells,
        'form_contrasts':contrasts,'history_contrasts':history_contrasts,'batch_latency':batch_records,'rows':rows,
        'inference_limits':'No confidence intervals or significance tests for three AI-authored development families. Semantic scoring by qualified humans remains outstanding.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if Path(__file__).resolve()!=a.release.resolve()/'analyze.py': raise SystemExit('Run the frozen release copy')
    result=analyze(a.release,a.output);save(a.output/'analysis.json',result)
    print(json.dumps({k:result[k] for k in ['status','n_initial','n_final','overall']},indent=2))
