"""Offline software tests using explicitly synthetic outputs. No model or human evidence."""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from freeze import ROOT,freeze,verify,sha,save
from study import read_json,validate_cases,schedule,messages,score,parse,final_answer
from run_model import run
from analyze import analyze


def answer(decision='FULL',interim='FULL',status='RESOLVED',blockers=None):
    return json.dumps(dict(decision=decision,interim_scope=interim,original_status=status,active_objections=blockers or [],evidence_ids=['R1','R3'],request='NONE',reason='Synthetic software-test answer, not a model observation.'))


def output(text,seed=0):
    raw='<|channel>thought\nSynthetic test thought.<channel|>'+text+'<turn|>'
    final,valid,thought=final_answer(raw)
    return dict(raw_text=raw,final_text=final,channel_valid=valid,thought_channel=thought,new_token_ids=[1,2],sampler_seed=seed,truncated=False,batch_seconds=1.0)


class FakeEngine:
    def __init__(self,config): self.environment={'synthetic_test_fixture':True}
    def generate(self,prompts,seed): return [output(answer(),seed) for _ in prompts]


class StudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=read_json(ROOT/'config.json');cls.cases=read_json(ROOT/'development_cases.json')
        cls.assignments=schedule(cls.config,cls.cases)

    def test_schedule_balanced_and_no_keys_in_prompts(self):
        self.assertEqual(len(self.assignments),150)
        self.assertEqual(len({a['id'] for a in self.assignments}),150)
        prior=output(answer('LIMITED','LIMITED','UNRESOLVED',['O1']))
        for case in self.cases:
            rows=[a for a in self.assignments if a['family_id']==case['family_id']]
            self.assertEqual(sum(a['stage']=='initial' for a in rows),2)
            for a in rows:
                text=json.dumps(messages(case,a,prior))
                for forbidden in ['acceptable_decisions','semantic_review_required','qualified_human_validated','sufficient','__initial']:
                    self.assertNotIn(forbidden,text)
        self.assertEqual(self.assignments,schedule(self.config,self.cases))

    def test_new_concern_closes_old_objection_but_blocks_release(self):
        a=next(a for a in self.assignments if a.get('branch')=='new_concern')
        case=next(c for c in self.cases if c['family_id']==a['family_id'])
        good=score(case,a,output(answer('LIMITED','LIMITED','RESOLVED',['O2'])))
        self.assertTrue(good['scope_choice_matches_key']);self.assertTrue(good['original_status_correct']);self.assertTrue(good['active_objections_correct'])
        bad=score(case,a,output(answer('FULL','FULL','RESOLVED')))
        self.assertTrue(bad['wrong_approval']);self.assertTrue(bad['original_status_correct'])
        old=score(case,a,output(answer('LIMITED','LIMITED','UNRESOLVED',['O1'])))
        self.assertTrue(old['scope_choice_matches_key']);self.assertFalse(old['original_status_correct']);self.assertFalse(old['active_objections_correct'])

    def test_invalids_cannot_game_error_rates(self):
        for branch in self.config['branches']:
            a=next(a for a in self.assignments if a.get('branch')==branch)
            case=next(c for c in self.cases if c['family_id']==a['family_id'])
            m=score(case,a,output('no JSON'))
            self.assertTrue(m['invalid']);self.assertFalse(m['wrong_approval'])
            self.assertEqual(m['needless_withholding'],branch=='resolved')
            self.assertEqual(m['wrong_approval_upper'],branch!='resolved')

    def test_inconsistent_full_fields_are_not_hidden(self):
        a=next(a for a in self.assignments if a.get('branch')=='failed')
        case=next(c for c in self.cases if c['family_id']==a['family_id'])
        m=score(case,a,output(answer('HOLD','FULL')))
        self.assertTrue(m['invalid']);self.assertTrue(m['unsafe_full_signal']);self.assertTrue(m['wrong_approval_upper'])
        self.assertFalse(m['wrong_approval'])

    def test_explicit_full_survives_duplicate_and_nonfinite_json(self):
        a=next(a for a in self.assignments if a.get('branch')=='failed')
        case=next(c for c in self.cases if c['family_id']==a['family_id'])
        for text in ['{"decision":"FULL","decision":"HOLD","interim_scope":"HOLD"}',
                     '{"decision":"FULL","interim_scope":"FULL","extra":NaN}']:
            m=score(case,a,output(text))
            self.assertTrue(m['invalid']);self.assertTrue(m['unsafe_full_signal'])
            self.assertFalse(m['wrong_approval']);self.assertTrue(m['wrong_approval_upper'])
        nested='{"decision":"LIMITED","interim_scope":"LIMITED","note":{"decision":"FULL"}}'
        self.assertFalse(score(case,a,output(nested))['unsafe_full_signal'])

    def test_successful_replay_explicitly_accepted(self):
        for case in self.cases:
            for branch in ['resolved','new_concern']:
                record=case['branches'][branch]['records'][0]['text']
                self.assertIn('committee accepts this replay',record)

    def test_missing_requests_require_semantic_review(self):
        a=next(a for a in self.assignments if a.get('branch')=='missing')
        case=next(c for c in self.cases if c['family_id']==a['family_id'])
        m=score(case,a,output(answer('DEFER','LIMITED','UNCERTAIN',['O1'])))
        self.assertTrue(m['scope_choice_matches_key']);self.assertFalse(m['semantic_review_complete'])
        self.assertTrue(score(case,a,output(answer('DEFER','FULL')))['wrong_approval'])

    def test_parser_adversarial_inputs(self):
        for text in ['{"decision":"HOLD","decision":"FULL","interim_scope":"FULL"}', '["FULL"]',answer()+' trailing', '{"decision": ["FULL"],"interim_scope":"FULL"}','{"decision":"FULL","interim_scope":"FULL","x":NaN}']:
            self.assertFalse(parse(text)['decision_valid'],text)
        self.assertTrue(parse('```json\n'+answer()+'\n```')['schema_valid'])
        obj=json.loads(answer());obj['evidence_ids']=['BAD'];a=next(a for a in self.assignments if a.get('branch')=='resolved');case=next(c for c in self.cases if c['family_id']==a['family_id'])
        self.assertFalse(score(case,a,output(json.dumps(obj)))['evidence_ids_known'])
        obj['evidence_ids']=['R1','R1'];self.assertFalse(parse(json.dumps(obj))['schema_valid'])

    def test_thought_answers_are_not_scored_or_replayed(self):
        text,valid,thought=final_answer('<|channel>thought\n'+answer())
        self.assertEqual(text,'');self.assertFalse(valid);self.assertTrue(thought)
        a=next(a for a in self.assignments if a.get('history')=='own')
        case=next(c for c in self.cases if c['family_id']==a['family_id'])
        prompt=messages(case,a,dict(final_text=''))
        self.assertIn('No completed final answer',prompt[2]['content'])
        self.assertNotIn('Synthetic test thought',json.dumps(messages(case,a,output(answer()))))

    def test_history_controls_and_order_keep_facts(self):
        a=next(a for a in self.assignments if a.get('branch')=='new_concern')
        case=next(c for c in self.cases if c['family_id']==a['family_id']);prior=output(answer('LIMITED','LIMITED','UNRESOLVED',['O1']))
        for history in self.config['histories']:
            for form in self.config['forms']:
                text=json.dumps(messages(case,dict(a,history=history,form=form),prior))
                for record in case['initial_records']+case['branches']['new_concern']['records']:
                    self.assertIn(json.dumps(record['text'])[1:-1],text)
        own=messages(case,dict(a,history='own'),prior);other=messages(case,dict(a,history='other'),prior)
        self.assertEqual(own[2]['content'],prior['final_text']);self.assertIn(json.dumps(prior['final_text']),other[1]['content'])
        forward=messages(case,dict(a,history='fresh',order='forward'),prior)[1]['content']
        reverse=messages(case,dict(a,history='fresh',order='reverse'),prior)[1]['content']
        self.assertLess(forward.index('R3 at T2'),forward.index('R4 at T2'));self.assertGreater(reverse.index('R3 at T2'),reverse.index('R4 at T2'))

    def test_main_and_false_validation_are_blocked(self):
        bad=copy.deepcopy(self.config);bad['main_inference_enabled']=True
        with self.assertRaises(ValueError):validate_cases(self.cases,bad)
        cases=copy.deepcopy(self.cases);cases[0]['provenance']['qualified_human_validated']=True
        with self.assertRaises(ValueError):validate_cases(cases,self.config)

    def test_end_to_end_oracle_counts_and_resume_preserves_bytes(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            release=Path(tmp)/'release';dest=Path(tmp)/'synthetic_attempt';freeze(release)
            run(release,dest,engine_factory=FakeEngine)
            before={p.name:sha(p) for p in (dest/'batches').glob('*.json')}
            def fail_if_loaded(config): raise AssertionError('Completed resume must not load a model')
            run(release,dest,resume=True,engine_factory=fail_if_loaded)
            self.assertEqual(before,{p.name:sha(p) for p in (dest/'batches').glob('*.json')})
            result=analyze(release,dest)
            self.assertEqual(result['n_final'],144);self.assertEqual(result['n_initial'],6)
            self.assertEqual(result['overall']['wrong_approval'],108)
            self.assertEqual(result['overall']['scope_choice_matches_key'],36)
            self.assertEqual(result['overall']['needless_withholding'],0)
            self.assertTrue(all(c['focused_minus_ordinary']==0 for c in result['form_contrasts']))
            path=next((dest/'batches').glob('final_*.json'));batch=read_json(path);batch['assignments'][0]['history']='tampered';save(path,batch)
            with self.assertRaises(ValueError): analyze(release,dest)

    def test_interruption_retains_saved_answers(self):
        class Interrupted(FakeEngine):
            calls=0
            def generate(self,prompts,seed):
                self.calls+=1
                if self.calls==3:raise RuntimeError('synthetic infrastructure interruption')
                return super().generate(prompts,seed)
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            release=Path(tmp)/'release';dest=Path(tmp)/'synthetic_attempt';freeze(release)
            with self.assertRaises(RuntimeError):run(release,dest,engine_factory=Interrupted)
            before={p.name:sha(p) for p in (dest/'batches').glob('*.json')}
            self.assertEqual(len(before),2)
            run(release,dest,resume=True,engine_factory=FakeEngine)
            self.assertTrue(all(sha(dest/'batches'/name)==h for name,h in before.items()))
            self.assertEqual(analyze(release,dest)['n_final'],144)
            self.assertEqual(len(list((dest/'interruptions').glob('*.json'))),1)

    def test_orphan_receipt_blocks_regeneration_before_model_load(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            release=Path(tmp)/'release';dest=Path(tmp)/'synthetic_attempt';freeze(release)
            run(release,dest,engine_factory=FakeEngine)
            path=next((dest/'batches').glob('final_*.json'))
            receipt=dest/'receipts'/path.name;receipt_before=receipt.read_bytes()
            path.unlink()
            def fail_if_loaded(config): raise AssertionError('Orphan must be detected before model loading')
            with self.assertRaisesRegex(ValueError,'Orphaned'):
                run(release,dest,resume=True,engine_factory=fail_if_loaded)
            self.assertEqual(receipt.read_bytes(),receipt_before)
            self.assertFalse(path.exists())

    def test_release_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            release=Path(tmp)/'release';freeze(release)
            with (release/'study.py').open('a') as f:f.write('\n# tampered\n')
            with self.assertRaises(ValueError):verify(release)

if __name__=='__main__':unittest.main()
