"""Offline channel tests plus exact compatibility with all archived Gemma diagnostic outputs."""
from pathlib import Path
import json,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_resolution_diagnostic import score
from run_gemma_reasoning_check import final_answer
ROOT=Path(__file__).resolve().parents[2]

class ParserTests(unittest.TestCase):
    def test_thought_json_never_scores(self):
        raw='<|channel>thought\n{"decision":"FULL"}'
        answer,meta=final_answer(raw)
        self.assertEqual(answer,''); self.assertFalse(meta['valid_channel'])
    def test_only_final_answer(self):
        raw='<|channel>thought\n{"decision":"FULL"}<channel|>{"decision":"HOLD"}<turn|>'
        answer,meta=final_answer(raw)
        self.assertEqual(json.loads(answer)['decision'],'HOLD');self.assertTrue(meta['valid_channel'])
    def test_unknown_and_repeated_channels(self):
        for raw in ['<|channel>other\n{"decision":"FULL"}', '<|channel>thought\nx<channel|>y<channel|>z']:
            self.assertFalse(final_answer(raw)[1]['valid_channel'])
    def test_truncation_remains_error(self):
        row=json.loads((ROOT/'results/incident_resolution_diagnostic/assignments.json').read_text())[0]
        text=json.dumps(row['gold'])
        self.assertFalse(score(row,{'text':text,'truncated':True})[1]['correct_decision'])
    def test_all_288_archived_outputs(self):
        n=0
        for p in sorted((ROOT/'results/incident_resolution_diagnostic/gemma_v1/batches').glob('*.json')):
            for row in json.loads(p.read_text()):
                text,meta=final_answer(row['output']['raw_text'])
                self.assertTrue(meta['valid_channel'])
                self.assertEqual(text,row['output']['text'])
                self.assertEqual(score(row['assignment'],{**row['output'],'text':text})[1],row['metrics'])
                n+=1
        self.assertEqual(n,288)
    def test_factorial_assignments(self):
        rows=json.loads((ROOT/'results/incident_resolution_diagnostic/assignments.json').read_text())
        self.assertEqual(len(rows),288)
        keys={(r['base_id'],r['sufficient'],r['position'],r['form'],r['rep']) for r in rows}
        self.assertEqual(len(keys),288)
        self.assertEqual(len({r['base_id'] for r in rows}),12)
        for sufficient in [False,True]:
            for position in ['first','last']:
                for form in ['primary','first_listed','decision_only']:
                    cell=[r for r in rows if r['sufficient']==sufficient and r['position']==position and r['form']==form]
                    self.assertEqual(len(cell),24)

if __name__=='__main__': unittest.main()
