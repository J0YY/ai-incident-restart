"""Descriptive plot of an exploratory follow-up chosen after the main study."""
from pathlib import Path
import json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'tmp/pdfs/who_asks_the_model/focused_revision/figures';OUT.mkdir(parents=True,exist_ok=True)
p=ROOT/'results/incident_resolution_diagnostic/gemma_v1/analysis.json'
a=json.loads(p.read_text());forms=['primary','first_listed','decision_only']
cells=[next(c for c in a['cells'] if c['form']==form and c['position']=='last' and c['sufficient']) for form in forms]
counts=[c['false_withhold_n'] for c in cells];assert counts==[9,1,0];assert all(c['n']==24 for c in cells)
other=[next(c for c in a['cells'] if c['form']==form and c['position']=='last' and not c['sufficient']) for form in forms]
assert all(c['false_full_n']==0 for c in other)
plt.rcParams.update({'font.family':'STIXGeneral','font.size':9.5,'pdf.fonttype':42})
f,ax=plt.subplots(figsize=(5.15,1.80));f.subplots_adjust(left=.34,right=.98,top=.90,bottom=.28)
y=[2,1,0]
for yi,n in zip(y,counts):
 ax.hlines(yi,0,24,color='#dce0e3',linewidth=.6,zorder=1)
 ax.scatter(n,yi,s=32,color='#34556a',edgecolors='white',linewidths=.5,zorder=3)
 ax.text(n+.7,yi,f'{n} / 24',va='center',fontsize=9.5,color='#263943')
ax.set(yticks=y,yticklabels=['First test called primary','First-listed wording','No first-test fields'],xticks=[0,6,12,18,24],xlim=(-.6,24.5),ylim=(-.45,2.45))
ax.set_xlabel('Needless refusals among 24 responses',labelpad=4)
for side in ['top','left','right']:ax.spines[side].set_visible(False)
ax.spines['bottom'].set_color('#afb7bd');ax.spines['bottom'].set_linewidth(.6)
ax.tick_params(axis='y',length=0,pad=8);ax.tick_params(axis='x',length=3,width=.6,color='#afb7bd')
f.canvas.draw();ren=f.canvas.get_renderer()
for txt in [*ax.texts,*ax.get_xticklabels(),*ax.get_yticklabels(),ax.xaxis.label]:
 b=txt.get_window_extent(ren);r=f.bbox
 assert b.x0>=r.x0 and b.x1<=r.x1 and b.y0>=r.y0 and b.y1<=r.y1,txt.get_text()
f.savefig(OUT/'gemma_form_misses.pdf');f.savefig(OUT/'gemma_form_misses.png',dpi=190);plt.close(f)
record={'source':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'selection':{'model':a['model'],'position':'last','sufficient':True,'forms':forms},'counts':counts,'denominators':[c['n'] for c in cells],'base_cases_per_cell':12,'samples_per_base':2,'wrong_full_approvals_in_corresponding_insufficient_cells':[c['false_full_n'] for c in other],'interpretation':'Descriptive observed counts, no new test or interval. Appendix reports all original paired contrasts and their small-base limitations.'}
(OUT/'dotplot_data.json').write_text(json.dumps(record,indent=2)+'\n')
print('Gemma dot plot verified against retained analysis, with full 0-24 count scale.')
