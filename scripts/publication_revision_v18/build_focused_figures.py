"""Compact serif diagrams for the incident-governance manuscript, revision 12."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties,findfont
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'tmp/pdfs/who_asks_the_model/focused_revision/figures'
OUT.mkdir(parents=True,exist_ok=True)
# STIX matches conventional journal typography and ships with Matplotlib.
plt.rcParams.update({'font.family':'STIXGeneral','font.size':9.2,'pdf.fonttype':42,'ps.fonttype':42})
INK='#20262d';BLUE='#435c70';GREEN='#486659';RED='#7b5148';BG='#f7f8f9'
checks=[]
def canvas(height):
 f=plt.figure(figsize=(6.35,height));ax=f.add_axes([0,0,1,1]);ax.set(xlim=(0,6.35),ylim=(0,height));ax.axis('off');return f,ax

def box(ax,x,y,w,h,lines,color=BLUE,fill=BG,size=9.2):
 rect=Rectangle((x,y),w,h,facecolor=fill,edgecolor=color,lw=.65)
 ax.add_patch(rect)
 txt=ax.text(x+w/2,y+h/2,lines,ha='center',va='center',fontsize=size,color=INK,linespacing=1.35)
 checks.append((ax,rect,txt))

def arrow(ax,x1,y1,x2,y2):
 ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops={'arrowstyle':'->','lw':.7,'color':INK,'shrinkA':4,'shrinkB':4})

def save(f,name):
 f.canvas.draw();renderer=f.canvas.get_renderer()
 for ax,rect,txt in checks:
  if ax.figure is not f:continue
  r=rect.get_window_extent(renderer);t=txt.get_window_extent(renderer)
  # Require at least 4 pt of padding on every side.
  pad=f.dpi*4/72
  assert t.x0>=r.x0+pad and t.x1<=r.x1-pad and t.y0>=r.y0+pad and t.y1<=r.y1-pad,(name,txt.get_text(),r.bounds,t.bounds)
 for ax in f.axes:
  for txt in ax.texts:
   if not txt.get_text():continue
   t=txt.get_window_extent(renderer);r=f.bbox
   assert t.x0>=r.x0 and t.x1<=r.x1 and t.y0>=r.y0 and t.y1<=r.y1,(name,'canvas',txt.get_text())
 f.savefig(OUT/(name+'.pdf'))
 f.savefig(OUT/(name+'.png'),dpi=190)
 plt.close(f)

f,ax=canvas(2.85)
ax.text(.12,2.66,'1   Main comparison of review instructions',fontsize=10.2,color=INK)
box(ax,.12,1.74,1.57,.70,'36 scenarios\n4 input configurations each\n144 incident packets')
box(ax,2.02,1.74,2.43,.70,'Same packet under three guides\nAssess evidence  /  List records\nMatch test to intended work')
box(ax,4.78,1.74,1.45,.70,'Model recommends\nFull operation,\nlimited work, or hold')
arrow(ax,1.69,2.09,2.02,2.09);arrow(ax,4.45,2.09,4.78,2.09)
ax.text(.12,1.42,'2   Exploratory diagnostic after the main results',fontsize=10.2,color=INK)
box(ax,.12,.50,1.57,.70,'12 cases with two tests\nAdditional test either\nsufficient or insufficient',size=9.2)
box(ax,2.02,.50,2.43,.70,'Additional record first or last\nThree answer formats\nPrimary  /  First-listed  /  Overall')
box(ax,4.78,.50,1.45,.70,'Model recommends\nFull operation,\nlimited work, or hold')
arrow(ax,1.69,.85,2.02,.85);arrow(ax,4.45,.85,4.78,.85)
ax.text(3.175,.15,'Twelve diagnostic bases reuse the main study templates; two responses per condition.',ha='center',fontsize=9.1,color=INK)
save(f,'experimental_design')

# Preserve the same transparent post-outcome example selection as revision 11.
rows=[]
quotes=['Followup validation content matches requested operation content.','Followup validation is not the primary validation.']
for p in sorted((ROOT/'results/incident_review_study/gemma_main_v3_merged/batches').glob('*.json')):
 for r in json.loads(p.read_text()):
  a=r['assignment'];t=r['output']['text']
  if a['kind']=='resolution' and a['gold']['decision']=='FULL' and not r['metrics']['decision'] and all(q in t for q in quotes):
   rows.append((a['case_id'],a['guide'],a['style'],a['rep'],p,r))
chosen=sorted(rows,key=lambda x:x[:4])[0];r=chosen[-1];p=chosen[-2]
f,ax=canvas(2.35)
ax.text(.12,2.15,'The rule permits any qualifying test; the form calls the first test primary.',fontsize=10,color=INK)
box(ax,.12,1.29,1.85,.57,'First test, called primary\nPasses under different conditions',color=RED,fill='#faf7f5',size=9.1)
box(ax,2.25,1.29,1.85,.57,'Additional test\nPasses under intended conditions',color=GREEN,fill='#f5f8f6',size=9.1)
box(ax,4.38,1.29,1.85,.57,'Correct recommendation\nRestore full operation',color=GREEN,fill='#f5f8f6',size=9.1)
arrow(ax,4.10,1.575,4.38,1.575)
ax.text(.12,1.01,'Gemma reports a match but follows the primary framing',fontsize=10,color=INK)
ax.text(.22,.70,'“'+quotes[0]+'”',fontsize=9.1,color=INK)
ax.text(.22,.44,'“'+quotes[1]+'”',fontsize=9.1,color=INK)
ax.text(.12,.13,'Gemma recommends limited work, even though the packet meets the approval rule.',fontsize=9.2,color=RED)
save(f,'evidence_example')
meta={'type':'Conceptual experimental-design diagram and exact raw-output illustration. No new analysis.','example_selection':'Lexicographically first (case_id, guide, style, rep) incorrect resolution response with FULL gold containing both displayed quotations.','eligible_n':len(rows),'assignment':{k:r['assignment'][k] for k in ['case_id','guide','style','rep']},'batch':str(p.relative_to(ROOT)),'batch_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'quotes':quotes,'output':r['output']['text'],'reference_decision':'FULL','model_decision':r['parsed']['values']['decision'],'layout':{'font':'STIXGeneral','minimum_box_padding_points':4,'design_inches':[6.35,2.85],'example_inches':[6.35,2.35]},'design':{'core_bases':36,'configurations':4,'guides':3,'diagnostic_bases':12,'sufficiency_conditions':2,'orders':2,'forms':3,'repetitions':2,'models':2}}
(OUT/'diagram_record.json').write_text(json.dumps(meta,indent=2)+'\n')
print('Created two compact serif diagrams; all labels pass canvas and box-padding checks.')
