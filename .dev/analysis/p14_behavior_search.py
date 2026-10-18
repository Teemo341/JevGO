"""Replay recorded search I/O through unchanged formal MCTS; no environment/model."""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
from dataclasses import dataclass
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo',type=Path,default=next(p for p in Path(__file__).resolve().parents if (p/'jevgo/core.py').exists()))
parser.add_argument('--artifacts',type=Path)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
R=args.repo.resolve();sys.path.insert(0,str(R));B=(args.artifacts or R/'.dev/artifacts/P14-20261017-pretrial').resolve();O=args.output.resolve();O.mkdir(parents=True,exist_ok=False)
from jevgo.core import Candidate,RunConfig,BudgetLedger,BudgetLimits,UsagePhase,ValueEstimate,ResearchState,GenerationResponse,JudgeResponse,ModelIdentity,Usage
from jevgo.search.mcts import MCTS
from jevgo.research import ResearchPolicy
@dataclass
class State:
 branch:str
 terminal:bool=False
class Replay:
 def __init__(self,ops):
  self.ops=ops;self.pos=0;self.ledger=BudgetLedger(BudgetLimits());self.capabilities=SimpleNamespace(require_branching=lambda:None);self.trace=[];self.modes={}
 def pop(self,kind,state):
  line,e=self.ops[self.pos];assert e['event_type']==kind,(line,kind,e['event_type']);self.pos+=1
  self.trace.append({'line':line,'operation':kind,'from_branch':state.branch});return line,e
 def candidates(self,state,*,phase):
  line,e=self.pop('candidates',state);assert e['branch_id']==state.branch
  return tuple(Candidate.model_validate(c) for c in e['payload']['candidates'])
 def transition(self,state,candidate,*,phase):
  line,e=self.pop('transition',state);d=e['payload'];assert d['candidate']['candidate_id']==candidate.candidate_id
  # Search successor branches are the complete parent branch plus one new suffix.
  assert e['branch_id'].rsplit('/',1)[0]==state.branch,(line,state.branch,e['branch_id'])
  s=d['state'];return SimpleNamespace(state=State(e['branch_id'],s['success'] or s['terminated'] or s['truncated']),reward=d['step']['reward'])
 def evaluate(self,state,*,phase):return ValueEstimate(value=0,source='heuristic',rationale='Recorded formal zero bootstrap')
class TracedMCTS(MCTS):
 def _expand(self,domain,node):
  before=domain.pos;super()._expand(domain,node);domain.modes[domain.ops[before][0]]='tree_expand'
 def _rollout(self,domain,node,rewards):
  before=domain.pos;answer=super()._rollout(domain,node,rewards)
  for line,e in domain.ops[before:domain.pos]:
   if e['event_type']=='candidates':domain.modes.setdefault(line,'greedy_rollout')
  return answer
records=[]
for label,path in {'9B4096':'a/4096-B3','9B8192':'a/8192-B3','SemIf8':'c/loops/horizon20-output256-seed0-sim8','SemIf2':'c/loops/horizon20-output256-seed0-sim2'}.items():
 cfg=RunConfig.model_validate(json.loads((B/path/'config.json').read_text())['run']);ops=[];n=0
 for line,text in enumerate((B/path/'events.jsonl').read_text().splitlines(),1):
  e=json.loads(text);k=e['event_type']
  if k=='candidates' or (k=='transition' and e['payload']['phase']=='search'):ops.append((line,e))
  if k=='search':
   n+=1;expected=e['payload']['result']
   if expected['stop_reason']=='budget_exhausted':
    records.append({'condition':label,'search_number':n,'search_line':line,'status':'not_replayed_with_unlimited_fixture_budget','reason':'Original budget stop retained; arithmetic/root selection verified separately'});ops=[];continue
   replay=Replay(ops);actual=TracedMCTS(cfg).search(replay,State('root')).model_dump(mode='json')
   assert actual==expected,(label,line,actual,expected);assert replay.pos==len(ops)
   multi=[{'candidate_line':l,'mode':replay.modes[l],'candidates':e2['payload']['candidates']} for l,e2 in ops if e2['event_type']=='candidates' and len(e2['payload']['candidates'])>1]
   records.append({'condition':label,'search_number':n,'search_line':line,'status':'exact_MCTS_result_and_call_order_match','candidate_calls':sum(e2['event_type']=='candidates' for l,e2 in ops),'search_transitions':sum(e2['event_type']=='transition' for l,e2 in ops),'multi_candidate_modes':multi,'trace':replay.trace});ops=[]
# Explicit singleton score counterfactual through actual ResearchPolicy.
initial=next(json.loads(s)['payload']['state'] for s in (B/'a/4096-B3/events.jsonl').read_text().splitlines() if json.loads(s)['event_type']=='initial_state')
state=ResearchState.model_validate(initial);identity=ModelIdentity(backend='cpu_fixture',model='no_model_loaded')
class Gen:
 def generate(self,request,**kwargs):return GenerationResponse(texts=tuple(json.dumps({'action':{'name':'forward','arguments':{}},'hypothesis':f'variant{i}','rationale':'fixture'}) for i in range(request.n)),identity=identity,usage=Usage())
class Judge:
 def __init__(self,value):self.value=value
 def score(self,request,**kwargs):return JudgeResponse(scores=tuple(self.value for _ in request.items),identity=identity,usage=Usage())
fixture=[]
for score in [0.,1e-9,.01,.5,1.]:
 policy=ResearchPolicy(Gen(),Judge(score),RunConfig(method='B3',candidate_count=4))
 candidates=policy.candidates(state,ledger=BudgetLedger(BudgetLimits()))
 assert len(candidates)==1 and candidates[0].prior==1 and candidates[0].judge_score==score
 fixture.append({'raw_judge_score':score,'unique_actions':len(candidates),'prior':candidates[0].prior,'sample_count':candidates[0].sample_count,'first_hypothesis_retained':candidates[0].hypothesis})
(O/'search-replay.json').write_text(json.dumps({'unchanged_formal_MCTS_replay':records,'singleton_score_counterfactual':fixture,'models_loaded':0,'GPU_calls':0,'environment_steps':0},ensure_ascii=False,indent=2)+'\n')
print('Exact search replays',sum(x['status'].startswith('exact') for x in records),'budget stopped retained',sum(not x['status'].startswith('exact') for x in records),'singleton fixtures',fixture)
