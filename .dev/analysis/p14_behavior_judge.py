"""CPU-only arithmetic/source-evidence audit; never constructs a model."""
from pathlib import Path
from collections import Counter, defaultdict
import argparse,hashlib,json,math,statistics,sys,subprocess
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo',type=Path,default=next(p for p in Path(__file__).resolve().parents if (p/'jevgo/core.py').exists()))
parser.add_argument('--artifacts',type=Path)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
REPO=args.repo.resolve();sys.path.insert(0,str(REPO))
BASE=(args.artifacts or REPO/'.dev/artifacts/P14-20261017-pretrial').resolve()
OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
from jevgo.core import ResearchState
from jevgo.research import public_state
from transformers import AutoTokenizer
from semif_phase1.core import direct_messages
load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def soft(x):
 ex=[math.exp(z-max(x)) for z in x];return [z/sum(ex) for z in ex]
def close(a,b):assert len(a)==len(b) and all(math.isclose(x,y,abs_tol=1e-12,rel_tol=1e-12) for x,y in zip(a,b)),(a,b)
TOK=AutoTokenizer.from_pretrained(str(REPO/'models/Qwen3.5-4B'),local_files_only=True,trust_remote_code=False)
head=load(REPO/'models/JEV-9B/adapter_vllm/decision_head.json');temps=load(REPO/'models/JEV-9B/calibration.json')['per_kind']
paths={'9B4096':'a/4096-B3','9B8192':'a/8192-B3','SemIf8':'c/loops/horizon20-output256-seed0-sim8','SemIf2':'c/loops/horizon20-output256-seed0-sim2'}
summary={};examples={};allrows=[]
for label,rel in paths.items():
 p=BASE/rel; lines=p.joinpath('events.jsonl').read_text().splitlines();events=[(i,json.loads(s)) for i,s in enumerate(lines,1)]; cfg=load(p/'config.json')
 h=[]
 if label.startswith('9B'):
  h=[(i,json.loads(s)) for i,s in enumerate((BASE/(rel+'-http')/'http.jsonl').read_text().splitlines(),1) if 'request_finished' in s]
  h=[(i,x) for i,x in h if x['role']!='generation']; hi=0
 branch_state={}; pending=None; last_judge=None; last_gen=None; counts=Counter(); roles=defaultdict(list); roots=[]; transition_counts=Counter(); multi=[]; allcandidate=[]; stopped=[]; memory_noop=0; errors=[]
 examples[label]=[]
 for line,e in events:
  k=e['event_type'];b=e['branch_id'];d=e['payload']
  if k=='initial_state':branch_state[b]=d['state']
  if k=='transition':
   assert d['state']['observation']==d['step']['observation']
   assert d['state']['actions'][-1]==d['candidate']['action']
   branch_state[b]=d['state'];transition_counts[(d['phase'],d['candidate']['action']['name'])]+=1
   assert d['step']['reward']==0 and not d['step']['success']
  if k in ('generation_request','judge_request'):
   text=d['request'].get('context',d['request'].get('prompt'))
   actual=json.JSONDecoder().raw_decode(text.split('\nPublic state:\n',1)[1])[0]
   expected=public_state(ResearchState.model_validate(branch_state[b]))
   assert actual==expected,(label,line,'state mismatch')
   counts['public_state_matches_current_branch']+=1
   if k=='generation_request':last_gen=(line,actual)
   else:
    assert pending is None,(label,line,'unpaired judge');pending=(line,d['request'],actual)
  if k=='search_postprocessing_interrupted':
   assert pending is not None;counts['unsent_budget_blocked_judge_calls']+=1;pending=None
  if k=='judge':
   jl,req,state=pending;pending=None;assert req['role']==d['role'] and req['items']==d['items']; meta=d['metadata']; decisions=meta['decisions']; category=d['role']=='candidate' and len(req['items'])>1
   expected_scores=[]
   for di,dec in enumerate(decisions):
    if label.startswith('9B'):
     hl,http=h[hi];hi+=1;assert http['role']==d['role'];raw=load_response=json.loads(http['raw_response'])
     kind='choice' if category else 'noul';assert dec['kind']==kind
     lo,high=head['slots']['ranges'][kind];opts=[f'[item-{i}] {json.dumps(item,ensure_ascii=False)}' for i,item in enumerate(req['items'])] if category else ['false','true']
     assert dec['options']==opts
     ids=head['verbalizer_ids'][lo:lo+len(opts)];bias=head['bias'][lo:lo+len(opts)];assert ids==dec['option_token_ids'] and bias==dec['bias'] and temps[kind]==dec['temperature']
     rawlp=[raw['choices'][0]['logprobs']['top_logprobs'][0][f'token_id:{t}'] for t in ids];assert rawlp==dec['raw_logprobs']
     vals=[(x+y)/temps[kind] for x,y in zip(rawlp,bias)];probs=soft(vals);close(probs,dec['probabilities'])
     promptstate=req['context'] if category else req['context']+f'\nItem [item-{di}]:\n'+req['items'][di]
     optlines=[f'{chr(65+i)}) {s}' for i,s in enumerate(opts)] if category else opts
     prompt=f'[kind] {kind}\n[state] {promptstate}\n[question] {meta["question"]}\n[options]\n'+'\n'.join(optlines)+'\n[decision]:'
     request=http['request'];assert request['prompt']==prompt and request['allowed_token_ids']==ids and request['model']=='jev-decision'
     assert request['temperature']==1.0 and request['top_k']==0 and request['top_p']==1 and request['repetition_penalty']==1
     assert d['identity']['adapter']=='jev-decision'
     for name,digest in d['identity']['metadata']['asset_sha256'].items():assert sha(REPO/'models/JEV-9B'/name)==digest
     source={'http_line':hl,'request_number':http['request_number'],'raw_logprobs':rawlp,'bias':bias,'temperature':temps[kind],'token_ids':ids}
    else:
     opts=[{'id':f'item-{i}','description':item} for i,item in enumerate(req['items'])] if category else [{'id':'no','description':'No'},{'id':'yes','description':'Yes'}]
     row={'id':'candidate' if category else f'{d["role"]}:{di}','state':{'context':req['context']} if category else {'context':req['context'],'item':{'id':f'item-{di}','text':req['items'][di]}},'question':meta['question'],'options':opts}
     assert dec['option_ids']==[o['id'] for o in opts]
     prompt=TOK.apply_chat_template(direct_messages(row),tokenize=False,add_generation_prompt=True,enable_thinking=False)
     assert hashlib.sha256(prompt.encode()).hexdigest()==dec['prompt_sha256'],(label,line,di,'prompt digest')
     probs=soft(dec['option_logits']);close(probs,dec['probabilities']);assert d['identity']['adapter'] is None and meta['calibrated'] is False
     source={'option_logits':dec['option_logits'],'prompt_sha256':dec['prompt_sha256'],'slot_semantics':'A=no/B=yes for binary; A/B follow declared candidate order'}
    expected_scores.extend(probs if category else [probs[1]])
    counts['decision_prompt_and_score_reconciled']+=1
   close(expected_scores,d['scores']);roles[d['role']].extend(d['scores']);counts['judge_calls_reconciled']+=1
   chain={'condition':label,'judge_request_line':jl,'judge_result_line':line,'branch':b,'role':d['role'],'actions_before':len(state['actions']),'items':d['items'],'scores':d['scores'],'category':category,'decisions':decisions,'public_observation':state['observation'],'public_memory':state['memory'],'source_detail':source}
   if d['role']=='candidate':allcandidate.append(chain)
   if category:multi.append(chain)
   if not any(x['role']==d['role'] for x in examples[label]):examples[label].append(chain|{'full_context':req['context'],'full_rendered_prompt':prompt})
   last_judge=(line,d,chain);allrows.append({k:v for k,v in chain.items() if k not in ('public_observation','public_memory','items','decisions')})
  if k=='candidates':
   cs=d['candidates'];assert last_judge[1]['role']=='candidate';scores=last_judge[1]['scores'];assert len(cs)==len(scores)
   freq=[c['sample_count']/sum(z['sample_count'] for z in cs) for c in cs];prefs=[s/sum(scores) for s in scores] if sum(scores) else [1/len(scores)]*len(scores)
   close([(x+y)/2 for x,y in zip(freq,prefs)],[c['prior'] for c in cs]);assert [c['judge_score'] for c in cs]==scores
   counts[f'unique_candidates_{len(cs)}']+=1
   last_judge[2]['candidate_event_line']=line;last_judge[2]['candidate_priors']=[{'action':c['action'],'prior':c['prior'],'sample_count':c['sample_count'],'candidate_id':c['candidate_id']} for c in cs]
  if k=='memory_selection':
   rawstate=branch_state[b];records=[r for r in rawstate['evidence'] if r['kind']!='judge_opinion'];maximum=cfg['run']['options']['memory_records'];pool=records[-max(maximum,cfg['run']['candidate_count']):]
   assert [r['content'] for r in pool]==last_judge[1]['items'];selected=[r['evidence_id'] for r in pool];assert d['evidence_ids']==selected
   memory_noop+=1;rawstate['working_memory']=d['evidence_ids']
  if k=='completion_advice':
   assert d['objective_success'] is False;stopped.append((line,d['score']))
  if k=='search':
   res=d['result'];stats=res['rootstats'];assert len(stats)==1 and stats[0]['candidate']['action']['name']=='forward'
   assert stats[0]['P']==1 and stats[0]['Q']==stats[0]['W']==0 and stats[0]['N']==res['simulations']
   assert res['candidate']['candidate_id']==stats[0]['candidate']['candidate_id']
   roots.append({'line':line,'state_id':d['state_id'],**res})
 if h:assert hi==len(h)
 assert pending is None or label=='9B8192'
 s={'events_path':str(p/'events.jsonl'),'events_sha256':sha(p/'events.jsonl'),'counts':dict(counts),'raw_judge_HTTP_count':len(h) if h else None,'roles':{r:{'n':len(v),'min':min(v),'max':max(v),'mean':statistics.mean(v),'above_0_5':sum(x>.5 for x in v)} for r,v in roles.items()},'all_root_searches_single_forward_P1_Q0':True,'root_searches':len(roots),'transition_actions':{str(k):v for k,v in transition_counts.items()},'memory_selection_score_invariant_calls':memory_noop,'stop_advice_n':len(stopped),'stop_advice_above0_5':sum(v>.5 for l,v in stopped),'stop_does_not_change_objective_success':True,'unanswered_final_judge_request':pending,'multi_candidate_count':len(multi),'root_details':roots}
 summary[label]=s
 lowest=min(allcandidate,key=lambda c:min(c['scores']));examples[label].append(lowest|{'example':'minimum candidate raw judge score'})
 (OUT/(label+'-candidate-chain.json')).write_text(json.dumps(allcandidate,ensure_ascii=False,indent=2)+'\n')
 (OUT/(label+'-multi-chain.json')).write_text(json.dumps(multi,ensure_ascii=False,indent=2)+'\n')
 print(label,{k:v for k,v in s.items() if k not in ('root_details','unanswered_final_judge_request')},flush=True)
(OUT/'score-chain-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
(OUT/'raw-chain-examples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2)+'\n')
(OUT/'judge-readouts.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in allrows))
files=['jevgo/research.py','jevgo/runner.py','jevgo/search/mcts.py','jevgo/models/autotrust.py','jevgo/models/semif.py','jevgo/core.py']
version={f:{'sha256':sha(REPO/f),'matches_actual_gpu_commit':subprocess.check_output(['git','show','955df143e7e2db059cb870e8d4fed9e9741621a2:'+f],cwd=REPO)==(REPO/f).read_bytes()} for f in files}
assert all(v['matches_actual_gpu_commit'] for v in version.values())
(OUT/'source-identities.json').write_text(json.dumps(version,indent=2)+'\n')
