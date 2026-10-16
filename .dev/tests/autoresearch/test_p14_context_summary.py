"""Read-only P14a reporting regressions; no GPU or inference."""
import importlib.util
import json
from pathlib import Path

import pytest

path=Path(__file__).resolve().parents[2]/'analysis/p14_context_summary.py'
spec=importlib.util.spec_from_file_location('p14_context_summary',path)
summary=importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def test_zero_http_failure_remains_reportable_without_peak(tmp_path):
    cost=summary.http_summary(tmp_path)
    assert cost['http_requests']==0 and cost['peaks_including_rejected_attempts']=={}
    row={'condition':'4096-B0','window':4096,'memory_records':4,'method':'B0',
         'committed_steps':0,'objective_success':False,'termination_reason':'model_failure',
         'cost':cost,'peak_request':None,'original_role_frontier_9_actions_19_memories':False,
         'technical_closed_loop_pass':False,'sampled_window_gpu_peak_mib':None,
         'raw_error':'Failure before any HTTP preflight'}
    text=summary.markdown({'conditions':[row],'protocols':[],'missing':[],'limitations':[]})
    assert 'not observed' in text and 'Failure before any HTTP preflight' in text


def test_unknown_protocol_usage_cannot_print_complete_zero_cost(tmp_path):
    (tmp_path/'http.jsonl').write_text(json.dumps({'event':'request_finished','role':'generation',
        'network_sent':True,'request_number':1,'raw_response':'{"choices":[]}',
        'offline_input_tokens':65,'requested_output_tokens_per_sequence':128,
        'total_window_tokens':193,'remaining_tokens':3903})+'\n')
    cost=summary.http_summary(tmp_path)
    assert cost['http_requests']==1 and not cost['usage_complete']
    assert cost['roles']['generation']['unknown_input_requests']==1
    assert cost['roles']['generation']['unknown_output_requests']==1
    text=summary.markdown({'conditions':[],'protocols':[{'window':4096,'ok':False,'cost':cost}],
                           'missing':[],'limitations':[]})
    assert '>=0 (unknown remainder) input / >=0 (unknown remainder) output' in text


def test_protocol_declared_complete_with_unknown_raw_usage_rejected():
    cost={'http_requests':1,'known_input_tokens':0,'known_output_tokens':0,'usage_complete':False}
    usage={'model_requests':1,'input_tokens':0,'output_tokens':0}
    result={'usage':usage,'usage_complete':True}
    with pytest.raises(AssertionError,match='Result cost completeness'):
        summary.reconcile_totals(cost,{'used':usage},result)


@pytest.mark.parametrize('stale_source',['ledger','result','audit'])
def test_stale_ledger_result_or_audit_totals_rejected(stale_source):
    cost={'http_requests':1,'known_input_tokens':10,'known_output_tokens':2,'usage_complete':True}
    usage={'model_requests':1,'input_tokens':10,'output_tokens':2}
    ledger={'used':dict(usage)}
    result={'usage':dict(usage),'termination_reason':'model_failure','error':'failed','final_state':{'actions':[]}}
    audit={'usage_from_events':dict(usage),'usage_complete':True,'termination_reason':'model_failure',
           'raw_error':'failed','committed_steps':0}
    summary.reconcile_totals(cost,ledger,result,audit)
    selected={'ledger':ledger['used'],'result':result['usage'],'audit':audit['usage_from_events']}[stale_source]
    selected['input_tokens']=11
    with pytest.raises(AssertionError,match='current ledger/result/audit'):
        summary.reconcile_totals(cost,ledger,result,audit)
