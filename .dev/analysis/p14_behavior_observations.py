"""CPU-only MiniGrid observation audit. Private state is diagnostic output, never model input."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / 'jevgo').is_dir() and (parent / 'configs').is_dir())
sys.path.insert(0, str(ROOT))
import gymnasium as gym
import minigrid
import numpy as np
from minigrid.core.actions import Actions
from minigrid.core.constants import DIR_TO_VEC, IDX_TO_COLOR, IDX_TO_OBJECT
from jevgo.core import Action, BudgetLedger, EpisodeSpec, UsagePhase
from jevgo.envs.minigrid import MiniGridAdapter, _ACTION_INDEX
from jevgo.research import initialize_state, public_state


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def diagnostic_world(env):
    """Only this audit reads private simulator position; never passed to a policy/model."""
    world = env.unwrapped
    front = world.grid.get(*world.front_pos)
    goals = [(x, y) for x in range(world.width) for y in range(world.height)
             if world.grid.get(x, y) is not None and world.grid.get(x, y).type == 'goal']
    return {'agent_pos': [int(x) for x in world.agent_pos], 'direction': int(world.agent_dir),
            'step_count': int(world.step_count), 'front_pos': [int(x) for x in world.front_pos],
            'front_object': front.type if front else 'empty',
            'goal_world_positions': [list(p) for p in goals],
            'goal_view_positions': [[int(x) for x in world.get_view_coords(*p)] for p in goals],
            'private_diagnostic_only': True}


def observation(raw):
    """Verify the adapter against upstream array cells, not a row-major interpretation."""
    result = MiniGridAdapter._observation(raw).model_dump(mode='json')
    image = np.asarray(raw['image'])
    assert result['data'] == {'image': image.tolist(), 'direction': int(raw['direction']), 'mission': raw['mission']}
    text_rows = result['text'].splitlines()[3:]
    assert len(text_rows) == 7
    for y, line in enumerate(text_rows):
        cells = line.split(' | ')
        assert len(cells) == 7
        for x, cell in enumerate(cells):
            kind, color, state = map(int, image[x, y])
            expected = IDX_TO_OBJECT[kind]
            if expected not in ('unseen', 'empty'):
                expected = IDX_TO_COLOR[color] + ' ' + expected
            if kind == 4:
                expected += ' ' + ('open', 'closed', 'locked')[state]
            assert cell == expected, (x, y, cell, expected)
    return result


class Oracle:
    def __init__(self, env_id='MiniGrid-Empty-5x5-v0', seed=0, max_steps=20):
        self.env_id, self.seed, self.max_steps = env_id, seed, max_steps
        self.cache = {}

    def replay(self, actions):
        key = tuple(action['name'] for action in actions)
        assert all(not action['arguments'] for action in actions)
        if key in self.cache:
            return self.cache[key]
        env = gym.make(self.env_id, agent_view_size=7, render_mode=None, max_steps=self.max_steps)
        try:
            raw, _ = env.reset(seed=self.seed)
            step = {'observation': observation(raw), 'action_success': None, 'reward': 0.0,
                    'terminated': False, 'truncated': False, 'success': False, 'objective_score': None, 'info': {}}
            for name in key:
                before = diagnostic_world(env)
                raw, reward, terminated, truncated, info = env.step(Actions[name])
                after = diagnostic_world(env)
                # Empty5x5 has no manipulable objects: position/orientation captures world changes.
                changed = before['agent_pos'] != after['agent_pos'] or before['direction'] != after['direction']
                step = {'observation': observation(raw), 'action_success': changed, 'reward': float(reward),
                        'terminated': bool(terminated), 'truncated': bool(truncated),
                        'success': bool(terminated and reward > 0), 'objective_score': float(reward), 'info': {}}
            value = {'step': step, 'diagnostic_world': diagnostic_world(env)}
            self.cache[key] = value
            return value
        finally:
            env.close()


def actions(names):
    return [{'name': name, 'arguments': {}} for name in names]


def inspect_coordinates():
    env = gym.make('MiniGrid-Empty-5x5-v0', agent_view_size=7, render_mode=None, max_steps=20)
    try:
        env.reset(seed=0)
        rows = []
        for direction in range(4):
            world = env.unwrapped
            world.agent_dir = direction
            raw = world.gen_obs()
            current = observation(raw)
            agent = world.get_view_coords(*world.agent_pos)
            front = world.get_view_coords(*world.front_pos)
            right = world.get_view_coords(*(np.asarray(world.agent_pos) + world.right_vec))
            assert tuple(agent) == (3, 6) and tuple(front) == (3, 5) and tuple(right) == (4, 6)
            rows.append({'world_direction': direction, 'world_forward_vector': DIR_TO_VEC[direction].tolist(),
                         'agent_view_xy': [int(x) for x in agent], 'front_view_xy': [int(x) for x in front],
                         'right_view_xy': [int(x) for x in right], 'observation': current})
        raw = env.unwrapped.gen_obs()
        poisoned = {**raw, 'agent_pos': [999, 999], 'grid': 'PRIVATE_SENTINEL', 'info': {'secret': 'PRIVATE_SENTINEL'}}
        adapter_observation = MiniGridAdapter._observation(poisoned)
        assert 'PRIVATE_SENTINEL' not in adapter_observation.model_dump_json()
        return {'directions': rows, 'action_mapping': _ACTION_INDEX,
                'action_mapping_matches_installed_enum': _ACTION_INDEX == {a.name: int(a) for a in Actions},
                'raw_extra_fields_excluded': True,
                'array_axes': 'image[x][y][object/color/door-state]; text rows iterate y, columns iterate x',
                'numeric_image_axis_or_code_legend_in_prompt': False,
                'local_up_vs_world_north': 'Local upward is forward for every world direction; direction=0 is world east.',
                'agent_cell': 'The upstream observation has no agent token at (3,6); it encodes carried object or empty.',
                'empty_env_see_through_walls': bool(env.unwrapped.see_through_walls),
                'off_grid_cells': 'Upstream Grid.slice fills off-world cells with Wall; this is not an adapter hallucination.'}
    finally:
        env.close()


def inspect_restore():
    env, ledger = MiniGridAdapter(max_steps=20), BudgetLedger()
    try:
        initial = env.reset(EpisodeSpec(environment='minigrid', task='MiniGrid-Empty-5x5-v0', seed=0), ledger=ledger)
        initial_world = diagnostic_world(env._env)
        assert initial_world['agent_pos'] == [1, 1] and initial_world['direction'] == 0 and initial_world['step_count'] == 0
        for _ in range(2):
            env.step(Action(name='forward'), ledger=ledger)
        expected_fingerprints = env._fingerprints()
        expected_public = env.observe().model_dump(mode='json')
        expected_world = diagnostic_world(env._env)
        snapshot = env.snapshot(ledger=ledger)
        branch = []
        for name in ['right', 'forward', 'forward']:
            result = env.step(Action(name=name), ledger=ledger, phase=UsagePhase.SEARCH)
            branch.append({'action': name, 'step': result.model_dump(mode='json'), 'diagnostic_world': diagnostic_world(env._env)})
        assert branch[-1]['step']['success']
        restored = env.restore(snapshot, ledger=ledger, phase=UsagePhase.COMMIT)
        after = diagnostic_world(env._env)
        assert env._fingerprints() == expected_fingerprints
        assert restored.observation.model_dump(mode='json') == expected_public and after == expected_world
        committed = env.step(Action(name='forward'), ledger=ledger)
        assert committed.action_success is False and not committed.success
        assert diagnostic_world(env._env)['step_count'] == 3
        public = public_state(initialize_state(EpisodeSpec(environment='minigrid', task='MiniGrid-Empty-5x5-v0', seed=0), initial))
        assert set(public) == {'task', 'success_criteria', 'observation', 'actions', 'memory'}
        assert set(public['observation']['data']) == {'image', 'direction', 'mission'}
        return {'reset_capability_probe_restored_seed_start': True, 'initial_private_diagnostic': initial_world,
                'before_branch_private_diagnostic': expected_world, 'branch_to_objective_success': branch,
                'restored_private_diagnostic': after, 'public_and_private_fingerprints_match': True,
                'post_restore_commit': committed.model_dump(mode='json'), 'cost': ledger.used.model_dump(),
                'public_state_top_level_keys': sorted(public),
                'feedback_fields_not_exposed': ['action_success', 'reward', 'objective_score', 'success', 'terminated', 'truncated'],
                'restoration_result': 'An alternative successful search branch does not contaminate restored commit state.'}
    finally:
        env.close()


def parse_public(text):
    return json.JSONDecoder().raw_decode(text.split('Public state:\n', 1)[1])[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', '--root', dest='root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True, help='New output directory; existing directories are refused.')
    args = parser.parse_args()
    out, root = args.output, args.root
    out.mkdir(parents=True, exist_ok=False)
    source = root / '.dev/artifacts/P14-20261017-pretrial/a'
    oracle = Oracle()
    forward = [oracle.replay(actions(['forward'] * i)) for i in range(21)]
    route_names = ['forward', 'forward', 'right', 'forward', 'forward']
    route = [oracle.replay(actions(route_names[:i])) for i in range(6)]
    assert all(row['step']['action_success'] is False for row in forward[3:])
    assert all(row['step']['observation'] == forward[2]['step']['observation'] for row in forward[3:])
    assert forward[20]['step']['truncated'] and route[-1]['step']['success']
    write(out / 'cpu-trajectories.json', {'forward20': forward, 'successful_control_actions': route_names,
          'successful_control': route, 'control_scope': 'CPU diagnostic only; this route was never sent to a model.'})
    write(out / 'coordinate-and-visibility.json', inspect_coordinates())
    write(out / 'snapshot-restore.json', inspect_restore())
    summaries, sources = [], []
    all_prompt_rows, all_commit_rows, all_transition_rows = [], [], []
    for condition in ['4096-B0', '4096-B3', '8192-B0', '8192-B3']:
        directory = source / condition
        config = json.loads((directory / 'config.json').read_text())
        assert config['episode']['seed'] == 0 and config['run']['max_steps'] == 20
        assert config['episode']['options'].get('max_steps', 20) == 20
        path = directory / 'events.jsonl'
        content = path.read_bytes()
        events = [(number, line, json.loads(line)) for number, line in enumerate(content.splitlines(keepends=True), 1)]
        sources.append({'path': str(path.relative_to(root)), 'sha256': sha(content), 'bytes': len(content)})
        event_prompt_rows, commits, transitions = [], [], []
        for number, line, event in events:
            payload = event['payload']
            reference = {'source_line': number, 'event_id': event['event_id'], 'source_line_sha256': sha(line)}
            if event['event_type'] in ('generation_request', 'judge_request'):
                request = payload['request']
                role = 'generation' if event['event_type'] == 'generation_request' else request['role']
                text = request['prompt'] if role == 'generation' else request['context']
                public = parse_public(text)
                expected = oracle.replay(public['actions'])
                assert public['observation'] == expected['step']['observation'], (condition, reference)
                assert set(public) == {'task', 'success_criteria', 'observation', 'actions', 'memory'}
                row = {'condition': condition, 'transport': 'logical_event', 'role': role,
                       'branch_id': event['branch_id'], 'phase': payload['phase'],
                       'actions': public['actions'], 'action_count': len(public['actions']),
                       'observation_sha256': sha(canonical(public['observation'])),
                       'observation_matches_fresh_direct_env': True,
                       'public_state_sha256': sha(canonical(public)),
                       'explicit_action_success_in_public_state': 'action_success' in public,
                       'memory_kind_counts': dict(Counter(row['kind'] for row in public['memory'])), **reference}
                event_prompt_rows.append(row)
                all_prompt_rows.append(row)
            if event['event_type'] == 'transition':
                expected = oracle.replay(payload['state']['actions'])
                assert payload['step'] == expected['step'], (condition, reference, 'step mismatch')
                assert payload['state']['observation'] == expected['step']['observation']
                row = {'condition': condition, 'phase': payload['phase'], 'action_count': len(payload['state']['actions']),
                       'action': payload['candidate']['action'], 'action_success': payload['step']['action_success'],
                       'fresh_direct_replay_matches_logged_step': True, **reference}
                transitions.append(row)
                all_transition_rows.append(row)
                if payload['phase'] == 'commit':
                    before = oracle.replay(payload['state']['actions'][:-1])
                    commits.append({**row, 'candidate': payload['candidate'], 'before': before,
                                    'after': expected, 'observation_changed': before['step']['observation'] != expected['step']['observation'],
                                    'logged_state_id': payload['state']['state_id']})
        # Exact committed root context references before generation and after execution.
        for row in commits:
            depth = row['action_count']
            row['root_prompt_references_before_action'] = [r for r in event_prompt_rows if r['branch_id'] == 'root'
                and r['action_count'] == depth - 1 and r['role'] in ('generation', 'candidate')]
            row['root_prompt_references_after_action'] = [r for r in event_prompt_rows if r['branch_id'] == 'root'
                and r['action_count'] == depth and r['role'] in ('evidence', 'stop')]
            all_commit_rows.append(row)
        journal_path = source / f'{condition}-http/http.jsonl'
        journal_bytes = journal_path.read_bytes()
        sources.append({'path': str(journal_path.relative_to(root)), 'sha256': sha(journal_bytes), 'bytes': len(journal_bytes)})
        http_counts = Counter()
        for number, line in enumerate(journal_bytes.splitlines(keepends=True), 1):
            request_row = json.loads(line)
            if request_row['event'] != 'request_preflight':
                continue
            body = request_row['request']
            text = next(message['content'] for message in body['messages'] if message['role'] == 'user') if 'messages' in body else body['prompt']
            public = parse_public(text)
            expected = oracle.replay(public['actions'])
            assert public['observation'] == expected['step']['observation'], (condition, number, 'HTTP stale observation')
            assert set(public['observation']['data']) == {'image', 'direction', 'mission'}
            http_counts[request_row['role']] += 1
            all_prompt_rows.append({'condition': condition, 'transport': 'actual_http_preflight', 'role': request_row['role'],
                'request_number': request_row['request_number'], 'action_count': len(public['actions']),
                'source_line': number, 'source_line_sha256': sha(line),
                'observation_sha256': sha(canonical(public['observation'])),
                'public_state_sha256': sha(canonical(public)), 'observation_matches_fresh_direct_env': True})
        summaries.append({'condition': condition, 'commits': len(commits), 'transitions_checked': len(transitions),
                          'logical_model_requests_checked': len(event_prompt_rows), 'actual_http_payloads_checked': sum(http_counts.values()),
                          'http_roles': dict(http_counts), 'first_blocked_commit': next((r['action_count'] for r in commits if not r['action_success']), None),
                          'all_commit_actions': dict(Counter(r['action']['name'] for r in commits)),
                          'unchanged_observation_commits': [r['action_count'] for r in commits if not r['observation_changed']],
                          'all_logged_transitions_match_fresh_direct_replay': True,
                          'all_model_visible_observations_match_fresh_direct_replay': True})
    for name, rows in [('prompt-observation-audit.jsonl', all_prompt_rows), ('commit-observation-audit.jsonl', all_commit_rows),
                       ('transition-replay-audit.jsonl', all_transition_rows)]:
        (out / name).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    relevant_sources = [root / 'jevgo/envs/minigrid.py', root / 'jevgo/research.py', root / 'jevgo/runner.py', root / 'jevgo/factory.py',
                        Path(minigrid.__file__).parent / 'minigrid_env.py', Path(minigrid.__file__).parent / 'core/grid.py',
                        Path(minigrid.__file__).parent / 'core/constants.py', Path(minigrid.__file__).parent / 'core/actions.py',
                        Path(minigrid.__file__).parent / 'envs/empty.py']
    summary = {'conditions': summaries, 'model_requests_made_by_analysis': 0, 'gpu_used': False,
               'private_state_only_for_diagnosis_not_model_input': True,
               'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
               'inference_commit': '955df143e7e2db059cb870e8d4fed9e9741621a2',
               'formal_code_diff_from_inference': subprocess.check_output(['git', 'diff', '--stat', '955df143e7e2db059cb870e8d4fed9e9741621a2', 'HEAD', '--', 'jevgo', 'configs'], cwd=root, text=True),
               'versions': {name: importlib.metadata.version(name) for name in ['minigrid', 'gymnasium', 'numpy']},
               'sources': sources, 'code_sources': [{'path': str(p), 'sha256': sha(p.read_bytes())} for p in relevant_sources],
               'script_sha256': sha(Path(__file__).read_bytes()),
               'first_wall_collision_action_number': 3,
               'forward20_end': forward[-1]['diagnostic_world'],
               'five_action_control_success': route[-1]['step']['success'],
               'confirmed_observation_conversion_or_replay_bug': False,
               'input_contract_gaps': ['Numeric image uses upstream x-major axes without an explicit numeric-axis/code legend.',
                    'Global compass and local facing-up are both correct but the model explicitly conflates them.',
                    'Public state drops action_success and reward/terminal fields; unchanged public observation and action history are available.',
                    'Memory content preserves observations/hypotheses but omits raw action_success and raw action links.'],
               'scope': 'These checks establish transport/environment fidelity for P14a; they do not measure model competence under a corrected prompt.'}
    write(out / 'summary.json', summary)
    write(out / 'artifact-index.json', {'files': [{'path': str(p.relative_to(root)), 'bytes': p.stat().st_size,
           'sha256': sha(p.read_bytes())} for p in sorted(out.iterdir()) if p.is_file() and p.name != 'artifact-index.json']})
    print(json.dumps({'conditions': summaries, 'model_requests': 0, 'assertions_passed': True}, indent=2))


if __name__ == '__main__':
    main()
