"""DiscoveryWorld real-worker acceptance plus deterministic protocol fixtures."""
from dataclasses import replace
import io
import json
from pathlib import Path
from uuid import uuid4

import pytest

from jevgo.core import (
    Action, BudgetExceeded, BudgetLedger, BudgetLimits, EnvironmentFailure,
    EnvironmentSnapshot, EpisodeSpec, UnsupportedCapability, Usage, UsagePhase,
)
from jevgo.envs import discoveryworld as module
from jevgo.envs.discoveryworld import DiscoveryWorldAdapter, _public_ui, _task_status


ROOT = Path(__file__).resolve().parents[3]


def episode(**options):
    return EpisodeSpec(environment="discoveryworld", task="Tutorial", seed=0, options=options)


def rotate(direction="north"):
    return Action(name="ROTATE_DIRECTION", arguments={"arg1": direction})


@pytest.fixture
def output():
    # Persistent logs/PNGs remain available as development acceptance evidence.
    return ROOT / ".dev/artifacts/P10/test-workers" / uuid4().hex


def event_rows(session):
    return [json.loads(line) for line in (session.directory / "events.jsonl").read_text().splitlines()]


def test_real_tutorial_private_rng_mismatch_is_online_only_and_root_is_untouched(output):
    adapter = DiscoveryWorldAdapter(output_dir=output)
    ledger = BudgetLedger()
    try:
        initial = adapter.reset(episode(), ledger=ledger)
        session = adapter._session
        assert initial.info["world_steps"] == 1
        assert not adapter.capabilities.can_branch
        assert adapter.capabilities.details["reason"] == "private_state_or_rng_mismatch"
        assert adapter.capabilities.details["verification_stage"] == "initial"
        assert [row["operation"] for row in event_rows(session)] == ["load", "tick", "observe"]
        assert len(list(output.glob("discoveryworld-worker-*"))) == 2
        assert ledger.used.env_resets == ledger.used.env_observations == 2
        assert ledger.used.env_snapshots == ledger.used.env_restores == 1
        assert ledger.used.env_steps == ledger.used.replay_steps == 0
        assert all(entry.metadata.get("purpose") == "capability_verification" for entry in ledger.entries)
        with pytest.raises(UnsupportedCapability):
            adapter.snapshot(ledger=ledger)
        before_events = (session.directory / "events.jsonl").read_bytes()
        before_usage = ledger.used
        assert adapter.observe() is initial.observation
        assert adapter.observe() is initial.observation
        assert (session.directory / "events.jsonl").read_bytes() == before_events
        assert ledger.used == before_usage
        result = adapter.step(rotate(), ledger=ledger)
        assert result.action_success is True and result.success is False
        assert result.info["world_steps"] == 2
        assert result.observation.data["ui"]["agentLocation"]["faceDirection"] == "north"
        assert result.observation.image_paths == ()
        assert ledger.used.env_steps == 1 and ledger.used.env_observations == 3
        assert [row["operation"] for row in event_rows(session)] == [
            "load", "tick", "observe", "action", "tick", "observe"]
        assert len([row for row in event_rows(session) if row["operation"] == "observe"]) == 2
        serialized = result.observation.model_dump_json().lower()
        assert "fullscorecard" not in serialized and "criticalhypotheses" not in serialized
        assert "completedsuccessfully" not in serialized and '"completed"' not in serialized
    finally:
        session = adapter._session
        adapter.close()
        if session:
            assert session.process.poll() is not None


def test_real_workers_have_distinct_processes_cwds_and_frame_files(output):
    first = DiscoveryWorldAdapter(output_dir=output)
    second = DiscoveryWorldAdapter(output_dir=output)
    ledger1, ledger2 = BudgetLedger(), BudgetLedger()
    try:
        first.reset(episode(verify_replay=False), ledger=ledger1)
        one = first._session
        marker = one.directory / "video/frames-thread0/preserved.png"
        marker.write_bytes(b"isolated sentinel")
        initial2 = second.reset(episode(verify_replay=False), ledger=ledger2)
        two = second._session
        assert one.process.pid != two.process.pid
        assert one.directory != two.directory
        assert Path(f"/proc/{one.process.pid}/cwd").resolve() == one.directory
        assert Path(f"/proc/{two.process.pid}/cwd").resolve() == two.directory
        assert marker.read_bytes() == b"isolated sentinel"
        before = {path.name: path.stat().st_mtime_ns for path in
                  (two.directory / "video/frames-thread0").glob("*.png")}
        first.step(rotate(), ledger=ledger1)
        assert second.observe() is initial2.observation
        assert before == {path.name: path.stat().st_mtime_ns for path in
                          (two.directory / "video/frames-thread0").glob("*.png")}
        assert marker.read_bytes() == b"isolated sentinel"
    finally:
        first.close()
        second.close()


def test_real_vl_uses_cached_agent_view_png(output):
    from PIL import Image

    adapter = DiscoveryWorldAdapter(output_dir=output)
    try:
        result = adapter.reset(episode(verify_replay=False, modality="vl"), ledger=BudgetLedger())
        path, = result.observation.image_paths
        assert Path(path).is_relative_to(adapter._session.directory)
        with Image.open(path) as image:
            assert image.format == "PNG" and image.width > 0 and image.height > 0
            assert any(low != high for low, high in image.convert("RGB").getextrema())
        before = Path(path).stat().st_mtime_ns
        assert adapter.observe().image_paths == (path,)
        assert Path(path).stat().st_mtime_ns == before
    finally:
        adapter.close()


class FakeProcess:
    """JSON transport fixture. It does not stand in for real-worker acceptance."""
    instances = []
    mismatch = False
    fail_step = False

    def __init__(self, command, *, cwd, **kwargs):
        self.directory = Path(cwd)
        self.stdout = io.StringIO()
        self.stdin = self
        self.returncode = None
        self.config = None
        self.steps = 0
        self.observations = 0
        self.requests = []
        self.pid = len(type(self).instances) + 1
        type(self).instances.append(self)

    def write(self, line):
        request = json.loads(line)
        self.requests.append(request["operation"])
        if request["operation"] == "load":
            self.config = request["config"]
            self.steps = 1
        else:
            self.steps += 1
        if request["operation"] == "step" and self.fail_step:
            reply = {"ok": False, "error": "fixture worker exception"}
        else:
            self.observations += 1
            failed_task = self.config["scenario"] == "failed-task" and self.steps == 2
            success_task = self.config["scenario"] == "success-task" and self.steps == 2
            direction = "south" if request["operation"] == "load" else request["action"]["arguments"]["arg1"]
            reply = {"ok": True, "response": {
                "ui": {"world_steps": self.steps, "agentLocation": {"faceDirection": direction},
                       "extended_action_message": f"queued message {self.observations}",
                       "taskProgress": [{"taskName": "public task", "description": "public instruction"}]},
                "completed": failed_task or success_task, "success": success_task,
                "action_success": None if request["operation"] == "load" else True,
                "action_errors": [], "pixels": "same-pixels",
                "private": f"state:{self.steps}:{direction}" + (str(self.pid) if self.mismatch else ""),
                "image_path": str(self.directory / "view.png") if self.config["modality"] == "vl" else None,
                "schemas": {"ROTATE_DIRECTION": {"args": ["arg1"], "desc": "rotate"}},
                "locations": []}}
        self.stdout.seek(0)
        self.stdout.truncate()
        self.stdout.write(json.dumps(reply) + "\n")
        self.stdout.seek(0)

    def flush(self):
        pass

    def close(self):
        pass

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


@pytest.fixture
def protocol(monkeypatch):
    FakeProcess.instances = []
    FakeProcess.mismatch = FakeProcess.fail_step = False
    monkeypatch.setattr(module.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(module.select, "select", lambda readers, _w, _x, timeout: (readers, [], []))
    return FakeProcess


def test_constructor_is_lazy_and_cached_observe_does_not_consume_messages(output, protocol):
    adapter = DiscoveryWorldAdapter(output_dir=output)
    assert not output.exists()
    ledger = BudgetLedger()
    try:
        initial = adapter.reset(episode(verify_replay=False), ledger=ledger)
        assert initial.observation.data["ui"]["extended_action_message"] == "queued message 1"
        for _ in range(4):
            assert adapter.observe() is initial.observation
        assert protocol.instances[0].observations == 1
        assert ledger.used.env_observations == 1
        result = adapter.step(rotate(), ledger=ledger)
        assert result.observation.data["ui"]["extended_action_message"] == "queued message 2"
        assert protocol.instances[0].observations == 2
        assert protocol.instances[0].requests == ["load", "step"]
    finally:
        adapter.close()


def test_protocol_replay_fresh_workers_costs_and_sibling_isolation(output, protocol):
    adapter = DiscoveryWorldAdapter(output_dir=output)
    ledger = BudgetLedger()
    try:
        initial = adapter.reset(episode(), ledger=ledger)
        original = adapter._session
        assert adapter.capabilities.can_branch and adapter._session is original
        assert original.history == () and original.process.requests == ["load"]
        assert (ledger.used.env_resets, ledger.used.env_steps, ledger.used.replay_steps) == (3, 2, 1)
        assert (ledger.used.env_snapshots, ledger.used.env_restores, ledger.used.env_observations) == (2, 2, 5)
        root = adapter.snapshot(ledger=ledger)
        north = adapter.step(rotate("north"), ledger=ledger)
        north_snapshot = adapter.snapshot(ledger=ledger)
        assert adapter.restore(root, ledger=ledger) == initial
        west = adapter.step(rotate("west"), ledger=ledger)
        before = ledger.used
        assert adapter.restore(north_snapshot, ledger=ledger) == north
        assert adapter.observe() != west.observation
        assert ledger.used.env_steps - before.env_steps == 1
        assert ledger.used.replay_steps - before.replay_steps == 1
        assert ledger.used.env_resets - before.env_resets == 1
        assert ledger.used.env_restores - before.env_restores == 1
        assert ledger.used.env_observations - before.env_observations == 2
        replay_entries = [entry for entry in ledger.entries if entry.usage.replay_steps and not entry.metadata.get("purpose")]
        assert replay_entries and all(entry.phase == UsagePhase.REPLAY and entry.budget_phase == "search" for entry in replay_entries)
        assert len({item.directory for item in protocol.instances}) == len(protocol.instances)
    finally:
        adapter.close()


def test_protocol_vl_replay_compares_pixels_not_worker_image_paths(output, protocol):
    adapter = DiscoveryWorldAdapter(output_dir=output)
    ledger = BudgetLedger()
    try:
        initial = adapter.reset(episode(modality="vl"), ledger=ledger)
        assert adapter.capabilities.can_branch
        root = adapter.snapshot(ledger=ledger)
        restored = adapter.restore(root, ledger=ledger)
        assert initial.observation.image_paths != restored.observation.image_paths
        assert initial.observation.data == restored.observation.data
    finally:
        adapter.close()


@pytest.mark.parametrize("task,success", [("failed-task", False), ("success-task", True)])
def test_completed_is_not_equivalent_to_success(output, protocol, task, success):
    adapter = DiscoveryWorldAdapter(output_dir=output)
    ledger = BudgetLedger()
    try:
        adapter.reset(episode(verify_replay=False).model_copy(update={"task": task}), ledger=ledger)
        result = adapter.step(rotate(), ledger=ledger)
        assert result.terminated and result.done
        assert result.success is success and result.reward == int(success)
        with pytest.raises(EnvironmentFailure, match="finished"):
            adapter.step(rotate(), ledger=ledger)
    finally:
        adapter.close()


def test_task_status_and_recursive_oracle_filter():
    assert _task_status([]) == (False, False)
    assert _task_status([{"completed": True, "completedSuccessfully": False}]) == (True, False)
    assert _task_status([{"completed": True, "completedSuccessfully": True}]) == (True, True)
    raw = {"world_steps": 1, "fullscorecard": "SECRET",
           "agentLocation": {"x": 2, "scoreCard": "SECRET_NESTED"},
           "taskProgress": [{"taskName": "task", "description": "visible",
                             "completed": True, "completedSuccessfully": False,
                             "criticalHypotheses": "SECRET_TASK"}]}
    encoded = json.dumps(_public_ui(raw))
    assert "SECRET" not in encoded and "completed" not in encoded
    assert "visible" in encoded


def test_protocol_restore_mismatch_disables_capability_without_claiming_restored(output, protocol):
    adapter = DiscoveryWorldAdapter(output_dir=output)
    ledger = BudgetLedger()
    try:
        adapter.reset(episode(), ledger=ledger)
        root = adapter.snapshot(ledger=ledger)
        previous = adapter.step(rotate(), ledger=ledger)
        process = adapter._session.process
        protocol.mismatch = True
        with pytest.raises(EnvironmentFailure, match="branching disabled"):
            adapter.restore(root, ledger=ledger)
        assert not adapter.capabilities.can_branch
        assert adapter._session.process is process
        assert adapter.observe() == previous.observation
        assert protocol.instances[-1].poll() is not None
    finally:
        adapter.close()


def test_protocol_worker_exception_is_environment_failure_not_online_fallback(output, protocol):
    protocol.fail_step = True
    adapter = DiscoveryWorldAdapter(output_dir=output)
    with pytest.raises(EnvironmentFailure, match="worker failed"):
        adapter.reset(episode(), ledger=BudgetLedger())
    assert adapter._session is None
    assert all(process.poll() is not None for process in protocol.instances)


def test_protocol_step_exception_closes_worker_and_keeps_costs(output, protocol):
    adapter = DiscoveryWorldAdapter(output_dir=output)
    ledger = BudgetLedger()
    adapter.reset(episode(verify_replay=False), ledger=ledger)
    protocol.fail_step = True
    with pytest.raises(EnvironmentFailure, match="worker failed"):
        adapter.step(rotate(), ledger=ledger)
    assert ledger.used.env_steps == 1
    assert adapter._session is None and protocol.instances[-1].poll() is not None
    assert any(entry.failed for entry in ledger.entries)


def test_budget_preflight_creates_no_worker_or_directory(output, protocol):
    adapter = DiscoveryWorldAdapter(output_dir=output)
    ledger = BudgetLedger(BudgetLimits(env_resets=1))
    with pytest.raises(BudgetExceeded):
        adapter.reset(episode(), ledger=ledger)
    assert not output.exists() and protocol.instances == []
    assert ledger.used == Usage()


def test_search_restore_preserves_commit_reserve_and_invalid_snapshot_costs_nothing(output, protocol):
    adapter = DiscoveryWorldAdapter(output_dir=output)
    ledger = BudgetLedger(BudgetLimits(env_steps=4), commit_reserve=Usage(env_steps=1))
    try:
        adapter.reset(episode(), ledger=ledger)
        result = adapter.step(rotate(), ledger=ledger)
        saved = adapter.snapshot(ledger=ledger)
        for bad in (object(), EnvironmentSnapshot(owner_id="foreign", payload=saved.payload),
                    replace(saved, payload=replace(saved.payload, history=()))):
            before = ledger.used
            with pytest.raises(EnvironmentFailure, match="Invalid, foreign, or expired"):
                adapter.restore(bad, ledger=ledger)
            assert ledger.used == before
        count = len(protocol.instances)
        before = ledger.used
        with pytest.raises(BudgetExceeded):
            adapter.restore(saved, ledger=ledger)
        assert ledger.used == before and len(protocol.instances) == count
        assert adapter.restore(saved, ledger=ledger, phase=UsagePhase.COMMIT) == result
        entries = [entry for entry in ledger.entries if entry.usage.replay_steps and not entry.metadata.get("purpose")]
        assert entries[-1].budget_phase == "commit"
        assert ledger.used.env_steps == 4
    finally:
        adapter.close()


def test_invalid_action_and_max_steps_are_distinct_from_success(output, protocol):
    adapter = DiscoveryWorldAdapter(output_dir=output, max_steps=1)
    ledger = BudgetLedger()
    try:
        adapter.reset(episode(verify_replay=False), ledger=ledger)
        invalid = adapter.step(Action(name="teleport"), ledger=ledger)
        assert invalid.action_success is False and not invalid.success
        assert ledger.used.env_steps == 0
        last = adapter.step(rotate(), ledger=ledger)
        assert last.truncated and not last.success
        assert ledger.used.env_steps == 1
    finally:
        adapter.close()


def test_timeout_terminates_worker(output, protocol, monkeypatch):
    monkeypatch.setattr(module.select, "select", lambda *args: ([], [], []))
    adapter = DiscoveryWorldAdapter(output_dir=output, timeout=0.01)
    with pytest.raises(EnvironmentFailure, match="timed out"):
        adapter.reset(episode(verify_replay=False), ledger=BudgetLedger())
    assert adapter._session is None and protocol.instances[-1].poll() is not None


def test_public_dialog_choices_and_feed_ids_are_executable_actions():
    ui = {"dialog_box": {"is_in_dialog": True, "dialogOptions": {"1": "Yes", "2": "No"}}}
    actions = module._admissible(ui, {}, [])
    assert actions == (Action(name="DIALOG_CHOICE", arguments={"chosen_dialog_option_int": 1}),
                       Action(name="DIALOG_CHOICE", arguments={"chosen_dialog_option_int": 2}))
    ui = {"discoveryFeed": {"posts": [{"postID": 7}], "scientific_articles": [{"postID": 11}]}}
    schemas = {"DISCOVERY_FEED_GET_POST_BY_ID": {"args": ["arg1"], "desc": "read public post"}}
    assert {a.arguments["arg1"] for a in module._admissible(ui, schemas, [])} == {7, 11}


def test_replay_startup_exception_is_environment_failure_and_preserves_old_worker(output, protocol, monkeypatch):
    adapter = DiscoveryWorldAdapter(output_dir=output)
    ledger = BudgetLedger()
    try:
        adapter.reset(episode(), ledger=ledger)
        saved = adapter.snapshot(ledger=ledger)
        old_session = adapter._session

        def fail_start(*args, **kwargs):
            raise OSError("fixture startup failure")

        monkeypatch.setattr(module.subprocess, "Popen", fail_start)
        with pytest.raises(EnvironmentFailure, match="startup failed"):
            adapter.restore(saved, ledger=ledger)
        assert not adapter.capabilities.can_branch
        assert adapter._session is old_session and old_session.process.poll() is None
    finally:
        adapter.close()
