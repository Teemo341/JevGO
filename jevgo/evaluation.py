"""Reproducible matrices and ledger-checked summaries; fixture results stay labeled."""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import mean
from typing import Any

import yaml

from .core import EpisodeSpec, JSONLEventStore, RunConfig, RunResult, Usage


ABLATIONS = {
    "full": {}, "no_candidate_judge": {"judge_candidates": False},
    "no_evidence_judge": {"judge_evidence": False}, "no_stop_judge": {"judge_stop": False},
    "no_working_memory": {"use_memory": False},
}


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _write(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def build_matrix(spec: dict) -> list[dict]:
    """Cross explicit tasks/seeds/models/methods; ablations apply only to B3.

    Modality belongs to each episode's public observation condition, not to model
    size. To compare text and visual inputs, provide separate episode conditions.
    A different model or modality always produces a separate comparison group.
    """
    unknown = set(spec) - {"episodes", "seeds", "models", "methods", "ablations", "run"}
    if unknown:
        raise ValueError(f"Unknown matrix keys: {sorted(unknown)}")
    episodes, models = spec["episodes"], spec["models"]
    methods = spec.get("methods", ["B0", "B1", "B2", "B3"])
    seeds = spec.get("seeds", [0])
    ablations = spec.get("ablations", ["full"])
    if not episodes or not models or not methods or not seeds or not ablations:
        raise ValueError("Matrix axes must be nonempty")
    if len(set(methods)) != len(methods) or len(set(seeds)) != len(seeds) or len(set(ablations)) != len(ablations):
        raise ValueError("Repeated seeds/methods/ablations would duplicate samples")
    if any(type(seed) is not int for seed in seeds):
        raise ValueError("Seeds must be integers")
    if set(ablations) - set(ABLATIONS):
        raise ValueError("Unknown ablation")
    rows, identities = [], set()
    for task in episodes:
        for seed in seeds:
            condition = deepcopy(task)
            condition["seed"] = seed
            condition["episode_id"] = str(task.get("episode_id", task["task"])) + f":seed={seed}"
            episode = EpisodeSpec.model_validate(condition)
            for model in models:
                if not isinstance(model, dict) or not isinstance(model.get("backend"), str):
                    raise ValueError("Each model must explicitly name its backend")
                for method in methods:
                    for ablation in ablations if method == "B3" else ["full"]:
                        config = RunConfig.model_validate({**spec.get("run", {}), "method": method,
                                                          "seed": seed, **ABLATIONS[ablation]})
                        payload = {"episode": episode.model_dump(mode="json"),
                                   "run": config.model_dump(mode="json"), "model": deepcopy(model)}
                        digest = hashlib.sha256(_canonical(payload).encode()).hexdigest()[:16]
                        label = re.sub(r"[^A-Za-z0-9_-]", "_", model["backend"])
                        run_id = f"{label}-{method}-{ablation}-{digest}"
                        if digest in identities:
                            raise ValueError("Matrix contains duplicate run conditions")
                        identities.add(digest)
                        rows.append({"run_id": run_id, "ablation": ablation, "payload": payload})
    return rows


def run_matrix(spec: dict, *, output_dir: Path | str, assets_root: Path | str,
               execute: bool = False) -> Path:
    """Plan by default. Execute only when explicitly requested; failures are retained."""
    rows = build_matrix(spec)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "source": deepcopy(spec), "execute": execute, "runs": []}
    _write(output / "matrix-source.json", spec)
    for row in rows:
        entry = {**row, "directory": str(output / row["run_id"]), "status": "planned"}
        manifest["runs"].append(entry)
    manifest_path = output / "manifest.json"
    _write(manifest_path, manifest)
    if execute:
        from .factory import run_from_dict
        for entry in manifest["runs"]:
            try:
                result = run_from_dict(entry["payload"], assets_root=assets_root, output_dir=entry["directory"])
                entry["status"] = "finished"
                entry["termination_reason"] = result.termination_reason.value
            except Exception as exc:
                entry["status"] = "error"
                entry["error"] = f"{type(exc).__name__}: {exc}"
                failure_dir = Path(entry["directory"])
                failure_dir.mkdir(parents=True, exist_ok=True)
                _write(failure_dir / "failure.json", {"error": entry["error"], "payload": entry["payload"]})
            finally:
                _write(manifest_path, manifest)
    return manifest_path


def wilson(successes: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z, p = 1.959963984540054, successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0, center - margin), min(1, center + margin)]


def audit_run(directory: Path | str) -> dict:
    """Recompute every cost from usage events, independently compare the ledger/result."""
    directory = Path(directory)
    result = RunResult.model_validate_json((directory / "result.json").read_text())
    ledger = json.loads((directory / "ledger.json").read_text())
    events = JSONLEventStore(directory / "events.jsonl").read()
    usage_events = [e.payload for e in events if e.event_type == "usage"]
    total, phases = Usage(), defaultdict(Usage)
    complete = True
    for i, event in enumerate(usage_events):
        if event["sequence"] != i + 1:
            raise ValueError("Usage sequence is incomplete or reordered")
        usage = Usage.model_validate(event["usage"])
        total = total + usage
        phase = event["phase"] + "/" + event["budget_phase"]
        phases[phase] = phases[phase] + usage
        detail = event["metadata"]
        if detail.get("usage_complete") is False or detail.get("unknown_usage"):
            complete = False
    if len(usage_events) != len(ledger["entries"]):
        raise ValueError("Event ledger entry count differs from ledger.json")
    for actual, expected in zip(usage_events, ledger["entries"]):
        # model_dump converts the deeply frozen event payload into ordinary JSON.
        thawed = json.loads(_serialize_payload(actual))
        if thawed != expected:
            raise ValueError("Event ledger entries differ from ledger.json")
    for expected in (result.usage, Usage.model_validate(ledger["used"])):
        for key in Usage.model_fields:
            if not math.isclose(getattr(total, key), getattr(expected, key), rel_tol=1e-10, abs_tol=1e-8):
                raise ValueError(f"Recomputed usage disagrees on {key}")
    committed = [e for e in events if e.event_type == "transition" and e.payload["phase"] == "commit"]
    if len(committed) != len(result.final_state.actions):
        raise ValueError("Committed action count differs from transition events")
    return {"result": result, "usage": total.model_dump(mode="json"),
            "phase_usage": {k: v.model_dump(mode="json") for k, v in phases.items()},
            "cost_complete": complete, "within_budget": not ledger["over_budget"],
            "committed_steps": len(committed)}


def _serialize_payload(payload) -> str:
    # Reuse the public serializer for nested immutable JSON mappings.
    from .core import EventRecord
    return json.dumps(EventRecord(event_type="audit", payload=payload).model_dump(mode="json")["payload"])


def _comparison_condition(payload: dict) -> dict:
    episode = EpisodeSpec.model_validate(payload["episode"])
    config = RunConfig.model_validate(payload["run"])
    model = payload["model"]
    return {"environment": episode.environment, "model": model,
            "modality": episode.options.get("modality", "text"), "split": episode.split,
            "budget": config.budget.model_dump(mode="json"),
            "commit_reserve": config.commit_reserve.model_dump(mode="json"),
            "max_steps": config.max_steps,
            "search_parameters": {k: getattr(config, k) for k in
                ("candidate_count", "simulations", "max_depth", "rollout_depth", "c_puct", "gamma")}}


def summarize(manifest_path: Path | str) -> dict:
    manifest = json.loads(Path(manifest_path).read_text())
    groups = defaultdict(list)
    for entry in manifest["runs"]:
        payload = entry["payload"]
        condition = _comparison_condition(payload)
        key = _canonical({**condition, "method": payload["run"]["method"], "ablation": entry["ablation"]})
        row = {"run_id": entry["run_id"], "status": entry["status"], "objective_success": False,
               "within_budget_success": False, "cost_complete": False,
               "task": payload["episode"]["task"], "seed": payload["episode"]["seed"],
               "episode_options": payload["episode"].get("options", {})}
        if entry["status"] == "finished":
            try:
                audited = audit_run(entry["directory"])
                result = audited.pop("result")
                # Verify that no run silently changed its declared experiment condition.
                if result.episode != EpisodeSpec.model_validate(payload["episode"]) or result.config != RunConfig.model_validate(payload["run"]):
                    raise ValueError("Executed episode/config differs from the matrix")
                row.update(audited)
                row.update(objective_success=result.success,
                    within_budget_success=result.success and audited["within_budget"],
                    termination_reason=result.termination_reason.value,
                    validation_level=result.metadata.get("validation_level", "unclassified"))
            except (ValueError, OSError) as exc:
                row.update(status="invalid_artifact", error=str(exc))
        elif "error" in entry:
            row["error"] = entry["error"]
        groups[key].append(row)
    summary = []
    for key, rows in groups.items():
        attempted = [r for r in rows if r["status"] not in ("planned", "skipped")]
        finished = [r for r in rows if r["status"] == "finished"]
        successes = sum(r["within_budget_success"] for r in rows)
        costs = {field: mean(r["usage"][field] for r in finished) for field in Usage.model_fields} if finished else {}
        summary.append({**json.loads(key), "planned": len(rows), "attempted": len(attempted),
            "finished": len(finished), "objective_successes": sum(r["objective_success"] for r in rows),
            "within_budget_successes": successes,
            "success_rate_of_planned": successes / len(rows),
            "success_rate_of_attempted": successes / len(attempted) if attempted else None,
            "success_rate_wilson95": wilson(successes, len(attempted)), "mean_usage": costs,
            "cost_complete_runs": sum(r["cost_complete"] for r in rows),
            "validation_levels": sorted({r.get("validation_level", "unverified") for r in rows}),
            "rows": rows})
    # A fair pair requires the same complete task/seed/permissions set as well as
    # the same model/modality/budget; missing or unknown-cost runs are conspicuous.
    comparison_sets = defaultdict(list)
    for group in summary:
        condition = {k: group[k] for k in ("environment", "model", "modality", "split", "budget",
                     "commit_reserve", "max_steps", "search_parameters")}
        comparison_sets[_canonical(condition)].append(group)
    comparisons = []
    for key, choices in comparison_sets.items():
        sets = [{_canonical({k: row[k] for k in ("task", "seed", "episode_options")}) for row in g["rows"]}
                for g in choices]
        comparisons.append({"condition": json.loads(key),
            "methods": [g["method"] + "/" + g["ablation"] for g in choices],
            "same_task_conditions": all(s == sets[0] for s in sets),
            "all_runs_finished_with_known_cost": all(g["finished"] == g["planned"] == g["cost_complete_runs"] for g in choices)})
    return {"schema_version": 1, "groups": summary, "comparisons": comparisons,
            "interpretation": "CPU fixture results validate implementation, not pretrained model quality. "
                              "Separate models/modalities/environments; report missing, failed and unknown-cost runs."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    matrix = sub.add_parser("matrix")
    matrix.add_argument("--config", required=True, type=Path)
    matrix.add_argument("--output", required=True, type=Path)
    matrix.add_argument("--assets-root", required=True, type=Path)
    matrix.add_argument("--execute", action="store_true")
    analysis = sub.add_parser("summarize")
    analysis.add_argument("--manifest", required=True, type=Path)
    analysis.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "matrix":
        path = run_matrix(yaml.safe_load(args.config.read_text()), output_dir=args.output,
                          assets_root=args.assets_root, execute=args.execute)
        print(path)
    else:
        if args.output.exists():
            raise FileExistsError(args.output)
        _write(args.output, summarize(args.manifest))
        print(args.output)


if __name__ == "__main__":
    main()
