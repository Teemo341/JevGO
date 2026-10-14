"""Export an audited JevGO summary to a flat CSV for downstream plotting."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = []
    for group in json.loads(args.summary.read_text())["groups"]:
        for run in group["rows"]:
            rows.append({"environment": group["environment"], "model": group["model"]["backend"],
                "modality": group["modality"], "method": group["method"], "ablation": group["ablation"],
                "task": run["task"], "seed": run["seed"], "status": run["status"],
                "validation_level": run.get("validation_level", "unverified"),
                "objective_success": run["objective_success"],
                "within_budget_success": run["within_budget_success"],
                "cost_complete": run["cost_complete"], **run.get("usage", {})})
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with args.output.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
