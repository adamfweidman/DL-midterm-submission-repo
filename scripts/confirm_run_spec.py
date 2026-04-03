#!/usr/bin/env python3
"""Validate a YAML run spec against the current Slurm job and repo state."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    return parser.parse_args()


def load_spec(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def parse_scontrol_job(job_id: str) -> dict[str, str]:
    output = subprocess.check_output(["scontrol", "show", "job", job_id, "-o"], text=True).strip()
    fields: dict[str, str] = {}
    for token in output.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = value
    return fields


def tracked_changes_against_head(repo_root: Path) -> list[str]:
    output = subprocess.check_output(["git", "diff", "--name-only", "HEAD"], cwd=repo_root, text=True)
    return [line.strip() for line in output.splitlines() if line.strip()]


def latest_preflight_passed(repo_root: Path) -> bool:
    status_path = repo_root / "state" / "preflight-status.md"
    if not status_path.exists():
        return False
    text = status_path.read_text(encoding="utf-8")
    return "Latest result: passed" in text


def check(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def main() -> int:
    args = parse_args()
    spec = load_spec(args.spec)
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise SystemExit("SLURM_JOB_ID is not set; run this from within sbatch.")

    job_fields = parse_scontrol_job(job_id)
    failures: list[str] = []

    expected_job = spec["job"]
    expected_env = spec["environment"]

    check(job_fields.get("Account") == expected_job["account"], "Slurm account does not match spec.", failures)
    check(job_fields.get("Partition") == expected_job["partition"], "Slurm partition does not match spec.", failures)
    check(job_fields.get("JobName") == expected_job["name"], "Slurm job name does not match spec.", failures)
    check(job_fields.get("TimeLimit") == expected_job["time_limit"], "Slurm time limit does not match spec.", failures)
    check(job_fields.get("CPUs/Task") == str(expected_job["cpus_per_task"]), "Slurm CPUs/Task does not match spec.", failures)
    check(job_fields.get("MinMemoryNode") == expected_job["mem"], "Slurm memory request does not match spec.", failures)
    expected_repo_root = expected_env.get("repo_root")
    if expected_repo_root not in {None, "", "__REPO_ROOT__"}:
        check(str(args.repo_root) == expected_repo_root, "Repo root does not match spec.", failures)

    if expected_env.get("require_preflight_status_pass", False):
        check(latest_preflight_passed(args.repo_root), "Latest repo-tracked preflight status is not passing.", failures)

    if expected_env.get("require_clean_tracked_tree_except_experiment", False):
        changed = set(tracked_changes_against_head(args.repo_root))
        allowed = {"experiment.py", "state/preflight-status.md"}
        disallowed = sorted(changed - allowed)
        check(not disallowed, f"Tracked changes outside allowed set {sorted(allowed)}: {disallowed}", failures)

    payload = {
        "spec_path": str(args.spec),
        "repo_root": str(args.repo_root),
        "job_id": job_id,
        "confirmed": not failures,
        "runner": spec["runner"],
        "job": {
            "account": job_fields.get("Account"),
            "partition": job_fields.get("Partition"),
            "job_name": job_fields.get("JobName"),
            "time_limit": job_fields.get("TimeLimit"),
            "cpus_per_task": job_fields.get("CPUs/Task"),
            "mem": job_fields.get("MinMemoryNode"),
        },
        "failures": failures,
    }
    print(json.dumps(payload, indent=2))

    if failures:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
