# State Files

This directory holds runtime-generated experiment state used by `runner.py`.

Common generated files:

- `best-run.json`
- `run-results.jsonl`
- `preflight-status.md`

The tracked file `manual-loop-validation.yaml` defines the expected Slurm configuration for `slurm/run_experiment.slurm`.
