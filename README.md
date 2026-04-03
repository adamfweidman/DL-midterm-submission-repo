# Text-to-SVG Midterm Project

Public code release for the Spring 2026 deep learning midterm project on text-to-SVG generation.

## Main Files

- `experiment.py`: final training and decoding configuration
- `runner.py`: training loop plus local validation scoring
- `slurm/run_experiment.slurm`: main NYU HPC train+eval entrypoint
- `scripts/generate_submission.py`: generate `submission.csv` from a saved adapter
- `notebooks/submission-notebook.ipynb`: Kaggle inference notebook
- `splits/validation_ids.txt`: fixed local validation split used for project evaluation

## Setup

```bash
source setup_env.sh
uv pip install -r requirements.txt
python scripts/download_kaggle_data.py
```

The code expects Kaggle `train.csv`, `test.csv`, and `sample_submission.csv` under `$MIDTERM_DATA_DIR`.

## Training and Evaluation

The main workflow used in the project was:

```bash
sbatch slurm/run_experiment.slurm
```

This launches `runner.py`, which trains the LoRA adapter and evaluates it on the fixed local validation split.

## Submission Generation

After training, generate a Kaggle submission with:

```bash
python scripts/generate_submission.py \
  --adapter-dir /path/to/run/adapter \
  --output submission.csv \
  --max-new-tokens 2048
```

## Notes

- The Kaggle notebook is inference-only. Training happens offline.
- The notebook includes the required disclosure for AI-assisted coding/debugging.
