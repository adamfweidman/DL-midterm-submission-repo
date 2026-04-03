#!/usr/bin/env python3
"""Download Kaggle competition CSVs into the canonical scratch data directory."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


COMPETITION = "dl-spring-2026-svg-generation"
EXPECTED_FILES = ["train.csv", "test.csv", "sample_submission.csv"]


def default_paths() -> tuple[Path, Path]:
    user = os.environ.get("USER", "unknown")
    scratch_root = Path(os.environ.get("MIDTERM_SCRATCH_ROOT", f"/scratch/{user}/midterm-project"))
    data_dir = Path(
        os.environ.get(
            "MIDTERM_DATA_DIR",
            str(scratch_root / "data" / "kaggle" / "svg-generation"),
        )
    )
    cache_dir = Path(
        os.environ.get(
            "KAGGLEHUB_CACHE",
            str(scratch_root / "cache" / "kagglehub"),
        )
    )
    return data_dir, cache_dir


def parse_args() -> argparse.Namespace:
    data_dir, cache_dir = default_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", default=COMPETITION)
    parser.add_argument("--data-dir", type=Path, default=data_dir)
    parser.add_argument("--cache-dir", type=Path, default=cache_dir)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def ensure_auth() -> None:
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if token:
        return

    token_path = Path.home() / ".kaggle" / "access_token"
    if token_path.exists() and token_path.read_text(encoding="utf-8").strip():
        return

    raise SystemExit(
        "Missing Kaggle auth. Set KAGGLE_API_TOKEN or write the token to ~/.kaggle/access_token."
    )


def main() -> int:
    args = parse_args()
    ensure_auth()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("KAGGLEHUB_CACHE", str(args.cache_dir))

    try:
        import kagglehub
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "kagglehub is not installed. Run with `uv run --with kagglehub python scripts/download_kaggle_data.py`."
        ) from exc

    print(f"competition={args.competition}")
    print(f"data_dir={args.data_dir}")
    print(f"cache_dir={args.cache_dir}")
    result_dir = kagglehub.competition_download(
        args.competition,
        force_download=args.force,
        output_dir=str(args.data_dir),
    )
    print(f"download_result={result_dir}")

    missing = [name for name in EXPECTED_FILES if not (args.data_dir / name).exists()]
    if missing:
        print("missing_files=" + ",".join(missing))
        return 1

    for name in EXPECTED_FILES:
        path = args.data_dir / name
        print(f"{name}={path.stat().st_size}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
