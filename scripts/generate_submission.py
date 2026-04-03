#!/usr/bin/env python3
"""Generate submission.csv from test.csv using a trained LoRA adapter.

Usage (local or HPC):
    python scripts/generate_submission.py \
        --adapter-dir /path/to/run/adapter \
        --output submission.csv

Usage (Kaggle notebook — paths would differ):
    python scripts/generate_submission.py \
        --base-model /kaggle/input/qwen25-coder-3b/... \
        --adapter-dir /kaggle/input/best-adapter/adapter \
        --data-dir /kaggle/input/svg-generation \
        --output /kaggle/working/submission.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Config — mirrors experiment.py defaults, overridable via CLI
# ---------------------------------------------------------------------------

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
DEFAULT_SYSTEM_PROMPT = (
    "You generate compact, valid SVG markup for 256x256 canvases. "
    "Return only SVG code with a single root <svg> element."
)
DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_BATCH_SIZE = 16

SVG_REGEX = re.compile(r"<svg[\s\S]*?</svg>", flags=re.IGNORECASE)
BLANK_FALLBACK = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256"></svg>'


# ---------------------------------------------------------------------------
# Core inference functions
# ---------------------------------------------------------------------------

def prompt_prefix(prompt: str, system_prompt: str) -> str:
    return (
        "<|im_start|>system\n"
        f"{system_prompt}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def extract_svg(text: str) -> str:
    match = SVG_REGEX.search(text)
    return match.group(0).strip() if match else text.strip()


def validate_svg_basic(svg_text: str, max_length: int = 16000, max_paths: int = 256) -> bool:
    """Quick validity check against Kaggle constraints — fallback on failure."""
    import xml.etree.ElementTree as ET

    raw = svg_text.strip()
    if not raw or not raw.lower().startswith("<svg"):
        return False
    if len(raw) > max_length:
        return False
    if 'viewBox="0 0 256 256"' not in raw:
        return False
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return False
    path_count = sum(1 for elem in root.iter() if elem.tag.split("}")[-1] == "path")
    if path_count > max_paths:
        return False
    return True


def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    system_prompt: str,
    max_new_tokens: int,
    device: torch.device,
) -> list[str]:
    """Generate SVGs for a batch of prompts."""
    batch_text = [prompt_prefix(p, system_prompt) for p in prompts]
    inputs = tokenizer(
        batch_text,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1e-6,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    results = []
    for gen in generated_ids:
        text = tokenizer.decode(gen, skip_special_tokens=False)
        svg = extract_svg(text).strip()
        if not validate_svg_basic(svg):
            svg = BLANK_FALLBACK
        results.append(svg)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate submission.csv")
    parser.add_argument("--adapter-dir", type=Path, required=True,
                        help="Path to the LoRA adapter directory")
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL,
                        help="Base model name or path")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Directory containing test.csv (default: MIDTERM_DATA_DIR or scratch)")
    parser.add_argument("--output", type=Path, default=Path("submission.csv"),
                        help="Output path for submission.csv")
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--local-files-only", action="store_true", default=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Resolve data dir
    if args.data_dir is None:
        data_dir = os.environ.get("MIDTERM_DATA_DIR")
        if data_dir:
            args.data_dir = Path(data_dir)
        else:
            user = Path.home().name
            args.data_dir = Path(f"/scratch/{user}/midterm-project/data/kaggle/svg-generation")

    test_csv = args.data_dir / "test.csv"
    if not test_csv.exists():
        print(f"ERROR: test.csv not found at {test_csv}", file=sys.stderr)
        return 1

    if not args.adapter_dir.exists():
        print(f"ERROR: adapter dir not found at {args.adapter_dir}", file=sys.stderr)
        return 1

    # Load test prompts
    test_rows: list[dict[str, str]] = []
    with test_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_rows.append(row)
    print(f"Loaded {len(test_rows)} test prompts from {test_csv}")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"Device: {device}, dtype: {dtype}")

    print(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, local_files_only=args.local_files_only
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        local_files_only=args.local_files_only,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Loading adapter from: {args.adapter_dir}")
    model = PeftModel.from_pretrained(model, str(args.adapter_dir))
    model.to(device)
    model.eval()
    model.config.use_cache = True
    print("Model loaded and ready")

    # Generate
    prompts = [row["prompt"] for row in test_rows]
    ids = [row["id"] for row in test_rows]
    all_svgs: list[str] = []
    total_batches = (len(prompts) + args.batch_size - 1) // args.batch_size

    t0 = time.time()
    for i in range(0, len(prompts), args.batch_size):
        batch_num = i // args.batch_size + 1
        batch_prompts = prompts[i : i + args.batch_size]
        batch_svgs = generate_batch(
            model, tokenizer, batch_prompts,
            args.system_prompt, args.max_new_tokens, device,
        )
        all_svgs.extend(batch_svgs)

        valid_count = sum(1 for s in batch_svgs if s != BLANK_FALLBACK)
        elapsed = time.time() - t0
        print(f"  batch {batch_num}/{total_batches}  "
              f"generated={len(all_svgs)}/{len(prompts)}  "
              f"valid={valid_count}/{len(batch_svgs)}  "
              f"elapsed={elapsed:.0f}s")

    elapsed = time.time() - t0
    total_valid = sum(1 for s in all_svgs if s != BLANK_FALLBACK)
    total_fallback = len(all_svgs) - total_valid
    print(f"\nGeneration complete: {len(all_svgs)} rows, "
          f"{total_valid} valid, {total_fallback} fallback, "
          f"{elapsed:.0f}s total")

    # Verify row count matches test set
    assert len(all_svgs) == len(test_rows), (
        f"Row count mismatch: generated {len(all_svgs)} but test has {len(test_rows)}"
    )

    # Write submission — strip newlines to match sample format
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "svg"])
        for row_id, svg in zip(ids, all_svgs):
            writer.writerow([row_id, svg.replace("\n", " ").replace("\r", "")])

    # Verify output is parseable and has correct row count
    with args.output.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["id", "svg"], f"Bad header: {reader.fieldnames}"
        output_rows = list(reader)
    assert len(output_rows) == len(test_rows), (
        f"Output row count {len(output_rows)} != expected {len(test_rows)}"
    )

    print(f"Submission written to {args.output} ({len(output_rows)} rows, verified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
