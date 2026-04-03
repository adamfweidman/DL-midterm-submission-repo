from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import random
import subprocess
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

import experiment
from data import SvgSftDataset, default_data_dir, load_competition_frames, load_validation_ids, split_train_validation, subset_frame, validation_ids_path
from eval import extract_svg, score_svg_pair, summarize_proxy_scores, validate_svg


REPO_ROOT = Path(__file__).resolve().parent
STATE_DIR = REPO_ROOT / "state"
EXPERIMENT_PATH = REPO_ROOT / "experiment.py"
BEST_RUN_STATE_PATH = STATE_DIR / "best-run.json"
RUN_RESULTS_PATH = STATE_DIR / "run-results.jsonl"
PROMOTION_EPSILON = 0.005
INFERENCE_BATCH_SIZE = 16
VAL_SCORE_LIMIT = 1000


@dataclass
class ExperimentStateArtifacts:
    base_commit_sha: str
    attempted_experiment_text: str
    best_experiment_text: str
    attempted_experiment_sha256: str
    best_experiment_sha256: str
    attempted_snapshot_path: Path
    best_snapshot_path: Path
    diff_path: Path
    current_best_state: dict[str, object] | None


@dataclass
class SelectionDecision:
    outcome: str
    reason: str

    @property
    def promoted(self) -> bool:
        return self.outcome == "promoted"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--sample-generations", type=int, default=None)
    return parser.parse_args()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def git_show_head_file(path: str) -> str:
    return subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=REPO_ROOT, text=True)


def tracked_changes_against_head() -> set[str]:
    output = subprocess.check_output(["git", "diff", "--name-only", "HEAD"], cwd=REPO_ROOT, text=True)
    return {line.strip() for line in output.splitlines() if line.strip()}


ALLOWED_WORKTREE_CHANGES = {"experiment.py", "state/preflight-status.md"}


def ensure_iteration_worktree_clean() -> None:
    modified = tracked_changes_against_head()
    disallowed = sorted(path for path in modified if path not in ALLOWED_WORKTREE_CHANGES)
    if disallowed:
        raise RuntimeError(
            "Runner refuses to start with tracked changes outside experiment.py: " + ", ".join(disallowed)
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_and_dtype() -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16
    return torch.device("cpu"), torch.float32


def load_tokenizer_and_model(dtype: torch.dtype):
    candidates = [
        experiment.MODEL["name_or_path"],
        *experiment.MODEL["fallback_chain"],
    ]
    errors: list[str] = []

    for candidate in candidates:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                candidate,
                local_files_only=experiment.MODEL["local_files_only"],
            )
            model = AutoModelForCausalLM.from_pretrained(
                candidate,
                local_files_only=experiment.MODEL["local_files_only"],
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
            model.config.use_cache = False
            if experiment.MODEL["gradient_checkpointing"] and hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            return candidate, tokenizer, model
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")

    raise RuntimeError("Failed to load any configured base model:\n" + "\n".join(errors))


def prompt_prefix(prompt: str) -> str:
    return (
        "<|im_start|>system\n"
        f"{experiment.PROMPT['system']}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def full_example(prompt: str, svg: str) -> str:
    return prompt_prefix(prompt) + svg.strip() + "<|im_end|>\n"


class SvgCollator:
    def __init__(self, tokenizer, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, batch):
        prefix_texts = [prompt_prefix(record.prompt) for record in batch]
        full_texts = [full_example(record.prompt, record.svg) for record in batch]

        full_tokens = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prefix_tokens = self.tokenizer(
            prefix_texts,
            padding=False,
            truncation=True,
            max_length=self.max_seq_length,
            add_special_tokens=False,
        )

        labels = full_tokens["input_ids"].clone()
        labels[full_tokens["attention_mask"] == 0] = -100
        for idx, prefix_ids in enumerate(prefix_tokens["input_ids"]):
            prefix_len = min(len(prefix_ids), labels.shape[1])
            labels[idx, :prefix_len] = -100

        return {
            "input_ids": full_tokens["input_ids"],
            "attention_mask": full_tokens["attention_mask"],
            "labels": labels,
        }


def _compute_average_loss(model, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            total_loss += float(outputs.loss.item())
            count += 1
    return total_loss / max(count, 1)


_eval_batch_size: int | None = None


def average_loss(model, dataloader: DataLoader, device: torch.device) -> float:
    global _eval_batch_size
    if _eval_batch_size is not None and _eval_batch_size != dataloader.batch_size:
        dataloader = DataLoader(
            dataloader.dataset, batch_size=_eval_batch_size, shuffle=False,
            collate_fn=dataloader.collate_fn, num_workers=0, pin_memory=torch.cuda.is_available(),
        )
    for fallback_bs in [dataloader.batch_size, 4, 2, 1]:
        try:
            if fallback_bs != dataloader.batch_size:
                torch.cuda.empty_cache()
                dataloader = DataLoader(
                    dataloader.dataset, batch_size=fallback_bs, shuffle=False,
                    collate_fn=dataloader.collate_fn, num_workers=0, pin_memory=torch.cuda.is_available(),
                )
            result = _compute_average_loss(model, dataloader, device)
            _eval_batch_size = fallback_bs
            return result
        except torch.cuda.OutOfMemoryError:
            if fallback_bs == 1:
                raise
            continue
    raise RuntimeError("average_loss: all batch sizes exhausted")


def generate_svg_batch(model, tokenizer, prompts: list[str], device: torch.device) -> list[str]:
    model.eval()
    outputs: list[str] = []
    original_padding_side = tokenizer.padding_side
    try:
        tokenizer.padding_side = "left"
        for start in range(0, len(prompts), INFERENCE_BATCH_SIZE):
            batch_prompts = prompts[start : start + INFERENCE_BATCH_SIZE]
            batch_text = [prompt_prefix(prompt) for prompt in batch_prompts]
            inputs = tokenizer(
                batch_text,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=experiment.DECODE["max_new_tokens"],
                    do_sample=experiment.DECODE["do_sample"],
                    temperature=max(float(experiment.DECODE["temperature"]), 1e-6),
                    top_p=float(experiment.DECODE["top_p"]),
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
            for generated in generated_ids:
                text = tokenizer.decode(generated, skip_special_tokens=False)
                svg = extract_svg(text) if experiment.POSTPROCESS["extract_svg_regex"] else text
                outputs.append(svg.strip() if experiment.POSTPROCESS["strip_whitespace"] else svg)
    finally:
        tokenizer.padding_side = original_padding_side

    return outputs


def blank_fallback_svg() -> str:
    return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256"></svg>'


def maybe_apply_fallback(predicted_svg: str) -> tuple[str, bool, dict[str, object] | None]:
    initial_validation = validate_svg(predicted_svg)
    if initial_validation.valid or not experiment.FALLBACK.get("enabled", False):
        return predicted_svg, False, None

    fallback_mode = experiment.FALLBACK.get("mode", "blank")
    if fallback_mode != "blank":
        raise RuntimeError(f"Unsupported fallback mode: {fallback_mode}")

    return blank_fallback_svg(), True, initial_validation.to_dict()


def trainable_parameter_summary(model) -> dict[str, int]:
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    return {
        "trainable": trainable,
        "total": total,
    }


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    scratch_root = Path(os.environ.get("MIDTERM_OUTPUT_ROOT", f"/scratch/{Path.home().name}/midterm-project/outputs"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    sha = git_sha()[:7]
    kind = "smoke" if args.smoke_test else "run"
    job_id = os.environ.get("SLURM_JOB_ID")
    suffix = f"-job{job_id}" if job_id else ""
    return scratch_root / kind / f"{stamp}-{sha}{suffix}"


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def snapshot_experiment_state(output_dir: Path) -> ExperimentStateArtifacts:
    ensure_iteration_worktree_clean()
    base_commit_sha = git_sha()
    attempted_experiment_text = EXPERIMENT_PATH.read_text(encoding="utf-8")
    best_experiment_text = git_show_head_file("experiment.py")
    attempted_experiment_sha256 = sha256_text(attempted_experiment_text)
    best_experiment_sha256 = sha256_text(best_experiment_text)
    current_best_state = load_best_run_state()

    if current_best_state is not None:
        expected_hash = current_best_state.get("best_experiment_sha256")
        if expected_hash and expected_hash != best_experiment_sha256:
            raise RuntimeError(
                "HEAD:experiment.py does not match the repo-tracked best experiment state. "
                "Restore the best state before launching a new run."
            )

    attempted_snapshot_path = output_dir / "attempted_experiment.py"
    best_snapshot_path = output_dir / "best_experiment_before_run.py"
    diff_path = output_dir / "attempted_vs_best.patch"

    attempted_snapshot_path.write_text(attempted_experiment_text, encoding="utf-8")
    best_snapshot_path.write_text(best_experiment_text, encoding="utf-8")
    diff_text = "".join(
        difflib.unified_diff(
            best_experiment_text.splitlines(keepends=True),
            attempted_experiment_text.splitlines(keepends=True),
            fromfile="a/experiment.py",
            tofile="b/experiment.py",
        )
    )
    diff_path.write_text(diff_text, encoding="utf-8")

    return ExperimentStateArtifacts(
        base_commit_sha=base_commit_sha,
        attempted_experiment_text=attempted_experiment_text,
        best_experiment_text=best_experiment_text,
        attempted_experiment_sha256=attempted_experiment_sha256,
        best_experiment_sha256=best_experiment_sha256,
        attempted_snapshot_path=attempted_snapshot_path,
        best_snapshot_path=best_snapshot_path,
        diff_path=diff_path,
        current_best_state=current_best_state,
    )


def restore_best_experiment(best_experiment_text: str, attempted_experiment_text: str) -> bool:
    if best_experiment_text == attempted_experiment_text:
        return False
    EXPERIMENT_PATH.write_text(best_experiment_text, encoding="utf-8")
    return True


def load_best_run_state() -> dict[str, object] | None:
    if not BEST_RUN_STATE_PATH.exists():
        return None
    return json.loads(BEST_RUN_STATE_PATH.read_text(encoding="utf-8"))


def decide_promotion(metrics: dict[str, object], current_best_state: dict[str, object] | None) -> SelectionDecision:
    if current_best_state is None:
        return SelectionDecision("promoted", "No prior best run is recorded; promoting the first scored run.")

    current_score = float(metrics["mean_proxy_score"])
    best_score = float(current_best_state["mean_proxy_score"])
    if current_score > best_score + PROMOTION_EPSILON:
        return SelectionDecision(
            "promoted",
            f"mean_proxy_score improved from {best_score:.6f} to {current_score:.6f}.",
        )
    if current_score < best_score - PROMOTION_EPSILON:
        return SelectionDecision(
            "rejected",
            f"mean_proxy_score regressed from {best_score:.6f} to {current_score:.6f}.",
        )

    current_validity = float(metrics["validity_rate"])
    best_validity = float(current_best_state["validity_rate"])
    if current_validity > best_validity:
        return SelectionDecision(
            "promoted",
            "mean_proxy_score was within the tie band and validity_rate improved.",
        )
    if current_validity < best_validity:
        return SelectionDecision(
            "rejected",
            "mean_proxy_score was within the tie band and validity_rate regressed.",
        )

    current_fallback = float(metrics["fallback_rate"])
    best_fallback = float(current_best_state["fallback_rate"])
    if current_fallback < best_fallback:
        return SelectionDecision(
            "promoted",
            "mean_proxy_score and validity_rate tied closely, and fallback_rate improved.",
        )
    return SelectionDecision(
        "rejected",
        "mean_proxy_score did not clear the promotion threshold and tie-breakers did not improve.",
    )


def score_validation_rows(model, tokenizer, device: torch.device, val_dataset: SvgSftDataset, sample_generations: int, score_limit: int | None = None):
    row_scores = []
    sample_records = []
    total_rows = min(len(val_dataset), score_limit) if score_limit is not None else len(val_dataset)
    total_batches = (total_rows + INFERENCE_BATCH_SIZE - 1) // INFERENCE_BATCH_SIZE
    print(f"[score] starting validation scoring: {total_rows} rows (of {len(val_dataset)} total), batch_size={INFERENCE_BATCH_SIZE}, {total_batches} batches")

    for start in range(0, total_rows, INFERENCE_BATCH_SIZE):
        batch_num = start // INFERENCE_BATCH_SIZE + 1
        batch_records = [val_dataset[index] for index in range(start, min(start + INFERENCE_BATCH_SIZE, total_rows))]
        raw_predicted_svgs = generate_svg_batch(model, tokenizer, [record.prompt for record in batch_records], device)

        for record, raw_predicted_svg in zip(batch_records, raw_predicted_svgs):
            predicted_svg, fallback_used, pre_fallback_validation = maybe_apply_fallback(raw_predicted_svg)
            row_score = score_svg_pair(
                row_id=record.row_id,
                reference_svg=record.svg,
                predicted_svg=predicted_svg,
                fallback_used=fallback_used,
            )
            row_scores.append(row_score)

            if len(sample_records) < sample_generations:
                sample_records.append(
                    {
                        "id": record.row_id,
                        "prompt": record.prompt,
                        "reference_svg_chars": len(record.svg),
                        "predicted_svg": predicted_svg,
                        "raw_predicted_svg": raw_predicted_svg,
                        "fallback_used": fallback_used,
                        "pre_fallback_validation": pre_fallback_validation,
                        "row_score": row_score.to_dict(),
                    }
                )

        scored_so_far = min(start + INFERENCE_BATCH_SIZE, total_rows)
        valid_so_far = sum(1 for rs in row_scores if rs.validation.valid)
        print(f"[score] batch {batch_num}/{total_batches}  scored={scored_so_far}/{total_rows}  valid={valid_so_far}/{scored_so_far}")

    return row_scores, sample_records


def write_row_scores(path: Path, row_scores: list) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row_score in row_scores:
            handle.write(json.dumps(row_score.to_dict(), sort_keys=True) + "\n")


def build_best_run_state(
    metrics: dict[str, object],
    state_artifacts: ExperimentStateArtifacts,
    output_dir: Path,
    selection_reason: str,
) -> dict[str, object]:
    return {
        "updated_at": utc_timestamp(),
        "best_run_id": output_dir.name,
        "base_commit_sha": state_artifacts.base_commit_sha,
        "mean_proxy_score": metrics["mean_proxy_score"],
        "validity_rate": metrics["validity_rate"],
        "fallback_rate": metrics["fallback_rate"],
        "blank_like_rate": metrics["blank_like_rate"],
        "eval_loss": metrics["eval_loss"],
        "output_dir": str(output_dir),
        "attempted_experiment_snapshot_path": str(state_artifacts.attempted_snapshot_path),
        "best_experiment_snapshot_path": str(state_artifacts.best_snapshot_path),
        "experiment_diff_path": str(state_artifacts.diff_path),
        "best_experiment_sha256": state_artifacts.attempted_experiment_sha256,
        "selection_reason": selection_reason,
    }


def run_training_and_scoring(args: argparse.Namespace, output_dir: Path, state_artifacts: ExperimentStateArtifacts) -> dict[str, object]:
    seed = int(experiment.TRAIN["seed"])
    set_seed(seed)

    train_df, _, _ = load_competition_frames(args.data_dir)
    validation_ids = load_validation_ids(validation_ids_path(REPO_ROOT))
    train_split, val_split = split_train_validation(train_df, validation_ids)

    if args.smoke_test:
        train_limit = args.train_limit or int(experiment.SMOKE_TEST["train_rows"])
        val_limit = args.val_limit or int(experiment.SMOKE_TEST["val_rows"])
        max_steps = args.max_steps or int(experiment.SMOKE_TEST["max_steps"])
        sample_generations = args.sample_generations or int(experiment.SMOKE_TEST["sample_generations"])
    else:
        train_limit = args.train_limit
        val_limit = args.val_limit
        max_steps = args.max_steps or int(experiment.TRAIN["max_steps"])
        sample_generations = args.sample_generations or 4

    train_dataset = SvgSftDataset(subset_frame(train_split, train_limit), normalize=True)
    val_dataset = SvgSftDataset(subset_frame(val_split, val_limit))

    device, dtype = device_and_dtype()
    resolved_model_name, tokenizer, model = load_tokenizer_and_model(dtype)

    lora_config = LoraConfig(
        r=experiment.LORA["r"],
        lora_alpha=experiment.LORA["alpha"],
        lora_dropout=experiment.LORA["dropout"],
        target_modules=experiment.LORA["target_modules"],
        bias=experiment.LORA["bias"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.to(device)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    collator = SvgCollator(tokenizer, int(experiment.TRAIN["max_seq_length"]))
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(experiment.TRAIN["per_device_batch_size"]),
        shuffle=True,
        generator=train_generator,
        collate_fn=collator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=float(experiment.TRAIN["learning_rate"]),
        weight_decay=float(experiment.TRAIN["weight_decay"]),
    )

    warmup_ratio = float(experiment.TRAIN.get("warmup_ratio", 0.1))
    num_warmup_steps = int(max_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_steps,
    )

    grad_accum = int(experiment.TRAIN["gradient_accumulation_steps"])
    step = 0
    optimizer.zero_grad(set_to_none=True)
    train_losses: list[float] = []
    eval_losses: list[dict[str, float]] = []

    # Early stopping
    early_stop_patience = int(experiment.TRAIN.get("early_stop_patience", 3))
    best_eval_loss = float("inf")
    best_eval_step = 0
    eval_patience_counter = 0
    early_stopped = False
    best_adapter_dir = output_dir / "best_adapter"
    best_adapter_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] starting: {max_steps} steps, grad_accum={grad_accum}, warmup={num_warmup_steps}, {len(train_dataset)} train rows, {len(val_dataset)} val rows")
    for epoch in range(int(experiment.TRAIN["epochs"])):
        batches_since_step = 0
        for batch_index, batch in enumerate(train_loader, start=1):
            model.train()
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / grad_accum
            loss.backward()
            train_losses.append(float(outputs.loss.item()))
            batches_since_step += 1

            if batch_index % grad_accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                batches_since_step = 0
                print(f"[train] step {step}/{max_steps}  train_loss={outputs.loss.item():.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

                if step % int(experiment.TRAIN["eval_every_steps"]) == 0:
                    eval_loss = average_loss(model, val_loader, device)
                    eval_losses.append({"step": step, "loss": eval_loss})
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        best_eval_step = step
                        eval_patience_counter = 0
                        model.save_pretrained(best_adapter_dir)
                        tokenizer.save_pretrained(best_adapter_dir)
                        print(f"[eval]  step {step}/{max_steps}  eval_loss={eval_loss:.4f}  (new best, saved)")
                    else:
                        eval_patience_counter += 1
                        print(f"[eval]  step {step}/{max_steps}  eval_loss={eval_loss:.4f}  (patience {eval_patience_counter}/{early_stop_patience})")
                        if eval_patience_counter >= early_stop_patience:
                            print(f"[train] early stopping at step {step} — best eval_loss={best_eval_loss:.4f} at step {best_eval_step}")
                            early_stopped = True

                if step >= max_steps or early_stopped:
                    break
        if batches_since_step > 0 and step < max_steps:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            print(f"[train] step {step}/{max_steps}  train_loss={outputs.loss.item():.4f}  lr={scheduler.get_last_lr()[0]:.2e}  (partial accum)")
            if step % int(experiment.TRAIN["eval_every_steps"]) == 0:
                eval_loss = average_loss(model, val_loader, device)
                eval_losses.append({"step": step, "loss": eval_loss})
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    best_eval_step = step
                    eval_patience_counter = 0
                    model.save_pretrained(best_adapter_dir)
                    tokenizer.save_pretrained(best_adapter_dir)
                    print(f"[eval]  step {step}/{max_steps}  eval_loss={eval_loss:.4f}  (new best, saved)")
                else:
                    eval_patience_counter += 1
                    print(f"[eval]  step {step}/{max_steps}  eval_loss={eval_loss:.4f}  (patience {eval_patience_counter}/{early_stop_patience})")
                    if eval_patience_counter >= early_stop_patience:
                        print(f"[train] early stopping at step {step} — best eval_loss={best_eval_loss:.4f} at step {best_eval_step}")
                        early_stopped = True
        if step >= max_steps or early_stopped:
            break

    if early_stopped and best_adapter_dir.exists() and (best_adapter_dir / "adapter_model.safetensors").exists():
        print(f"[train] restoring best adapter from step {best_eval_step} (eval_loss={best_eval_loss:.4f})")
        model = PeftModel.from_pretrained(
            model.base_model.model,
            str(best_adapter_dir),
        )
        model.to(device)
        model.eval()

    print(f"[train] training complete: {step} steps, last_train_loss={train_losses[-1]:.4f}" if train_losses else "[train] training complete: 0 steps")
    if early_stopped:
        print(f"[train] early stopped at step {step}, using best checkpoint from step {best_eval_step}")
    final_eval_loss = average_loss(model, val_loader, device)
    print(f"[eval]  final eval_loss={final_eval_loss:.4f}")
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.config.use_cache = True

    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    if experiment.TRAIN["save_adapter"]:
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        print(f"[save] adapter saved to {adapter_dir}")

    score_limit = VAL_SCORE_LIMIT if not args.smoke_test else None
    row_scores, sample_records = score_validation_rows(model, tokenizer, device, val_dataset, sample_generations, score_limit=score_limit)
    proxy_summary = summarize_proxy_scores(row_scores)
    if proxy_summary.scored_rows == 0:
        raise RuntimeError("Proxy scoring produced zero scored rows; cannot rank this experiment safely.")

    metrics = {
        "run_id": output_dir.name,
        "run_kind": "smoke" if args.smoke_test else "train",
        "experiment_name": experiment.EXPERIMENT_META["name"],
        "git_sha": state_artifacts.base_commit_sha,
        "output_dir": str(output_dir),
        "model_name_or_path": resolved_model_name,
        "train_rows": len(train_dataset),
        "val_rows": len(val_dataset),
        "max_steps": max_steps,
        "completed_steps": step,
        "early_stopped": early_stopped,
        "best_eval_step": best_eval_step,
        "best_eval_loss": best_eval_loss,
        "device": str(device),
        "dtype": str(dtype),
        "train_loss_last": train_losses[-1] if train_losses else None,
        "eval_loss": final_eval_loss,
        "mean_proxy_score": proxy_summary.mean_proxy_score,
        "validity_rate": proxy_summary.validity_rate,
        "fallback_rate": proxy_summary.fallback_rate,
        "blank_like_rate": proxy_summary.blank_like_rate,
        "skipped_reference_rows": proxy_summary.skipped_reference_rows,
        "reference_render_failures": proxy_summary.reference_render_failures,
        "prediction_render_failures": proxy_summary.prediction_render_failures,
        "scored_rows": proxy_summary.scored_rows,
        "sample_count": len(sample_records),
        "parameter_counts": trainable_parameter_summary(model),
        "train_losses": train_losses,
        "eval_losses": eval_losses,
    }
    save_json(output_dir / "metrics.json", metrics)
    save_json(
        output_dir / "config_snapshot.json",
        {
            "MODEL": experiment.MODEL,
            "PROMPT": experiment.PROMPT,
            "LORA": experiment.LORA,
            "TRAIN": experiment.TRAIN,
            "DECODE": experiment.DECODE,
            "POSTPROCESS": experiment.POSTPROCESS,
            "FALLBACK": experiment.FALLBACK,
            "SMOKE_TEST": experiment.SMOKE_TEST,
            "EXPERIMENT_META": experiment.EXPERIMENT_META,
        },
    )
    (output_dir / "generations.json").write_text(json.dumps(sample_records, indent=2) + "\n", encoding="utf-8")
    write_row_scores(output_dir / "row_scores.jsonl", row_scores)
    (output_dir / "commit_sha.txt").write_text(state_artifacts.base_commit_sha + "\n", encoding="ascii")
    return metrics


def finalize_run(
    metrics: dict[str, object],
    state_artifacts: ExperimentStateArtifacts,
    output_dir: Path,
) -> tuple[dict[str, object], bool]:
    selection = decide_promotion(metrics, state_artifacts.current_best_state)
    restored_experiment = False

    if selection.promoted:
        best_run_state = build_best_run_state(metrics, state_artifacts, output_dir, selection.reason)
        save_json(BEST_RUN_STATE_PATH, best_run_state)
    else:
        restored_experiment = restore_best_experiment(
            state_artifacts.best_experiment_text,
            state_artifacts.attempted_experiment_text,
        )

    run_record = {
        "timestamp_utc": utc_timestamp(),
        "run_id": output_dir.name,
        "run_kind": metrics["run_kind"],
        "run_status": "completed",
        "selection_outcome": selection.outcome,
        "selection_reason": selection.reason,
        "base_commit_sha": state_artifacts.base_commit_sha,
        "attempted_experiment_sha256": state_artifacts.attempted_experiment_sha256,
        "best_experiment_sha256_at_launch": state_artifacts.best_experiment_sha256,
        "attempted_experiment_snapshot_path": str(state_artifacts.attempted_snapshot_path),
        "best_experiment_snapshot_path": str(state_artifacts.best_snapshot_path),
        "experiment_diff_path": str(state_artifacts.diff_path),
        "restored_experiment": restored_experiment,
        "metrics": {
            "mean_proxy_score": metrics["mean_proxy_score"],
            "validity_rate": metrics["validity_rate"],
            "fallback_rate": metrics["fallback_rate"],
            "blank_like_rate": metrics["blank_like_rate"],
            "eval_loss": metrics["eval_loss"],
            "skipped_reference_rows": metrics["skipped_reference_rows"],
            "reference_render_failures": metrics["reference_render_failures"],
            "prediction_render_failures": metrics["prediction_render_failures"],
            "scored_rows": metrics["scored_rows"],
            "val_rows": metrics["val_rows"],
            "completed_steps": metrics["completed_steps"],
        },
        "output_dir": str(output_dir),
    }
    save_json(output_dir / "run_record.json", run_record)
    append_jsonl(RUN_RESULTS_PATH, run_record)
    return run_record, restored_experiment


def write_failure_record(
    output_dir: Path,
    state_artifacts: ExperimentStateArtifacts | None,
    exc: Exception,
) -> dict[str, object]:
    restored_experiment = False
    if state_artifacts is not None:
        restored_experiment = restore_best_experiment(
            state_artifacts.best_experiment_text,
            state_artifacts.attempted_experiment_text,
        )

    run_record = {
        "timestamp_utc": utc_timestamp(),
        "run_id": output_dir.name,
        "run_kind": "smoke" if "smoke" in output_dir.parts else "train",
        "run_status": "failed",
        "selection_outcome": "crashed",
        "selection_reason": f"{type(exc).__name__}: {exc}",
        "base_commit_sha": state_artifacts.base_commit_sha if state_artifacts is not None else git_sha(),
        "attempted_experiment_sha256": state_artifacts.attempted_experiment_sha256 if state_artifacts is not None else None,
        "best_experiment_sha256_at_launch": state_artifacts.best_experiment_sha256 if state_artifacts is not None else None,
        "attempted_experiment_snapshot_path": str(state_artifacts.attempted_snapshot_path) if state_artifacts is not None else None,
        "best_experiment_snapshot_path": str(state_artifacts.best_snapshot_path) if state_artifacts is not None else None,
        "experiment_diff_path": str(state_artifacts.diff_path) if state_artifacts is not None else None,
        "restored_experiment": restored_experiment,
        "output_dir": str(output_dir),
        "traceback_path": str(output_dir / "failure.txt"),
    }
    (output_dir / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
    save_json(output_dir / "run_record.json", run_record)
    append_jsonl(RUN_RESULTS_PATH, run_record)
    return run_record


def main() -> int:
    args = parse_args()
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_artifacts: ExperimentStateArtifacts | None = None

    try:
        state_artifacts = snapshot_experiment_state(output_dir)
        metrics = run_training_and_scoring(args, output_dir, state_artifacts)
        run_record, _ = finalize_run(metrics, state_artifacts, output_dir)
        print(json.dumps(run_record, indent=2))
        return 0
    except Exception as exc:
        if state_artifacts is None:
            print(
                json.dumps(
                    {
                        "run_status": "failed_prelaunch",
                        "selection_outcome": "not_started",
                        "reason": f"{type(exc).__name__}: {exc}",
                    },
                    indent=2,
                )
            )
            return 1
        run_record = write_failure_record(output_dir, state_artifacts, exc)
        print(json.dumps(run_record, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
