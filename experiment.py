"""Single editable experiment surface for the text-to-SVG project."""

MODEL = {
    "name_or_path": "Qwen/Qwen2.5-Coder-3B-Instruct",
    "fallback_chain": [
        "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    ],
    "local_files_only": False,
    "gradient_checkpointing": True,
}

PROMPT = {
    "system": (
        "You generate compact, valid SVG markup for 256x256 canvases. "
        "Return only SVG code with a single root <svg> element."
    ),
    "template_version": "qwen-svg-v1",
}

LORA = {
    "r": 16,
    "alpha": 48,
    "dropout": 0.05,
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "bias": "none",
}

TRAIN = {
    "seed": 42,
    "learning_rate": 4e-4,
    "warmup_ratio": 0.95,
    "weight_decay": 0.05,
    "epochs": 1,
    "max_steps": 256,
    "per_device_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "max_seq_length": 8192,
    "eval_every_steps": 32,
    "early_stop_patience": 999,
    "save_adapter": True,
}

DECODE = {
    "max_new_tokens": 2048,
    "do_sample": False,
    "temperature": 0.0,
    "top_p": 1.0,
}

POSTPROCESS = {
    "extract_svg_regex": True,
    "strip_whitespace": True,
}

FALLBACK = {
    "enabled": False,
    "mode": "blank",
}

SMOKE_TEST = {
    "train_rows": 32,
    "val_rows": 4,
    "max_steps": 2,
    "sample_generations": 2,
}

EXPERIMENT_META = {
    "name": "final-256step-2048tok",
    "description": "Final project configuration: Qwen2.5-Coder-3B-Instruct with LoRA fine-tuning, 256 training steps, and 2048-token decoding for complete SVG generation.",
}
