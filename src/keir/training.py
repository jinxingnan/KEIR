import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import torch

from .io import read_jsonl, seed_everything, write_json


@dataclass
class RoleTrainingConfig:
    model_name_or_path: str
    train_file: str
    validation_file: str
    output_dir: str
    max_length: int = 2048
    epochs: float = 3.0
    max_steps: int = -1
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    label_smoothing_factor: float = 0.0
    lr_scheduler_type: str = "linear"
    logging_steps: int = 10
    save_steps: int = 250
    eval_steps: int = 250
    early_stopping_patience: int = 0
    early_stopping_threshold: float = 0.0
    seed: int = 42
    qlora: bool = True
    bf16: bool = True
    gradient_checkpointing: bool = True
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: Any = "all-linear"
    trust_remote_code: bool = False
    use_chat_template: bool = False
    resume_from_checkpoint: Optional[str] = None
    init_adapter_path: Optional[str] = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RoleTrainingConfig":
        known = {field.name for field in __import__("dataclasses").fields(cls)}
        unknown = set(value) - known
        if unknown:
            raise ValueError("Unknown training keys: %s" % ", ".join(sorted(unknown)))
        return cls(**dict(value))


class CompletionOnlyCollator:


    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_length = max(len(item["input_ids"]) for item in features)
        pad_id = self.tokenizer.pad_token_id
        side = self.tokenizer.padding_side
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in features:
            length = len(item["input_ids"])
            padding = max_length - length
            if side == "left":
                batch["input_ids"].append([pad_id] * padding + item["input_ids"])
                batch["attention_mask"].append([0] * padding + item["attention_mask"])
                batch["labels"].append([-100] * padding + item["labels"])
            else:
                batch["input_ids"].append(item["input_ids"] + [pad_id] * padding)
                batch["attention_mask"].append(item["attention_mask"] + [0] * padding)
                batch["labels"].append(item["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def _tokenize_record(
    record: Mapping[str, Any],
    tokenizer: Any,
    max_length: int,
    use_chat_template: bool = False,
) -> Dict[str, Any]:
    prompt = str(record["prompt"])
    response = str(record["response"])
    eos = tokenizer.eos_token or ""
    if use_chat_template:
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError("use_chat_template requires a tokenizer chat_template")
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
        response_ids = tokenizer(
            response + eos, add_special_tokens=False, truncation=False
        )["input_ids"]
        full_ids = list(prompt_ids) + response_ids
        prefix_length = len(prompt_ids)
    else:
        prompt_ids = tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"]
        full_ids = tokenizer(prompt + response + eos, add_special_tokens=True, truncation=False)[
            "input_ids"
        ]

        prefix_length = min(len(prompt_ids), len(full_ids))
        while prefix_length and prompt_ids[:prefix_length] != full_ids[:prefix_length]:
            prefix_length -= 1
    if len(full_ids) > max_length:
        raise ValueError("Record %s exceeds max_length=%d; increase the context limit instead "
                         "of truncating the problem or its solution history"
                         % (record.get("id", "unknown"), max_length))
    labels = [-100] * prefix_length + full_ids[prefix_length:]
    if not any(label != -100 for label in labels):
        raise ValueError("No response tokens to supervise for record %s" % record.get("id", "unknown"))
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def _load_records(path: str) -> List[Dict[str, Any]]:
    records = list(read_jsonl(path))
    if not records:
        raise ValueError("No training records found in %s" % path)
    for record in records:
        if "prompt" not in record or "response" not in record:
            raise ValueError("Training records require prompt and response fields")
    return records


def train_role(config: RoleTrainingConfig) -> Dict[str, Any]:

    try:
        from datasets import Dataset
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError("Install the full training requirements before running train") from exc

    seed_everything(config.seed)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if config.qlora and not torch.cuda.is_available():
        raise RuntimeError("QLoRA training requires a visible CUDA GPU")
    runtime_dtype = (
        torch.bfloat16
        if config.bf16
        else (torch.float16 if torch.cuda.is_available() else torch.float32)
    )
    model_kwargs = {
        "trust_remote_code": config.trust_remote_code,
        "torch_dtype": runtime_dtype,
    }
    if config.qlora:
        try:
            import bitsandbytes
        except ImportError as exc:
            raise RuntimeError(
                "QLoRA requested but bitsandbytes is unavailable. Install bitsandbytes>=0.43."
            ) from exc
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if config.bf16 else torch.float16,
        )
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        model_kwargs["device_map"] = {"": local_rank}
    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, **model_kwargs)
    model.config.use_cache = False
    if config.qlora:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config.gradient_checkpointing
        )
    elif config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    if config.init_adapter_path:
        model = PeftModel.from_pretrained(
            model,
            config.init_adapter_path,
            is_trainable=True,
        )
    else:
        lora = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
    trainable, total = model.get_nb_trainable_parameters()

    def encode(record: Dict[str, Any]) -> Dict[str, Any]:
        return _tokenize_record(
            record, tokenizer, config.max_length, use_chat_template=config.use_chat_template
        )

    train_records = _load_records(config.train_file)
    train_dataset = Dataset.from_list(train_records).map(
        encode, remove_columns=list(train_records[0].keys())
    )
    validation_records = _load_records(config.validation_file)
    validation_dataset = Dataset.from_list(validation_records).map(
        encode, remove_columns=list(validation_records[0].keys())
    )
    if config.early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be non-negative")
    callbacks = []
    if config.early_stopping_patience:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience,
                early_stopping_threshold=config.early_stopping_threshold,
            )
        )

    bf16_enabled = bool(config.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    arguments = TrainingArguments(
        output_dir=str(output),
        num_train_epochs=config.epochs,
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        label_smoothing_factor=config.label_smoothing_factor,
        lr_scheduler_type=config.lr_scheduler_type,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=False,
        bf16=bf16_enabled,
        fp16=bool(torch.cuda.is_available() and not bf16_enabled),
        gradient_checkpointing=config.gradient_checkpointing,
        optim="paged_adamw_32bit" if config.qlora else "adamw_torch",
        report_to=[],
        seed=config.seed,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=CompletionOnlyCollator(tokenizer),
        callbacks=callbacks,
    )
    result = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    trainer.save_model(str(output / "adapter"))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(output / "adapter"))
    metrics = dict(result.metrics)
    metrics.update(
        {
            "trainable_parameters": trainable,
            "total_parameters": total,
            "trainable_fraction": trainable / float(total),
            "config": asdict(config),
        }
    )
    if trainer.is_world_process_zero():
        write_json(str(output / "training_metrics.json"), metrics)
    return metrics


def train_keir(config: Mapping[str, Any]) -> Dict[str, Any]:

    common = dict(config.get("common", {}))
    results = {}
    roles = tuple(config.get("roles", ("decomposer", "executor")))
    if roles == ("decomposer", "executor"):
        from .balanced_training import train_interleaved
        return train_interleaved(config)
    if roles != ("cot",):
        raise ValueError("Use roles=[decomposer, executor] for KEIR or roles=[cot] for the control")
    for role in roles:
        role_config = dict(common)
        role_config.update(config.get(role, {}))
        results[role] = train_role(RoleTrainingConfig.from_mapping(role_config))
    return results
