import math
import os
import random
import tempfile
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import torch

from .io import seed_everything, write_json
from .training import CompletionOnlyCollator, RoleTrainingConfig, _load_records, _tokenize_record

ROLES = ("decomposer", "executor")


def balanced_epoch_batches(sizes, batch_size, accumulation, world_size=1, rank=0,
                           seed=42, epoch=0):


    if set(sizes) != set(ROLES) or min(sizes.values()) < 1:
        raise ValueError("Both roles require nonempty training streams")
    if min(batch_size, accumulation, world_size) < 1 or not 0 <= rank < world_size:
        raise ValueError("Invalid batch or distributed configuration")
    effective = batch_size * accumulation * world_size
    updates = math.ceil(max(sizes.values()) / effective)
    streams = {}
    for role_index, role in enumerate(ROLES):
        rng = random.Random(seed + 1000003 * epoch + 7919 * role_index)
        stream = []
        while len(stream) < updates * effective:
            permutation = list(range(sizes[role]))
            rng.shuffle(permutation)
            stream.extend(permutation)
        streams[role] = stream
    for update in range(updates):
        for role in ROLES:
            batches = []
            for micro in range(accumulation):
                offset = update * effective + (micro * world_size + rank) * batch_size
                batches.append(streams[role][offset:offset + batch_size])
            yield role, batches


def role_parameters(model):

    parameters = {}
    for role in ROLES:
        model.set_adapter(role)
        parameters[role] = [p for p in model.parameters() if p.requires_grad]
        if not parameters[role]:
            raise ValueError("No trainable parameters for " + role)
    if {id(p) for p in parameters[ROLES[0]]} & {id(p) for p in parameters[ROLES[1]]}:
        raise ValueError("Role adapters must not share trainable parameters")
    return parameters


def run_balanced_epoch(model, records, collator, optimizers, schedulers, parameters,
                       *, device, batch_size, accumulation, seed, epoch,
                       world_size=1, rank=0, bf16=False):


    model.train()
    counts = {role: 0 for role in ROLES}
    losses = {role: 0.0 for role in ROLES}
    batches = balanced_epoch_batches(
        {role: len(records[role]) for role in ROLES}, batch_size, accumulation,
        world_size, rank, seed, epoch,
    )
    for role, microbatches in batches:
        model.set_adapter(role)
        model.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for indices in microbatches:
            batch = collator([records[role][i] for i in indices])
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=bf16 and device.type == "cuda"):
                loss = model(**batch).loss
            (loss / accumulation).backward()
            loss_sum += float(loss.detach()) / accumulation

        for parameter in parameters[role]:
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            if world_size > 1:
                torch.distributed.all_reduce(parameter.grad)
                parameter.grad.div_(world_size)
        if world_size > 1:
            mean_loss = torch.tensor(loss_sum, device=device)
            torch.distributed.all_reduce(mean_loss)
            loss_sum = float(mean_loss / world_size)
        torch.nn.utils.clip_grad_norm_(parameters[role], 1.0, error_if_nonfinite=True)
        optimizers[role].step()
        schedulers[role].step()
        counts[role] += 1
        losses[role] += loss_sum
    return {role: {"updates": counts[role], "mean_loss": losses[role] / counts[role]}
            for role in ROLES}


def paired_configs(config):
    result = {}
    for role in ROLES:
        settings = dict(config.get("common", {}))
        settings.update(config.get(role, {}))
        result[role] = RoleTrainingConfig.from_mapping(settings)
    excluded = {"train_file", "validation_file", "output_dir"}
    left, right = (asdict(result[role]) for role in ROLES)
    differing = [key for key in left if key not in excluded and left[key] != right[key]]
    if differing:
        raise ValueError("Paired training requires common role settings: " + ", ".join(differing))
    common = result[ROLES[0]]
    if common.epochs < 1 or not float(common.epochs).is_integer():
        raise ValueError("Balanced training requires a positive integer epoch count")
    if common.max_steps != -1 or common.early_stopping_patience or common.label_smoothing_factor:
        raise ValueError("Use epoch-based completion-only training without early stopping")
    if common.resume_from_checkpoint or common.init_adapter_path:
        raise ValueError("Paired resume/init is not implemented; start both adapters together")
    if Path(result[ROLES[0]].output_dir).resolve() == Path(result[ROLES[1]].output_dir).resolve():
        raise ValueError("Role output directories must differ")
    for role in ROLES:
        if (Path(result[role].output_dir) / "adapter").exists():
            raise FileExistsError("Refusing to replace an existing adapter: " + result[role].output_dir)
    return result


def train_interleaved(config):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_scheduler

    configs = paired_configs(config)
    common = configs[ROLES[0]]
    seed_everything(common.seed)
    cuda = torch.cuda.is_available()
    if common.qlora and not cuda:
        raise RuntimeError("QLoRA training requires CUDA")
    if cuda and not common.bf16:
        raise ValueError("The KEIR paper GPU configuration requires bfloat16 computation")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if cuda:
        torch.cuda.set_device(local_rank)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("The configured GPU must support bfloat16")
    device = torch.device("cuda", local_rank) if cuda else torch.device("cpu")
    owned_group = False
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl" if cuda else "gloo")
        owned_group = True
    try:
        distributed = torch.distributed.is_initialized()
        world_size = torch.distributed.get_world_size() if distributed else 1
        rank = torch.distributed.get_rank() if distributed else 0
        tokenizer = AutoTokenizer.from_pretrained(common.model_name_or_path,
                                                  trust_remote_code=common.trust_remote_code)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        records = {role: [_tokenize_record(row, tokenizer, common.max_length,
                                          common.use_chat_template)
                          for row in _load_records(configs[role].train_file)] for role in ROLES}
        kwargs = {"torch_dtype": torch.bfloat16 if cuda else torch.float32,
                  "trust_remote_code": common.trust_remote_code}
        if common.qlora:
            kwargs.update(quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16), device_map={"": local_rank})
        backbone = AutoModelForCausalLM.from_pretrained(common.model_name_or_path, **kwargs)
        backbone.config.use_cache = False
        if common.qlora:
            backbone = prepare_model_for_kbit_training(backbone, use_gradient_checkpointing=False)
        else:
            backbone.to(device)
        def adapter_config():
            return LoraConfig(r=common.lora_rank, lora_alpha=common.lora_alpha,
                              lora_dropout=common.lora_dropout, target_modules=common.target_modules,
                              bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(backbone, adapter_config(), adapter_name=ROLES[0])


        model.add_adapter(ROLES[1], deepcopy(model.peft_config[ROLES[0]]))
        if common.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
        parameters = role_parameters(model)
        optimizer_class = torch.optim.AdamW
        if common.qlora:
            from bitsandbytes.optim import PagedAdamW32bit
            optimizer_class = PagedAdamW32bit
        optimizers = {role: optimizer_class(parameters[role], lr=common.learning_rate,
                                            weight_decay=common.weight_decay) for role in ROLES}
        effective = common.per_device_train_batch_size * common.gradient_accumulation_steps * world_size
        per_epoch = math.ceil(max(map(len, records.values())) / effective)
        total_steps = int(common.epochs) * per_epoch
        schedulers = {role: get_scheduler(common.lr_scheduler_type, optimizer=optimizers[role],
                         num_warmup_steps=math.ceil(total_steps * common.warmup_ratio),
                         num_training_steps=total_steps) for role in ROLES}
        history = []
        for epoch in range(int(common.epochs)):
            metrics = run_balanced_epoch(
                model, records, CompletionOnlyCollator(tokenizer), optimizers, schedulers, parameters,
                device=device, batch_size=common.per_device_train_batch_size,
                accumulation=common.gradient_accumulation_steps, seed=common.seed, epoch=epoch,
                world_size=world_size, rank=rank, bf16=cuda,
            )
            history.append(metrics)
            if rank == 0:
                print({"epoch": epoch + 1, "roles": metrics}, flush=True)
        results = {role: {"config": asdict(configs[role]), "role_batch_ratio": "1:1",
                          "updates": total_steps, "effective_batch_size": effective,
                          "checkpoint_selection": "final_epoch", "epochs": history,
                          "epoch_definition": "longer_stream_with_cycling_and_full_batch_padding"}
                   for role in ROLES}
        if rank == 0:
            for role in ROLES:
                output = Path(configs[role].output_dir)
                output.mkdir(parents=True, exist_ok=True)


                with tempfile.TemporaryDirectory(prefix="export-", dir=output) as temporary:
                    model.save_pretrained(temporary, selected_adapters=[role])
                    Path(temporary, role).rename(output / "adapter")
                tokenizer.save_pretrained(output / "adapter")
                write_json(str(output / "training_metrics.json"), results[role])
        if distributed:
            torch.distributed.barrier()
        return results
    finally:
        if owned_group:
            torch.distributed.destroy_process_group()
