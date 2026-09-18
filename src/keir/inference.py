import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .answers import majority_answer, normalize_benchmark_prediction
from .consolidation import SemanticConsolidator
from .prompts import cot_prompt, decomposition_prompt, execution_prompt, parse_subtasks
from .schemas import GenerationStats, Prediction
from .io import read_json, stable_hash
from .costs import estimate_lm_cost


@dataclass
class GenerationResult:
    text: str
    stats: GenerationStats
    prompt_fingerprint: Optional[str] = None


class HFMultiAdapterRuntime:


    def __init__(self, model_name_or_path: str, adapters: Dict[str, str],
                 qlora: bool = True, bf16: bool = True,
                 trust_remote_code: bool = False, use_chat_template: bool = False,
                 nominal_parameters: Optional[int] = None):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not adapters:
            raise ValueError("Provide trained role adapters; pretrained checkpoints alone are not KEIR")
        if nominal_parameters is not None:
            estimate_lm_cost(nominal_parameters, 0, 0, "processed")
        self.nominal_parameters = nominal_parameters
        if qlora and not torch.cuda.is_available():
            raise RuntimeError("4-bit inference requires a CUDA GPU")
        self.torch = torch
        self.roles = set(adapters)
        seeds = []
        for path in adapters.values():
            metadata = Path(path).parent / "training_metrics.json"
            seeds.append(read_json(str(metadata)).get("config", {}).get("seed") if metadata.exists() else None)
        known_seeds = {seed for seed in seeds if seed is not None}
        if len(known_seeds) > 1:
            raise ValueError("Role adapters have different training seeds")
        self.training_seed = seeds[0] if all(seed is not None for seed in seeds) else None
        self.use_chat_template = use_chat_template
        first_role, first_path = next(iter(adapters.items()))
        self.tokenizer = AutoTokenizer.from_pretrained(first_path, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        if use_chat_template and not getattr(self.tokenizer, "chat_template", None):
            raise ValueError("use_chat_template requires a tokenizer chat template")
        dtype = torch.bfloat16 if bf16 else (torch.float16 if torch.cuda.is_available() else torch.float32)
        kwargs = {"trust_remote_code": trust_remote_code, "torch_dtype": dtype}
        if qlora:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
            )
            kwargs["device_map"] = {"": 0}
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        if not qlora and torch.cuda.is_available():
            model = model.to("cuda:0")
        self.model = PeftModel.from_pretrained(model, first_path, adapter_name=first_role, is_trainable=False)
        for role, path in list(adapters.items())[1:]:
            self.model.load_adapter(path, adapter_name=role, is_trainable=False)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.parameter_count = self.model.num_parameters(exclude_embeddings=False)

    def start_request(self) -> float:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
            self.torch.cuda.reset_peak_memory_stats(self.device)
        return time.perf_counter()

    def end_request(self, start: float):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
        latency = time.perf_counter() - start
        peak = self.torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0
        return latency, peak

    def generate(self, role: str, prompt: str, max_new_tokens: int = 256,
                 do_sample: bool = False, temperature: float = 0.7,
                 top_p: float = 0.9, seed: int = 42) -> GenerationResult:
        if role not in self.roles:
            raise ValueError("Adapter role not loaded: " + role)
        self.model.set_adapter(role)
        if self.use_chat_template:
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
            )
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=not self.use_chat_template)
        prompt_fingerprint = stable_hash(inputs["input_ids"][0].tolist())
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        self.torch.manual_seed(seed)
        if self.device.type == "cuda":
            self.torch.cuda.manual_seed_all(seed)
            self.torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample, num_beams=1,
                      pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
        kwargs.update(temperature=temperature if do_sample else None,
                      top_p=top_p if do_sample else None, top_k=0 if do_sample else None)
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **kwargs)
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
        latency = time.perf_counter() - start
        prompt_tokens = int(inputs["input_ids"].shape[1])
        generated_tokens = int(output.shape[1] - prompt_tokens)
        text = self.tokenizer.decode(output[0, prompt_tokens:], skip_special_tokens=True)
        return GenerationResult(text, GenerationStats(
            prompt_tokens=prompt_tokens, generated_tokens=generated_tokens,
            latency_seconds=latency,
            estimated_flops=estimate_lm_cost(self.parameter_count, prompt_tokens, generated_tokens,
                                             "processed", "loaded_model")["estimated_lm_flops"],
            nominal_parameters=self.nominal_parameters, loaded_parameters=self.parameter_count,
        ), prompt_fingerprint=prompt_fingerprint)


def _start(runtime):
    return runtime.start_request() if hasattr(runtime, "start_request") else time.perf_counter()


def _finish(runtime, start, parts):
    latency, peak = runtime.end_request(start) if hasattr(runtime, "end_request") else (time.perf_counter() - start, 0)
    unique = {}
    for part in parts:
        if part.prompt_fingerprint is not None:
            unique[part.prompt_fingerprint] = part.stats.prompt_tokens
    unique_tokens = sum(unique.values()) if all(x.prompt_fingerprint is not None for x in parts) else None
    nominal = {x.stats.nominal_parameters for x in parts}
    loaded = {x.stats.loaded_parameters for x in parts}
    if len(nominal) > 1 or len(loaded) > 1:
        raise ValueError("Generation calls use different parameter bases")
    return GenerationStats(
        prompt_tokens=sum(x.stats.prompt_tokens for x in parts),
        unique_prompt_tokens=unique_tokens,
        generated_tokens=sum(x.stats.generated_tokens for x in parts),
        latency_seconds=latency, peak_memory_bytes=peak,
        estimated_flops=sum(x.stats.estimated_flops for x in parts),
        nominal_parameters=next(iter(nominal)) if nominal else None,
        loaded_parameters=next(iter(loaded)) if loaded else None,
    )


class KEIRReasoner:
    def __init__(self, runtime: Any, consolidator: Optional[SemanticConsolidator],
                 max_subtasks: int = 10, max_new_tokens: int = 256):
        if max_subtasks < 1 or max_new_tokens < 1:
            raise ValueError("Generation limits must be positive")
        self.runtime = runtime
        self.consolidator = consolidator
        self.max_subtasks = max_subtasks
        self.max_new_tokens = max_new_tokens

    def solve(self, example_id: str, statement: str, question: str,
              gold: Optional[str] = None, consolidate: bool = True, seed: int = 42,
              dataset: str = "") -> Prediction:
        start = _start(self.runtime)
        decomposition = self.runtime.generate(
            "decomposer", decomposition_prompt(statement, question, self.max_subtasks),
            max_new_tokens=self.max_new_tokens, do_sample=False, seed=seed,
        )
        raw = parse_subtasks(decomposition.text, self.max_subtasks)
        if not raw:

            return Prediction(
                id=example_id, prediction="", raw_subtasks=[], retained_subtasks=[], solutions=[],
                stats=_finish(self.runtime, start, [decomposition]), gold=gold,
                metadata={"dataset": dataset, "failure": "empty_decomposition",
                          "latency_scope": "end_to_end_single_problem"},
            )
        retained = list(raw)
        info = {"predicted_count": len(raw), "executed_count": len(raw), "reduction": 0.0}
        if consolidate and self.consolidator is not None:
            retained, info = self.consolidator.consolidate(raw)
        solutions, parts = [], [decomposition]
        for index, task in enumerate(retained):
            result = self.runtime.generate(
                "executor", execution_prompt(statement, question, task, retained[:index], solutions,
                                              is_terminal=index == len(retained) - 1),
                max_new_tokens=self.max_new_tokens, do_sample=False, seed=seed + index + 1,
            )
            solutions.append(result.text.strip())
            parts.append(result)
        answer = normalize_benchmark_prediction(solutions[-1], dataset)
        stats = _finish(self.runtime, start, parts)
        return Prediction(
            id=example_id, prediction=answer, raw_subtasks=raw, retained_subtasks=retained,
            solutions=solutions, stats=stats, gold=gold,
            metadata={"dataset": dataset, "consolidation": info,
                      "latency_scope": "end_to_end_single_problem",
                      "generation_latency_seconds": sum(x.stats.latency_seconds for x in parts)},
        )


def solve_cot(runtime, example_id, statement, question, samples=1, max_new_tokens=256,
              seed=42, gold=None, dataset=""):
    if samples < 1:
        raise ValueError("samples must be positive")
    start = _start(runtime)
    answers, traces, parts = [], [], []
    for index in range(samples):
        result = runtime.generate(
            "cot", cot_prompt(statement, question), max_new_tokens=max_new_tokens,
            do_sample=samples > 1, temperature=0.7, top_p=0.9, seed=seed + index,
        )
        traces.append(result.text)
        answers.append(normalize_benchmark_prediction(result.text, dataset))
        parts.append(result)

    answer = majority_answer(traces, dataset)
    return Prediction(
        id=example_id, prediction=answer, raw_subtasks=[], retained_subtasks=[], solutions=traces,
        stats=_finish(runtime, start, parts), gold=gold,
        metadata={"dataset": dataset, "samples": samples, "sample_answers": answers,
                  "latency_scope": "end_to_end_single_problem",
                  "generation_latency_seconds": sum(x.stats.latency_seconds for x in parts)},
    )
