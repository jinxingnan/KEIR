import argparse
import json


def _print(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _runtime_from_config(config, mode="keir"):
    from .consolidation import OfficialSimCSEEncoder, SemanticConsolidator
    from .inference import HFMultiAdapterRuntime, KEIRReasoner

    model = config["model"]
    inference = config.get("inference", {})
    if mode == "cot":
        if not model.get("cot_adapter"):
            raise ValueError("CoT evaluation requires model.cot_adapter trained on unified CoT records")
        adapters = {"cot": model["cot_adapter"]}
    else:
        adapters = {role: model[role + "_adapter"] for role in ("decomposer", "executor")}
    runtime = HFMultiAdapterRuntime(
        model["name_or_path"], adapters,
        qlora=model.get("qlora", True), bf16=model.get("bf16", True),
        trust_remote_code=model.get("trust_remote_code", False),
        use_chat_template=model.get("use_chat_template", False),
        nominal_parameters=model.get("nominal_parameters"),
    )
    consolidator = None
    if mode == "keir" and inference.get("consolidation", True):
        encoder = OfficialSimCSEEncoder(
            model_name=inference.get("embedding_model", "princeton-nlp/sup-simcse-roberta-base"),
            device=str(runtime.device),
            batch_size=inference.get("embedding_batch_size", 32),
        )
        consolidator = SemanticConsolidator(encoder, threshold=inference.get("threshold", 0.25))
    return KEIRReasoner(runtime, consolidator, max_subtasks=inference.get("max_subtasks", 10),
                        max_new_tokens=inference.get("max_new_tokens", 256))


def build_parser():
    parser = argparse.ArgumentParser(prog="keir", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser("download-gsm8k")
    download.add_argument("--output-dir", required=True)
    download.add_argument("--split", choices=("train", "test"), default="train")
    download.add_argument("--config", choices=("main", "socratic"), default="main")

    benchmark = sub.add_parser("download-benchmark")
    benchmark.add_argument("--benchmark", required=True, choices=("asdiv", "gsm8k", "math", "mawps", "svamp"))
    benchmark.add_argument("--output", required=True)

    normalize = sub.add_parser("normalize-benchmark")
    normalize.add_argument("--benchmark", required=True, choices=("asdiv", "gsm8k", "math", "mawps", "svamp"))
    normalize.add_argument("--input", required=True)
    normalize.add_argument("--output", required=True)
    normalize.add_argument("--source-url")
    normalize.add_argument("--source-revision")
    normalize.add_argument("--source-split")

    validate = sub.add_parser("validate-benchmark")
    validate.add_argument("--benchmark", required=True, choices=("asdiv", "gsm8k", "math", "mawps", "svamp"))
    validate.add_argument("--input", required=True)
    validate.add_argument("--allow-size-mismatch", action="store_true")

    prepare_socratic = sub.add_parser("prepare-socratic-v0")
    prepare_socratic.add_argument("--socratic-input", required=True)
    prepare_socratic.add_argument("--main-input", required=True)
    prepare_socratic.add_argument("--output", required=True)
    prepare_socratic.add_argument("--rejected", required=True)
    prepare_socratic.add_argument("--stats", required=True)
    prepare_socratic.add_argument("--split", choices=("train", "test"), default="train")

    plan_augmentation = sub.add_parser("plan-gpt4o-augmentation")
    plan_augmentation.add_argument("--input", required=True)
    plan_augmentation.add_argument("--requests", required=True)
    plan_augmentation.add_argument("--manifest", required=True)
    plan_augmentation.add_argument("--limit", type=int)
    plan_augmentation.add_argument("--variants", type=int, default=3)
    plan_augmentation.add_argument("--model", default="gpt-4o-2024-08-06")
    plan_augmentation.add_argument("--stage-b-model")
    plan_augmentation.add_argument("--near-duplicate-ratio", type=float, default=0.82)

    teacher_stage_a = sub.add_parser("run-teacher-stage-a")
    teacher_stage_a.add_argument("--requests", required=True)
    teacher_stage_a.add_argument("--v0-input", required=True)
    teacher_stage_a.add_argument("--responses", required=True)
    teacher_stage_a.add_argument("--candidates", required=True)
    teacher_stage_a.add_argument("--stats", required=True)
    teacher_stage_a.add_argument("--cache-dir", required=True)
    teacher_stage_a.add_argument("--model")
    teacher_stage_a.add_argument("--base-url")
    teacher_stage_a.add_argument("--workers", type=int, default=8)
    teacher_stage_a.add_argument("--timeout", type=float, default=240.0)
    teacher_stage_a.add_argument("--max-retries", type=int, default=3)
    teacher_stage_a.add_argument("--variants", type=int, default=3)
    teacher_stage_a.add_argument("--near-duplicate-ratio", type=float, default=0.82)

    teacher_stage_b = sub.add_parser("run-teacher-stage-b")
    teacher_stage_b.add_argument("--candidates", required=True)
    teacher_stage_b.add_argument("--v0-input", required=True)
    teacher_stage_b.add_argument("--responses", required=True)
    teacher_stage_b.add_argument("--accepted", required=True)
    teacher_stage_b.add_argument("--rejected", required=True)
    teacher_stage_b.add_argument("--stats", required=True)
    teacher_stage_b.add_argument("--cache-dir", required=True)
    teacher_stage_b.add_argument("--model")
    teacher_stage_b.add_argument("--base-url")
    teacher_stage_b.add_argument("--workers", type=int, default=16)
    teacher_stage_b.add_argument("--timeout", type=float, default=300.0)
    teacher_stage_b.add_argument("--max-retries", type=int, default=3)
    teacher_stage_b.add_argument("--limit", type=int)

    export = sub.add_parser("export-training")
    export.add_argument("--input", required=True)
    export.add_argument("--output-dir", required=True)
    export.add_argument("--validation-fraction", type=float, default=0.02)
    export.add_argument("--seed", type=int, default=42)
    export.add_argument("--max-subtasks", type=int, default=10)
    export.add_argument("--executor-variants-per-source", type=int, default=0)
    export.add_argument("--require-target-review", action="store_true")
    export.add_argument(
        "--no-terminal-answer-marker",
        action="store_true",
        help="Preserve raw teacher terminal solutions instead of enforcing the inference format.",
    )

    audit = sub.add_parser("sample-audit")
    audit.add_argument("--input", required=True)
    audit.add_argument("--output", required=True)
    audit.add_argument("--count", type=int, default=200)
    audit.add_argument("--seed", type=int, default=42)

    train = sub.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--set", action="append", default=[])

    infer = sub.add_parser("evaluate")
    infer.add_argument("--config", required=True)
    infer.add_argument("--input", required=True)
    infer.add_argument("--output", required=True)
    infer.add_argument("--mode", choices=("keir", "cot"), default="keir")
    infer.add_argument("--samples", type=int, default=1)
    infer.add_argument("--no-consolidation", action="store_true")
    infer.add_argument("--limit", type=int)
    infer.add_argument("--seed", type=int, default=42)
    infer.add_argument("--training-seed", type=int, help="Training seed, required for aggregation if adapter metadata is absent")
    infer.add_argument(
        "--resume",
        action="store_true",
        help="Resume a contiguous, validated .partial.jsonl for the same output path.",
    )
    infer.add_argument("--set", action="append", default=[])

    metrics = sub.add_parser("score")
    metrics.add_argument("--input", required=True)
    metrics.add_argument("--output")

    cost = sub.add_parser("estimate-cost", help="Analytical LM cost from explicitly supplied token counts")
    cost.add_argument("--nominal-parameters", type=int, required=True)
    cost.add_argument("--prefill-tokens", type=float, required=True)
    cost.add_argument("--generated-tokens", type=float, required=True)
    cost.add_argument("--prefill-accounting", choices=("reported", "processed", "exact_full_prompt_once"), required=True)
    cost.add_argument("--output")

    significance = sub.add_parser("mcnemar")
    significance.add_argument("--first", required=True)
    significance.add_argument("--second", required=True)
    significance.add_argument("--output")

    aggregate = sub.add_parser("aggregate-seeds")
    aggregate.add_argument("metrics", nargs="+")
    aggregate.add_argument("--output")
    macro = sub.add_parser("aggregate-paper")
    macro.add_argument("metrics", nargs="+")
    macro.add_argument("--output")
    review_queue = sub.add_parser("review-targets")
    review_queue.add_argument("--input", required=True)
    review_queue.add_argument("--output", required=True)
    review_apply = sub.add_parser("apply-target-reviews")
    review_apply.add_argument("--input", required=True)
    review_apply.add_argument("--reviews", required=True)
    review_apply.add_argument("--output", required=True)

    download.add_argument("--revision", default="main")
    benchmark.add_argument("--revision", default="main")
    benchmark.add_argument("--trust-remote-code", action="store_true")
    merge = sub.add_parser("merge-corpus")
    merge.add_argument("--originals", required=True)
    merge.add_argument("--variants", required=True)
    merge.add_argument("--output", required=True)
    measure = sub.add_parser("measure")
    measure.add_argument("--config", required=True)
    measure.add_argument("--input", required=True)
    measure.add_argument("--output", required=True)
    measure.add_argument("--mode", choices=("keir", "cot"), default="keir")
    measure.add_argument("--samples", type=int, default=1)
    measure.add_argument("--limit", type=int)
    measure.add_argument("--seed", type=int, default=42)
    measure.add_argument("--warmup", type=int, default=10)
    measure.add_argument("--repetitions", type=int, default=3)
    measure.add_argument("--set", action="append", default=[])
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "download-gsm8k":
        from .data import download_gsm8k

        _print({"written": download_gsm8k(args.output_dir, args.split, args.config, args.revision)})
    elif args.command == "download-benchmark":
        from .benchmarks import download_benchmark

        _print({"written": download_benchmark(args.benchmark, args.output, args.revision, args.trust_remote_code)})
    elif args.command == "normalize-benchmark":
        from .benchmarks import normalize_local_benchmark

        _print({"written": normalize_local_benchmark(args.benchmark, args.input, args.output,
                args.source_url, args.source_revision, args.source_split)})
    elif args.command == "validate-benchmark":
        from .benchmarks import validate_benchmark_file

        result = validate_benchmark_file(
            args.benchmark, args.input, strict_size=not args.allow_size_mismatch
        )
        _print(result)
        if not result["valid"]:
            raise SystemExit(1)
    elif args.command == "prepare-socratic-v0":
        from .socratic import build_official_socratic_v0

        _print(
            build_official_socratic_v0(
                socratic_path=args.socratic_input,
                main_path=args.main_input,
                output_path=args.output,
                rejected_path=args.rejected,
                stats_path=args.stats,
                split=args.split,
            )
        )
    elif args.command == "plan-gpt4o-augmentation":
        from .augmentation import plan_gpt4o_augmentation

        _print(
            plan_gpt4o_augmentation(
                v0_path=args.input,
                requests_path=args.requests,
                manifest_path=args.manifest,
                limit=args.limit,
                variants=args.variants,
                model=args.model,
                stage_b_model=args.stage_b_model,
                near_duplicate_ratio=args.near_duplicate_ratio,
            )
        )
    elif args.command == "export-training":
        from .data import export_training_data

        _print(
            export_training_data(
                args.input,
                args.output_dir,
                validation_fraction=args.validation_fraction,
                seed=args.seed,
                max_subtasks=args.max_subtasks,
                executor_variants_per_source=args.executor_variants_per_source,
                terminal_answer_marker=not args.no_terminal_answer_marker,
                require_target_review=args.require_target_review,
            )
        )
    elif args.command == "run-teacher-stage-a":
        from .teacher_stage_a import run_teacher_stage_a

        _print(
            run_teacher_stage_a(
                requests_path=args.requests,
                v0_path=args.v0_input,
                responses_path=args.responses,
                candidates_path=args.candidates,
                stats_path=args.stats,
                cache_dir=args.cache_dir,
                model=args.model,
                base_url=args.base_url,
                workers=args.workers,
                timeout=args.timeout,
                max_retries=args.max_retries,
                variants=args.variants,
                near_duplicate_ratio=args.near_duplicate_ratio,
            )
        )
    elif args.command == "sample-audit":
        from .data import sample_manual_audit

        _print({"written": sample_manual_audit(args.input, args.output, args.count, args.seed)})
    elif args.command == "run-teacher-stage-b":
        from .teacher_stage_b import run_teacher_stage_b

        _print(
            run_teacher_stage_b(
                candidates_path=args.candidates,
                v0_path=args.v0_input,
                responses_path=args.responses,
                accepted_path=args.accepted,
                rejected_path=args.rejected,
                stats_path=args.stats,
                cache_dir=args.cache_dir,
                model=args.model,
                base_url=args.base_url,
                workers=args.workers,
                timeout=args.timeout,
                max_retries=args.max_retries,
                limit=args.limit,
            )
        )
    elif args.command == "train":
        from .config import apply_overrides, load_config
        from .training import train_keir

        config = apply_overrides(load_config(args.config), args.set)
        _print(train_keir(config["training"]))
    elif args.command == "evaluate":
        from .config import apply_overrides, load_config
        from .evaluation import run_benchmark

        config = apply_overrides(load_config(args.config), args.set)


        if args.no_consolidation:
            config.setdefault("inference", {})["consolidation"] = False
        reasoner = _runtime_from_config(config, args.mode)
        from .io import stable_hash
        training_settings = {
            role: {k: v for k, v in values.items() if k not in {"seed", "output_dir"}}
            if isinstance(values, dict) else values
            for role, values in config.get("training", {}).items()
        }
        model_settings = {k: v for k, v in config["model"].items() if not k.endswith("_adapter")}
        configuration_id = stable_hash({"model": model_settings, "training": training_settings,
                                       "inference": config.get("inference", {}), "mode": args.mode,
                                       "samples": args.samples, "no_consolidation": args.no_consolidation})
        _print(
            run_benchmark(
                reasoner,
                args.input,
                args.output,
                mode=args.mode,
                samples=args.samples,
                consolidate=not args.no_consolidation,
                limit=args.limit,
                seed=args.seed,
                resume=args.resume,
                training_seed=args.training_seed,
                configuration_id=configuration_id,
            )
        )
    elif args.command == "score":
        from .evaluation import evaluate_prediction_file

        _print(evaluate_prediction_file(args.input, args.output))
    elif args.command == "estimate-cost":
        from .costs import estimate_lm_cost
        from .io import write_json
        result = estimate_lm_cost(args.nominal_parameters, args.prefill_tokens,
                                  args.generated_tokens, args.prefill_accounting)
        if args.output:
            write_json(args.output, result)
        _print(result)
    elif args.command == "mcnemar":
        from .evaluation import mcnemar_files

        _print(mcnemar_files(args.first, args.second, args.output))
    elif args.command == "aggregate-seeds":
        from .evaluation import aggregate_seeds

        _print(aggregate_seeds(args.metrics, args.output))
    elif args.command == "aggregate-paper":
        from .evaluation import aggregate_paper
        _print(aggregate_paper(args.metrics, args.output))
    elif args.command == "review-targets":
        from .target_review import write_target_review_queue
        _print({"written": write_target_review_queue(args.input, args.output)})
    elif args.command == "apply-target-reviews":
        from .target_review import apply_target_reviews
        _print(apply_target_reviews(args.input, args.reviews, args.output))
    elif args.command == "merge-corpus":
        from .data import merge_corpus
        _print(merge_corpus(args.originals, args.variants, args.output))
    elif args.command == "measure":
        from .config import apply_overrides, load_config
        from .profiling import measure
        config = apply_overrides(load_config(args.config), args.set)
        reasoner = _runtime_from_config(config, args.mode)
        _print(measure(reasoner, args.input, args.output, args.warmup, args.repetitions,
                       args.limit, args.seed, args.mode, args.samples))


if __name__ == "__main__":
    main()
