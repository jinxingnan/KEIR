import random
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence, Tuple

from .answers import extract_answer
from .io import read_jsonl, write_json, write_jsonl
from .prompts import decomposition_prompt, execution_prompt, format_subtasks, cot_prompt
from .schemas import OriginalExample, RefinedExample


def split_statement_question(problem: str, strict: bool = False) -> Tuple[str, str]:


    text = re.sub(r"\s+", " ", str(problem)).strip()
    if not text:
        if strict:
            raise ValueError("ambiguous_statement_question_split: empty problem")
        return "", ""
    abbreviations = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs",
                     "e.g", "i.e", "a.m", "p.m", "u.s", "u.k", "no"}
    sentences, start = [], 0
    for match in re.finditer(r"[.!?]+(?=\s|$)", text):
        if match.group() == "." and match.end() < len(text):
            token = re.search(r"([A-Za-z.]+)$", text[:match.start()])
            word = token.group(1) if token else ""
            next_target = re.match(r"\s+(?:how|what|which|calculate|compute|determine|find|evaluate|solve)\b",
                                   text[match.end():], re.I)
            title = word.lower() in {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st"}
            if ((word.lower() in abbreviations and (title or not next_target))
                    or (len(word) == 1 and word.isupper())):
                continue
        sentences.append((start, text[start:match.end()].strip()))
        start = match.end()
    if text[start:].strip():
        sentences.append((start, text[start:].strip()))
    questions = [sentence for _, sentence in sentences if "?" in sentence]
    if sentences:
        offset, terminal = sentences[-1]
        explicit_target = terminal.endswith("?") or bool(re.match(
            r"(?:calculate|compute|determine|find|evaluate|solve|give|state)\b", terminal, re.I))
        if explicit_target and len(questions) == int(terminal.endswith("?")):


            return text[:offset].strip() or text, terminal
    if strict:
        raise ValueError("ambiguous_statement_question_split: review the target sentence")
    return text, "What is the final answer?"


def parse_gsm_answer(raw_answer: str) -> Tuple[str, str]:
    raw = str(raw_answer).strip()
    if "####" in raw:
        rationale, answer = raw.rsplit("####", 1)
        return rationale.strip(), answer.strip()
    return "", extract_answer(raw)


def download_gsm8k(
    output_dir: str,
    split: str = "train",
    config: str = "main",
    revision: str = "main",
) -> int:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required to download GSM8K") from exc
    if config not in {"main", "socratic"}:
        raise ValueError("GSM8K config must be main or socratic")
    from huggingface_hub import HfApi
    from hashlib import sha256
    resolved = HfApi().dataset_info("openai/gsm8k", revision=revision).sha
    dataset = load_dataset("openai/gsm8k", config, split=split, revision=resolved)
    rows = []
    for record in dataset:

        rows.append(
            {"question": str(record["question"]), "answer": str(record["answer"])}
        )
    filename = "gsm8k_%s_%s.jsonl" % (config, split)
    target = str(Path(output_dir) / filename)
    count = write_jsonl(target, rows)
    write_json(target + ".manifest.json", {
        "source": "openai/gsm8k", "config": config, "split": split,
        "resolved_revision": resolved, "count": count,
        "sha256": sha256(Path(target).read_bytes()).hexdigest(),
    })
    return count


def load_originals(path: str) -> Iterator[OriginalExample]:
    for row in read_jsonl(path):
        yield OriginalExample.from_dict(row)


def load_refined(path: str) -> Iterator[RefinedExample]:
    for row in read_jsonl(path):
        yield RefinedExample.from_dict(row)


def _text_key(value: str) -> str:
    return re.sub(r"\W+", " ", value.lower()).strip()


def _number_signature(value: str) -> Tuple[str, ...]:
    numeric = [
        "digit:" + item
        for item in re.findall(
            r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?", value
        )
    ]
    number_words = {
        "zero", "one", "two", "three", "four", "five", "six", "seven",
        "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
        "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
        "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
        "hundred", "thousand", "million", "billion", "trillion", "half",
        "halves", "quarter", "quarters", "third", "thirds", "fourth", "fourths",
        "fifth", "fifths", "sixth", "sixths", "seventh", "sevenths", "eighth",
        "eighths", "ninth", "ninths", "tenth", "tenths", "dozen", "dozens",
        "first", "second", "eleventh", "twelfth", "single", "both", "pair",
        "couple", "trio", "quartet", "quintet", "sextet", "septet", "octet",
        "nonet", "once", "double", "twice", "triple", "thrice",
    }
    words = [
        "word:" + item
        for item in re.findall(r"[a-z]+", value.lower())
        if item in number_words
    ]
    return tuple(numeric + words)


def build_training_records(
    examples: Sequence[RefinedExample],
    max_subtasks: int = 10,
    terminal_answer_marker: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    decomposition = []
    execution = []
    for example in examples:
        selected_chain = list(example.chain)
        if not 1 <= len(selected_chain) <= max_subtasks:
            raise ValueError("Chain must contain 1..%d steps; refusing to remove dependencies: %s"
                             % (max_subtasks, example.id))
        subtasks = [step.task for step in selected_chain]
        solutions = [step.solution for step in selected_chain]
        common = {
            "example_id": example.id,
            "source_id": example.source_id,
            "variant_index": example.variant_index,
        }
        decomposition.append(
            {
                "id": example.id,
                "prompt": decomposition_prompt(
                    example.statement, example.question, max_subtasks=max_subtasks
                ),
                "response": format_subtasks(subtasks),
                "metadata": common,
            }
        )
        for index, (subtask, solution) in enumerate(zip(subtasks, solutions)):
            is_terminal = index == len(subtasks) - 1
            response = solution
            if (
                is_terminal
                and terminal_answer_marker
                and not re.search(r"(?:final\s+answer|answer)\s*(?:is|:|=)", solution, re.I)
            ):

                response = "%s\nFinal answer: %s" % (solution.rstrip(), example.answer)
            execution.append(
                {
                    "id": "%s-step-%02d" % (example.id, index + 1),
                    "prompt": execution_prompt(
                        statement=example.statement,
                        question=example.question,
                        current_subtask=subtask,
                        previous_subtasks=subtasks[:index],
                        previous_solutions=solutions[:index],
                        is_terminal=is_terminal,
                    ),
                    "response": response,
                    "metadata": dict(common, step=index + 1, terminal=is_terminal),
                }
            )
    return decomposition, execution


def select_executor_variants(
    examples: Sequence[RefinedExample],
    max_variants_per_source: int,
) -> List[RefinedExample]:


    if max_variants_per_source <= 0:
        return list(examples)
    grouped: Dict[str, List[RefinedExample]] = {}
    for item in examples:
        grouped.setdefault(item.source_id, []).append(item)
    retained_ids = set()
    for items in grouped.values():
        ordered = sorted(
            items,
            key=lambda item: (not item.is_original, item.variant_index, item.id),
        )
        retained_ids.update(item.id for item in ordered[:max_variants_per_source])
    return [item for item in examples if item.id in retained_ids]


def split_by_source(
    examples: Sequence[RefinedExample],
    validation_fraction: float = 0.02,
    seed: int = 42,
) -> Tuple[List[RefinedExample], List[RefinedExample]]:

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must lie between 0 and 1")
    source_ids = sorted({item.source_id for item in examples})
    if len(source_ids) < 2:
        raise ValueError("At least two source problems are required for disjoint train/validation sets")
    random.Random(seed).shuffle(source_ids)
    validation_count = min(len(source_ids) - 1, max(1, round(len(source_ids) * validation_fraction)))
    validation_ids = set(source_ids[:validation_count])
    train = [item for item in examples if item.source_id not in validation_ids]
    validation = [item for item in examples if item.source_id in validation_ids]
    return train, validation


def export_training_data(
    refined_path: str,
    output_dir: str,
    validation_fraction: float = 0.02,
    seed: int = 42,
    max_subtasks: int = 10,
    executor_variants_per_source: int = 0,
    terminal_answer_marker: bool = True,
    require_target_review: bool = False,
) -> Dict[str, int]:
    from .target_review import has_current_approval
    examples = list(load_refined(refined_path))
    pending = [item.id for item in examples if not item.is_original
               and not has_current_approval(item.to_dict())]
    if require_target_review and pending:
        raise ValueError("Unreviewed variant targets remain: %d; apply target reviews before export" % len(pending))
    train, validation = split_by_source(examples, validation_fraction, seed)
    output = Path(output_dir)
    counts = {"variants_without_human_target_approval": len(pending)}
    for split_name, split_examples in (("train", train), ("validation", validation)):
        decomposer, _ = build_training_records(
            split_examples,
            max_subtasks=max_subtasks,
            terminal_answer_marker=terminal_answer_marker,
        )
        executor_examples = select_executor_variants(
            split_examples, executor_variants_per_source
        )
        _, executor = build_training_records(
            executor_examples,
            max_subtasks=max_subtasks,
            terminal_answer_marker=terminal_answer_marker,
        )
        counts["decomposer_%s" % split_name] = write_jsonl(
            str(output / ("decomposer_%s.jsonl" % split_name)), decomposer
        )
        counts["executor_%s" % split_name] = write_jsonl(
            str(output / ("executor_%s.jsonl" % split_name)), executor
        )
        cot = [{"id": item.id, "prompt": cot_prompt(item.statement, item.question),
                "response": "\n".join(step.solution for step in item.chain) + "\nFinal answer: " + item.answer,
                "metadata": {"source_id": item.source_id, "variant_index": item.variant_index}}
               for item in split_examples]
        counts["cot_%s" % split_name] = write_jsonl(str(output / ("cot_%s.jsonl" % split_name)), cot)
    write_json(str(output / "manifest.json"), counts)
    return counts


def merge_corpus(original_path: str, variant_path: str, output_path: str):
    originals = list(load_refined(original_path))
    variants = list(load_refined(variant_path))
    by_source = {item.source_id: item for item in originals}
    if len(by_source) != len(originals) or any(not item.is_original for item in originals):
        raise ValueError("Expected one original record per source")
    for item in variants:
        source = by_source.get(item.source_id)
        if source is None or item.is_original or item.variant_index < 1:
            raise ValueError("Variant has no corresponding original: " + item.id)
        if item.statement != source.statement or item.answer != source.answer:
            raise ValueError("Variant changes the statement or gold answer: " + item.id)
    rows = originals + variants
    if len({item.id for item in rows}) != len(rows):
        raise ValueError("Duplicate record IDs")
    count = write_jsonl(output_path, (item.to_dict() for item in rows))
    return {"originals": len(originals), "variants": len(variants), "total": count}


def sample_manual_audit(
    refined_path: str,
    output_path: str,
    count: int = 200,
    seed: int = 42,
) -> int:
    examples = list(load_refined(refined_path))
    sample = random.Random(seed).sample(examples, min(count, len(examples)))
    rows = []
    for item in sample:
        row = item.to_dict()
        row["audit"] = {
            "statement_preserved": None,
            "target_preserved": None,
            "answer_verified": None,
            "subquestions_natural": None,
            "subquestions_necessary": None,
            "answers_local": None,
            "dependency_ordered": None,
            "terminal_target_matches": None,
            "chain_executable": None,
            "notes": "",
        }
        rows.append(row)
    return write_jsonl(output_path, rows)
