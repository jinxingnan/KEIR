from __future__ import annotations

import ast
import hashlib
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .answers import answers_equal, normalize_answer
from .data import parse_gsm_answer, split_statement_question
from .io import read_jsonl, write_json, write_jsonl
from .schemas import RefinedExample, SubtaskSolution


class SocraticValidationError(ValueError):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__("%s%s" % (code, (": " + detail) if detail else ""))


def _safe_fraction(expression: str) -> Fraction:
    tree = ast.parse(
        expression.replace(",", "").replace("$", "").strip(),
        mode="eval",
    )

    def evaluate(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return Fraction(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp):
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.FloorDiv):
                return Fraction(left // right)
            if isinstance(node.op, ast.Pow) and right.denominator == 1:
                return left**right.numerator
        raise SocraticValidationError("unsupported_calculation", ast.dump(node))

    return evaluate(tree)


def _annotation_values(text: str) -> List[str]:
    values = []
    for payload in re.findall(r"<<([^<>]+)>>", text):
        if "=" not in payload:
            raise SocraticValidationError("malformed_calculation", payload)
        expression, claimed = payload.rsplit("=", 1)
        try:
            actual = _safe_fraction(expression)
            expected = _safe_fraction(claimed)
        except (SyntaxError, ZeroDivisionError, ValueError) as exc:
            if isinstance(exc, SocraticValidationError):
                raise
            raise SocraticValidationError("unsupported_calculation", payload) from exc
        if actual != expected:
            raise SocraticValidationError(
                "incorrect_calculation",
                "%s (computed %s, claimed %s)" % (payload, actual, expected),
            )
        values.append(claimed.strip())
    return values


_NUMBER_PATTERN = r"[-+]?(?:\d{1,3}(?:[ ,]\d{3})+|\d+(?:\.\d+)?|\.\d+)%?"


def _numeric_candidates(text: str) -> List[str]:

    return [value.replace(" ", "") for value in re.findall(_NUMBER_PATTERN, text)]


def _validate_unannotated_calculations(text: str) -> None:


    if "<<" in text or ">>" in text:
        return
    if text.count("=") > 1:


        return
    plain = text
    equation = re.compile(
        r"(?P<expr>(?:\$?(?:\d[\d,]*(?:\.\d+)?|\.\d+)\s*"
        r"(?:[+\-*/xX×]\s*))+\$?(?:\d[\d,]*(?:\.\d+)?|\.\d+))"
        r"\s*=\s*\$?(?P<claimed>[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)"
        r"(?:\s*/\s*(?:\d[\d,]*(?:\.\d+)?|\.\d+))?)"
    )
    for match in equation.finditer(plain):
        prefix = plain[max(0, match.start() - 12) : match.start()]
        if re.search(r"(?:\d\s+|\d\s+and\s+)$", prefix):

            continue
        suffix = plain[match.end() : match.end() + 12].lower()
        if "%" in suffix or "percent" in suffix:
            continue
        raw_operands = re.findall(r"\d+", match.group("expr"))
        if any(len(value) > 1 and value.startswith("0") for value in raw_operands):

            continue
        expression = re.sub(r"[xX×]", "*", match.group("expr"))
        claimed = match.group("claimed")
        try:
            divided_by_fraction = re.fullmatch(
                r"(.+?)\s*/\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)",
                expression,
            )
            if divided_by_fraction:
                numerator = _safe_fraction(divided_by_fraction.group(2))
                denominator = _safe_fraction(divided_by_fraction.group(3))
                actual = _safe_fraction(divided_by_fraction.group(1)) / (
                    numerator / denominator
                )
            else:
                actual = _safe_fraction(expression)
            expected = _safe_fraction(claimed)
        except (SyntaxError, ZeroDivisionError, ValueError) as exc:
            if isinstance(exc, SocraticValidationError):
                raise
            raise SocraticValidationError(
                "unsupported_calculation", match.group(0)
            ) from exc
        cent_scaled = (
            actual * 100 == expected
            and "$" in match.group("expr")
            and "cent" in plain[match.end() :].lower()
        )
        if actual != expected and not cent_scaled:
            raise SocraticValidationError(
                "incorrect_unannotated_calculation",
                "%s (computed %s, claimed %s)"
                % (match.group(0), actual, expected),
            )


def _candidate_matches(candidate: str, gold: str) -> bool:
    return bool(candidate) and answers_equal(candidate, gold, dataset="gsm8k")


def _scaled_unit_match(candidate: str, gold: str, context: str) -> bool:

    try:
        candidate_value = _safe_fraction(candidate.rstrip("%"))
        gold_value = _safe_fraction(normalize_answer(gold).rstrip("%"))
    except (SyntaxError, ZeroDivisionError, ValueError):
        return False
    if candidate_value * 100 != gold_value:
        return False
    lowered = context.lower()
    return ("cent" in lowered and "$" in context) or "%" in context or "percent" in lowered


def step_answer_matches_gold(answer: str, gold: str, context: str = "") -> bool:
    annotations = _annotation_values(answer)
    candidates = _numeric_candidates(answer)
    if not candidates:
        return False
    if not annotations:
        return any(_candidate_matches(candidate, gold) for candidate in candidates)

    final_annotation = annotations[-1]
    if _candidate_matches(final_annotation, gold):
        return True
    if _scaled_unit_match(final_annotation, gold, answer + " " + context):
        return True


    suffix = answer.rsplit(">>", 1)[-1]
    return any(_candidate_matches(candidate, gold) for candidate in _numeric_candidates(suffix))


def parse_official_socratic_answer(
    raw_answer: str, problem_context: str = ""
) -> Tuple[List[SubtaskSolution], str]:
    raw = str(raw_answer).strip()
    if raw.count("####") != 1:
        raise SocraticValidationError("invalid_gold_delimiter", "expected exactly one ####")
    body, gold = raw.rsplit("####", 1)
    gold = gold.strip()
    if not gold or normalize_answer(gold) == "":
        raise SocraticValidationError("empty_gold")
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    if not lines:
        raise SocraticValidationError("empty_chain")
    chain = []
    for position, line in enumerate(lines, 1):
        if line.count(" ** ") != 1:
            raise SocraticValidationError(
                "invalid_step_delimiter",
                "step %d must contain exactly one spaced ** delimiter" % position,
            )
        subquestion, answer = (part.strip() for part in line.split(" ** ", 1))
        if not subquestion:
            raise SocraticValidationError("empty_subquestion", "step %d" % position)
        if not subquestion.endswith("?"):
            raise SocraticValidationError(
                "non_interrogative_subquestion", "step %d: %s" % (position, subquestion)
            )
        if not answer:
            raise SocraticValidationError("empty_step_answer", "step %d" % position)
        _annotation_values(answer)
        chain.append(SubtaskSolution(task=subquestion, solution=answer))
    if len(chain) > 10:
        raise SocraticValidationError("too_many_steps", str(len(chain)))
    _validate_unannotated_calculations(chain[-1].solution)
    terminal_context = chain[-1].task + " " + problem_context
    if not step_answer_matches_gold(chain[-1].solution, gold, terminal_context):
        raise SocraticValidationError(
            "terminal_answer_mismatch",
            "gold=%s; terminal=%s" % (gold, chain[-1].solution),
        )
    return chain, gold


_NON_QUANTITY_FOCUS = {
    "additional", "are", "combined", "did", "do", "does", "different",
    "have", "more", "of", "total", "were", "will", "would",
}
_TEMPORAL_FOCUS = {
    "day", "week", "month", "year", "hour", "minute", "minut", "second",
}


def _how_many_focus(text: str) -> str:
    match = re.search(r"\bhow many\s+([a-z]+)", text.lower())
    if not match:
        return ""
    token = match.group(1)
    if token in _NON_QUANTITY_FOCUS:
        return ""
    for suffix in ("ies", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            return token[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return token


def _target_type(text: str) -> str:
    lowered = text.lower()
    if re.search(r"how many\s+days?\s+(?:a|per)\s+\w+\s+(?:do|does|did)\b", lowered):
        return "count"
    duration = re.search(
        r"(?:after\s+)?how many(?:\s+(?:additional|total|combined))*\s+"
        r"(?:days?|weeks?|months?|years?|hours?|minutes?|seconds?)\b",
        lowered,
    )
    if duration:
        duration_suffix = lowered[duration.end() :]
        after_how_many = lowered[: duration.start()].rstrip().endswith("after")
        if after_how_many or re.search(
            r"will it take|would it take|\bago\b|\buntil\b|\bbefore\b",
            duration_suffix,
        ):
            return "temporal"
    focus = _how_many_focus(text)
    if not focus:
        return ""
    return "temporal" if focus in _TEMPORAL_FOCUS else "count"


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_official_socratic_v0(
    socratic_path: str,
    main_path: str,
    output_path: str,
    rejected_path: str,
    stats_path: str,
    split: str = "train",
) -> Dict[str, Any]:

    socratic_rows = list(read_jsonl(socratic_path))
    main_rows = list(read_jsonl(main_path))
    if len(socratic_rows) != len(main_rows):
        raise ValueError(
            "Official main/Socratic row count mismatch: %d != %d"
            % (len(main_rows), len(socratic_rows))
        )
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    reasons = Counter()
    warnings = Counter()
    for index, (socratic, main) in enumerate(zip(socratic_rows, main_rows)):
        example_id = "gsm8k-%s-%05d-v0" % (split, index)
        try:
            if set(socratic) != {"question", "answer"}:
                raise SocraticValidationError("unexpected_socratic_schema")
            if set(main) != {"question", "answer"}:
                raise SocraticValidationError("unexpected_main_schema")
            full_problem = str(socratic["question"])
            if full_problem != str(main["question"]):
                raise SocraticValidationError("main_socratic_question_mismatch")
            chain, gold = parse_official_socratic_answer(
                str(socratic["answer"]), problem_context=full_problem
            )
            _, main_gold = parse_gsm_answer(str(main["answer"]))
            if not answers_equal(gold, main_gold, dataset="gsm8k"):
                raise SocraticValidationError(
                    "main_socratic_gold_mismatch", "%s != %s" % (main_gold, gold)
                )
            try:
                statement, question = split_statement_question(full_problem, strict=True)
            except ValueError as exc:
                raise SocraticValidationError("ambiguous_statement_question_split", str(exc)) from exc
            question_source = "official_terminal_sentence"
            original_type = _target_type(question)
            terminal_type = _target_type(chain[-1].task)
            if original_type and terminal_type and original_type != terminal_type:
                raise SocraticValidationError(
                    "terminal_target_type_mismatch",
                    "original=%s; terminal=%s" % (original_type, terminal_type),
                )
            keys = [re.sub(r"\W+", " ", step.task.lower()).strip() for step in chain]
            duplicate_count = len(keys) - len(set(keys))
            if duplicate_count:
                warnings["exact_duplicate_subquestions"] += 1
            record = RefinedExample(
                id=example_id,
                source_id="gsm8k-%s-%05d" % (split, index),
                variant_index=0,
                statement=statement,
                question=question,
                answer=gold,
                chain=chain,
                target_quantity=chain[-1].task,
                is_original=True,
                metadata={
                    "dataset": "gsm8k",
                    "config": "socratic",
                    "split": split,
                    "dataset_index": index,
                    "provenance": "openai/grade-school-math",
                    "chain_format": "official_socratic_subquestion_answer",
                    "validation": "strict_v1",
                    "question_source": question_source,
                    "exact_duplicate_subquestions": duplicate_count,
                },
            )
            accepted.append(record.to_dict())
        except SocraticValidationError as exc:
            reasons[exc.code] += 1
            rejected.append(
                {
                    "id": example_id,
                    "dataset_index": index,
                    "reason": exc.code,
                    "detail": exc.detail,
                    "source": socratic,
                }
            )
    write_jsonl(output_path, accepted)
    write_jsonl(rejected_path, rejected)
    stats = {
        "source_instances": len(socratic_rows),
        "accepted_instances": len(accepted),
        "rejected_instances": len(rejected),
        "acceptance_rate": len(accepted) / float(len(socratic_rows) or 1),
        "rejection_reasons": dict(sorted(reasons.items())),
        "warning_instances": dict(sorted(warnings.items())),
        "split": split,
        "socratic_source": socratic_path,
        "socratic_sha256": _sha256(socratic_path),
        "main_source": main_path,
        "main_sha256": _sha256(main_path),
        "output": output_path,
        "rejected_output": rejected_path,
    }
    write_json(stats_path, stats)
    return stats
