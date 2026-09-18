import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Optional


_NUMBER = r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:[eE][-+]?\d+)?%?"
_FINAL_PATTERNS = [
    re.compile(r"####\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(
        r"(?:final\s+answer|answer)\s*(?:is|:|=)\s*(.+?)\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
]


SCORING_PROTOCOL = "keir-normalized-em-v2"
_FINAL_MARKER = re.compile(r"(?:####|\b(?:final\s+answer|answer)\s*(?:is\b|:|=))[ \t]*([^\n]*)", re.I)
_DECIMAL = r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_SCALAR = r"(?:" + _DECIMAL + r"|\(\s*" + _DECIMAL + r"\s*\))"
_NUMERIC = re.compile(r"(?P<number>" + _SCALAR + r"(?:\s*/\s*" + _SCALAR + r")?)(?P<percent>\s*%)?")


_UNIT = re.compile(r"(?:dollars?|cents?|euros?|pounds?|kg|g|kilograms?|grams?|"
                   r"miles?|km|kilometers?|meters?|metres?|cm|centimeters?|feet|foot|inches?|yards?|"
                   r"hours?|minutes?|seconds?|days?|weeks?|months?|years?|"
                   r"liters?|litres?|gallons?|cups?|degrees?|percent|apples?|oranges?|"
                   r"books?|pages?|people|persons?|students?|children|items?|pieces?|units?)", re.I)


def numeric_answer(value: str) -> str:

    text = str(value).strip().replace("−", "-").replace("–", "-")
    if text.startswith("$") and text.endswith("$") and len(text) > 1:
        text = text[1:-1].strip()
    elif text.startswith("$"):
        text = text[1:].strip()
    if text.startswith(r"\(") and text.endswith(r"\)"):
        text = text[2:-2].strip()
    text = text.replace(r"\%", "%").replace(r"\$", "")
    text = _latex_fraction(text).strip()
    if text.endswith(".."):
        return ""
    text = text.removesuffix(".").strip()
    match = _NUMERIC.match(text)
    if match is None:
        return ""
    suffix = text[match.end():]
    percent_unit = False
    if suffix:
        if not suffix[0].isspace():
            return ""
        unit = suffix.strip()
        if unit.startswith("(") and unit.endswith(")"):
            unit = unit[1:-1].strip()
        if not _UNIT.fullmatch(unit):
            return ""
        percent_unit = unit.lower() == "percent"
    number = match["number"].replace(",", "").replace("(", "").replace(")", "").replace(" ", "")

    if len(number) > 256 or any(abs(int(e)) > 1000 for e in re.findall(r"[eE]([-+]?\d+)", number)):
        return ""
    try:
        pieces = number.split("/")
        parsed = Fraction(pieces[0])
        if len(pieces) == 2:
            parsed /= Fraction(pieces[1])
    except (ValueError, ZeroDivisionError, OverflowError):
        return ""
    return match["number"].strip() + ("%" if match["percent"] or percent_unit else "")


def _last_boxed(text: str) -> Optional[str]:

    markers = [r"\boxed{", r"\fbox{"]
    candidates = []
    for marker in markers:
        start = 0
        while True:
            index = text.find(marker, start)
            if index < 0:
                break
            content_start = index + len(marker)
            depth = 1
            cursor = content_start
            while cursor < len(text) and depth:
                if text[cursor] == "{":
                    depth += 1
                elif text[cursor] == "}":
                    depth -= 1
                cursor += 1
            if depth == 0:
                candidates.append((index, text[content_start : cursor - 1]))
            start = content_start
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def extract_answer(text: str, dataset: str = "") -> str:

    if text is None:
        return ""
    value = str(text).strip()
    boxed = _last_boxed(value)
    if boxed is not None:
        return boxed.strip()
    for pattern in _FINAL_PATTERNS:
        matches = pattern.findall(value)
        if matches:
            return str(matches[-1]).strip().rstrip(".")


    if dataset.lower() in ("gsm8k", "math"):
        return value
    complete = numeric_answer(value)
    if complete:
        return complete
    numbers = re.findall(_NUMBER, value)
    if numbers:
        return numbers[-1].strip()
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def extract_benchmark_prediction(text: str, dataset: str) -> str:


    if text is None:
        return ""
    value = str(text).strip()
    name = dataset.lower()
    if name == "math":
        boxed = _last_boxed(value)
        if boxed is not None:
            return boxed.strip()
        for pattern in _FINAL_PATTERNS:
            matches = pattern.findall(value)
            if matches:
                return str(matches[-1]).strip().rstrip(".")
        return ""
    markers = list(_FINAL_MARKER.finditer(value))
    if markers:

        return numeric_answer(markers[-1].group(1))
    boxed = _last_boxed(value)
    if boxed is not None:
        return numeric_answer(boxed)


    return "" if name == "gsm8k" else numeric_answer(value)


def _latex_fraction(value: str) -> str:
    pattern = re.compile(r"\\(?:d?frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
    previous = None
    while value != previous:
        previous = value
        value = pattern.sub(r"(\1)/(\2)", value)
    return value


def _canonical_number(value: str) -> Optional[str]:
    cleaned = value.replace(",", "").strip()
    try:
        if "/" in cleaned:
            left, right = cleaned.replace("(", "").replace(")", "").replace(" ", "").split("/")
            fraction = Fraction(left) / Fraction(right)
            return "%d/%d" % (fraction.numerator, fraction.denominator)
        number = Decimal(cleaned)
        if not number.is_finite():
            return None
        if number == number.to_integral():
            return str(int(number))
        normalized = format(number, "f")
        return normalized.rstrip("0").rstrip(".")
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


def normalize_answer(answer: str) -> str:

    if answer is None:
        return ""
    value = str(answer).strip()
    value = value.replace("−", "-").replace("–", "-")
    value = value.replace(r"\$", "").replace("$", "")
    value = value.replace(r"\%", "%")
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = value.replace(r"\,", "").replace(r"\!", "")
    value = value.replace(" ", " ")
    value = _latex_fraction(value)
    value = re.sub(r"^\s*(?:the\s+)?(?:final\s+)?answer\s*(?:is|:|=)?\s*", "", value, flags=re.I)
    value = value.strip().strip("`").strip()
    value = re.sub(r"[.!?,;:]+\s*$", "", value)
    value = re.sub(r"\s+", "", value)
    if value.endswith("%"):
        numeric = _canonical_number(value[:-1])
        return (numeric + "%") if numeric is not None else value.lower()
    numeric = _canonical_number(value)
    if numeric is not None:
        return numeric

    value = value.replace(r"\text{", "").replace("}", "") if value.startswith(r"\text{") else value
    return value.lower()


def _math_fix_fracs(value: str) -> str:
    parts = value.split(r"\frac")
    rebuilt = parts[0]
    for part in parts[1:]:
        rebuilt += r"\frac"
        if not part:
            return value
        if part[0] == "{":
            rebuilt += part
            continue
        if len(part) < 2:
            return value
        numerator, denominator = part[0], part[1]
        suffix = part[2:]
        if denominator != "{":
            rebuilt += "{%s}{%s}%s" % (numerator, denominator, suffix)
        else:
            rebuilt += "{%s}%s%s" % (numerator, denominator, suffix)
    return rebuilt


def _math_fix_slash_fraction(value: str) -> str:
    parts = value.split("/")
    if len(parts) != 2:
        return value
    try:
        numerator, denominator = int(parts[0]), int(parts[1])
    except ValueError:
        return value
    if value != "%d/%d" % (numerator, denominator):
        return value
    return r"\frac{%d}{%d}" % (numerator, denominator)


def _math_remove_right_units(value: str) -> str:
    if r"\text{ " not in value:
        return value
    parts = value.split(r"\text{ ")
    return parts[0] if len(parts) == 2 else value


def _math_fix_sqrt(value: str) -> str:
    if r"\sqrt" not in value:
        return value
    parts = value.split(r"\sqrt")
    rebuilt = parts[0]
    for part in parts[1:]:
        if not part:
            return value
        rebuilt += (
            r"\sqrt" + part
            if part[0] == "{"
            else r"\sqrt{" + part[0] + "}" + part[1:]
        )
    return rebuilt


def normalize_math_answer(answer: str) -> str:

    value = str(answer).replace("\n", "").replace(r"\!", "")
    value = value.replace(r"\\", "\\")
    value = value.replace("tfrac", "frac").replace("dfrac", "frac")
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = value.replace(r"^{\circ}", "").replace(r"^\circ", "")
    value = value.replace(r"\$", "")
    value = _math_remove_right_units(value)
    value = value.replace(r"\%", "").replace(r"\%", "")
    value = value.replace(" .", " 0.").replace("{.", "{0.")
    if not value:
        return value
    if value[0] == ".":
        value = "0" + value
    if len(value.split("=")) == 2 and len(value.split("=")[0]) <= 2:
        value = value.split("=")[1]
    value = _math_fix_sqrt(value)
    value = value.replace(" ", "")
    value = _math_fix_fracs(value)
    if value == "0.5":
        value = r"\frac{1}{2}"
    return _math_fix_slash_fraction(value)


def normalize_benchmark_prediction(text: str, dataset: str) -> str:

    extracted = extract_benchmark_prediction(text, dataset)
    if not extracted:
        return ""
    normalizer = normalize_math_answer if dataset.lower() == "math" else normalize_answer
    return normalizer(extracted)


def gsm8k_official_exact_match(prediction: str, gold: str) -> bool:

    extracted = extract_benchmark_prediction(prediction, "gsm8k")
    if not extracted:
        return False
    target = extract_answer(gold, "gsm8k")
    return extracted.replace(",", "") == target.replace(",", "")


def answers_equal(prediction: str, gold: str, dataset: str = "") -> bool:
    name = dataset.lower()
    if name == "math":
        raw_pred, raw_target = extract_answer(prediction, name), extract_answer(gold, name)
        if not raw_pred or not raw_target:
            return False
        try:
            return normalize_math_answer(raw_pred) == normalize_math_answer(raw_target)
        except (AssertionError, IndexError, ValueError):
            return raw_pred == raw_target

    raw_pred = numeric_answer(prediction) or extract_benchmark_prediction(prediction, name)
    raw_target = numeric_answer(gold) or extract_benchmark_prediction(gold, name)
    if not raw_pred or not raw_target:
        return False
    pred, target = normalize_answer(raw_pred), normalize_answer(raw_target)
    if pred == target:
        return True


    if name == "gsm8k" and pred.endswith("%") != target.endswith("%"):
        left = pred[:-1] if pred.endswith("%") else pred
        right = target[:-1] if target.endswith("%") else target
        try:
            return Fraction(left) == Fraction(right)
        except (ValueError, ZeroDivisionError):
            pass

    try:
        left = Fraction(pred)
        right = Fraction(target)
        return left == right
    except (ValueError, ZeroDivisionError):
        return False


def majority_answer(traces, dataset: str) -> str:

    answers = [normalize_benchmark_prediction(text, dataset) for text in traces]
    valid = [answer for answer in answers if answer]
    if dataset.lower() == "math":
        keys = valid
    else:
        keys = []
        for answer in valid:
            percent = answer.endswith("%")
            number = answer[:-1] if percent else answer
            key = str(Fraction(number))
            keys.append(key + ("%" if percent and dataset.lower() != "gsm8k" else ""))
    counts = Counter(keys)
    if not counts:
        return ""
    winner = max(counts, key=lambda key: (counts[key], -keys.index(key)))
    return valid[keys.index(winner)]
