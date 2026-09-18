import re
from pathlib import Path

from .io import read_jsonl, stable_hash, write_jsonl

COLORS = set("red blue green yellow black white orange purple pink brown grey gray".split())
ALIASES = {"sphere": "ball", "spheres": "ball", "balls": "ball", "items": "item",
           "objects": "item", "object": "item", "beads": "bead", "marbles": "marble",
           "apples": "apple", "oranges": "orange", "books": "book", "notebooks": "notebook",
           "people": "person", "persons": "person", "students": "student",
           "meters": "meter", "metres": "meter", "metre": "meter", "feet": "foot",
           "inches": "inch", "centimeters": "centimeter", "cm": "centimeter",
           "kilometers": "kilometer", "km": "kilometer", "dollars": "dollar",
           "cents": "cent", "hours": "hour", "minutes": "minute", "seconds": "second",
           "days": "day", "weeks": "week", "kilograms": "kilogram", "grams": "gram",
           "liters": "liter", "litres": "liter", "litre": "liter"}
ENTITIES = set("ball bead marble apple orange book notebook person student".split())
UNITS = set("meter foot inch centimeter kilometer dollar cent hour minute second day week kilogram gram liter".split())


def record_digest(record):
    from .schemas import RefinedExample
    payload = RefinedExample.from_dict(record).to_dict()
    payload["metadata"].pop("target_review", None)
    return stable_hash(payload)


def has_current_approval(record):
    review = record.get("metadata", {}).get("target_review", {})
    return (review.get("status") == "approved" and isinstance(review.get("reviewer"), str)
            and bool(review["reviewer"].strip())
            and review.get("reviewed_record_sha256") == record_digest(record))


def _focus(question):

    match = re.search(r"\b(?:how\s+(?:many|much)|(?:number|count|amount)\s+of)\s+(.+)", question, re.I)
    if not match:
        return set()
    phrase = re.split(r"\b(?:does|do|did|is|are|was|were|will|would|can|could|has|have|had|"
                      r"remain|remains|inside|in|after|before|at|per|until|if|when)\b|[?;]",
                      match.group(1), maxsplit=1, flags=re.I)[0]
    return {ALIASES.get(token, token) for token in re.findall(r"[a-z]+", phrase.lower())}


def check_target(original_question, candidate_question):
    left, right = _focus(original_question), _focus(candidate_question)
    conflicts = []
    for name, vocabulary in (("color", COLORS - {"orange"}), ("entity", ENTITIES), ("unit", UNITS)):
        a, b = left & vocabulary, right & vocabulary
        if len(a) == len(b) == 1 and a != b:
            conflicts.append(name + "_changed")


    return {"status": "conflict" if conflicts else "needs_review",
            "conflicts": conflicts, "original_focus": sorted(left), "candidate_focus": sorted(right),
            "semantic_equivalence_verified": False}


def write_target_review_queue(input_path, output_path):
    if Path(input_path).resolve() == Path(output_path).resolve():
        raise ValueError("Write the review queue to a new path, not over the input corpus")
    rows = []
    for record in read_jsonl(input_path):
        if record.get("is_original"):
            continue
        rows.append({"id": record["id"], "record_sha256": record_digest(record),
                     "statement": record["statement"],
                     "original_question": record.get("metadata", {}).get("original_question", ""),
                     "variant_question": record["question"], "gold": record["answer"],
                     "chain": record["chain"], "decision": "pending", "reviewer": "", "notes": ""})
    return write_jsonl(output_path, rows)


def apply_target_reviews(input_path, reviews_path, output_path):
    outputs = {Path(output_path).resolve(), Path(str(output_path) + ".rejected.jsonl").resolve()}
    if outputs & {Path(input_path).resolve(), Path(reviews_path).resolve()}:
        raise ValueError("Write reviewed records to new paths, not over the input corpus or reviews")
    records = list(read_jsonl(input_path))
    by_id = {str(row["id"]): row for row in records}
    if len(by_id) != len(records):
        raise ValueError("Duplicate corpus IDs")
    decisions = {}
    for review in read_jsonl(reviews_path):
        key = str(review["id"])
        if key in decisions or key not in by_id:
            raise ValueError("Unknown or duplicate review ID: " + key)
        if review.get("record_sha256") != record_digest(by_id[key]):
            raise ValueError("Review does not match current record content: " + key)
        if (review.get("decision") not in {"approved", "rejected"}
                or not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip()):
            raise ValueError("Every submitted decision requires approved/rejected and a reviewer")
        decisions[key] = review
    accepted, rejected = [], []
    for row in records:
        review = decisions.get(str(row["id"]))
        if review:
            row.setdefault("metadata", {})["target_review"] = {
                "status": review["decision"], "reviewer": review["reviewer"],
                "notes": review.get("notes", ""), "reviewed_record_sha256": review["record_sha256"],
            }
        (rejected if review and review["decision"] == "rejected" else accepted).append(row)
    write_jsonl(output_path, accepted)
    write_jsonl(str(output_path) + ".rejected.jsonl", rejected)
    return {"retained": len(accepted), "rejected": len(rejected), "reviewed": len(decisions)}
