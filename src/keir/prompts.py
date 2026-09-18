import json
from typing import List, Sequence

from .schemas import OriginalExample, RefinedExample


VARIANT_TEACHER_SYSTEM = "You are an expert math dataset writer."


SOCRATIC_TEACHER_SYSTEM = "You are an expert math teacher and data annotator."


def variant_teacher_prompt(example: OriginalExample, variants: int = 3) -> str:

    return """Your task is to rewrite the QUESTION of a math word problem into alternative phrasings,
while preserving the exact same underlying problem, givens, and final answer.

You will be given:
1. A problem statement
2. The original question

Requirements:
- Keep the meaning exactly unchanged.
- Keep all quantities, entities, and constraints unchanged.
- Rewrite only the original question sentence. Do not restate, summarize, or copy facts from the
  problem statement into a rewritten question.
- Preserve exactly the multiset and surface form of every number already present in the original
  question. If the original question contains no number, every rewrite must also contain no number.
- Do not import a number from the problem statement, even when that number is relevant.
- Do NOT introduce new numbers, assumptions, steps, hints, or intermediate results.
- Do NOT simplify or make the problem harder.
- Do NOT turn it into a different problem type.
- The rewritten question must still be fully answerable using the same statement.
- The rewritten question should sound natural and fluent.
- Each rewrite must be a direct interrogative sentence ending in `?`; do not use commands such as
  `calculate`, `determine`, `find`, or `compute`.
- Keep each rewrite concise and approximately the same length as the original question.
- Each rewritten question must be semantically equivalent to the original question, but phrased differently.
- Avoid trivial edits such as only changing one word or punctuation.
- Do NOT include the solution.
- Do NOT include explanations.
- Do NOT repeat the original question verbatim.
- Generate exactly {variants} rewritten questions.

Output format:
Return only a JSON object in the following format:
{{"rewritten_questions": ["...", "...", "..."]}}

Problem statement:
{statement}

Original question:
{question}""".format(
        variants=variants,
        statement=example.statement,
        question=example.question,
    )


def variant_socratic_teacher_prompt(
    original: RefinedExample,
    variant_question: str,
) -> str:

    reference_chain = [step.to_dict() for step in original.chain]
    schema = {
        "steps": [
            {
                "subquestion": "a concise natural-language question ending in ?",
                "answer": "the local answer to this subquestion",
            }
        ],
        "final_answer": original.answer,
    }
    return """Adapt a verified GSM8K Socratic question-answer chain to an answer-equivalent
variant of the original question. The problem statement and gold answer are unchanged.

Use the verified official chain as the reasoning anchor. Preserve every correct and useful
intermediate dependency that still applies. Change wording or step boundaries only when needed to
make the chain natural for the variant question. The final subquestion must ask exactly the target
of the variant question.

Requirements:
- Return a small ordered sequence of necessary Socratic subquestions and local answers.
- Every subquestion must be a genuine question ending in `?`.
- Do not introduce any new quantity, entity, constraint, assumption, or alternate target.
- Do not copy an intermediate numerical result into the variant question.
- Do not omit a prerequisite used by a later answer.
- Do not add redundant or rhetorical questions.
- Verify every calculation and unit conversion.
- The last answer must resolve the variant question and equal the supplied gold answer.
- Return only valid JSON with exactly this shape:
{schema}

Problem statement:
{statement}

Original question:
{original_question}

Answer-equivalent variant question:
{variant_question}

Gold final answer:
{answer}

Verified official Socratic chain:
{reference_chain}""".format(
        schema=json.dumps(schema, ensure_ascii=False),
        statement=original.statement,
        original_question=original.question,
        variant_question=variant_question,
        answer=original.answer,
        reference_chain=json.dumps(reference_chain, ensure_ascii=False),
    )


def decomposition_prompt(statement: str, question: str, max_subtasks: int = 10) -> str:
    return """You are the Socratic question decomposer in a mathematical reasoning system.
Decompose the problem into an ordered list of necessary, non-overlapping subquestions. Each
subquestion must be a concise natural-language question ending in `?`, and later subquestions may
depend on earlier answers. The last subquestion must ask for exactly the original target.
Use at most {max_subtasks} subquestions. Do not answer them.
Output exactly one subquestion per line as `<index>. <subquestion>` and no other text.

Problem statement:
{statement}

Question:
{question}

Subquestions:
""".format(max_subtasks=max_subtasks, statement=statement, question=question)


def format_subtasks(subtasks: Sequence[str]) -> str:
    return "\n".join("%d. %s" % (index, task.strip()) for index, task in enumerate(subtasks, 1))


def parse_subtasks(text: str, max_subtasks: int = 10) -> List[str]:
    tasks = []
    for raw_line in text.strip().splitlines():
        line = raw_line.strip()
        line = line.strip("`")
        line = line.replace("•", "-")
        line = __import__("re").sub(r"^\s*(?:[-*]|\d+\s*[\).:-])\s*", "", line).strip()
        if line and line.lower() not in {
            "subtasks:",
            "subtasks",
            "subquestions:",
            "subquestions",
        }:
            tasks.append(line)
    if len(tasks) == 1 and ";" in tasks[0]:
        tasks = [item.strip() for item in tasks[0].split(";") if item.strip()]
    return tasks[:max_subtasks]


def execution_prompt(
    statement: str,
    question: str,
    current_subtask: str,
    previous_subtasks: Sequence[str],
    previous_solutions: Sequence[str],
    is_terminal: bool,
) -> str:
    history = "None."
    if previous_solutions:
        history = "\n".join(
            "Step %d\nSubquestion: %s\nAnswer: %s" % (index, task, solution)
            for index, (task, solution) in enumerate(
                zip(previous_subtasks, previous_solutions), start=1
            )
        )
    terminal = (
        "This is the terminal subquestion. End with `Final answer: <answer>`."
        if is_terminal
        else "Answer only this subquestion; do not jump to the final question."
    )
    return """You are the Socratic subquestion answerer in a mathematical reasoning system.
Use the original problem and established answers. Answer the current local subquestion accurately.
Do not revise established results unless they are inconsistent. {terminal}

Problem statement:
{statement}

Original question:
{question}

Established subquestion-answer pairs:
{history}

Current subquestion:
{subtask}

Answer:
""".format(
        terminal=terminal,
        statement=statement,
        question=question,
        history=history,
        subtask=current_subtask,
    )


def cot_prompt(statement: str, question: str) -> str:
    return """Solve the following mathematical problem step by step. End with
`Final answer: <answer>`.

Problem statement:
{statement}

Question:
{question}

Solution:
""".format(statement=statement, question=question)
