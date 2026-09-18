from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional


class SchemaError(ValueError):
    pass


@dataclass
class OriginalExample:
    id: str
    statement: str
    question: str
    answer: str
    rationale: str = ""
    source: str = "gsm8k"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OriginalExample":
        required = ("id", "statement", "question", "answer")
        missing = [key for key in required if key not in value]
        if missing:
            raise SchemaError("Original example missing fields: %s" % ", ".join(missing))
        return cls(
            id=str(value["id"]),
            statement=str(value["statement"]).strip(),
            question=str(value["question"]).strip(),
            answer=str(value["answer"]).strip(),
            rationale=str(value.get("rationale", "")).strip(),
            source=str(value.get("source", "gsm8k")),
            metadata=dict(value.get("metadata", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def problem(self) -> str:
        return ("%s %s" % (self.statement.rstrip(), self.question.lstrip())).strip()


@dataclass
class SubtaskSolution:


    task: str
    solution: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SubtaskSolution":
        task = str(
            value.get("subquestion", value.get("task", value.get("subtask", "")))
        ).strip()
        solution = str(value.get("solution", value.get("answer", ""))).strip()
        if not task or not solution:
            raise SchemaError("Every chain step requires a non-empty subquestion and answer")
        return cls(task=task, solution=solution)

    def to_dict(self) -> Dict[str, str]:
        return {"subquestion": self.task, "answer": self.solution}

    @property
    def subquestion(self) -> str:
        return self.task

    @property
    def step_answer(self) -> str:
        return self.solution


@dataclass
class RefinedExample:
    id: str
    source_id: str
    variant_index: int
    statement: str
    question: str
    answer: str
    chain: List[SubtaskSolution]
    target_quantity: str = ""
    is_original: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RefinedExample":
        chain_value = value.get("chain", value.get("subtasks", []))
        chain = [SubtaskSolution.from_dict(item) for item in chain_value]
        if not chain:
            raise SchemaError("Refined example requires at least one subtask-solution step")
        required = ("id", "source_id", "statement", "question", "answer")
        missing = [key for key in required if key not in value]
        if missing:
            raise SchemaError("Refined example missing fields: %s" % ", ".join(missing))
        return cls(
            id=str(value["id"]),
            source_id=str(value["source_id"]),
            variant_index=int(value.get("variant_index", 0)),
            statement=str(value["statement"]).strip(),
            question=str(value["question"]).strip(),
            answer=str(value["answer"]).strip(),
            chain=chain,
            target_quantity=str(value.get("target_quantity", "")).strip(),
            is_original=bool(value.get("is_original", False)),
            metadata=dict(value.get("metadata", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "variant_index": self.variant_index,
            "statement": self.statement,
            "question": self.question,
            "answer": self.answer,
            "chain": [step.to_dict() for step in self.chain],
            "target_quantity": self.target_quantity,
            "is_original": self.is_original,
            "metadata": self.metadata,
        }

    @property
    def problem(self) -> str:
        return ("%s %s" % (self.statement.rstrip(), self.question.lstrip())).strip()


@dataclass
class GenerationStats:
    prompt_tokens: int = 0
    generated_tokens: int = 0
    latency_seconds: float = 0.0
    peak_memory_bytes: int = 0
    estimated_flops: float = 0.0
    unique_prompt_tokens: Optional[int] = None
    nominal_parameters: Optional[int] = None
    loaded_parameters: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Prediction:
    id: str
    prediction: str
    raw_subtasks: List[str]
    retained_subtasks: List[str]
    solutions: List[str]
    stats: GenerationStats
    gold: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
