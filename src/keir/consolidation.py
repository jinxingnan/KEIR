from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Embeddings must have shape [items, dimensions]")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return values / norms


def average_linkage_clusters(
    embeddings: np.ndarray,
    threshold: float,
) -> List[List[int]]:


    if threshold < 0.0 or threshold > 2.0:
        raise ValueError("Cosine-distance threshold must lie in [0, 2]")
    normalized = _normalize_rows(embeddings)
    count = normalized.shape[0]
    if count == 0:
        return []
    pair_distance = 1.0 - np.clip(normalized @ normalized.T, -1.0, 1.0)
    clusters = [[index] for index in range(count)]

    while len(clusters) > 1:
        best = None
        for left in range(len(clusters)):
            for right in range(left + 1, len(clusters)):
                block = pair_distance[np.ix_(clusters[left], clusters[right])]
                distance = float(block.mean())
                key = (
                    distance,
                    min(clusters[left]),
                    min(clusters[right]),
                )
                if best is None or key < best[0]:
                    best = (key, left, right)
        if best is None:
            break
        distance = best[0][0]
        if distance > threshold:
            break
        left, right = best[1], best[2]
        merged = sorted(clusters[left] + clusters[right])
        clusters = [
            cluster for index, cluster in enumerate(clusters) if index not in (left, right)
        ]
        clusters.append(merged)
        clusters.sort(key=min)
    return sorted(clusters, key=min)


class OfficialSimCSEEncoder:


    def __init__(
        self,
        model_name: str = "princeton-nlp/sup-simcse-roberta-base",
        device: Optional[str] = None,
        batch_size: int = 32,
        prefix: str = "",
        local_files_only: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers and torch are required for official SimCSE encoding"
            ) from exc
        self.torch = torch
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.batch_size = batch_size
        self.prefix = prefix
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        self.model = AutoModel.from_pretrained(
            model_name, local_files_only=local_files_only
        ).to(self.device).eval()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        values = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch = self.tokenizer(
                    [self.prefix + item for item in texts[start : start + self.batch_size]],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                batch = {key: value.to(self.device) for key, value in batch.items()}
                pooled = self.model(**batch, return_dict=True).pooler_output
                pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
                values.append(pooled.cpu().numpy())
        if not values:
            hidden_size = int(getattr(self.model.config, "hidden_size", 0))
            return np.empty((0, hidden_size), dtype=np.float32)
        return np.concatenate(values, axis=0)


class SemanticConsolidator:


    def __init__(self, encoder: Any, threshold: float = 0.25) -> None:
        self.encoder = encoder
        self.threshold = threshold

    def consolidate(self, subtasks: Sequence[str]) -> Tuple[List[str], Dict[str, Any]]:
        values = [str(item).strip() for item in subtasks if str(item).strip()]
        if not values:
            return [], {"clusters": [], "retained_indices": [], "reduction": 0.0}
        clusters = average_linkage_clusters(self.encoder.encode(values), self.threshold)
        retained_indices = sorted(min(cluster) for cluster in clusters)
        retained = [values[index] for index in retained_indices]
        return retained, {
            "clusters": clusters,
            "retained_indices": retained_indices,
            "predicted_count": len(values),
            "executed_count": len(retained),
            "reduction": 1.0 - len(retained) / len(values),
            "threshold": self.threshold,
            "representative_policy": "earliest",
        }
