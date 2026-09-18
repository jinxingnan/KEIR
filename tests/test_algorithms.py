import numpy as np
import pytest

from keir.benchmarks import download_benchmark
from keir.cli import build_parser
from keir.consolidation import average_linkage_clusters
from keir.evaluation import mcnemar
from keir.prompts import parse_subtasks


def test_average_linkage_uses_cross_cluster_mean():
    angles = np.deg2rad([0, 40, 80])
    points = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    assert sorted(map(len, average_linkage_clusters(points, 0.30))) == [1, 2]
    assert len(average_linkage_clusters(points, 0.60)) == 1


def test_parser_preserves_exact_duplicates():
    assert parse_subtasks("1. Same?\n2. Same?") == ["Same?", "Same?"]


def test_mcnemar_requires_matched_questions():
    a = [{"id": "x", "prediction": "1", "gold": "1"}]
    b = [{"id": "x", "prediction": "0", "gold": "1"}]
    result = mcnemar(a, b)
    assert result["n10"] == 1 and result["n01"] == 0
    with pytest.raises(ValueError):
        mcnemar(a, [{"id": "y", "prediction": "1", "gold": "1"}])


def test_mawps_requires_an_explicit_evaluation_file(tmp_path):
    with pytest.raises(ValueError, match="355"):
        download_benchmark("mawps", str(tmp_path / "out.jsonl"))


def test_evaluation_cli_requires_configuration_and_data():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["evaluate"])
    args = parser.parse_args(["evaluate", "--config", "config.yaml", "--input", "data.jsonl",
                              "--output", "predictions.jsonl"])
    assert args.mode == "keir"


def test_local_source_metadata_options_are_explicit():
    args = build_parser().parse_args([
        "normalize-benchmark", "--benchmark", "mawps", "--input", "mawps.jsonl",
        "--output", "normalized.jsonl", "--source-url", "https://example.org/dataset",
        "--source-revision", "revision-1", "--source-split", "full-collection"])
    assert args.source_revision == "revision-1"
    assert args.source_split == "full-collection"
