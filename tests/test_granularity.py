from __future__ import annotations

import json

import pytest

from agent_retrieval_bench.granularity import (
    build_report,
    gold_file_paths,
    interval_line_count,
    iter_jsonl,
    percentile,
    report_markdown,
)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_gold_file_paths_supports_all_v2_task_schemas():
    assert gold_file_paths({"task_type": "code2test", "gold": {"related_tests": ["test_a.py"]}}) == ["test_a.py"]
    assert gold_file_paths(
        {"task_type": "comment2context", "gold": {"must_context_files": [{"path": "src/a.py"}]}}
    ) == ["src/a.py"]
    assert gold_file_paths({"task_type": "trace2code", "gold": {"root_cause_files": ["src/b.py"]}}) == ["src/b.py"]
    assert gold_file_paths({"task_type": "edit2ripple", "gold": {"files": ["src/c.py"]}}) == ["src/c.py"]


def test_percentile_and_interval_union_are_deterministic():
    assert percentile([1, 2, 3, 4], 0.5) == pytest.approx(2.5)
    assert percentile([1, 2, 3, 4], 0.95) == pytest.approx(3.85)
    assert interval_line_count([(1, 5), (4, 10), (20, 21)]) == 12


def test_iter_jsonl_streams_nonempty_rows(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"id": 1}\n\n{"id": 2}\n', encoding="utf-8")

    rows = iter_jsonl(path)

    assert iter(rows) is rows
    assert list(rows) == [{"id": 1}, {"id": 2}]


def test_build_report_connects_corpus_sizes_spans_and_detail_metrics(tmp_path):
    chunks_path = tmp_path / "data/corpus/repo/base.chunks.jsonl"
    write_jsonl(
        chunks_path,
        [
            {"path": "src/large.py", "start_line": 1, "end_line": 500},
            {"path": "src/large.py", "start_line": 501, "end_line": 1200},
        ],
    )
    manifest_path = tmp_path / "data/corpus/corpus_manifest.jsonl"
    write_jsonl(
        manifest_path,
        [
            {
                "status": "ok",
                "repo": "org/repo",
                "base_commit": "base",
                "chunks_path": "data/corpus/repo/base.chunks.jsonl",
            }
        ],
    )
    sample = {
        "id": "s1",
        "task_type": "trace2code",
        "repo": "org/repo",
        "base_commit": "base",
        "gold": {"root_cause_files": ["src/large.py"]},
        "gold_spans": [{"path": "src/large.py", "start_line": 100, "end_line": 119}],
    }
    details_path = tmp_path / "details.jsonl"
    write_jsonl(
        details_path,
        [
            {
                "metrics": {"Recall@20": 1.0, "gold_coverage@8k": 1.0},
                "line_metrics": {
                    "line_recall@8k": 0.5,
                    "line_precision@8k": 0.1,
                    "line_f1@8k": 1 / 6,
                    "line_predicted_count": 100,
                },
            }
        ],
    )

    report = build_report(
        [sample],
        [sample],
        [manifest_path],
        {"method": details_path},
        tmp_path,
    )

    assert report["missing_gold_file_lengths"] == []
    assert report["file_sizes"]["overall"]["file_lines"]["median"] == 1200
    assert report["span_evidence"]["overall"]["evidence_lines"]["median"] == 20
    assert report["span_evidence"]["overall"]["evidence_to_file_ratio"]["median"] == pytest.approx(1 / 60)
    assert report["method_sensitivity"][0]["line_f1@8k_chars"] == pytest.approx(1 / 6)
    assert "File exposure versus line overlap" in report_markdown(report)
