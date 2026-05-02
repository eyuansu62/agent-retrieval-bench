import json
import tempfile
import unittest
from pathlib import Path

from agent_retrieval_bench.hardmine import export_hardmine_candidates


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def sample(sample_id, task_type="code2test", base_commit="base", **overrides):
    gold = {"related_tests": [f"tests/{sample_id}_test.py"]}
    query = {"changed_file": f"src/{sample_id}.py", "pr_title": f"change {sample_id} behavior"}
    if task_type == "comment2context":
        query = {"path": f"src/{sample_id}.py", "comment": f"Does the {sample_id} behavior need another context file?"}
        gold = {
            "given_files": [f"src/{sample_id}.py"],
            "must_context_files": [{"path": f"tests/{sample_id}_behavior_test.py"}],
            "root_cause_files": [f"tests/{sample_id}_behavior_test.py"],
        }
    elif task_type == "trace2code":
        query = {"trace": f"Traceback (most recent call last): File \"src/{sample_id}.py\", line 1, in f"}
        gold = {"root_cause_files": [f"src/{sample_id}.py"]}
    record = {
        "id": sample_id,
        "version": 2,
        "task_type": task_type,
        "repo": "o/r",
        "base_commit": base_commit,
        "query": query,
        "gold": gold,
        "candidate_corpus": {"type": "repo_at_base_commit", "base_commit": base_commit},
    }
    record.update(overrides)
    return record


class HardmineExportTests(unittest.TestCase):
    def test_export_merges_sources_filters_noise_and_writes_task_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_a = root / "source_a"
            source_b = root / "source_b"
            out = root / "out"
            corpus_manifest = root / "corpus_manifest.jsonl"
            write_jsonl(
                corpus_manifest,
                [
                    {"repo": "o/r", "base_commit": "base", "status": "ok", "chunks_path": "unused"},
                    {"repo": "o/r", "base_commit": "trace-base", "status": "ok", "chunks_path": "unused"},
                ],
            )
            write_jsonl(
                source_a / "code2test.jsonl",
                [
                    sample("keep-code"),
                    sample("drop-testlog", task_type="testlog2code", gold={"root_cause_files": ["src/a.py"]}),
                    sample("drop-leak", query={"changed_file": "src/a.py", "patch": "diff --git a/src/a.py b/src/a.py"}),
                    sample("drop-missing-gold", gold={"related_tests": []}),
                    sample("drop-schema", candidate_corpus={"type": "repo_at_base_commit", "base_commit": "other"}),
                    sample("drop-corpus", base_commit="missing-base"),
                ],
            )
            write_jsonl(
                source_b / "samples.jsonl",
                [
                    sample("keep-code"),
                    sample("keep-comment", task_type="comment2context"),
                    sample("keep-trace", task_type="trace2code", base_commit="trace-base"),
                    sample(
                        "drop-weak-trace",
                        task_type="trace2code",
                        query={"trace": "please inspect src/drop-weak-trace.py"},
                    ),
                ],
            )

            result = export_hardmine_candidates(
                sources=[source_a, source_b],
                out_dir=out,
                corpus_manifest=corpus_manifest,
                require_corpus=True,
            )
            rows = [json.loads(line) for line in (out / "samples.jsonl").read_text().splitlines()]
            code_rows = [json.loads(line) for line in (out / "code2test.jsonl").read_text().splitlines()]
            comment_rows = [json.loads(line) for line in (out / "comment2context.jsonl").read_text().splitlines()]
            trace_rows = [json.loads(line) for line in (out / "trace2code.jsonl").read_text().splitlines()]
            manifest = json.loads((out / "manifest.json").read_text())

            self.assertEqual(result["total"], 3)
            self.assertEqual([row["id"] for row in rows], ["keep-code", "keep-comment", "keep-trace"])
            self.assertEqual(len(code_rows), 1)
            self.assertEqual(len(comment_rows), 1)
            self.assertEqual(len(trace_rows), 1)
            self.assertEqual(manifest["counts_by_task"], {"code2test": 1, "comment2context": 1, "trace2code": 1})
            self.assertEqual(manifest["dropped"]["duplicate_sample_id"], 1)
            self.assertEqual(manifest["dropped"]["excluded_task"], 1)
            self.assertEqual(manifest["dropped"]["query_leakage"], 1)
            self.assertEqual(manifest["dropped"]["missing_gold"], 1)
            self.assertEqual(manifest["dropped"]["schema_invalid"], 1)
            self.assertEqual(manifest["dropped"]["missing_corpus_pair"], 1)
            self.assertEqual(manifest["dropped"]["weak_trace"], 1)
            self.assertEqual(rows[0]["metadata"]["hardmine_source"], str(source_a / "code2test.jsonl"))


if __name__ == "__main__":
    unittest.main()
