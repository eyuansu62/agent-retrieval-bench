import csv
import json
import tempfile
import unittest
from pathlib import Path

from agent_retrieval_bench.audit import AUDIT_FIELDS, build_audit_rows, summarize_audit, write_audit_sample


def sample(sample_id, task_type, repo, query=None, gold=None):
    return {
        "id": sample_id,
        "task_type": task_type,
        "repo": repo,
        "base_commit": "base",
        "query": query or {"raw_signal": "failure in auth"},
        "gold": gold
        or {
            "root_cause_files": ["src/auth.py"],
            "related_tests": ["tests/test_auth.py"],
            "supporting_files": [],
            "negative_distractors": [],
            "fix_commit": "abc123fix",
        },
    }


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class AuditTests(unittest.TestCase):
    def test_audit_sampling_is_stable_and_uses_fixed_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            derived = Path(tmp) / "derived"
            write_jsonl(
                derived / "comment2context.jsonl",
                [
                    sample("c1", "comment2context", "o/a"),
                    sample("c2", "comment2context", "o/a"),
                    sample("c3", "comment2context", "o/b"),
                    sample("c4", "comment2context", "o/b"),
                ],
            )

            first = build_audit_rows(derived, per_task=2, seed=7, tasks=["comment2context"])
            second = build_audit_rows(derived, per_task=2, seed=7, tasks=["comment2context"])

            self.assertEqual(first, second)
            self.assertEqual(len(first), 2)
            self.assertEqual(tuple(first[0].keys()), AUDIT_FIELDS)

    def test_audit_outputs_jsonl_csv_and_removes_patch_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            derived = Path(tmp) / "derived"
            fix_commit = "ff00ff00ff00"
            write_jsonl(
                derived / "trace2code.jsonl",
                [
                    sample(
                        "t1",
                        "trace2code",
                        "o/r",
                        query={"raw_signal": f"diff --git a/x b/x\n+++ b/x\nerror {fix_commit}"},
                        gold={"root_cause_files": ["src/x.py"], "related_tests": [], "fix_commit": fix_commit},
                    )
                ],
            )

            result = write_audit_sample(derived, Path(tmp) / "audit", per_task=20, tasks=["trace2code"])
            jsonl_row = json.loads((Path(result["outputs"]["jsonl"])).read_text(encoding="utf-8").strip())
            with Path(result["outputs"]["csv"]).open("r", encoding="utf-8", newline="") as handle:
                csv_row = next(csv.DictReader(handle))

            self.assertEqual(set(jsonl_row), set(AUDIT_FIELDS))
            self.assertEqual(set(csv_row), set(AUDIT_FIELDS))
            self.assertNotIn("diff --git", jsonl_row["query_excerpt"])
            self.assertNotIn(fix_commit, jsonl_row["query_excerpt"])

    def test_audit_summary_counts_verdicts_and_keep_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit_path = Path(tmp) / "audit.csv"
            with audit_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
                writer.writeheader()
                writer.writerow({"sample_id": "a", "task_type": "comment2context", "repo": "o/r", "verdict": "valid"})
                writer.writerow({"sample_id": "b", "task_type": "comment2context", "repo": "o/r", "verdict": "noisy", "keep": "yes"})
                writer.writerow({"sample_id": "c", "task_type": "trace2code", "repo": "o/r", "verdict": "leaked"})

            keep_path = Path(tmp) / "keep.jsonl"
            summary = summarize_audit(audit_path, keep_list_path=keep_path)
            keep_ids = [json.loads(line)["sample_id"] for line in keep_path.read_text(encoding="utf-8").splitlines()]

            self.assertEqual(summary["by_task"]["comment2context"]["counts"]["valid"], 1)
            self.assertEqual(summary["by_task"]["comment2context"]["counts"]["noisy"], 1)
            self.assertEqual(summary["by_task"]["trace2code"]["counts"]["leaked"], 1)
            self.assertEqual(keep_ids, ["a", "b"])


if __name__ == "__main__":
    unittest.main()
