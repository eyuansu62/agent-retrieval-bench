import json
import tempfile
import unittest
from pathlib import Path

from agent_retrieval_bench.selective_embedding_eval import evaluate_selective_embedding_baseline


class ToyEmbedder:
    model_name = "toy-embedder"

    def encode(self, texts, batch_size=32):
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "positive query" in lowered:
                vectors.append([1.0, 0.0])
            elif "no gold query" in lowered:
                vectors.append([0.1, 0.0])
            elif "gold implementation" in lowered:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return vectors


class SelectiveEmbeddingEvalTests(unittest.TestCase):
    def test_selective_embedding_rejects_missing_benchmark_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            with self.assertRaisesRegex(ValueError, "No benchmark samples found"):
                evaluate_selective_embedding_baseline(
                    sample_paths=[root / "missing" / "samples.jsonl"],
                    corpus_dir=root / "corpus",
                    model_name="toy-embedder",
                    embedder=ToyEmbedder(),
                    progress=False,
                )

    def test_selective_embedding_scores_positive_and_no_gold_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark = root / "benchmark"
            corpus = root / "corpus"
            benchmark.mkdir()
            corpus.mkdir()
            chunks_path = corpus / "owner__repo" / "abc.chunks.jsonl"
            chunks_path.parent.mkdir()
            chunks = [
                {
                    "repo": "owner/repo",
                    "base_commit": "abc",
                    "path": "src/gold.py",
                    "chunk_id": "gold",
                    "kind": "file",
                    "symbol": "",
                    "text": "gold implementation",
                },
                {
                    "repo": "owner/repo",
                    "base_commit": "abc",
                    "path": "src/other.py",
                    "chunk_id": "other",
                    "kind": "file",
                    "symbol": "",
                    "text": "unrelated helper",
                },
            ]
            with chunks_path.open("w", encoding="utf-8") as handle:
                for chunk in chunks:
                    handle.write(json.dumps(chunk) + "\n")
            (corpus / "corpus_manifest.jsonl").write_text(
                json.dumps(
                    {
                        "repo": "owner/repo",
                        "base_commit": "abc",
                        "status": "ok",
                        "chunks_path": str(chunks_path),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            samples = [
                {
                    "id": "positive",
                    "task_type": "edit2ripple",
                    "repo": "owner/repo",
                    "base_commit": "abc",
                    "query": {"intent": "positive query"},
                    "gold": {"files": ["src/gold.py"]},
                },
                {
                    "id": "no_gold",
                    "task_type": "abstention",
                    "repo": "owner/repo",
                    "base_commit": "abc",
                    "query": {"intent": "no gold query"},
                    "gold": {"no_gold": True, "reason": "external_dependency"},
                },
            ]
            sample_path = benchmark / "samples.jsonl"
            with sample_path.open("w", encoding="utf-8") as handle:
                for sample in samples:
                    handle.write(json.dumps(sample) + "\n")

            out = root / "summary.json"
            details = root / "details.jsonl"
            sweep = root / "sweep.jsonl"
            report = root / "report.md"
            result = evaluate_selective_embedding_baseline(
                sample_paths=[sample_path],
                corpus_dir=corpus,
                model_name="toy-embedder",
                out_path=out,
                details_path=details,
                sweep_path=sweep,
                report_path=report,
                embedder=ToyEmbedder(),
                batch_size=2,
                query_batch_size=2,
                progress=False,
            )

            self.assertEqual(result["mode"], "selective_embedding")
            self.assertEqual(result["evaluated"], 2)
            self.assertEqual(result["positive_evaluated"], 1)
            self.assertEqual(result["no_gold_evaluated"], 1)
            self.assertEqual(result["skipped"], {})
            self.assertEqual(result["positive_retrieval_metrics"]["Recall@20"], 1.0)
            target = result["operating_points"]["target_positive_pass_90"]
            self.assertEqual(target["positive_pass_rate"], 1.0)
            self.assertEqual(target["no_gold_abstain_rate"], 1.0)
            self.assertTrue(out.exists())
            self.assertEqual(sum(1 for _ in details.open()), 2)
            self.assertGreater(sum(1 for _ in sweep.open()), 1)
            self.assertIn("Selective embedding Threshold Report", report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
