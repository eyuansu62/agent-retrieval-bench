import json
import io
import tempfile
import unittest
from pathlib import Path

from agent_retrieval_bench.embedding_eval import (
    VoyageAPIEmbedder,
    chunk_text_for_embedding,
    default_embedding_cache_dir,
    default_embedding_summary_path,
    evaluate_embedding_baseline,
    load_or_encode_chunk_vectors,
    model_slug,
    rank_chunks_by_vectors,
)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class KeywordEmbedder:
    model_name = "keyword"

    def encode(self, texts, batch_size=32):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    1.0 if "auth" in lowered else 0.0,
                    1.0 if "noise" in lowered else 0.0,
                    1.0 if "diff --git" in lowered else 0.0,
                ]
            )
        return vectors


class CountingKeywordEmbedder(KeywordEmbedder):
    def __init__(self):
        self.calls = 0

    def encode(self, texts, batch_size=32):
        self.calls += 1
        return super().encode(texts, batch_size=batch_size)


class TypedKeywordEmbedder(KeywordEmbedder):
    def __init__(self):
        self.input_types = []

    def encode(self, texts, batch_size=32, input_type=None):
        self.input_types.append(input_type)
        return super().encode(texts, batch_size=batch_size)


class EmbeddingEvalTests(unittest.TestCase):
    def test_rank_chunks_by_vectors_sorts_by_similarity(self):
        chunks = [
            {"chunk_id": "c1", "path": "src/noise.py"},
            {"chunk_id": "c2", "path": "tests/test_auth.py"},
        ]

        ranked = rank_chunks_by_vectors([1.0, 0.0], [[0.0, 1.0], [1.0, 0.0]], chunks)

        self.assertEqual(ranked[0]["path"], "tests/test_auth.py")

    def test_embedding_baseline_evaluates_file_level_recall(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = root / "samples.jsonl"
            corpus_dir = root / "corpus"
            chunks_path = corpus_dir / "o__r" / "base.chunks.jsonl"
            write_jsonl(
                samples,
                [
                    {
                        "id": "s1",
                        "task_type": "code2test",
                        "repo": "o/r",
                        "base_commit": "base",
                        "query": {"changed_file": "src/auth.py", "intent": "auth failure"},
                        "gold": {"related_tests": ["tests/test_auth.py"], "fix_commit": "fix"},
                    },
                    {
                        "id": "s2",
                        "task_type": "code2test",
                        "repo": "o/r",
                        "base_commit": "base",
                        "query": {"raw_signal": "diff --git a/x b/x"},
                        "gold": {"related_tests": ["tests/test_x.py"], "fix_commit": "fix"},
                    },
                ],
            )
            write_jsonl(
                chunks_path,
                [
                    {
                        "chunk_id": "c1",
                        "repo": "o/r",
                        "base_commit": "base",
                        "path": "src/noise.py",
                        "kind": "file",
                        "text": "noise",
                    },
                    {
                        "chunk_id": "c2",
                        "repo": "o/r",
                        "base_commit": "base",
                        "path": "tests/test_auth.py",
                        "kind": "file",
                        "text": "auth assertion",
                    },
                ],
            )
            write_jsonl(
                corpus_dir / "corpus_manifest.jsonl",
                [{"repo": "o/r", "base_commit": "base", "status": "ok", "chunks_path": str(chunks_path)}],
            )

            result = evaluate_embedding_baseline(
                sample_paths=[samples],
                corpus_dir=corpus_dir,
                model_name="keyword",
                embedder=KeywordEmbedder(),
                cache_dir=None,
                details_path=root / "details.jsonl",
            )
            detail = json.loads((root / "details.jsonl").read_text(encoding="utf-8").splitlines()[0])

            self.assertEqual(result["evaluated"], 1)
            self.assertEqual(result["candidate_filter"], "all_files")
            self.assertEqual(result["skipped"], {"query_leakage": 1})
            self.assertEqual(detail["candidate_filter"], "all_files")
            self.assertEqual(detail["gold_ranks"], {"tests/test_auth.py": 1})
        self.assertEqual(result["metrics"]["code2test"]["Recall@5"], 1.0)
        self.assertEqual(result["metrics"]["code2test"]["MRR"], 1.0)

    def test_embedding_baseline_reports_progress_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = root / "samples.jsonl"
            corpus_dir = root / "corpus"
            chunks_path = corpus_dir / "o__r" / "base.chunks.jsonl"
            write_jsonl(
                samples,
                [
                    {
                        "id": "s1",
                        "task_type": "code2test",
                        "repo": "o/r",
                        "base_commit": "base",
                        "query": {"changed_file": "src/auth.py", "intent": "auth failure"},
                        "gold": {"related_tests": ["tests/test_auth.py"], "fix_commit": "fix"},
                    }
                ],
            )
            write_jsonl(
                chunks_path,
                [
                    {
                        "chunk_id": "c1",
                        "repo": "o/r",
                        "base_commit": "base",
                        "path": "tests/test_auth.py",
                        "kind": "file",
                        "text": "auth assertion",
                    }
                ],
            )
            write_jsonl(
                corpus_dir / "corpus_manifest.jsonl",
                [{"repo": "o/r", "base_commit": "base", "status": "ok", "chunks_path": str(chunks_path)}],
            )
            stream = io.StringIO()

            evaluate_embedding_baseline(
                sample_paths=[samples],
                corpus_dir=corpus_dir,
                model_name="keyword",
                embedder=KeywordEmbedder(),
                cache_dir=None,
                progress=True,
                progress_stream=stream,
                candidate_filter="tests_only",
            )

            output = stream.getvalue()
            self.assertIn("loading embedding model", output)
            self.assertIn("encoding chunks without cache", output)
            self.assertIn("evaluating samples", output)

    def test_embedding_baseline_passes_query_and_document_input_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = root / "samples.jsonl"
            corpus_dir = root / "corpus"
            chunks_path = corpus_dir / "o__r" / "base.chunks.jsonl"
            write_jsonl(
                samples,
                [
                    {
                        "id": "s1",
                        "task_type": "code2test",
                        "repo": "o/r",
                        "base_commit": "base",
                        "query": {"changed_file": "src/auth.py", "intent": "auth failure"},
                        "gold": {"related_tests": ["tests/test_auth.py"], "fix_commit": "fix"},
                    }
                ],
            )
            write_jsonl(
                chunks_path,
                [
                    {
                        "chunk_id": "c1",
                        "repo": "o/r",
                        "base_commit": "base",
                        "path": "tests/test_auth.py",
                        "kind": "file",
                        "text": "auth assertion",
                    }
                ],
            )
            write_jsonl(
                corpus_dir / "corpus_manifest.jsonl",
                [{"repo": "o/r", "base_commit": "base", "status": "ok", "chunks_path": str(chunks_path)}],
            )
            embedder = TypedKeywordEmbedder()

            evaluate_embedding_baseline(
                sample_paths=[samples],
                corpus_dir=corpus_dir,
                model_name="typed",
                embedder=embedder,
                cache_dir=None,
                query_input_type="query",
                passage_input_type="document",
            )

            self.assertEqual(embedder.input_types, ["document", "query"])

    def test_voyage_api_embedder_sends_typed_requests_and_normalizes(self):
        requests = []

        def fake_request(payload):
            requests.append(dict(payload))
            return {
                "data": [
                    {"index": index, "embedding": [3.0, 4.0]}
                    for index, _text in enumerate(payload["input"])
                ]
            }

        embedder = VoyageAPIEmbedder(
            api_key="test-key",
            request_func=fake_request,
            output_dimension=512,
            retry_base_seconds=0,
        )

        vectors = embedder.encode(["one", "two"], batch_size=1, input_type="document")

        self.assertEqual(len(vectors), 2)
        self.assertAlmostEqual(vectors[0][0], 0.6)
        self.assertAlmostEqual(vectors[0][1], 0.8)
        self.assertEqual([request["input_type"] for request in requests], ["document", "document"])
        self.assertEqual(requests[0]["model"], "voyage-code-3")
        self.assertEqual(requests[0]["output_dimension"], 512)

    def test_embedding_cache_is_keyed_by_passage_options(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy is optional")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunks = [
                {
                    "chunk_id": "c1",
                    "repo": "o/r",
                    "base_commit": "base",
                    "path": "src/auth.py",
                    "kind": "file",
                    "text": "auth",
                }
            ]
            chunks_path = root / "corpus" / "o__r" / "base.chunks.jsonl"
            embedder = CountingKeywordEmbedder()

            load_or_encode_chunk_vectors(chunks, chunks_path, embedder, "keyword", root / "cache")
            load_or_encode_chunk_vectors(chunks, chunks_path, embedder, "keyword", root / "cache")
            load_or_encode_chunk_vectors(chunks, chunks_path, embedder, "keyword", root / "cache", passage_prefix="noise ")
            load_or_encode_chunk_vectors(chunks, chunks_path, embedder, "keyword", root / "cache", candidate_filter="tests_only")

            self.assertEqual(embedder.calls, 3)

    def test_embedding_text_and_default_paths_are_stable(self):
        text = chunk_text_for_embedding(
            {"path": "src/auth.py", "kind": "symbol", "symbol": "refresh", "text": "return token"}
        )

        self.assertIn("path: src/auth.py", text)
        self.assertIn("symbol: refresh", text)
        self.assertEqual(model_slug("jinaai/jina-code-embeddings-0.5b"), "jinaai-jina-code-embeddings-0.5b")
        self.assertEqual(
            default_embedding_summary_path("jinaai/jina-code-embeddings-0.5b"),
            Path("data/eval/v0_1/jinaai-jina-code-embeddings-0.5b_summary.json"),
        )
        self.assertEqual(
            default_embedding_summary_path("jinaai/jina-code-embeddings-0.5b", candidate_filter="tests_only"),
            Path("data/eval/v0_1/jinaai-jina-code-embeddings-0.5b_tests_only_summary.json"),
        )
        self.assertEqual(
            default_embedding_cache_dir("jinaai/jina-code-embeddings-0.5b"),
            Path("data/embeddings/v0_1/jinaai-jina-code-embeddings-0.5b"),
        )


if __name__ == "__main__":
    unittest.main()
