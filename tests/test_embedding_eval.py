import json
import io
import tempfile
import unittest
from pathlib import Path

from agent_retrieval_bench.cli import resolve_embedding_eval_paths
from agent_retrieval_bench.embedding_eval import (
    TextEmbeddingCache,
    VoyageAPIEmbedder,
    chunk_text_for_embedding,
    default_embedding_cache_dir,
    default_embedding_summary_path,
    evaluate_embedding_baseline,
    load_sample_id_file,
    load_or_encode_chunk_vectors,
    model_slug,
    parse_retry_after,
    rank_chunks_by_vectors,
    shared_text_cache_encode_window_size,
    voyage_text_batches,
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
        self.encoded_texts = 0

    def encode(self, texts, batch_size=32):
        self.calls += 1
        self.encoded_texts += len(texts)
        return super().encode(texts, batch_size=batch_size)


class FailingAfterFirstCallEmbedder(CountingKeywordEmbedder):
    def encode(self, texts, batch_size=32):
        if self.calls >= 1:
            raise RuntimeError("stop after first batch")
        return super().encode(texts, batch_size=batch_size)


class FailingOnCallEmbedder(CountingKeywordEmbedder):
    def __init__(self, fail_on_call):
        super().__init__()
        self.fail_on_call = fail_on_call

    def encode(self, texts, batch_size=32):
        if self.calls + 1 == self.fail_on_call:
            raise RuntimeError(f"stop on call {self.fail_on_call}")
        return super().encode(texts, batch_size=batch_size)


class OomOnMultiQueryEmbedder(CountingKeywordEmbedder):
    def __init__(self):
        super().__init__()
        self.ooms = 0

    def encode(self, texts, batch_size=32):
        if len(texts) > 1 and not str(texts[0]).startswith("path:"):
            self.ooms += 1
            raise RuntimeError("CUDA out of memory")
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

    def test_embedding_baseline_can_select_sample_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = root / "samples.jsonl"
            corpus_dir = root / "corpus"
            chunks_path = corpus_dir / "o__r" / "base.chunks.jsonl"
            details_path = root / "details.jsonl"
            write_jsonl(
                samples,
                [
                    {
                        "id": f"s{index}",
                        "task_type": "code2test",
                        "repo": "o/r",
                        "base_commit": "base",
                        "query": {"changed_file": f"src/auth{index}.py", "intent": "auth failure"},
                        "gold": {"related_tests": ["tests/test_auth.py"], "fix_commit": "fix"},
                    }
                    for index in range(1, 5)
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

            result = evaluate_embedding_baseline(
                sample_paths=[samples],
                corpus_dir=corpus_dir,
                model_name="keyword",
                embedder=KeywordEmbedder(),
                cache_dir=None,
                details_path=details_path,
                shard_count=2,
                shard_index=1,
            )
            detail_ids = [json.loads(line)["sample_id"] for line in details_path.read_text(encoding="utf-8").splitlines()]

            self.assertEqual(detail_ids, ["s2", "s4"])
            self.assertEqual(result["evaluated"], 2)
            self.assertEqual(result["selection"]["input_samples"], 4)
            self.assertEqual(result["selection"]["selected_samples"], 2)
            self.assertEqual(result["selection"]["shard_count"], 2)
            self.assertEqual(result["selection"]["shard_index"], 1)

    def test_embedding_baseline_can_select_sample_id_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_ids_path = root / "ids.jsonl"
            write_jsonl(sample_ids_path, [{"sample_id": "s1"}, {"id": "s2"}])
            (root / "ids.txt").write_text("s3\n\n", encoding="utf-8")

            self.assertEqual(load_sample_id_file(sample_ids_path), {"s1", "s2"})
            self.assertEqual(load_sample_id_file(root / "ids.txt"), {"s3"})

    def test_embedding_baseline_flushes_details_after_each_completed_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = root / "samples.jsonl"
            corpus_dir = root / "corpus"
            chunks_path = corpus_dir / "o__r" / "base.chunks.jsonl"
            details_path = root / "details.jsonl"
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
                        "query": {"changed_file": "src/other.py", "intent": "other failure"},
                        "gold": {"related_tests": ["tests/test_other.py"], "fix_commit": "fix"},
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
                        "path": "tests/test_auth.py",
                        "kind": "file",
                        "text": "auth assertion",
                    },
                    {
                        "chunk_id": "c2",
                        "repo": "o/r",
                        "base_commit": "base",
                        "path": "tests/test_other.py",
                        "kind": "file",
                        "text": "noise assertion",
                    },
                ],
            )
            write_jsonl(
                corpus_dir / "corpus_manifest.jsonl",
                [{"repo": "o/r", "base_commit": "base", "status": "ok", "chunks_path": str(chunks_path)}],
            )

            with self.assertRaisesRegex(RuntimeError, "stop on call 3"):
                evaluate_embedding_baseline(
                    sample_paths=[samples],
                    corpus_dir=corpus_dir,
                    model_name="keyword",
                    embedder=FailingOnCallEmbedder(fail_on_call=3),
                    cache_dir=None,
                    details_path=details_path,
                    query_batch_size=1,
                )

            rows = [json.loads(line) for line in details_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["sample_id"] for row in rows], ["s1"])
            self.assertEqual(rows[0]["candidate_filter"], "all_files")

    def test_embedding_baseline_resume_details_skips_completed_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = root / "samples.jsonl"
            corpus_dir = root / "corpus"
            chunks_path = corpus_dir / "o__r" / "base.chunks.jsonl"
            details_path = root / "details.jsonl"
            sample_rows = [
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
                    "query": {"changed_file": "src/other.py", "intent": "other failure"},
                    "gold": {"related_tests": ["tests/test_other.py"], "fix_commit": "fix"},
                },
            ]
            write_jsonl(samples, sample_rows)
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
                    },
                    {
                        "chunk_id": "c2",
                        "repo": "o/r",
                        "base_commit": "base",
                        "path": "tests/test_other.py",
                        "kind": "file",
                        "text": "noise assertion",
                    },
                ],
            )
            write_jsonl(
                corpus_dir / "corpus_manifest.jsonl",
                [{"repo": "o/r", "base_commit": "base", "status": "ok", "chunks_path": str(chunks_path)}],
            )
            with self.assertRaisesRegex(RuntimeError, "stop on call 3"):
                evaluate_embedding_baseline(
                    sample_paths=[samples],
                    corpus_dir=corpus_dir,
                    model_name="keyword",
                    embedder=FailingOnCallEmbedder(fail_on_call=3),
                    cache_dir=None,
                    details_path=details_path,
                    query_batch_size=1,
                )

            retry_embedder = CountingKeywordEmbedder()
            result = evaluate_embedding_baseline(
                sample_paths=[samples],
                corpus_dir=corpus_dir,
                model_name="keyword",
                embedder=retry_embedder,
                cache_dir=None,
                details_path=details_path,
                resume_details=True,
                query_batch_size=1,
            )

            rows = [json.loads(line) for line in details_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["sample_id"] for row in rows], ["s1", "s2"])
            self.assertEqual(result["evaluated"], 2)
            self.assertEqual(retry_embedder.calls, 2)

    def test_embedding_baseline_batches_query_embeddings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = root / "samples.jsonl"
            corpus_dir = root / "corpus"
            chunks_path = corpus_dir / "o__r" / "base.chunks.jsonl"
            write_jsonl(
                samples,
                [
                    {
                        "id": f"s{index}",
                        "task_type": "code2test",
                        "repo": "o/r",
                        "base_commit": "base",
                        "query": {"changed_file": f"src/auth_{index}.py", "intent": "auth failure"},
                        "gold": {"related_tests": ["tests/test_auth.py"], "fix_commit": "fix"},
                    }
                    for index in range(3)
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
            embedder = CountingKeywordEmbedder()

            result = evaluate_embedding_baseline(
                sample_paths=[samples],
                corpus_dir=corpus_dir,
                model_name="keyword",
                embedder=embedder,
                cache_dir=None,
                query_batch_size=8,
            )

            self.assertEqual(result["evaluated"], 3)
            self.assertEqual(embedder.calls, 2)
            self.assertEqual(embedder.encoded_texts, 4)

    def test_embedding_baseline_splits_query_batch_after_oom(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = root / "samples.jsonl"
            corpus_dir = root / "corpus"
            chunks_path = corpus_dir / "o__r" / "base.chunks.jsonl"
            details_path = root / "details.jsonl"
            write_jsonl(
                samples,
                [
                    {
                        "id": "s1",
                        "task_type": "code2test",
                        "repo": "o/r",
                        "base_commit": "base",
                        "query": {"changed_file": "src/auth_1.py", "intent": "auth failure"},
                        "gold": {"related_tests": ["tests/test_auth.py"], "fix_commit": "fix"},
                    },
                    {
                        "id": "s2",
                        "task_type": "code2test",
                        "repo": "o/r",
                        "base_commit": "base",
                        "query": {"changed_file": "src/auth_2.py", "intent": "auth failure"},
                        "gold": {"related_tests": ["tests/test_auth.py"], "fix_commit": "fix"},
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
            embedder = OomOnMultiQueryEmbedder()

            result = evaluate_embedding_baseline(
                sample_paths=[samples],
                corpus_dir=corpus_dir,
                model_name="keyword",
                embedder=embedder,
                cache_dir=None,
                details_path=details_path,
                query_batch_size=2,
            )

            rows = [json.loads(line) for line in details_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(result["evaluated"], 2)
            self.assertEqual([row["sample_id"] for row in rows], ["s1", "s2"])
            self.assertEqual(embedder.ooms, 1)
            self.assertEqual(embedder.calls, 3)

    def test_embedding_baseline_resume_details_rebuilds_summary_without_model_or_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = root / "samples.jsonl"
            corpus_dir = root / "missing-corpus"
            details_path = root / "details.jsonl"
            summary_path = root / "summary.json"
            sample_rows = [
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
                    "task_type": "comment2context",
                    "repo": "o/r",
                    "base_commit": "base",
                    "query": {"comment": "auth context"},
                    "gold": {"context_files": ["src/auth.py"], "fix_commit": "fix"},
                },
            ]
            zero_metrics = {
                "Recall@5": 0.0,
                "Recall@10": 0.0,
                "Recall@20": 0.0,
                "MRR": 0.0,
                "gold_coverage@8k": 0.0,
            }
            write_jsonl(samples, sample_rows)
            write_jsonl(
                details_path,
                [
                    {
                        "sample_id": "s1",
                        "task_type": "code2test",
                        "repo": "o/r",
                        "base_commit": "base",
                        "candidate_filter": "all_files",
                        "gold_files": ["tests/test_auth.py"],
                        "gold_ranks": {},
                        "top_files": [],
                        "metrics": zero_metrics,
                    },
                    {
                        "sample_id": "s2",
                        "task_type": "comment2context",
                        "repo": "o/r",
                        "base_commit": "base",
                        "candidate_filter": "all_files",
                        "gold_files": ["src/auth.py"],
                        "gold_ranks": {},
                        "top_files": [],
                        "metrics": zero_metrics,
                    },
                ],
            )

            embedder = CountingKeywordEmbedder()
            result = evaluate_embedding_baseline(
                sample_paths=[samples],
                corpus_dir=corpus_dir,
                model_name="keyword",
                embedder=embedder,
                cache_dir=None,
                details_path=details_path,
                out_path=summary_path,
                resume_details=True,
            )

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(result["evaluated"], 2)
            self.assertEqual(result["skipped"], {})
            self.assertEqual(result["metrics"]["overall"]["samples"], 2)
            self.assertEqual(summary["evaluated"], 2)
            self.assertEqual(summary["metrics"]["overall"]["samples"], 2)
            self.assertEqual(embedder.calls, 0)

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

    def test_voyage_batches_split_by_count_and_character_budget(self):
        self.assertEqual(
            list(voyage_text_batches(["aa", "bb", "cc"], batch_size=2)),
            [["aa", "bb"], ["cc"]],
        )
        self.assertEqual(
            list(voyage_text_batches(["aaaa", "bb", "ccc"], batch_size=10, max_request_chars=5)),
            [["aaaa"], ["bb", "ccc"]],
        )

    def test_voyage_api_embedder_splits_requests_by_character_budget(self):
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
            max_request_chars=5,
            retry_base_seconds=0,
        )

        vectors = embedder.encode(["aaaa", "bb", "ccc"], batch_size=10, input_type="document")

        self.assertEqual(len(vectors), 3)
        self.assertEqual([request["input"] for request in requests], [["aaaa"], ["bb", "ccc"]])

    def test_parse_retry_after_header(self):
        self.assertEqual(parse_retry_after("2.5"), 2.5)
        self.assertEqual(parse_retry_after("-1"), 0.0)
        self.assertIsNone(parse_retry_after("soon"))
        self.assertIsNone(parse_retry_after(None))

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

    def test_embedding_cache_recovers_from_corrupt_vector_file(self):
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
            cache_dir = root / "cache"
            embedder = CountingKeywordEmbedder()

            first = load_or_encode_chunk_vectors(chunks, chunks_path, embedder, "keyword", cache_dir)
            vectors_path = next(cache_dir.rglob("*.npy"))
            vectors_path.write_bytes(b"not a numpy file")
            second = load_or_encode_chunk_vectors(chunks, chunks_path, embedder, "keyword", cache_dir)

            self.assertEqual(embedder.calls, 2)
            self.assertEqual(first.shape, second.shape)

    def test_shared_text_cache_reuses_duplicate_chunk_texts_across_commits(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy is optional")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunks_a = [
                {
                    "chunk_id": "c1",
                    "repo": "o/r",
                    "base_commit": "base-a",
                    "path": "src/auth.py",
                    "kind": "file",
                    "text": "auth",
                },
                {
                    "chunk_id": "c2",
                    "repo": "o/r",
                    "base_commit": "base-a",
                    "path": "src/auth.py",
                    "kind": "file",
                    "text": "auth",
                },
            ]
            chunks_b = [
                {
                    "chunk_id": "c1",
                    "repo": "o/r",
                    "base_commit": "base-b",
                    "path": "src/auth.py",
                    "kind": "file",
                    "text": "auth",
                }
            ]
            embedder = CountingKeywordEmbedder()
            with TextEmbeddingCache(
                root / "shared.sqlite",
                {
                    "model": "keyword",
                    "normalize_embeddings": True,
                    "passage_prefix": "",
                    "embedding_options": {},
                },
            ) as shared_cache:
                vectors_a = load_or_encode_chunk_vectors(
                    chunks_a,
                    root / "corpus" / "o__r" / "base-a.chunks.jsonl",
                    embedder,
                    "keyword",
                    root / "pair-cache",
                    shared_text_cache=shared_cache,
                )
                vectors_b = load_or_encode_chunk_vectors(
                    chunks_b,
                    root / "corpus" / "o__r" / "base-b.chunks.jsonl",
                    embedder,
                    "keyword",
                    root / "pair-cache",
                    shared_text_cache=shared_cache,
                )

            self.assertEqual(embedder.calls, 1)
            self.assertEqual(embedder.encoded_texts, 1)
            self.assertEqual(vectors_a.shape[0], 2)
            self.assertEqual(vectors_b.shape[0], 1)

    def test_shared_text_cache_batches_cache_misses_above_model_batch_size(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy is optional")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunks = [
                {
                    "chunk_id": f"c{index}",
                    "repo": "o/r",
                    "base_commit": "base",
                    "path": f"src/auth_{index}.py",
                    "kind": "file",
                    "text": f"auth token {index}",
                }
                for index in range(4)
            ]
            embedder = CountingKeywordEmbedder()
            with TextEmbeddingCache(
                root / "shared.sqlite",
                {
                    "model": "keyword",
                    "normalize_embeddings": True,
                    "passage_prefix": "",
                    "embedding_options": {},
                },
            ) as shared_cache:
                vectors = load_or_encode_chunk_vectors(
                    chunks,
                    root / "corpus" / "o__r" / "base.chunks.jsonl",
                    embedder,
                    "keyword",
                    root / "pair-cache",
                    shared_text_cache=shared_cache,
                    batch_size=1,
                )

            self.assertEqual(embedder.calls, 1)
            self.assertEqual(embedder.encoded_texts, 4)
            self.assertEqual(vectors.shape[0], 4)

    def test_shared_text_cache_writes_completed_batches_before_failure(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy is optional")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            encode_window_size = shared_text_cache_encode_window_size(1)
            chunks = [
                {
                    "chunk_id": f"c{index}",
                    "repo": "o/r",
                    "base_commit": "base",
                    "path": f"src/auth_{index}.py",
                    "kind": "file",
                    "text": f"auth token {index}",
                }
                for index in range(encode_window_size + 1)
            ]
            chunks_path = root / "corpus" / "o__r" / "base.chunks.jsonl"
            with TextEmbeddingCache(
                root / "shared.sqlite",
                {
                    "model": "keyword",
                    "normalize_embeddings": True,
                    "passage_prefix": "",
                    "embedding_options": {},
                },
            ) as shared_cache:
                with self.assertRaisesRegex(RuntimeError, "stop after first batch"):
                    load_or_encode_chunk_vectors(
                        chunks,
                        chunks_path,
                        FailingAfterFirstCallEmbedder(),
                        "keyword",
                        root / "pair-cache",
                        shared_text_cache=shared_cache,
                        batch_size=1,
                    )

                retry_embedder = CountingKeywordEmbedder()
                vectors = load_or_encode_chunk_vectors(
                    chunks,
                    chunks_path,
                    retry_embedder,
                    "keyword",
                    root / "pair-cache",
                    shared_text_cache=shared_cache,
                    batch_size=1,
                )

            self.assertEqual(retry_embedder.encoded_texts, 1)
            self.assertEqual(vectors.shape[0], encode_window_size + 1)

    def test_shared_text_cache_rejects_metadata_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with TextEmbeddingCache(root / "shared.sqlite", {"model": "one"}):
                pass

            with self.assertRaisesRegex(RuntimeError, "metadata mismatch"):
                TextEmbeddingCache(root / "shared.sqlite", {"model": "two"})

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

    def test_eval_embedding_version_paths_are_inferred(self):
        paths = resolve_embedding_eval_paths(
            model="/models/jina-code-embeddings-0.5b",
            version="v1_1",
            model_label=None,
            derived=None,
            corpus=None,
            out=None,
            details=None,
            cache=None,
            shared_text_cache=None,
            no_shared_text_cache=False,
            keep_list=None,
            no_keep_list=False,
            candidate_filter="all_files",
        )

        self.assertEqual(paths.model_label, "jina-code-embeddings-0.5b")
        self.assertEqual(paths.derived, Path("data/benchmark/v1_1"))
        self.assertEqual(paths.corpus, Path("data/corpus/v1_1"))
        self.assertEqual(paths.out, Path("data/eval/v1_1/jina-code-embeddings-0.5b_summary.json"))
        self.assertEqual(paths.details, Path("data/eval/v1_1/jina-code-embeddings-0.5b_details.jsonl"))
        self.assertEqual(paths.cache, Path("data/embeddings/v1_1/jina-code-embeddings-0.5b"))
        self.assertEqual(paths.shared_text_cache, Path("data/embeddings/v1_1/jina-code-embeddings-0.5b_texts.sqlite"))
        self.assertIsNone(paths.keep_list)

    def test_eval_embedding_legacy_paths_default_to_all_samples_without_version(self):
        paths = resolve_embedding_eval_paths(
            model="jinaai/jina-code-embeddings-0.5b",
            version=None,
            model_label=None,
            derived=None,
            corpus=None,
            out=None,
            details=None,
            cache=None,
            shared_text_cache=None,
            no_shared_text_cache=False,
            keep_list=None,
            no_keep_list=False,
            candidate_filter="all_files",
        )

        self.assertEqual(paths.model_label, "jinaai/jina-code-embeddings-0.5b")
        self.assertEqual(paths.derived, Path("data/benchmark/v0_1"))
        self.assertEqual(paths.corpus, Path("data/corpus/v0_1"))
        self.assertEqual(paths.out, Path("data/eval/v0_1/jinaai-jina-code-embeddings-0.5b_summary.json"))
        self.assertEqual(paths.cache, Path("data/embeddings/v0_1/jinaai-jina-code-embeddings-0.5b"))
        self.assertIsNone(paths.shared_text_cache)
        self.assertIsNone(paths.keep_list)

    def test_eval_embedding_version_paths_accept_model_label_and_overrides(self):
        paths = resolve_embedding_eval_paths(
            model="/models/Qwen3-Embedding-4B",
            version="v1_1",
            model_label="qwen3-embedding-4b",
            derived=Path("custom/benchmark"),
            corpus=Path("custom/corpus"),
            out=None,
            details=None,
            cache=None,
            shared_text_cache=None,
            no_shared_text_cache=True,
            keep_list=Path("keep.jsonl"),
            no_keep_list=False,
            candidate_filter="code_only",
        )

        self.assertEqual(paths.model_label, "qwen3-embedding-4b")
        self.assertEqual(paths.derived, Path("custom/benchmark"))
        self.assertEqual(paths.corpus, Path("custom/corpus"))
        self.assertEqual(paths.out, Path("data/eval/v1_1/qwen3-embedding-4b_code_only_summary.json"))
        self.assertEqual(paths.details, Path("data/eval/v1_1/qwen3-embedding-4b_code_only_details.jsonl"))
        self.assertEqual(paths.cache, Path("data/embeddings/v1_1/qwen3-embedding-4b"))
        self.assertIsNone(paths.shared_text_cache)
        self.assertEqual(paths.keep_list, Path("keep.jsonl"))

    def test_eval_embedding_version_paths_reject_conflicting_shared_text_cache_flags(self):
        with self.assertRaisesRegex(ValueError, "no-shared-text-cache"):
            resolve_embedding_eval_paths(
                model="/models/jina-code-embeddings-0.5b",
                version="v1_1",
                model_label=None,
                derived=None,
                corpus=None,
                out=None,
                details=None,
                cache=None,
                shared_text_cache=Path("texts.sqlite"),
                no_shared_text_cache=True,
                keep_list=None,
                no_keep_list=False,
                candidate_filter="all_files",
            )


if __name__ == "__main__":
    unittest.main()
