from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from agent_retrieval_bench.cli import main
from agent_retrieval_bench.release import (
    CURRENT_BENCHMARK_RELEASES,
    DEFAULT_RELEASE_ID,
    download_benchmark_release,
    merge_corpus_manifests,
    release_catalog,
    verify_release_checksum,
)


class ReleaseDownloadTests(unittest.TestCase):
    def test_cli_reports_package_version(self) -> None:
        stdout = StringIO()
        with redirect_stdout(stdout), self.assertRaises(SystemExit) as raised:
            main(["--version"])

        self.assertEqual(raised.exception.code, 0)
        self.assertEqual(stdout.getvalue().strip(), "arb 0.2.1")

    def test_release_catalog_identifies_the_five_current_subsets(self) -> None:
        catalog = release_catalog()

        self.assertEqual(
            [item["id"] for item in catalog["current"]],
            [
                "v2_code2test",
                "v2_comment2context",
                "v2_trace2code",
                "v2_edit2ripple",
                "v2_abstention",
            ],
        )
        self.assertEqual(sum(item["samples"] for item in catalog["current"]), 427)
        self.assertEqual(
            [item["id"] for item in catalog["auxiliary"]],
            ["v2_selective_retrieval_balanced", "v2_selective_retrieval_natural"],
        )

    def test_releases_command_prints_machine_readable_catalog(self) -> None:
        stdout = StringIO()

        with redirect_stdout(stdout):
            result = main(["releases", "--json"])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(result, 0)
        self.assertEqual(payload["current"][0]["id"], "v2_code2test")

    def test_download_all_processes_each_current_release(self) -> None:
        stdout = StringIO()
        with patch("agent_retrieval_bench.cli.download_benchmark_release") as download:
            download.side_effect = lambda **kwargs: {"version": kwargs["version"]}

            with redirect_stdout(stdout):
                result = main(["download-benchmark", "--all", "--local-dir", "custom-data"])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(result, 0)
        self.assertEqual(
            [call.kwargs["version"] for call in download.call_args_list],
            [release.id for release in CURRENT_BENCHMARK_RELEASES],
        )
        self.assertEqual(
            [item["version"] for item in payload["releases"]],
            [release.id for release in CURRENT_BENCHMARK_RELEASES],
        )
        self.assertTrue(all(call.kwargs["local_dir"] == Path("custom-data") for call in download.call_args_list))

    def test_download_single_release_preserves_single_result_shape(self) -> None:
        stdout = StringIO()
        with patch("agent_retrieval_bench.cli.download_benchmark_release", return_value={"version": "historical-v1"}):
            with redirect_stdout(stdout):
                result = main(["download-benchmark", "--version", "historical-v1"])

        self.assertEqual(result, 0)
        self.assertEqual(json.loads(stdout.getvalue()), {"version": "historical-v1"})

    def test_download_requires_version_or_all(self) -> None:
        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as raised:
                main(["download-benchmark"])

        self.assertEqual(raised.exception.code, 2)

    def test_merge_corpus_manifests_deduplicates_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            local_dir = Path(tmp) / "data"
            first = local_dir / "corpus" / "one" / "corpus_manifest.jsonl"
            second = local_dir / "corpus" / "two" / "corpus_manifest.jsonl"
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)
            first.write_text(
                "\n".join(
                    [
                        json.dumps({"repo": "o/a", "base_commit": "1", "status": "ok", "chunks_path": "one.jsonl"}),
                        json.dumps({"repo": "o/b", "base_commit": "2", "status": "ok", "chunks_path": "two.jsonl"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            second.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {"repo": "o/a", "base_commit": "1", "status": "ok", "chunks_path": "duplicate.jsonl"}
                        ),
                        json.dumps({"repo": "o/c", "base_commit": "3", "status": "ok", "chunks_path": "three.jsonl"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            output = local_dir / "corpus" / "merged" / "corpus_manifest.jsonl"

            summary = merge_corpus_manifests(local_dir=local_dir, versions=["one", "two"], output=output)

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(summary["input_rows"], 4)
            self.assertEqual(summary["unique_snapshots"], 3)
            self.assertEqual(summary["duplicates"], 1)
            self.assertEqual(summary["inputs"], {"one": 2, "two": 2})
            self.assertEqual(
                [(row["repo"], row["base_commit"]) for row in rows],
                [("o/a", "1"), ("o/b", "2"), ("o/c", "3")],
            )

    def test_merge_corpus_manifests_reports_missing_release(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "download-benchmark --all"):
                merge_corpus_manifests(local_dir=Path(tmp), versions=["missing"])

    def test_merge_corpus_manifests_command_accepts_repeated_versions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            local_dir = Path(tmp) / "data"
            for version, repo in (("one", "o/a"), ("two", "o/b")):
                manifest = local_dir / "corpus" / version / "corpus_manifest.jsonl"
                manifest.parent.mkdir(parents=True)
                manifest.write_text(
                    json.dumps(
                        {
                            "repo": repo,
                            "base_commit": version,
                            "status": "ok",
                            "chunks_path": f"{version}.jsonl",
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
            output = local_dir / "combined.jsonl"
            stdout = StringIO()

            with redirect_stdout(stdout):
                result = main(
                    [
                        "merge-corpus-manifests",
                        "--version",
                        "one",
                        "--version",
                        "two",
                        "--local-dir",
                        str(local_dir),
                        "--out",
                        str(output),
                    ]
                )

            self.assertEqual(result, 0)
            self.assertEqual(json.loads(stdout.getvalue())["unique_snapshots"], 2)
            self.assertTrue(output.exists())

    @unittest.skipUnless(shutil.which("zstd"), "zstd is required for release extraction tests")
    def test_download_benchmark_release_verifies_and_extracts_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            self._write_release_tree(source)
            local_dir = root / "data"
            archive = self._write_bundle(source, local_dir, "v1")
            self._write_checksum(local_dir, archive)
            stale = local_dir / "benchmark" / "v1" / "stale.txt"
            stale.parent.mkdir(parents=True)
            stale.write_text("old")

            result = download_benchmark_release(version="v1", local_dir=local_dir, skip_download=True, force=True)

            self.assertFalse(stale.exists())
            self.assertEqual(result["version"], "v1")
            self.assertEqual(result["manifest"]["total"], 1)
            self.assertTrue((local_dir / "benchmark" / "v1" / "samples.jsonl").exists())
            self.assertTrue((local_dir / "corpus" / "v1" / "corpus_manifest.jsonl").exists())
            self.assertTrue((local_dir / "eval" / "v1" / "lexical_summary.json").exists())
            self.assertTrue((local_dir / "reports" / "v1" / "status.md").exists())

    def test_verify_release_checksum_rejects_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            local_dir = Path(tmp) / "data"
            release_dir = local_dir / "releases" / "v1"
            release_dir.mkdir(parents=True)
            archive = release_dir / "agent_retrieval_bench_v1.tar.zst"
            archive.write_bytes(b"archive")
            checksum = release_dir / "agent_retrieval_bench_v1.tar.zst.sha256"
            checksum.write_text("0" * 64 + "  releases/v1/agent_retrieval_bench_v1.tar.zst\n")

            with self.assertRaises(ValueError):
                verify_release_checksum(local_dir=local_dir, checksum_path=checksum)

    @unittest.skipUnless(shutil.which("zstd"), "zstd is required for release extraction tests")
    def test_download_canonical_release_uses_unversioned_archive_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            release_id = DEFAULT_RELEASE_ID
            source = root / "source"
            for name, filename in (
                ("benchmark", "manifest.json"),
                ("corpus", "corpus_manifest.jsonl"),
                ("eval", "lexical_summary.json"),
                ("reports", "status.md"),
            ):
                path = source / name / release_id
                path.mkdir(parents=True)
                path.joinpath(filename).write_text("{}\n")
            local_dir = root / "data"
            release_dir = local_dir / "releases" / release_id
            release_dir.mkdir(parents=True)
            tar_path = release_dir / "agent_retrieval_bench.tar"
            archive_path = release_dir / "agent_retrieval_bench.tar.zst"
            with tarfile.open(tar_path, "w") as tar:
                for name in ("benchmark", "corpus", "eval", "reports"):
                    tar.add(source / name / release_id, arcname=f"{name}/{release_id}")
            subprocess.run(["zstd", "-q", "-f", str(tar_path), "-o", str(archive_path)], check=True)
            tar_path.unlink()
            self._write_checksum(local_dir, archive_path)

            result = download_benchmark_release(local_dir=local_dir, skip_download=True, force=True)

            self.assertEqual(result["release_id"], release_id)
            self.assertTrue((local_dir / "benchmark" / release_id / "manifest.json").exists())
            self.assertEqual(archive_path.name, "agent_retrieval_bench.tar.zst")

    def _write_release_tree(self, source: Path) -> None:
        (source / "benchmark" / "v1").mkdir(parents=True)
        (source / "benchmark" / "v1" / "manifest.json").write_text(json.dumps({"total": 1}))
        (source / "benchmark" / "v1" / "samples.jsonl").write_text("{}\n")
        (source / "corpus" / "v1").mkdir(parents=True)
        (source / "corpus" / "v1" / "corpus_manifest.jsonl").write_text("{}\n")
        (source / "eval" / "v1").mkdir(parents=True)
        (source / "eval" / "v1" / "lexical_summary.json").write_text(json.dumps({"evaluated": 1, "skipped": {}}))
        (source / "reports" / "v1").mkdir(parents=True)
        (source / "reports" / "v1" / "status.md").write_text("# status\n")

    def _write_bundle(self, source: Path, local_dir: Path, version: str) -> Path:
        release_dir = local_dir / "releases" / version
        release_dir.mkdir(parents=True)
        tar_path = release_dir / "agent_retrieval_bench_v1.tar"
        archive_path = release_dir / "agent_retrieval_bench_v1.tar.zst"
        with tarfile.open(tar_path, "w") as tar:
            for name in ("benchmark", "corpus", "eval", "reports"):
                tar.add(source / name / version, arcname=f"{name}/{version}")
        subprocess.run(["zstd", "-q", "-f", str(tar_path), "-o", str(archive_path)], check=True)
        tar_path.unlink()
        return archive_path

    def _write_checksum(self, local_dir: Path, archive: Path) -> None:
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        checksum = archive.with_suffix(archive.suffix + ".sha256")
        checksum.write_text(f"{digest}  {archive.relative_to(local_dir)}\n")


if __name__ == "__main__":
    unittest.main()
