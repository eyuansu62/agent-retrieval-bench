from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path

from agent_retrieval_bench.release import download_benchmark_release, verify_release_checksum


class ReleaseDownloadTests(unittest.TestCase):
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
