from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATASET_REPO = "eyuansu71/agent_retrieval_bench"
DEFAULT_RELEASE_ID = "agent_retrieval_bench"


@dataclass(frozen=True)
class ReleaseSpec:
    id: str
    samples: int | None
    description: str

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "samples": self.samples,
            "description": self.description,
        }


CURRENT_BENCHMARK_RELEASES = (
    ReleaseSpec("v2_code2test", 106, "Implementation change to regression-test retrieval."),
    ReleaseSpec("v2_comment2context", 80, "Review comment to missing repository context."),
    ReleaseSpec("v2_trace2code", 101, "Failure trace to root-cause implementation."),
    ReleaseSpec("v2_edit2ripple", 58, "Anchored edit to additional affected files."),
    ReleaseSpec("v2_abstention", 82, "Natural and counterfactual no-gold cases."),
)

AUXILIARY_RELEASES = (
    ReleaseSpec("v2_selective_retrieval_balanced", None, "Balanced positive/no-gold selective-evaluation input."),
    ReleaseSpec("v2_selective_retrieval_natural", None, "Natural-prevalence selective-evaluation input."),
)


def release_catalog() -> dict:
    return {
        "dataset_repo": DEFAULT_DATASET_REPO,
        "current": [release.as_dict() for release in CURRENT_BENCHMARK_RELEASES],
        "auxiliary": [release.as_dict() for release in AUXILIARY_RELEASES],
    }


def release_archive_stem(release_id: str) -> str:
    if release_id == DEFAULT_RELEASE_ID:
        return DEFAULT_RELEASE_ID
    return f"agent_retrieval_bench_{release_id}"


def download_benchmark_release(
    *,
    version: str = DEFAULT_RELEASE_ID,
    local_dir: Path = Path("data"),
    repo_id: str = DEFAULT_DATASET_REPO,
    revision: str | None = None,
    hf_token: str | None = None,
    skip_download: bool = False,
    no_extract: bool = False,
    force: bool = False,
    hf_bin: str = "hf",
    zstd_bin: str = "zstd",
    tar_bin: str = "tar",
) -> dict:
    release_name = release_archive_stem(version)
    release_dir = local_dir / "releases" / version
    archive_path = release_dir / f"{release_name}.tar.zst"
    checksum_path = release_dir / f"{release_name}.tar.zst.sha256"

    if not skip_download:
        _download_release_bundle(
            repo_id=repo_id,
            version=version,
            local_dir=local_dir,
            revision=revision,
            hf_token=hf_token,
            hf_bin=hf_bin,
        )

    digest = verify_release_checksum(local_dir=local_dir, checksum_path=checksum_path)

    extracted = []
    if not no_extract:
        extracted = _extract_release_bundle(
            local_dir=local_dir,
            version=version,
            archive_path=archive_path,
            force=force,
            zstd_bin=zstd_bin,
            tar_bin=tar_bin,
        )

    result = {
        "version": version,
        "release_id": version,
        "repo_id": repo_id,
        "local_dir": str(local_dir),
        "archive": str(archive_path),
        "checksum": str(checksum_path),
        "sha256": digest,
        "downloaded": not skip_download,
        "extracted": extracted,
    }
    manifest_path = local_dir / "benchmark" / version / "manifest.json"
    if manifest_path.exists():
        result["manifest"] = json.loads(manifest_path.read_text())
    return result


def merge_corpus_manifests(
    *,
    local_dir: Path = Path("data"),
    versions: list[str] | tuple[str, ...] | None = None,
    output: Path | None = None,
) -> dict:
    selected_versions = list(versions or [release.id for release in CURRENT_BENCHMARK_RELEASES])
    if not selected_versions:
        raise ValueError("At least one release version is required.")
    if len(selected_versions) != len(set(selected_versions)):
        raise ValueError("Release versions must be unique.")

    output = output or local_dir / "corpus" / "v2_selective_mixed" / "corpus_manifest.jsonl"
    rows_by_snapshot: dict[tuple[str, str], dict] = {}
    input_counts: dict[str, int] = {}
    duplicate_count = 0

    for version in selected_versions:
        manifest_path = local_dir / "corpus" / version / "corpus_manifest.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing corpus manifest for {version}: {manifest_path}. "
                "Download the current subsets with `arb download-benchmark --all` first."
            )
        count = 0
        for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            repo = row.get("repo")
            base_commit = row.get("base_commit")
            if not repo or not base_commit:
                raise ValueError(f"Invalid corpus row at {manifest_path}:{line_number}: missing repo or base_commit")
            count += 1
            key = (str(repo), str(base_commit))
            existing = rows_by_snapshot.get(key)
            if existing is not None:
                duplicate_count += 1
                if existing.get("status") != "ok" and row.get("status") == "ok":
                    rows_by_snapshot[key] = row
                continue
            rows_by_snapshot[key] = row
        input_counts[version] = count

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for key in sorted(rows_by_snapshot):
            handle.write(json.dumps(rows_by_snapshot[key], ensure_ascii=False) + "\n")
    temporary.replace(output)

    return {
        "output": str(output),
        "versions": selected_versions,
        "inputs": input_counts,
        "input_rows": sum(input_counts.values()),
        "unique_snapshots": len(rows_by_snapshot),
        "duplicates": duplicate_count,
    }


def verify_release_checksum(*, local_dir: Path, checksum_path: Path) -> str:
    if not checksum_path.exists():
        raise FileNotFoundError(f"Missing checksum file: {checksum_path}")

    line = checksum_path.read_text().strip().splitlines()[0]
    parts = line.split()
    if len(parts) < 2:
        raise ValueError(f"Invalid checksum line in {checksum_path}: {line!r}")
    expected, relative_path = parts[0], parts[1]
    archive_path = local_dir / relative_path
    if not archive_path.exists():
        raise FileNotFoundError(f"Checksum target is missing: {archive_path}")

    actual = _sha256_file(archive_path)
    if actual.lower() != expected.lower():
        raise ValueError(f"Checksum mismatch for {archive_path}: expected {expected}, got {actual}")
    return actual


def _download_release_bundle(
    *,
    repo_id: str,
    version: str,
    local_dir: Path,
    revision: str | None,
    hf_token: str | None,
    hf_bin: str,
) -> None:
    command = [
        hf_bin,
        "download",
        repo_id,
        "--repo-type",
        "dataset",
        "--local-dir",
        str(local_dir),
        "--include",
        f"releases/{version}/*",
    ]
    if revision:
        command.extend(["--revision", revision])
    env = os.environ.copy()
    if hf_token:
        env["HF_TOKEN"] = hf_token
    subprocess.run(command, check=True, env=env)


def _extract_release_bundle(
    *,
    local_dir: Path,
    version: str,
    archive_path: Path,
    force: bool,
    zstd_bin: str,
    tar_bin: str,
) -> list[str]:
    if not archive_path.exists():
        raise FileNotFoundError(f"Missing release archive: {archive_path}")

    targets = [local_dir / name / version for name in ("benchmark", "corpus", "eval", "reports")]
    existing = [path for path in targets if path.exists()]
    if existing and not force:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing extracted directories without --force: {names}")
    for path in existing:
        shutil.rmtree(path)

    local_dir.mkdir(parents=True, exist_ok=True)
    zstd = subprocess.Popen([zstd_bin, "-dc", str(archive_path)], stdout=subprocess.PIPE)
    try:
        tar = subprocess.run([tar_bin, "-xf", "-", "-C", str(local_dir)], stdin=zstd.stdout, check=False)
    finally:
        if zstd.stdout:
            zstd.stdout.close()
    zstd_return = zstd.wait()
    if zstd_return:
        raise subprocess.CalledProcessError(zstd_return, [zstd_bin, "-dc", str(archive_path)])
    if tar.returncode:
        raise subprocess.CalledProcessError(tar.returncode, [tar_bin, "-xf", "-", "-C", str(local_dir)])
    return [str(path) for path in targets if path.exists()]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
