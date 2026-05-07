from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


DEFAULT_DATASET_REPO = "eyuansu71/agent_retrieval_bench"


def download_benchmark_release(
    *,
    version: str = "v1",
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
    release_name = f"agent_retrieval_bench_{version}"
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
