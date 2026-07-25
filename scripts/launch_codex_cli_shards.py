#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


BENCH_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch sharded Codex CLI ARB trajectory runs.")
    parser.add_argument("--samples", type=Path, default=Path("data/benchmark/v1_3/samples.jsonl"))
    parser.add_argument("--run-root", type=Path, default=Path("data/trajectory_runs/v1_4"))
    parser.add_argument("--run-prefix", default="codex_cli_gpt55_corpus_full_budgeted")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    sample_ids = load_sample_ids(resolve(args.samples))
    shards = [[] for _ in range(args.shards)]
    for index, sample_id in enumerate(sample_ids):
        shards[index % args.shards].append(sample_id)

    run_root = resolve(args.run_root)
    launched = []
    for shard_index, ids in enumerate(shards):
        shard_name = f"{args.run_prefix}_shard{shard_index:02d}"
        run_dir = run_root / shard_name
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "sample_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
        cmd = [
            "bash",
            "-lc",
            "source ~/.nvm/nvm.sh && "
            f"cd {shell_quote(str(BENCH_ROOT))} && "
            "PYTHONPATH=src python scripts/run_codex_cli_full.py "
            f"--run-dir {shell_quote(str(run_dir))} "
            f"--run-name {shell_quote(shard_name)} "
            f"--model {shell_quote(args.model)} "
            f"--timeout-seconds {args.timeout_seconds} "
            + ("--force " if args.force else "")
            + " ".join(f"--sample-id {shell_quote(sample_id)}" for sample_id in ids),
        ]
        stdout = (run_dir / "runner.out").open("a", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            cwd=BENCH_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        (run_dir / "runner.pid").write_text(str(proc.pid) + "\n", encoding="utf-8")
        launched.append({"shard": shard_index, "run_dir": str(run_dir), "pid": proc.pid, "samples": len(ids)})

    manifest = {
        "mode": "codex_cli_sharded_launcher",
        "model": args.model,
        "run_prefix": args.run_prefix,
        "shards": args.shards,
        "timeout_seconds": args.timeout_seconds,
        "sample_count": len(sample_ids),
        "launched": launched,
        "launched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    manifest_path = run_root / f"{args.run_prefix}_launcher.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else BENCH_ROOT / path


def load_sample_ids(path: Path) -> list[str]:
    ids = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("id") or "")
            if sample_id:
                ids.append(sample_id)
    return ids


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


if __name__ == "__main__":
    raise SystemExit(main())
