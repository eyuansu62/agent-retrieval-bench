from __future__ import annotations

import re
from pathlib import PurePosixPath

GENERATED_PATTERNS = (
    "dist/",
    "build/",
    "vendor/",
    "third_party/",
    "node_modules/",
    "target/",
    ".tox/",
    ".venv/",
)

GENERATED_SUFFIXES = (
    ".lock",
    ".min.js",
    ".map",
    ".snap",
    ".snapshot",
    ".pb.go",
    ".generated.go",
    ".g.cs",
)

LOCKFILE_NAMES = {
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "poetry.lock",
    "Pipfile.lock",
    "Cargo.lock",
    "Gemfile.lock",
    "go.sum",
}

TEST_PATH_PARTS = {
    "test",
    "tests",
    "__tests__",
    "spec",
    "specs",
    "it",
    "integration-test",
    "integration-tests",
}

TEST_NAME_PATTERNS = (
    re.compile(r"(^|[_\-.])(test|spec)([_\-.]|$)", re.IGNORECASE),
    re.compile(r"(test|spec)\.(py|ts|tsx|js|jsx|java|go|rs)$", re.IGNORECASE),
    re.compile(r"(_test\.go|_test\.rs)$", re.IGNORECASE),
    re.compile(r"(Test|Tests)\.java$"),
)

SOURCE_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".go",
    ".rs",
    ".kt",
    ".scala",
}


def is_generated_or_lockfile(path: str) -> bool:
    normalized = path.replace("\\", "/")
    name = PurePosixPath(normalized).name
    if name in LOCKFILE_NAMES:
        return True
    if any(part in normalized for part in GENERATED_PATTERNS):
        return True
    return normalized.endswith(GENERATED_SUFFIXES)


def is_source_file(path: str) -> bool:
    return PurePosixPath(path).suffix in SOURCE_EXTENSIONS and not is_generated_or_lockfile(path)


def is_test_file(path: str) -> bool:
    normalized = path.replace("\\", "/")
    if is_generated_or_lockfile(normalized):
        return False
    parts = {part.lower() for part in PurePosixPath(normalized).parts}
    if parts & TEST_PATH_PARTS:
        return True
    return any(pattern.search(normalized) for pattern in TEST_NAME_PATTERNS)


def split_changed_files(paths: list[str]) -> tuple[list[str], list[str], list[str]]:
    implementation: list[str] = []
    tests: list[str] = []
    ignored: list[str] = []
    for path in paths:
        if is_generated_or_lockfile(path):
            ignored.append(path)
        elif is_test_file(path):
            tests.append(path)
        elif is_source_file(path):
            implementation.append(path)
        else:
            ignored.append(path)
    return implementation, tests, ignored


def should_skip_pr(changed_paths: list[str], max_changed_files: int = 20) -> bool:
    if not changed_paths or len(changed_paths) > max_changed_files:
        return True
    implementation, tests, _ignored = split_changed_files(changed_paths)
    return not implementation and not tests


def sanitize_diff_hunk(diff_hunk: str | None, max_lines: int = 16) -> str:
    if not diff_hunk:
        return ""
    kept: list[str] = []
    for line in diff_hunk.replace("\r\n", "\n").splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+") or line.startswith("-"):
            continue
        kept.append(line[:240])
        if len(kept) >= max_lines:
            break
    return "\n".join(kept)


def sanitize_review_body(body: str | None, limit: int = 2200) -> str:
    if not body:
        return ""
    sanitized_lines: list[str] = []
    in_removed_fence = False
    for line in body.replace("\r\n", "\n").splitlines():
        stripped = line.strip()
        if stripped.startswith("```diff") or stripped.startswith("```suggestion"):
            in_removed_fence = True
            continue
        if in_removed_fence:
            if stripped.startswith("```"):
                in_removed_fence = False
            continue
        if stripped.startswith("```"):
            sanitized_lines.append(line)
            continue
        if stripped.startswith(("diff --git", "index ", "--- ", "+++ ", "@@ ")):
            continue
        if line.startswith(("+", "-")):
            continue
        sanitized_lines.append(line)
    sanitized = "\n".join(sanitized_lines).strip()
    if len(sanitized) > limit:
        return sanitized[:limit].rstrip() + "\n...[truncated]"
    return sanitized


TRACE_LINE_RE = re.compile(
    r"(?P<path>(?:[A-Za-z]:)?[^\s:'\"]+\.(?:py|js|jsx|ts|tsx|java|go|rs|kt|scala))"
    r"[\"']?(?::|, line |\()(?P<line>\d+)?",
    re.IGNORECASE,
)

RAW_PATCH_MARKER_RE = re.compile(r"diff --git\b|(^|\n)(---\s+(?:a/|/dev/null)|\+\+\+\s+(?:b/|/dev/null))")


def extract_repo_trace_paths(text: str) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for match in TRACE_LINE_RE.finditer(text):
        path = match.group("path").replace("\\", "/")
        if "site-packages/" in path or "node_modules/" in path or "/.cargo/" in path:
            continue
        if path not in seen:
            paths.append(path)
            seen.add(path)
    return paths


def contains_raw_patch_marker(text: str) -> bool:
    return bool(RAW_PATCH_MARKER_RE.search(text.replace("\\n", "\n")))


def contains_review_leakage(text: str) -> bool:
    normalized = text.replace("\r\n", "\n")
    lowered = normalized.lower()
    return (
        "```suggestion" in lowered
        or "git --no-pager diff" in lowered
        or "<summary>🧩 analysis chain</summary>" in lowered
        or "learnings added" in lowered
        or "learnings used" in lowered
        or "learnt from:" in lowered
        or "auto-generated reply by coderabbit" in lowered
    )


def is_ignored_check_signal(check_name: str | None, raw_signal: str | None) -> bool:
    text = f"{check_name or ''}\n{raw_signal or ''}".lower()
    ignored_markers = ("dependabot", "codecov", "dco", "developer certificate of origin")
    return any(marker in text for marker in ignored_markers)


def is_job_name_only_signal(check_name: str | None, raw_signal: str | None) -> bool:
    normalized_signal = _compact(raw_signal or "")
    normalized_name = _compact(check_name or "")
    if not normalized_signal:
        return True
    if normalized_name and normalized_signal == normalized_name:
        return True
    return "\n" not in (raw_signal or "") and len(normalized_signal) < 120 and not has_failure_or_trace_signal(normalized_signal)


def has_failure_or_trace_signal(text: str) -> bool:
    normalized = text.replace("\\n", "\n")
    patterns = (
        r"\bFAILED?\s+[^\s]+::[A-Za-z0-9_.$:-]+",
        r"\bFAIL(?:ED)?\b",
        r"\bTraceback\b",
        r"\bAssertionError\b",
        r"\bRuntimeError\b",
        r"\bException\b",
        r"\bpanic\b",
        r"\bstack backtrace\b",
        r"\bError Trace:\b",
        r"\bCaused by\b",
        r"\bTest:\s*[A-Za-z0-9_.$:-]+",
        r"\.(?:py|js|jsx|ts|tsx|java|go|rs|kt|scala):\d+",
    )
    return any(re.search(pattern, normalized, re.IGNORECASE) for pattern in patterns)


def is_runner_setup_noise(text: str) -> bool:
    lowered = text.lower()
    setup_markers = (
        "current runner version",
        "runner image",
        "hosted compute agent",
        "actions/cache",
        "cache hit for:",
        "setup-python",
        "version ",
        "downloaded",
        "extract downloaded archive",
        "postinstall",
        "node_modules",
        "go: downloading",
    )
    return any(marker in lowered for marker in setup_markers) and not has_failure_or_trace_signal(text)


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
