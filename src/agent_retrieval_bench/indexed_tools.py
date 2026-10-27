"""Indexed lookups that an external provider adds to the closed-tool loop.

The ``indexed_interactive`` mode of ``closed_tool_eval`` offers the policy
``symbol_search`` and ``references`` next to ``list_dir``, ``grep``,
``read_file`` and ``submit``. ARB does not depend on any index library. The
lookups come from a provider object, loaded from a ``module:factory`` spec,
with two methods:

``prepare(repo=, base_commit=, paths=) -> dict``
    Build or load the index for one repository snapshot. ``paths`` is the
    released file list. The result must set ``available`` and may report
    ``build_seconds``, ``cached``, ``size`` and ``reason``. The evaluator calls
    it once per snapshot in a run.

``call(tool=, arguments=, repo=, base_commit=, limit=) -> list[dict]``
    Run one lookup. ``symbol_search`` receives ``{"symbol"}`` and returns
    definitions with that name. ``references`` receives ``{"path", "symbol"}``
    and returns locations that reference, or are referenced by, that symbol.
    Each hit has ``path`` and may have ``start_line`` and ``end_line``
    (1-based, inclusive), ``symbol``, ``kind`` and ``relation``.

An optional ``describe() -> dict`` identifies the provider build in the run
summary. The provider never receives the sample id, query or gold. The
evaluator drops hits outside the released files, caps hit counts and
observation tokens, and charges each answered lookup as one tool call.
Lookups return locations only; content still costs ``read_file`` calls and
read tokens.
"""

from __future__ import annotations

import importlib
import time
from typing import Any, Iterable

from .bcy_curve import count_tokens

INDEXED_TOOLS = ("symbol_search", "references")
DEFAULT_MAX_INDEX_RESULTS = 10
DEFAULT_MAX_INDEX_OBSERVATION_TOKENS = 400
MAX_FIELD_CHARS = 120
# Echoed arguments shrink before a lookup's status line would break the cap:
# in full, cut to this many characters, then not at all.
SHORT_ECHO_CHARS = 24

TOOL_DESCRIPTIONS = {
    "symbol_search": "symbol_search: definitions named `symbol`, as path and line range.",
    "references": (
        "references: locations that reference, or are referenced by, `symbol` defined in `path`, "
        "as path and line range."
    ),
}
TOOL_EXAMPLES = {
    "symbol_search": (
        '{"action":"symbol_search","path":null,"pattern":null,"query":null,"symbol":"parse_config",'
        '"top_k":10,"files":[],"context_files":[]}'
    ),
    "references": (
        '{"action":"references","path":"src/config.py","pattern":null,"query":null,"symbol":"parse_config",'
        '"top_k":10,"files":[],"context_files":[]}'
    ),
}


def load_tool_provider(spec: str, options: dict[str, str] | None = None) -> Any:
    module_name, _, attribute = spec.partition(":")
    if not module_name or not attribute:
        raise ValueError(f"tool provider must be module:callable, got {spec!r}")
    factory = getattr(importlib.import_module(module_name), attribute)
    return factory(**(options or {}))


def parse_provider_options(values: Iterable[str]) -> dict[str, str]:
    options = {}
    for value in values:
        key, sep, item = value.partition("=")
        if not key or not sep:
            raise ValueError(f"provider option must be key=value, got {value!r}")
        options[key] = item
    return options


class IndexedTools:
    """Evaluator-side gate around one provider for one run."""

    def __init__(
        self,
        provider: Any,
        tools: Iterable[str],
        *,
        max_results: int = DEFAULT_MAX_INDEX_RESULTS,
        max_observation_tokens: int = DEFAULT_MAX_INDEX_OBSERVATION_TOKENS,
        provider_configuration: dict[str, Any] | None = None,
    ) -> None:
        requested = set(tools)
        unknown = requested - set(INDEXED_TOOLS)
        if unknown or not requested:
            raise ValueError(f"indexed tools must be a non-empty subset of {INDEXED_TOOLS}, got {sorted(requested)}")
        if max_results <= 0 or max_observation_tokens <= 0:
            raise ValueError("indexed-tool result and observation caps must be positive")
        for tool in requested:
            needed = minimal_observation_tokens(tool)
            if needed > max_observation_tokens:
                raise ValueError(
                    f"max_observation_tokens={max_observation_tokens} cannot hold a minimal {tool} "
                    f"response ({needed} tokens)"
                )
        self.provider = provider
        self.tools = tuple(tool for tool in INDEXED_TOOLS if tool in requested)
        self.max_results = max_results
        self.max_observation_tokens = max_observation_tokens
        self.provider_configuration = provider_configuration
        self.snapshots: dict[tuple[str, str], dict[str, Any]] = {}

    def describe(self) -> dict[str, Any]:
        describe = getattr(self.provider, "describe", None)
        return {
            "tools": list(self.tools),
            "max_results": self.max_results,
            "max_observation_tokens": self.max_observation_tokens,
            "provider": dict(describe()) if callable(describe) else {"class": type(self.provider).__qualname__},
        }

    def prepare(self, *, repo: str, base_commit: str, files: dict[str, str]) -> dict[str, Any]:
        key = (repo, base_commit)
        if key not in self.snapshots:
            started = time.monotonic()
            try:
                status = dict(self.provider.prepare(repo=repo, base_commit=base_commit, paths=sorted(files)) or {})
            except Exception as exc:  # A broken index leaves the minimal tools in place.
                status = {"reason": f"{type(exc).__name__}: {exc}"[:300]}
            status["available"] = status.get("available") is True
            status["prepare_seconds"] = time.monotonic() - started
            self.snapshots[key] = {"repo": repo, "base_commit": base_commit, **status}
        return self.snapshots[key]

    def lookup(
        self,
        tool: str,
        action: dict[str, Any],
        *,
        repo: str,
        base_commit: str,
        files: dict[str, str],
    ) -> dict[str, Any]:
        """Run one lookup; ``error`` means the action is rejected and not charged."""
        symbol = str(action.get("symbol") or action.get("query") or action.get("pattern") or "").strip()
        path = str(action.get("path") or "").strip()
        if not symbol or len(symbol) > MAX_FIELD_CHARS:
            return {"error": f"{tool} requires a symbol of 1 to {MAX_FIELD_CHARS} characters."}
        if tool == "references" and path not in files:
            return {"error": f"references path not found: {shorten(path, MAX_FIELD_CHARS)}"}
        arguments = {"symbol": symbol} if tool == "symbol_search" else {"path": path, "symbol": symbol}
        limit = clamp_limit(action.get("top_k"), self.max_results)
        started = time.monotonic()
        provider_error = ""
        try:
            raw = self.provider.call(tool=tool, arguments=dict(arguments), repo=repo, base_commit=base_commit, limit=limit)
        except Exception as exc:  # The lookup was made; its failure is the observation.
            raw, provider_error = [], f"{type(exc).__name__}: {exc}"[:300]
        latency = time.monotonic() - started
        hits, outside = released_hits(raw, files)
        observation, shown = fit_observation(
            tool, arguments, hits, hits[:limit], provider_error, self.max_observation_tokens
        )
        return {
            "error": "",
            "observation": observation,
            "trace": {
                "tool": tool,
                **arguments,
                "limit": limit,
                "hits": shown,
                "returned": len(raw) if isinstance(raw, list) else 0,
                "dropped_outside_release": outside,
                "omitted": len(hits) - len(shown),
                "observation_tokens": count_tokens(observation),
                "latency_seconds": latency,
                "provider_error": provider_error,
            },
        }


def clamp_limit(value: Any, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = maximum
    return max(1, min(maximum, parsed))


def released_hits(raw: Any, files: dict[str, str]) -> tuple[list[dict[str, Any]], int]:
    """Valid hits on released files, deduplicated in provider order."""
    hits: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    outside = 0
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            continue
        if item["path"] not in files:
            outside += 1
            continue
        start, end = line_number(item.get("start_line")), line_number(item.get("end_line"))
        if start is None or (end is not None and end < start):
            start, end = None, None
        hit = {
            "path": item["path"],
            "start_line": start,
            "end_line": end if start is not None else None,
            "symbol": field_text(item.get("symbol")),
            "kind": field_text(item.get("kind")),
            "relation": field_text(item.get("relation")),
        }
        key = tuple(hit.values())
        if key not in seen:
            seen.add(key)
            hits.append(hit)
    return hits, outside


def line_number(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 1 else None


def field_text(value: Any) -> str:
    return " ".join(str(value).split())[:MAX_FIELD_CHARS] if isinstance(value, str) else ""


def fit_observation(
    tool: str,
    arguments: dict[str, str],
    hits: list[dict[str, Any]],
    shown: list[dict[str, Any]],
    provider_error: str,
    cap: int,
) -> tuple[str, list[dict[str, Any]]]:
    """The complete observation within ``cap`` tokens, and the hits it shows.

    The fullest echo of the arguments whose hit-free response fits is kept,
    then hits are trimmed from the end. The constructor guarantees that the
    response without echo or counts fits.
    """
    for echo in (None, SHORT_ECHO_CHARS, 0):
        header = observation_header(tool, arguments, echo)
        if count_tokens(format_index_observation(header, [], len(hits), provider_error)) <= cap:
            break
    else:
        header = observation_header(tool, arguments, 0)
        return format_index_observation(header, [], len(hits), provider_error, counts=False), []
    observation = format_index_observation(header, shown, len(hits) - len(shown), provider_error)
    while shown and count_tokens(observation) > cap:
        shown = shown[:-1]
        observation = format_index_observation(header, shown, len(hits) - len(shown), provider_error)
    return observation, shown


def minimal_observation_tokens(tool: str) -> int:
    """Tokens of the largest response without echoed arguments or counts."""
    header = observation_header(tool, {}, 0)
    return max(
        count_tokens(format_index_observation(header, [], omitted, error, counts=False))
        for omitted, error in ((0, ""), (1, ""), (0, "failed"))
    )


def observation_header(tool: str, arguments: dict[str, str], echo: int | None) -> str:
    """``OBS tool`` and its arguments, each cut to ``echo`` characters (all, if None)."""
    if echo == 0:
        return f"OBS {tool}"
    shown = {key: value if echo is None else shorten(value, echo) for key, value in arguments.items()}
    return f"OBS {tool} " + " ".join(f"{key}={value!r}" for key, value in shown.items())


def shorten(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: max(limit - 1, 0)] + "…"


def format_index_observation(
    header: str,
    hits: list[dict[str, Any]],
    omitted: int,
    provider_error: str,
    *,
    counts: bool = True,
) -> str:
    if provider_error:
        return f"{header}\n(index lookup failed)"
    if not hits:
        if not omitted:
            return f"{header}\n(no hits)"
        detail = f"{omitted} hits" if counts else "hits"
        return f"{header}\n({detail} omitted to fit the observation cap)"
    lines = [header]
    for index, hit in enumerate(hits, start=1):
        span = ""
        if hit["start_line"] is not None:
            span = f":{hit['start_line']}" if hit["end_line"] in (None, hit["start_line"]) else (
                f":{hit['start_line']}-{hit['end_line']}"
            )
        label = " ".join(part for part in (hit["kind"], hit["symbol"]) if part)
        relation = f" [{hit['relation']}]" if hit["relation"] else ""
        lines.append(f"{index}. {hit['path']}{span}" + (f" {label}" if label else "") + relation)
    if omitted:
        lines.append(f"... {omitted} more hits omitted")
    return "\n".join(lines)
