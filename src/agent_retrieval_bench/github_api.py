from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from http.client import HTTPResponse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import read_json, write_json


@dataclass(frozen=True)
class GitHubResponse:
    body: Any
    headers: dict[str, str]
    status: int
    from_cache: bool = False


@dataclass(frozen=True)
class GitHubBytesResponse:
    body: bytes
    headers: dict[str, str]
    status: int


class GitHubAPI:
    def __init__(
        self,
        token: str | None = None,
        cache_dir: Path | None = None,
        rest_base: str = "https://api.github.com",
        graphql_url: str = "https://api.github.com/graphql",
        min_interval_seconds: float = 0.15,
    ) -> None:
        self.token = token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        self.cache_dir = cache_dir
        self.rest_base = rest_base.rstrip("/")
        self.graphql_url = graphql_url
        self.min_interval_seconds = min_interval_seconds
        self._last_request = 0.0

    @property
    def authenticated(self) -> bool:
        return bool(self.token)

    def graphql(self, query: str, variables: dict[str, Any] | None = None) -> GitHubResponse:
        payload = {"query": query, "variables": variables or {}}
        response = self._request("POST", self.graphql_url, body=payload, use_cache=False)
        if isinstance(response.body, dict) and response.body.get("errors"):
            messages = "; ".join(error.get("message", str(error)) for error in response.body["errors"])
            raise RuntimeError(f"GitHub GraphQL error: {messages}")
        return response

    def get(self, path: str, params: dict[str, Any] | None = None, accept: str | None = None) -> GitHubResponse:
        url = path if path.startswith("http") else f"{self.rest_base}{path}"
        if params:
            encoded = urllib.parse.urlencode({key: value for key, value in params.items() if value is not None})
            url = f"{url}?{encoded}"
        return self._request("GET", url, accept=accept, use_cache=True)

    def get_bytes(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        accept: str | None = None,
        max_bytes: int | None = None,
    ) -> GitHubBytesResponse:
        url = path if path.startswith("http") else f"{self.rest_base}{path}"
        if params:
            encoded = urllib.parse.urlencode({key: value for key, value in params.items() if value is not None})
            url = f"{url}?{encoded}"
        return self._request_bytes("GET", url, accept=accept, max_bytes=max_bytes)

    def paginate(self, path: str, params: dict[str, Any] | None = None, accept: str | None = None) -> list[Any]:
        page = 1
        items: list[Any] = []
        while True:
            page_params = dict(params or {})
            page_params.setdefault("per_page", 100)
            page_params["page"] = page
            response = self.get(path, page_params, accept=accept)
            if not isinstance(response.body, list):
                raise RuntimeError(f"Expected list from {path}, got {type(response.body).__name__}")
            items.extend(response.body)
            link = response.headers.get("link", "")
            if 'rel="next"' not in link or not response.body:
                return items
            page += 1

    def _request(
        self,
        method: str,
        url: str,
        body: dict[str, Any] | None = None,
        accept: str | None = None,
        use_cache: bool = False,
    ) -> GitHubResponse:
        cache_key = self._cache_key(method, url, body)
        cache_value = read_json(self.cache_dir / f"{cache_key}.json", None) if self.cache_dir and use_cache else None
        headers = {
            "Accept": accept or "application/vnd.github+json",
            "User-Agent": "agent-retrieval-bench/0.1",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if cache_value and cache_value.get("etag"):
            headers["If-None-Match"] = cache_value["etag"]

        encoded_body = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            encoded_body = json.dumps(body).encode("utf-8")

        backoff = 2.0
        for attempt in range(6):
            self._throttle()
            request = urllib.request.Request(url, data=encoded_body, headers=headers, method=method)
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    raw = response.read()
                    parsed = json.loads(raw.decode("utf-8")) if raw else None
                    response_headers = {key.lower(): value for key, value in response.headers.items()}
                    if self.cache_dir and use_cache:
                        write_json(
                            self.cache_dir / f"{cache_key}.json",
                            {"etag": response_headers.get("etag"), "body": parsed, "url": url},
                        )
                    return GitHubResponse(parsed, response_headers, response.status)
            except urllib.error.HTTPError as error:
                if error.code == 304 and cache_value:
                    return GitHubResponse(cache_value["body"], {"x-cache": "hit"}, 304, from_cache=True)
                raw_error = error.read().decode("utf-8", errors="replace")
                if self._should_retry(error, raw_error):
                    self._sleep_for_limit(error, raw_error, backoff)
                    backoff *= 2
                    continue
                raise RuntimeError(f"GitHub API {method} {url} failed: {error.code} {raw_error}") from error
            except (TimeoutError, urllib.error.URLError) as error:
                if attempt == 5:
                    raise RuntimeError(f"GitHub API {method} {url} failed after retries: {error}") from error
                time.sleep(backoff)
                backoff *= 2
        raise RuntimeError(f"GitHub API {method} {url} failed after retries")

    def _request_bytes(
        self,
        method: str,
        url: str,
        accept: str | None = None,
        max_bytes: int | None = None,
    ) -> GitHubBytesResponse:
        headers = {
            "Accept": accept or "application/vnd.github+json",
            "User-Agent": "agent-retrieval-bench/0.1",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        backoff = 2.0
        opener = urllib.request.build_opener(_NoRedirectHandler)
        for attempt in range(6):
            self._throttle()
            request = urllib.request.Request(url, headers=headers, method=method)
            try:
                response = opener.open(request, timeout=120)
                if response.status in {301, 302, 303, 307, 308}:
                    location = response.headers.get("Location")
                    if not location:
                        raise RuntimeError(f"GitHub API {method} {url} redirected without Location")
                    response.close()
                    return self._download_redirect(location, max_bytes)
                with response:
                    return self._read_bytes_response(response, max_bytes)
            except urllib.error.HTTPError as error:
                if error.code in {301, 302, 303, 307, 308}:
                    location = error.headers.get("Location")
                    if location:
                        return self._download_redirect(location, max_bytes)
                raw_error = error.read().decode("utf-8", errors="replace")
                if self._should_retry(error, raw_error):
                    self._sleep_for_limit(error, raw_error, backoff)
                    backoff *= 2
                    continue
                raise RuntimeError(f"GitHub API {method} {url} failed: {error.code} {raw_error}") from error
            except (TimeoutError, urllib.error.URLError) as error:
                if attempt == 5:
                    raise RuntimeError(f"GitHub API {method} {url} failed after retries: {error}") from error
                time.sleep(backoff)
                backoff *= 2
        raise RuntimeError(f"GitHub API {method} {url} failed after retries")

    def _download_redirect(self, location: str, max_bytes: int | None) -> GitHubBytesResponse:
        redirected = urllib.request.Request(location, headers={"User-Agent": "agent-retrieval-bench/0.1"}, method="GET")
        with urllib.request.urlopen(redirected, timeout=120) as response:
            return self._read_bytes_response(response, max_bytes)

    @staticmethod
    def _read_bytes_response(response: HTTPResponse, max_bytes: int | None) -> GitHubBytesResponse:
        raw = response.read(max_bytes + 1 if max_bytes else -1)
        response_headers = {key.lower(): value for key, value in response.headers.items()}
        return GitHubBytesResponse(raw, response_headers, response.status)

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - elapsed)
        self._last_request = time.monotonic()

    def _sleep_for_limit(self, error: urllib.error.HTTPError, raw_error: str, backoff: float) -> None:
        reset = error.headers.get("x-ratelimit-reset")
        remaining = error.headers.get("x-ratelimit-remaining")
        if remaining == "0" and reset:
            sleep_for = max(1.0, float(reset) - time.time() + 3.0)
        elif "secondary rate limit" in raw_error.lower() or "abuse" in raw_error.lower():
            sleep_for = max(backoff, 30.0)
        else:
            sleep_for = backoff
        time.sleep(sleep_for)

    @staticmethod
    def _should_retry(error: urllib.error.HTTPError, raw_error: str) -> bool:
        if error.code in {500, 502, 503, 504}:
            return True
        if error.code in {403, 429}:
            lowered = raw_error.lower()
            return "rate limit" in lowered or "abuse" in lowered or error.headers.get("x-ratelimit-remaining") == "0"
        return False

    @staticmethod
    def _cache_key(method: str, url: str, body: dict[str, Any] | None) -> str:
        raw = json.dumps({"method": method, "url": url, "body": body}, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None
