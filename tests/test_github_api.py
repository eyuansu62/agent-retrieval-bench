import time
import unittest
import urllib.error
from email.message import Message

from agent_retrieval_bench.github_api import GitHubAPI


def http_error(headers: dict[str, str]) -> urllib.error.HTTPError:
    message = Message()
    for key, value in headers.items():
        message[key] = value
    return urllib.error.HTTPError("https://api.github.test", 403, "rate limited", message, None)


class GitHubAPITests(unittest.TestCase):
    def test_rate_limit_sleep_cap_fails_fast(self):
        api = GitHubAPI(max_rate_limit_sleep_seconds=0)
        error = http_error({"x-ratelimit-remaining": "0", "x-ratelimit-reset": str(int(time.time()) + 3600)})

        with self.assertRaisesRegex(RuntimeError, "rate limit wait exceeds configured maximum"):
            api._sleep_for_limit(error, "API rate limit exceeded", backoff=2.0)

    def test_secondary_rate_limit_sleep_cap_fails_fast(self):
        api = GitHubAPI(max_rate_limit_sleep_seconds=1)
        error = http_error({})

        with self.assertRaisesRegex(RuntimeError, "rate limit wait exceeds configured maximum"):
            api._sleep_for_limit(error, "secondary rate limit", backoff=2.0)


if __name__ == "__main__":
    unittest.main()
