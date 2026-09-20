"""Stream a Chain response using credentials forwarded by the Baseten CLI.

Invoke with BASETEN_PROFILE set and `baseten truss --truss-executable
.venv-deploy/bin/python scripts/stream_baseten_chain.py --url ...`.
The request is JSON on stdin; stdout contains only the response stream. Tokens
never appear in command arguments, files, output, or exception messages.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from urllib.parse import urlsplit


MAX_REQUEST_BYTES = 32 * 1024 * 1024
MAX_RESPONSE_BYTES = 256 * 1024 * 1024


def validate_chain_url(value: str) -> str:
    parsed = urlsplit(value)
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("Invalid Chain port") from error
    if (
        parsed.scheme != "https"
        or not re.fullmatch(r"chain-[a-z0-9-]+\.api\.baseten\.co", parsed.hostname or "")
        or port not in (None, 443)
        or parsed.username or parsed.password or parsed.query or parsed.fragment
        or not re.fullmatch(r"/(production|development|deployment/[A-Za-z0-9_-]+|environments/[A-Za-z0-9_-]+)/run_remote", parsed.path)
    ):
        raise ValueError("Only a verified HTTPS Baseten Chain run_remote URL is allowed")
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--read-timeout", type=int, default=120)
    args = parser.parse_args(argv)
    try:
        url = validate_chain_url(args.url)
        if not 1 <= args.read_timeout <= 3600:
            raise ValueError("Read inactivity timeout must be between 1 and 3600 seconds")
        token = os.environ.get("BASETEN_TRUSS_AUTH_API_KEY", "")
        if not token:
            raise ValueError("Use the authenticated Baseten CLI wrapper to invoke this helper")
        raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
        if not raw or len(raw) > MAX_REQUEST_BYTES:
            raise ValueError("Request is empty or exceeds the bounded payload size")
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("Chain request must be a JSON object")
    except (ValueError, UnicodeError):
        print("Invalid Chain configuration or request; no inference was submitted.", file=sys.stderr)
        return 2

    import requests

    total = 0
    try:
        with requests.Session() as session:
            session.trust_env = False
            with session.post(
                url,
                headers={"Authorization": "Bearer " + token},
                json=payload,
                stream=True,
                allow_redirects=False,
                timeout=(10, args.read_timeout),
            ) as response:
                if response.status_code != 200:
                    print("Chain request returned HTTP %d; no retry was attempted." % response.status_code, file=sys.stderr)
                    return 3
                for block in response.iter_content(chunk_size=65536):
                    if not block:
                        continue
                    total += len(block)
                    if total > MAX_RESPONSE_BYTES:
                        print("Chain response exceeded the stream byte bound; outcome is incomplete.", file=sys.stderr)
                        return 4
                    sys.stdout.buffer.write(block)
                    sys.stdout.buffer.flush()
    except (requests.RequestException, BrokenPipeError, OSError):
        print("Chain stream interrupted; remote outcome may be ambiguous. No retry was attempted.", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
