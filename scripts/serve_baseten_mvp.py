"""Start the localhost console with an explicitly pinned cloud diagnostic.

No credential is loaded here: inference uses the named Baseten CLI profile.
This does not enable the qualified Baseten rollout backend or change any gate.
"""
import argparse
import os
import sys
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--deployment-id", required=True)
    parser.add_argument("--profile", default="plumb-api")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "live-integrated")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--live-demo-url", help="Loopback URL of the authenticated GPU worker tunnel.")
    parser.add_argument("--live-demo-token-file", type=Path, help="Private worker token file; never sent to the browser.")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 1 <= args.timeout_seconds <= 600:
        parser.error("port or timeout is outside its supported range")
    if bool(args.live_demo_url) != bool(args.live_demo_token_file):
        parser.error("live-demo URL and token file must be supplied together")
    if args.live_demo_url:
        endpoint = urlparse(args.live_demo_url)
        if endpoint.scheme != "http" or endpoint.hostname not in {"127.0.0.1", "localhost", "::1"} or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
            parser.error("live-demo URL must be a plain HTTP loopback tunnel URL")
        if not args.live_demo_token_file.is_file() or args.live_demo_token_file.is_symlink():
            parser.error("live-demo token must be an existing regular private file")
        os.environ["PLUMB_LIVE_DEMO_URL"] = args.live_demo_url.rstrip("/")
        os.environ["PLUMB_LIVE_DEMO_TOKEN_FILE"] = str(args.live_demo_token_file.resolve())
    os.environ.update({
        "PLUMB_DATA_DIR": str(args.data_dir.resolve()),
        "PLUMB_CLOUD_DIAGNOSTIC_MODEL_ID": args.model_id,
        "PLUMB_CLOUD_DIAGNOSTIC_DEPLOYMENT_ID": args.deployment_id,
        "PLUMB_CLOUD_DIAGNOSTIC_PROFILE": args.profile,
        "PLUMB_CLOUD_DIAGNOSTIC_MODEL_REVISION": "1a4c15dd9ad780a257e9494f0fac79cbe8e64793",
        "PLUMB_CLOUD_DIAGNOSTIC_CHECKPOINT_SHA256": "80b354db7a05d514d6df5b5a4395469902b0ff362d383edeeb4abb8c1c9d9e33",
        "PLUMB_CLOUD_DIAGNOSTIC_SOAR_REVISION": "eabd5f16a856e484884a22e257a941bb358cea08",
        "PLUMB_CLOUD_DIAGNOSTIC_TIMEOUT_SECONDS": str(args.timeout_seconds),
    })
    sys.path.insert(0, str(ROOT))
    import uvicorn
    uvicorn.run("plumb.api:create_app", factory=True, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
