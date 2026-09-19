"""Fetch only the pinned official Bridge smoke fixture, on cluster scratch."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path("/scratch/lchow432/plumb")
SOURCES = {
    "vendor_fixture.json": "https://raw.githubusercontent.com/NVIDIA/cosmos-framework/c23e51f2f157ae3e51cfcd86ebfb5464850894f2/inputs/omni/action_forward_dynamics_robot.json",
    "bridge_actions.json": "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/2b17a2413bd86b2cf9b03823637108851e4ddf2d/inputs/action/bridge_20260501_0.json",
    "bridge_video.mp4": "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/2b17a2413bd86b2cf9b03823637108851e4ddf2d/inputs/action/bridge_20260501_0.mp4",
}


def main():
    if not ROOT.parent.is_dir():
        raise RuntimeError("Cluster scratch is unavailable; refusing a local download")
    folder = ROOT / "fixtures"
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for name, url in SOURCES.items():
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read(200_000_001)
        if len(data) > 200_000_000:
            raise RuntimeError("Unexpectedly large fixture; review before downloading")
        (folder / name).write_bytes(data)
        records.append({"file": name, "source": url, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    (folder / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
