"""Derive the one-step input from actual OpenVLA evidence, not invented actions."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/scratch/lchow432/plumb"))
    parser.add_argument("--policy-report", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.policy_report.read_text())
    action = report.get("action", {}).get("values")
    if report.get("status") != "completed_unqualified" or not isinstance(action, list) or len(action) != 7:
        raise ValueError("An actual completed seven-dimensional policy smoke is required")
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in action):
        raise ValueError("Invalid policy action")
    video = Path(report["inputs"]["vision"]["path"])
    if sha(video) != report["inputs"]["vision"]["sha256"]:
        raise ValueError("Policy input video differs from recorded source")
    import imageio.v3 as iio
    folder = args.root / "fixtures"
    folder.mkdir(parents=True, exist_ok=True)
    image = folder / "irasim-openvla-first-frame.png"
    action_path = folder / "irasim-openvla-action.json"
    iio.imwrite(image, iio.imread(video, index=0))
    action_path.write_text(json.dumps(action) + "\n")
    provenance = {"source_uri": str(args.policy_report), "sha256": sha(args.policy_report),
                  "source_model": report["model"]["id"], "source_revision": report["model"]["revision"],
                  "source_video_sha256": sha(video), "source_frame_index": 0,
                  "image_sha256": sha(image), "action_file_sha256": sha(action_path),
                  "action_compatibility": "experimental_not_gate_b_qualified"}
    output = folder / "irasim-openvla-action-provenance.json"
    output.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"image": str(image), "action": str(action_path), "provenance": str(output)}))


if __name__ == "__main__":
    main()
