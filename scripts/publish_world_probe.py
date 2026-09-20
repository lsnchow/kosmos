"""Copy an explicitly inspected completed world probe into the saved gallery.

Never promotes a policy comparison or a gate. Original evidence is untouched.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--visually-inspected", action="store_true", required=True)
    args = parser.parse_args()
    report_path = args.probe / "report.json"
    report = json.loads(report_path.read_text())
    request = json.loads((args.probe / "request.json").read_text())["request"]
    if report.get("status") != "completed_unqualified" or len(report.get("frames", [])) != 16:
        raise ValueError("Only a fully received unqualified 16-frame probe may be archived")
    for record in report["frames"]:
        path = args.probe / record["path"]
        if path.parent.resolve() != args.probe.resolve() or sha(path) != record["sha256"]:
            raise ValueError("Probe frame provenance failed")
    destination = args.data_root / "cluster-evidence" / ("baseten-" + args.probe.name)
    destination.mkdir(parents=True, exist_ok=False)
    video = destination / ("drawer-" + request["direction"] + ".mp4")
    poster = destination / "poster.png"
    shutil.copyfile(args.probe / "probe.mp4", video)
    shutil.copyfile(args.probe / report["frames"][0]["path"], poster)
    entry = {"kind": "kosmos_baseten_world_probe", "status": "completed_unqualified",
        "qualified": False, "scored": False, "task": request["task_instruction"], "direction": request["direction"],
        "policy_calls": 0, "judge_calls": 0, "world": request["world"], "chain_url": report["url"],
        "parent_report_sha256": sha(report_path), "parent_attempt_id": report["attempt_id"],
        "source_png_sha256": request["source"]["png_sha256"], "world_seed": request["world"]["manual_seed"],
        "video": {"path": video.name, "sha256": sha(video), "frame_count": 16},
        "poster": {"path": poster.name, "sha256": sha(poster)},
        "notes": ["Explicit visual inspection for saved-demo playback only; not a dynamics gate.",
                  "Native 5 FPS; no interpolation or frame duplication; conditioning frame excluded."]}
    (destination / "report.json").write_text(json.dumps(entry, indent=2) + "\n")
    print(str(destination))


if __name__ == "__main__":
    main()
