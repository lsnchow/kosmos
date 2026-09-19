"""Derive a static first/final-video goal-conditioning diagnostic, never a task label."""
import argparse
import hashlib
import json
from pathlib import Path


def prepare(video: Path, output: Path) -> dict:
    import cv2
    from PIL import Image

    if output.exists():
        raise ValueError("fixture output must be fresh")
    video_bytes = video.read_bytes()
    capture = cv2.VideoCapture(str(video))
    first = last = None
    count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if first is None:
            first = frame.copy()
        last = frame
        count += 1
    capture.release()
    if count < 2:
        raise ValueError("video needs at least two decoded frames")
    converted = [cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (256, 256))
                 for frame in (first, last)]
    if (converted[0] == converted[1]).all():
        raise ValueError("current and demonstrated goal pixels must differ")
    output.mkdir(parents=True)
    records = []
    for index, rgb, name in zip((0, count - 1), converted, ("current.png", "goal.png")):
        path = output / name
        Image.fromarray(rgb).save(path)
        records.append({"source_frame_index": index, "output_file": name,
                        "output_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "output_rgb_shape": list(rgb.shape), "output_rgb_dtype": str(rgb.dtype),
                        "output_rgb_pixel_sha256": hashlib.sha256(rgb.tobytes()).hexdigest()})
    payload = {
        "schema": "plumb-octo-canonical-inputs-v1", "qualified": False,
        "source": {"path": str(video.resolve()), "sha256": hashlib.sha256(video_bytes).hexdigest(),
                   "frame_indices": [0, count - 1], "decoded_frame_count": count,
                   "decoder": "OpenCV VideoCapture", "opencv_version": cv2.__version__},
        "preprocessing": {"kind": "AutoEval ResizeObsImageWrapper-compatible cv2.resize",
                          "target_size": [256, 256], "interpolation": "OpenCV default INTER_LINEAR",
                          "color_conversion": "BGR→RGB before resize",
                          "limitation": "Static first/final decoded vendor-video frames; final image is demonstrated conditioning only, not a certified task goal, success label, or closed-loop feedback."},
        "frames": records,
    }
    (output / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.video, args.output), indent=2))
