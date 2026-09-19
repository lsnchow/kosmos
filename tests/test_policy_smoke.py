"""Offline fake-runtime tests for the cluster policy smoke entrypoint."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from cluster import policy_smoke as smoke
from plumb.adapters.contracts import PolicyObservation
from plumb.policies.judge import JudgeRequest, QwenJudgeProfile, ReferenceImage
from plumb.policies.openvla import OpenVLAPolicyProfile


REVISION = "a" * 40


def _manifest(path: Path, revision: str = REVISION) -> Path:
    path.write_text(json.dumps({"repo": "fixture/model", "revision": revision, "file_records": []}))
    return path


def _parsed(tmp_path: Path, *arguments: str):
    parser = smoke.build_parser()
    return smoke._apply_defaults(parser.parse_args(("--root", str(tmp_path)) + arguments))


@dataclass
class FakeActionReport:
    action: tuple = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
    unnorm_key: str = "bridge_orig"
    source_image_timestamp: float = 0.0
    backend_calls: int = 1
    wall_seconds: float = 0.012
    gpu_peak_memory_bytes: object = None


class FakeOpenVLA:
    observations = []

    def __init__(self, profile):
        self.profile = profile

    def capability(self):
        return None

    def predict_with_report(self, observation):
        type(self).observations.append(observation)
        return FakeActionReport()


class ExplodingOpenVLA(FakeOpenVLA):
    def predict_with_report(self, observation):
        raise RuntimeError("fixture model failure")


class FakeJudgeReport:
    judge_status = "evaluable"
    binary_success = True
    progress = 5.0

    def as_dict(self):
        return {
            "binary_success": True,
            "progress": 5.0,
            "judge_status": "evaluable",
            "samples": [{"raw_output": "fixture-json-vote"}],
        }


class FakeJudge:
    requests = []
    seeds = []

    def __init__(self, profile):
        self.profile = profile

    def capability(self):
        return None

    def evaluate(self, request, *, seeds):
        type(self).requests.append(request)
        type(self).seeds.append(tuple(seeds))
        return FakeJudgeReport()


def _openvla_dependencies():
    return OpenVLAPolicyProfile, object, PolicyObservation, 0.001


def _judge_dependencies():
    return QwenJudgeProfile, object, JudgeRequest, ReferenceImage, 0.001


def test_openvla_smoke_records_one_first_frame_action_and_local_manifest_metadata(tmp_path):
    model_dir = tmp_path / "models" / "openvla"
    model_dir.mkdir(parents=True)
    manifest = _manifest(tmp_path / "openvla-manifest.json", smoke.REVIEWED_OPENVLA_REMOTE_CODE_REVISION)
    video = tmp_path / "bridge.mp4"
    video.write_bytes(b"vendor-bridge-fixture")
    report_path = tmp_path / "evidence" / "openvla.json"
    FakeOpenVLA.observations.clear()
    args = _parsed(
        tmp_path,
        "openvla",
        "--model-path", str(model_dir),
        "--model-manifest", str(manifest),
        "--vision", str(video),
        "--task", "Put the pot to the left of the purple item.",
        "--report", str(report_path),
    )

    code, report = smoke.run_smoke(
        args,
        first_frame_loader=lambda path: "first-vendor-frame",
        openvla_dependency_loader=_openvla_dependencies,
        adapter_factory=lambda profile, unused: FakeOpenVLA(profile),
    )

    assert code == 0
    assert report["status"] == "completed_unqualified"
    assert report["qualified"] is False
    assert report["model"]["revision_from_metadata"] is True
    assert report["model"]["revision"] == smoke.REVIEWED_OPENVLA_REMOTE_CODE_REVISION
    assert report["inputs"]["first_frame_index"] == 0
    assert report["action"]["dimension"] == 7
    assert report["action"]["normalization"] == "bridge_orig"
    assert len(FakeOpenVLA.observations) == 1
    assert FakeOpenVLA.observations[0].image_history == ("first-vendor-frame",)
    persisted = json.loads(report_path.read_text())
    assert persisted["action"]["values"] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    assert persisted["offline_environment"] == {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}


def test_openvla_smoke_persists_failure_record_without_fabricating_action(tmp_path):
    model_dir = tmp_path / "models" / "openvla"
    model_dir.mkdir(parents=True)
    manifest = _manifest(tmp_path / "manifest.json", smoke.REVIEWED_OPENVLA_REMOTE_CODE_REVISION)
    video = tmp_path / "bridge.mp4"
    video.write_bytes(b"bridge")
    report_path = tmp_path / "evidence" / "failure.json"
    args = _parsed(
        tmp_path,
        "openvla",
        "--model-path", str(model_dir),
        "--model-manifest", str(manifest),
        "--vision", str(video),
        "--task", "Open the drawer",
        "--report", str(report_path),
    )

    code, report = smoke.run_smoke(
        args,
        first_frame_loader=lambda path: "frame",
        openvla_dependency_loader=_openvla_dependencies,
        adapter_factory=lambda profile, unused: ExplodingOpenVLA(profile),
    )

    assert code == 2
    assert report["status"] == "failed"
    assert report["qualified"] is False
    assert report["error"]["type"] == "RuntimeError"
    assert "action" not in report
    assert json.loads(report_path.read_text())["status"] == "failed"


def test_judge_smoke_uses_exactly_16_endpoint_inclusive_frames_and_scene_context_never_asserts_success(tmp_path):
    model_dir = tmp_path / "models" / "qwen"
    model_dir.mkdir(parents=True)
    manifest = _manifest(tmp_path / "qwen-manifest.json")
    video = tmp_path / "cosmos.mp4"
    reference = tmp_path / "context.mp4"
    video.write_bytes(b"cosmos-generated")
    reference.write_bytes(b"context-video")
    report_path = tmp_path / "evidence" / "judge.json"
    FakeJudge.requests.clear()
    FakeJudge.seeds.clear()
    args = _parsed(
        tmp_path,
        "judge",
        "--model-path", str(model_dir),
        "--model-manifest", str(manifest),
        "--video", str(video),
        "--reference", str(reference),
        "--reference-role", "scene",
        "--task", "Close the drawer",
        "--rubric", "Drawer: final visible state must be closed for completion.",
        "--diagnostic",
        "--seeds", "11", "12", "13", "14", "15",
        "--report", str(report_path),
    )
    frames = tuple("frame-%d" % index for index in range(17))

    code, report = smoke.run_smoke(
        args,
        video_loader=lambda path: frames,
        first_frame_loader=lambda path: "scene-context-first-frame",
        judge_dependency_loader=_judge_dependencies,
        judge_factory=lambda profile, unused: FakeJudge(profile),
    )

    assert code == 0
    assert report["status"] == "completed_unqualified"
    assert report["inputs"]["selected_frame_indices"][0] == 0
    assert report["inputs"]["selected_frame_indices"][-1] == 16
    assert len(report["inputs"]["selected_frame_indices"]) == 16
    request = FakeJudge.requests[0]
    assert len(request.frames) == len(request.frame_timestamps) == 16
    assert request.frames[0] == "frame-0" and request.frames[-1] == "frame-16"
    assert request.reference_images[0].image == "scene-context-first-frame"
    assert FakeJudge.seeds == [(11, 12, 13, 14, 15)]
    assert report["raw_judge_votes"]["binary_success"] is True
    assert report["outcome"]["asserted_success"] is None
    assert report["outcome"]["reason"] == "reference_role_scene_context_only"
    assert json.loads(report_path.read_text())["qualified"] is False


def test_cli_requires_reference_role_and_sampling_rejects_short_video():
    parser = smoke.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            (
                "judge", "--video", "video.mp4", "--reference", "reference.png", "--task", "Open", "--rubric", "rubric",
                "--seeds", "1", "2", "3", "4", "5",
            )
        )
    with pytest.raises(smoke.SmokeInputError, match="will not pad"):
        smoke.uniform_video_indices(15)
