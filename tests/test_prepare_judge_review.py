"""Development-review packet preparation never turns pilot media into labels."""
from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

import pytest

from cluster.prepare_judge_review import ReviewPreparationError, _read_candidate_manifest, prepare_review_packets


def digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def candidate_manifest(root: Path, source_root: Path, *, omit_last_frame: bool = False) -> tuple[Path, dict]:
    clips = []
    source = {}
    for clip_index in range(16):
        source_id = "bridge-episode-%06d" % clip_index
        video_relative = "clips/%s/source.mp4" % source_id
        video = ("video-%d" % clip_index).encode("ascii")
        write(source_root / video_relative, video)
        frames = []
        frame_values = []
        for frame_index in range(16):
            relative = "clips/%s/frame-%02d.png" % (source_id, frame_index)
            value = ("png-%d-%d" % (clip_index, frame_index)).encode("ascii")
            if not (omit_last_frame and clip_index == 15 and frame_index == 15):
                write(source_root / relative, value)
            frame_values.append((relative, value))
            frames.append(
                {
                    "path": relative,
                    "sha256": digest(value),
                    "pixel_sha256": digest(b"pixels-" + value),
                    "timestamp": frame_index * 0.2,
                }
            )
        clips.append(
            {
                "clip_id": source_id,
                "cohort": "development_validation" if clip_index % 2 else "train",
                "source_lineage_id": "private-lineage-%d" % clip_index,
                "task": "close_drawer",
                "video": {"path": video_relative, "sha256": digest(video)},
                "frames": frames,
                # Input-only bait: neither may ever appear in the public packet.
                "teacher_votes": [{"model": "teacher", "label": "not-a-review-label"}],
                "checkpoint": "private-checkpoint",
            }
        )
        source[source_id] = {"video": video, "frames": frame_values}
    manifest = {
        "schema": "judge-review-candidates-v1",
        "qualified": False,
        "task": "close_drawer",
        "clips": clips,
    }
    path = root / "candidate-inputs.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, source


def prepare_paths(tmp_path: Path):
    root = tmp_path / "workspace"
    root.mkdir()
    source_root = root / "judge-pilot-v1-review-source"
    source_root.mkdir()
    served_root = root / "data" / "live-integrated"
    public_out = served_root / "review" / "review-packets-v1"
    private_resolver = root / "coordinator-private" / "review-resolver.json"
    return root, source_root, served_root, public_out, private_resolver


def test_prepares_16_opaque_blank_review_packets_and_a_private_resolver(tmp_path: Path):
    root, source_root, served_root, public_out, private_resolver = prepare_paths(tmp_path)
    manifest, original = candidate_manifest(root, source_root)
    result = prepare_review_packets(
        candidate_manifest=manifest,
        dataset_root=source_root,
        public_out=public_out,
        private_resolver=private_resolver,
        served_root=served_root,
    )
    assert result["packets"] == 16
    assert result["qualified"] is False
    assert result["gate_d_eligible"] is False
    rows = [json.loads(line) for line in (public_out / "blank-worksheets.jsonl").read_text().splitlines()]
    assert len(rows) == 16
    assert all(row["purpose"] == "development_review_only" for row in rows)
    assert all(row["assignment_status"] == "unassigned" and row["reviewer_id"] == "" for row in rows)
    assert all(row["integrity"] == row["collision"] == row["progress"] == row["completion_evidence"] == "" for row in rows)
    assert all(row["evidence_frame_indices"] == [] and row["observable_reasons"] == "" for row in rows)
    assert all(row["clip_id"].startswith("review-clip-") for row in rows)
    assert all(row["media_id"].startswith("review-media-") for row in rows)
    assert all(len(row["sampled_frames"]) == 16 for row in rows)
    assert all(
        row["raw_video_url"].startswith("http://127.0.0.1:8787/api/artifacts/review/review-packets-v1/media/")
        for row in rows
    )
    assert all(
        frame["url"].startswith("http://127.0.0.1:8787/api/artifacts/review/review-packets-v1/media/")
        for row in rows
        for frame in row["sampled_frames"]
    )
    assert "/review-media/" not in "\n".join(
        [row["raw_video_url"] for row in rows]
        + [frame["url"] for row in rows for frame in row["sampled_frames"]]
    )
    public_text = (public_out / "blank-worksheets.jsonl").read_text() + (public_out / "INDEX.md").read_text()
    assert "bridge-episode" not in public_text
    assert "private-lineage" not in public_text
    assert "teacher" not in public_text.lower()
    assert "checkpoint" not in public_text.lower()
    for row in rows:
        video = public_out / "media" / row["media_id"] / "source.mp4"
        assert video.is_file()
        assert len(list((public_out / "media" / row["media_id"] / "frames").glob("*.png"))) == 16
    resolver = json.loads(private_resolver.read_text())
    assert resolver["reserved_source_clip_count"] == 16
    assert set(resolver["media"]) == {row["media_id"] for row in rows}
    first = next(iter(resolver["media"].values()))
    assert first["source_clip_id"].startswith("bridge-episode-")
    assert first["original_cohort"] in {"train", "development_validation"}
    assert first["source_lineage_id"].startswith("private-lineage-")
    assert len(first["sampled_frames"]) == 16
    for media_id, private_entry in resolver["media"].items():
        source = original[private_entry["source_clip_id"]]
        assert (public_out / "media" / media_id / "source.mp4").read_bytes() == source["video"]
        for frame in private_entry["sampled_frames"]:
            original_frame = source["frames"][frame["index"]][1]
            copied = public_out / frame["public_relative_path"]
            assert copied.read_bytes() == original_frame
    assert private_resolver.stat().st_mode & 0o777 == stat.S_IRUSR | stat.S_IWUSR
    assert not str(private_resolver).startswith(str(served_root))
    with pytest.raises(FileExistsError):
        prepare_review_packets(
            candidate_manifest=manifest,
            dataset_root=source_root,
            public_out=public_out,
            private_resolver=root / "coordinator-private" / "other.json",
            served_root=served_root,
        )


def test_missing_or_hash_mismatched_media_aborts_before_any_public_or_private_packet(tmp_path: Path):
    root, source_root, served_root, public_out, private_resolver = prepare_paths(tmp_path)
    manifest, _ = candidate_manifest(root, source_root, omit_last_frame=True)
    with pytest.raises(ReviewPreparationError, match="missing"):
        prepare_review_packets(
            candidate_manifest=manifest,
            dataset_root=source_root,
            public_out=public_out,
            private_resolver=private_resolver,
            served_root=served_root,
        )
    assert not public_out.exists()
    assert not private_resolver.exists()
    manifest, _ = candidate_manifest(root, source_root)
    payload = json.loads(manifest.read_text())
    payload["clips"][0]["video"]["sha256"] = "sha256:" + "0" * 64
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ReviewPreparationError, match="SHA-256 mismatch"):
        prepare_review_packets(
            candidate_manifest=manifest,
            dataset_root=source_root,
            public_out=public_out,
            private_resolver=private_resolver,
            served_root=served_root,
        )
    assert not public_out.exists()
    assert not private_resolver.exists()


def test_private_resolver_inside_served_root_is_rejected_before_media_copy(tmp_path: Path):
    root, source_root, served_root, public_out, _ = prepare_paths(tmp_path)
    manifest, _ = candidate_manifest(root, source_root)
    private_inside_served_root = served_root / "private-resolver.json"
    with pytest.raises(ReviewPreparationError, match="outside the served"):
        prepare_review_packets(
            candidate_manifest=manifest,
            dataset_root=source_root,
            public_out=public_out,
            private_resolver=private_inside_served_root,
            served_root=served_root,
        )
    assert not public_out.exists()
    assert not private_inside_served_root.exists()


def test_private_resolver_parent_symlink_is_rejected_before_media_copy(tmp_path: Path):
    root, source_root, served_root, public_out, _ = prepare_paths(tmp_path)
    manifest, _ = candidate_manifest(root, source_root)
    served_root.mkdir(parents=True)
    symlink_parent = root / "looks-private"
    symlink_parent.symlink_to(served_root, target_is_directory=True)
    private_via_symlink = symlink_parent / "resolver.json"
    with pytest.raises(ReviewPreparationError, match="contains a symlink"):
        prepare_review_packets(
            candidate_manifest=manifest,
            dataset_root=source_root,
            public_out=public_out,
            private_resolver=private_via_symlink,
            served_root=served_root,
        )
    assert not public_out.exists()
    assert not private_via_symlink.exists()


def test_manifest_task_descriptor_matches_the_real_candidate_input_shape():
    repository = Path(__file__).resolve().parents[1]
    clips, digest_value = _read_candidate_manifest(
        repository / "data" / "judge-pilot-v1-evidence" / "candidate-inputs.json"
    )
    assert len(clips) == 16
    assert all(clip.task == "close_drawer" for clip in clips)
    assert len(digest_value) == 64


def test_nonfinite_or_nonincreasing_timestamps_abort_before_writes(tmp_path: Path):
    root, source_root, served_root, public_out, private_resolver = prepare_paths(tmp_path)
    manifest, _ = candidate_manifest(root, source_root)
    payload = json.loads(manifest.read_text())
    payload["clips"][0]["frames"][2]["timestamp"] = payload["clips"][0]["frames"][1]["timestamp"]
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ReviewPreparationError, match="strictly increasing"):
        prepare_review_packets(
            candidate_manifest=manifest,
            dataset_root=source_root,
            public_out=public_out,
            private_resolver=private_resolver,
            served_root=served_root,
        )
    assert not public_out.exists()
    assert not private_resolver.exists()
