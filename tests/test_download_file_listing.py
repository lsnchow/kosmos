from types import SimpleNamespace

import pytest

from cluster.download_assets import selected_repository_files


def file(path, size=10):
    return SimpleNamespace(path=path, size=size, blob_id="git-object", lfs=None)


def test_exact_paths_use_complete_path_info_not_truncated_siblings():
    class API:
        def get_paths_info(self, **kwargs):
            return [file(value) for value in kwargs["paths"]]
    paths = ["videos/chunk-%03d/episode.mp4" % i for i in range(130)]
    result = selected_repository_files(API(), "repo", "dataset", "a" * 40, paths, [])
    assert len(result) == 130
    assert sum(value.size for value in result) == 1300


def test_missing_explicit_path_blocks_before_download():
    class API:
        def get_paths_info(self, **kwargs):
            return [file("first.mp4")]
    with pytest.raises(ValueError, match="missing"):
        selected_repository_files(API(), "repo", "dataset", "a" * 40, ["first.mp4", "second.mp4"], [])


def test_pattern_uses_paginated_tree_and_ignores_folders():
    class API:
        def list_repo_tree(self, **kwargs):
            assert kwargs["recursive"]
            yield SimpleNamespace(path="meta", size=0)
            yield file("meta/info.json")
            yield file("meta/ignored.json")
            yield file("videos/large.mp4", 900000)
    result = selected_repository_files(API(), "repo", "dataset", "a" * 40, ["meta/*"], ["*ignored*"])
    assert [value.rfilename for value in result] == ["meta/info.json"]


def test_unknown_size_cannot_bypass_explicit_ceiling():
    class API:
        def get_paths_info(self, **kwargs):
            return [file("video.mp4", None)]
    with pytest.raises(ValueError, match="byte length"):
        selected_repository_files(API(), "repo", "dataset", "a" * 40, ["video.mp4"], [])


def test_main_writes_receipt_not_last_downloaded_asset_and_refuses_overwrite(tmp_path, monkeypatch):
    import json
    import sys
    from cluster import download_assets as downloader

    root = tmp_path / "plumb"
    root.mkdir()
    monkeypatch.setattr(downloader, "validated_download_root", lambda _: root)
    class API:
        def repo_info(self, *args, **kwargs):
            return SimpleNamespace(sha="a" * 40, gated=False)

        def get_paths_info(self, **kwargs):
            return [file("video.mp4", 3)]

    def snapshot_download(**kwargs):
        from pathlib import Path
        destination = Path(kwargs["local_dir"])
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "video.mp4").write_bytes(b"raw")
        return str(destination)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=API, snapshot_download=snapshot_download))
    monkeypatch.setattr(sys, "argv", ["download", "test/repo", "--revision", "a" * 40,
                                     "--repo-type", "dataset", "--root", str(root), "--allow", "video.mp4",
                                     "--manifest-name", "pilot-receipt.json", "--execute"])
    assert downloader.main() == 0
    receipt = root / "evidence/pilot-receipt.json"
    recorded = receipt.read_bytes()
    assert json.loads(recorded)["file_records"][0]["path"] == "video.mp4"
    assert (root / "models/test--repo/video.mp4").read_bytes() == b"raw"
    with pytest.raises(SystemExit):
        downloader.main()
    assert receipt.read_bytes() == recorded
