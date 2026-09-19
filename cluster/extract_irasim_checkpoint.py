"""Extract one fixed TAR member from six hash-verified HF split parts.

No TAR member paths are trusted or materialized. Only the expected regular
checkpoint byte range is copied to a fixed destination; no pickle is loaded.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

NAME = "opensource_robotdata/bridge/checkpoints/frame_ada/0300000.pt"
SIZE = 10867774258
OFFSET = 1536
PART_SIZE = 2147483648


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/scratch/lchow432/plumb"))
    args = parser.parse_args()
    root = args.root.resolve()
    if not str(root).startswith(("/scratch/", "/global/scratch/")) or root.name != "plumb":
        parser.error("Use the cluster scratch project root")
    manifest_path = root / "evidence/fangqi--IRASim-download.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    records = {r["path"]: r for r in manifest["file_records"]}
    parts = ["bridge_checkpoints_data.tar.gz.sfpart_a" + letter for letter in "abcdef"]
    if set(records) != set(parts):
        raise ValueError("Expected exactly the first six verified checkpoint parts")
    source = root / "models/fangqi--IRASim"
    with (source / parts[0]).open("rb") as stream:
        stream.seek(1024)
        header = stream.read(512)
    name = header[:100].split(b"\x00", 1)[0].decode()
    raw_size = header[124:136]
    size = int.from_bytes(bytes([raw_size[0] & 0x7f]) + raw_size[1:], "big") if raw_size[0] & 0x80 else int(raw_size.strip(b" \x00"), 8)
    checksum = int(header[148:156].strip(b" \x00"), 8)
    if name != NAME or size != SIZE or header[156:157] not in (b"0", b"\x00"):
        raise ValueError("HF mirror TAR layout does not match the independently indexed source")
    if checksum != sum(header[:148]) + 8 * 32 + sum(header[156:]):
        raise ValueError("Invalid TAR member checksum")
    destination = root / "models/irasim/frame_ada_0300000.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError("Checkpoint already exists; inspect its provenance rather than overwrite it")
    digest = hashlib.sha256()
    copied = 0
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix="frame-ada-", suffix=".partial", delete=False) as output:
            temporary = Path(output.name)
            for index, name in enumerate(parts):
                record = records[name]
                path = source / name
                if path.stat().st_size != PART_SIZE:
                    raise ValueError("Part size mismatch")
                part_digest = hashlib.sha256()
                position = index * PART_SIZE
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        part_digest.update(block)
                        left = max(0, OFFSET - position)
                        right = min(len(block), OFFSET + SIZE - position)
                        if left < right:
                            selected = block[left:right]
                            output.write(selected)
                            digest.update(selected)
                            copied += len(selected)
                        position += len(block)
                if part_digest.hexdigest() != record["sha256"]:
                    raise ValueError("Part hash mismatch: " + name)
            output.flush()
            os.fsync(output.fileno())
        if copied != SIZE:
            raise ValueError("Extracted length mismatch")
        temporary.replace(destination)
        record = {"path": str(destination), "bytes": copied, "sha256": digest.hexdigest(),
                  "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                  "source_repository": manifest["repo"], "source_revision": manifest["revision"],
                  "tar_member": NAME, "tar_data_offset": OFFSET,
                  "deserialized": False, "extraction": "fixed byte range; no TAR paths or links extracted"}
        (root / "evidence/irasim-frame-ada-extracted.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)
    except Exception:
        if temporary is not None and temporary.exists():
            temporary.unlink()
        raise


if __name__ == "__main__":
    main()
