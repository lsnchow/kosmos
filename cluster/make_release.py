"""Make a small immutable source bundle; never includes weights or run data."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    project = args.project.resolve()
    paths = [project / "pyproject.toml"]
    for directory in ("plumb", "cluster", "deploy", "tests"):
        paths.extend(p for p in (project / directory).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts
                     and p.suffix in {".py", ".sh", ".sbatch", ".json", ".txt", ".md"})
    contents = {p.relative_to(project).as_posix(): p.read_bytes() for p in sorted(paths)}
    records = [{"path": path, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
               for path, data in contents.items()]
    manifest_bytes = json.dumps({"files": records}, sort_keys=True, separators=(",", ":")).encode()
    release = hashlib.sha256(manifest_bytes).hexdigest()
    temporary = Path(tempfile.mkdtemp(prefix="plumb-release-"))
    archive = temporary / (release + ".tar.gz")
    with tarfile.open(archive, "w:gz") as output:
        for name, data in list(contents.items()) + [("RELEASE.json", manifest_bytes)]:
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            entry.mode = 0o644
            output.addfile(entry, io.BytesIO(data))
    print(json.dumps({"release": release, "archive": str(archive), "bytes": archive.stat().st_size,
                      "files": len(contents)}))


if __name__ == "__main__":
    main()
