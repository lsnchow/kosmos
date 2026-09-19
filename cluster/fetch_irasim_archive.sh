#!/usr/bin/env bash
# Direct cluster download only. This does not unpack or deserialize the archive.
set -euo pipefail
PLUMB_ROOT=${PLUMB_ROOT:-/scratch/lchow432/plumb}
case "$PLUMB_ROOT" in /scratch/*/plumb|/global/scratch/*/plumb) ;; *) exit 2;; esac
mkdir -p "$PLUMB_ROOT/models/irasim" "$PLUMB_ROOT/evidence"
ARCHIVE="$PLUMB_ROOT/models/irasim/bridge_checkpoints_data.tar.gz"
SOURCE_URL=https://lf-robot-opensource.bytetos.com/obj/lab-robot-public/opensource_IRASim_v1/bridge_checkpoints_data.tar.gz
if [ ! -f "$ARCHIVE" ]; then
  curl --fail --location --retry 3 --continue-at - --max-filesize 36000000000 \
    --max-time 7200 --output "$ARCHIVE.partial" "$SOURCE_URL"
  mv "$ARCHIVE.partial" "$ARCHIVE"
fi
python3 - "$ARCHIVE" "$PLUMB_ROOT/evidence/irasim-archive-download.json" "$SOURCE_URL" <<'PY'
import hashlib,json,pathlib,sys,time
archive,manifest,url=pathlib.Path(sys.argv[1]),pathlib.Path(sys.argv[2]),sys.argv[3]
size=archive.stat().st_size
if size != 33456015360:
    raise SystemExit('Archive differs from recorded HTTP length; inspect before use: '+str(size))
h=hashlib.sha256()
with archive.open('rb') as stream:
    for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
record={'source':url,'path':str(archive),'bytes':size,'sha256':h.hexdigest(),
        'timestamp':time.time(),'upstream_sha256':None,'verification':'HTTP length checked; SHA-256 records downloaded bytes, not an upstream signature',
        'code_revision':'c72b6dade6fcd65971e0aa8ab49ea39b15108c90',
        'license_note':'Source Apache-2.0; inspect bundled checkpoint/VAE/data notices separately before deployment.'}
manifest.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
PY
