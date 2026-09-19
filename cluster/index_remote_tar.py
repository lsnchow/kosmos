"""Index the mislabeled IRASim TAR using HTTP ranges, without fetching all weights."""
import argparse
import json
import math
import time
import urllib.request
from pathlib import Path

DEFAULT_URL = "https://lf-robot-opensource.bytetos.com/obj/lab-robot-public/opensource_IRASim_v1/bridge_checkpoints_data.tar.gz"


def fetch_range(url, offset, length):
    request = urllib.request.Request(url, headers={"Range": "bytes=%d-%d" % (offset, offset + length - 1),
                                                   "Accept-Encoding": "identity"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                expected = "bytes %d-" % offset
                if response.status != 206 or not response.headers.get("Content-Range", "").startswith(expected):
                    raise ValueError("Server did not honor the exact byte range")
                data = response.read(length + 1)
                if len(data) != length:
                    raise ValueError("Unexpected HTTP range length")
                return data, response.headers.get("ETag")
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1 + attempt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    offset, entries, etag, long_name = 0, [], None, None
    for _ in range(10000):
        header, current_etag = fetch_range(args.url, offset, 512)
        if etag is None:
            etag = current_etag
        elif current_etag != etag:
            raise ValueError("Archive ETag changed during indexing")
        if header == bytes(512):
            break
        if offset == 0 and header[:2] == b"\x1f\x8b":
            raise ValueError("Actual gzip archive cannot be indexed with uncompressed TAR offsets")
        checksum = int(header[148:156].strip(b" \x00") or b"0", 8)
        if checksum != sum(header[:148]) + 8 * 32 + sum(header[156:]):
            raise ValueError("Invalid TAR checksum at %d" % offset)
        name = header[:100].split(b"\x00", 1)[0].decode("utf-8")
        prefix = header[345:500].split(b"\x00", 1)[0].decode("utf-8")
        if prefix:
            name = prefix + "/" + name
        raw_size = header[124:136]
        if raw_size[0] & 0x80:
            if raw_size[0] & 0x40:
                raise ValueError("Negative TAR member size")
            size = int.from_bytes(bytes([raw_size[0] & 0x7f]) + raw_size[1:], "big")
        else:
            size = int(raw_size.strip(b" \x00") or b"0", 8)
        kind = header[156:157]
        if kind == b"L":
            if size > 1024 * 1024:
                raise ValueError("Oversized GNU name")
            data, _ = fetch_range(args.url, offset + 512, size)
            long_name = data.rstrip(b"\x00\n").decode()
        else:
            if long_name is not None:
                name, long_name = long_name, None
            entry = {"name": name, "size": size, "header_offset": offset,
                     "data_offset": offset + 512, "type": kind.decode("ascii", errors="replace")}
            entries.append(entry)
            print(json.dumps(entry), flush=True)
        offset += 512 + math.ceil(size / 512) * 512
    else:
        raise ValueError("TAR entry limit exceeded")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"url": args.url, "etag": etag,
                                     "format": "uncompressed_tar_despite_gz_name", "entries": entries}, indent=2) + "\n")


if __name__ == "__main__":
    main()
