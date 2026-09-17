#!/usr/bin/env python3
"""Fetch one water-spray condition from Zenodo without downloading the 7 GB ZIP.

Dataset DOI: 10.5281/zenodo.17935932, CC-BY-4.0, Racz et al.
Downloads and provenance belong under ignored outputs/. ZIP CRCs and per-file
SHA256 identify the selected members; the full archive checksum is not verified.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import urllib.request
import zipfile
from collections import OrderedDict
from pathlib import Path

RECORD_URL = "https://zenodo.org/api/records/17935932"
CONDITION = "air-blast_1.28kgh_W_0.3bar_25C_1"


class RangeReader(io.RawIOBase):
    """Bounded HTTP range access for zipfile; reject servers ignoring Range."""

    def __init__(self, url, size, *, budget=256 * 1024**2):
        self.url, self.size, self.budget = url, size, budget
        self.position = self.transferred = 0
        self.blocks = OrderedDict()

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        bases = {0: 0, 1: self.position, 2: self.size}
        if whence not in bases or bases[whence] + offset < 0:
            raise ValueError("Invalid seek")
        self.position = bases[whence] + offset
        return self.position

    def read(self, size=-1):
        count = min(self.size - self.position, size if size >= 0 else self.size)
        if count > 8 * 1024**2:
            raise ValueError("Refusing a read larger than 8 MiB")
        chunks = []
        while count > 0:
            start = self.position // 1048576 * 1048576
            if start not in self.blocks:
                end = min(start + 1048576, self.size) - 1
                length = end - start + 1
                if self.transferred + length > self.budget:
                    raise ValueError("Transfer budget exceeded")
                request = urllib.request.Request(
                    self.url, headers={"Range": f"bytes={start}-{end}"}
                )
                with urllib.request.urlopen(request, timeout=60) as response:
                    expected = f"bytes {start}-{end}/{self.size}"
                    if (
                        response.status != 206
                        or response.headers.get("Content-Range") != expected
                    ):
                        raise ValueError(
                            "Server did not honor the requested byte range"
                        )
                    block = response.read(length + 1)
                if len(block) != length:
                    raise ValueError("Incomplete or oversized response")
                self.blocks[start] = block
                self.transferred += length
                if len(self.blocks) > 8:
                    self.blocks.popitem(last=False)
            self.blocks.move_to_end(start)
            chunk = self.blocks[start][
                self.position - start : self.position - start + count
            ]
            if not chunk:
                raise ValueError("Unexpected end of range")
            chunks.append(chunk)
            self.position += len(chunk)
            count -= len(chunk)
        return b"".join(chunks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", default=CONDITION)
    args = parser.parse_args()
    if not re.fullmatch(r"air-blast_1\.28kgh_W_[0-9.]+bar_25C_1", args.condition):
        raise ValueError("This importer qualifies only the 25 C water subsets")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata_bytes = urllib.request.urlopen(RECORD_URL, timeout=60).read()
    metadata = json.loads(metadata_bytes)
    (args.output / "zenodo_record.json").write_bytes(metadata_bytes)
    (item,) = metadata["files"]
    reader = RangeReader(item["links"]["self"], item["size"])
    members = []
    with zipfile.ZipFile(reader) as archive:
        selected = [
            i
            for i in archive.infolist()
            if (f"/{args.condition}/" in i.filename or "README" in i.filename)
            and not i.is_dir()
        ]
        if len([i for i in selected if f"/{args.condition}/" in i.filename]) != 90:
            raise ValueError("Expected exactly 90 measurement files")
        for i, member in enumerate(selected):
            destination = args.output / Path(member.filename).name
            data = archive.read(member)  # zipfile verifies CRC-32.
            destination.write_bytes(data)
            members.append(
                {
                    "archive_member": member.filename,
                    "local_name": destination.name,
                    "bytes": len(data),
                    "crc32": f"{member.CRC:08x}",
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
            if i % 15 == 0:
                print(f"Retrieved {i + 1}/{len(selected)} members", flush=True)
    provenance = {
        "dataset_doi": "10.5281/zenodo.17935932",
        "license": metadata["metadata"]["license"],
        "creators": metadata["metadata"]["creators"],
        "condition": args.condition,
        "record_url": RECORD_URL,
        "archive_url": item["links"]["self"],
        "archive_size": item["size"],
        "archive_checksum_reported": item["checksum"],
        "full_archive_checksum_verified": False,
        "record_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "transferred_bytes": reader.transferred,
        "members": members,
    }
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(
        f"Saved {len(members)} members; transferred {reader.transferred} bytes",
        flush=True,
    )


if __name__ == "__main__":
    main()
