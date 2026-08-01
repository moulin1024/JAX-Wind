#!/usr/bin/env python3
"""Download and safely extract the official GABLS1 12.5 m LES archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import ssl
import tarfile
import tempfile
from urllib.request import urlopen


HERE = Path(__file__).resolve().parent
URL = (
    "https://gabls.metoffice.gov.uk/"
    "gabls_data_zip/res_12.5m.tar.gz"
)
SHA256 = "4e3349e56f7c5460674984d40e0b6c12ccd3e826e9c474d0b943b5d48cddea8d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        help="Use an already downloaded archive instead of fetching it.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "reference" / "official_12p5m",
    )
    return parser.parse_args()


def download(destination: Path) -> None:
    # The legacy Met Office host currently presents a certificate for a
    # different metoffice.gov.uk hostname.  Integrity is enforced below by a
    # pinned SHA-256 before any archive member is opened.
    context = ssl._create_unverified_context()  # noqa: S323
    with urlopen(URL, context=context, timeout=60) as response:  # noqa: S310
        with destination.open("wb") as stream:
            shutil.copyfileobj(response, stream)


def verify(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256:
        raise SystemExit(
            f"ERROR: official archive hash mismatch: {digest} != {SHA256}"
        )


def extract(archive: Path, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with tarfile.open(archive, mode="r:gz") as bundle:
        for member in bundle.getmembers():
            path = PurePosixPath(member.name)
            if (
                not member.isfile()
                or path.suffix != ".dat"
                or len(path.parts) != 3
                or path.parts[0] != "res_12.5m"
            ):
                continue
            participant, filename = path.parts[1:]
            destination = output_dir / participant / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = bundle.extractfile(member)
            if source is None:
                raise SystemExit(f"ERROR: cannot read {member.name}")
            with destination.open("wb") as stream:
                shutil.copyfileobj(source, stream)
            extracted += 1
    metadata = {
        "source": URL,
        "sha256": SHA256,
        "files": extracted,
        "citation": (
            "Beare et al. (2006), Boundary-Layer Meteorology 118, "
            "247-272, doi:10.1007/s10546-004-2820-6"
        ),
    }
    (output_dir / "SOURCE.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return extracted


def main() -> None:
    args = parse_args()
    if args.archive is None:
        with tempfile.TemporaryDirectory(prefix="gabls1-") as temporary:
            archive = Path(temporary) / "res_12.5m.tar.gz"
            download(archive)
            verify(archive)
            count = extract(archive, args.output_dir)
    else:
        verify(args.archive)
        count = extract(args.archive, args.output_dir)
    print(f"[reference] extracted {count} official files to {args.output_dir}")


if __name__ == "__main__":
    main()
