#!/usr/bin/env python3
"""Download and safely extract an official GABLS1 LES archive."""

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
ARCHIVES = {
    "12.5m": {
        "sha256": "4e3349e56f7c5460674984d40e0b6c12ccd3e826e9c474d0b943b5d48cddea8d",
        "output": "official_12p5m",
    },
    "6.25m": {
        "sha256": "166ac6a20733269d285d3ae694df5573c222e89648eb346824a0947bc3b9f31f",
        "output": "official_6p25m",
    },
}
BASE_URL = "https://gabls.metoffice.gov.uk/gabls_data_zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resolution",
        choices=tuple(ARCHIVES),
        default="12.5m",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="Use an already downloaded archive instead of fetching it.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="defaults to reference/official_<resolution>",
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = HERE / "reference" / ARCHIVES[args.resolution]["output"]
    return args


def download(destination: Path, url: str) -> None:
    # The legacy Met Office host currently presents a certificate for a
    # different metoffice.gov.uk hostname.  Integrity is enforced below by a
    # pinned SHA-256 before any archive member is opened.
    context = ssl._create_unverified_context()  # noqa: S323
    with urlopen(url, context=context, timeout=60) as response:  # noqa: S310
        with destination.open("wb") as stream:
            shutil.copyfileobj(response, stream)


def verify(path: Path, expected_sha256: str) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise SystemExit(
            f"ERROR: official archive hash mismatch: {digest} != {expected_sha256}"
        )


def extract(
    archive: Path,
    output_dir: Path,
    *,
    resolution: str,
    url: str,
    sha256: str,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with tarfile.open(archive, mode="r:gz") as bundle:
        for member in bundle.getmembers():
            path = PurePosixPath(member.name)
            if (
                not member.isfile()
                or path.suffix != ".dat"
                or len(path.parts) != 3
                or path.parts[0] != f"res_{resolution}"
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
        "source": url,
        "sha256": sha256,
        "resolution_m": float(resolution.removesuffix("m")),
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
    specification = ARCHIVES[args.resolution]
    url = f"{BASE_URL}/res_{args.resolution}.tar.gz"
    sha256 = specification["sha256"]
    if args.archive is None:
        with tempfile.TemporaryDirectory(prefix="gabls1-") as temporary:
            archive = Path(temporary) / f"res_{args.resolution}.tar.gz"
            download(archive, url)
            verify(archive, sha256)
            count = extract(
                archive,
                args.output_dir,
                resolution=args.resolution,
                url=url,
                sha256=sha256,
            )
    else:
        verify(args.archive, sha256)
        count = extract(
            args.archive,
            args.output_dir,
            resolution=args.resolution,
            url=url,
            sha256=sha256,
        )
    print(f"[reference] extracted {count} official files to {args.output_dir}")


if __name__ == "__main__":
    main()
